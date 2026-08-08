"""Appearance features and candidate scoring (``tracking_detail.hpp``).

The only appearance feature is an 8x8x8 RGB color histogram
(``bin = R//32 + (G//32)*8 + (B//32)*64``).  Candidate boxes are scored by a
part-based figure/ground model; per-camera score grids are fused on the ground
plane and the argmax gives the object's 3-D position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NPART, NBINS
from .geometry import apply_homography, point_in_polygon


def sat(v, u):
    return np.minimum(v, u)


def sat2(v, lo, hi):
    return np.clip(v, lo, hi)


def rgb_bin_image(image: np.ndarray) -> np.ndarray:
    """Quantize an HxWx3 uint8 RGB image to the 512-bin index per pixel."""
    q = image.astype(np.int32) // 32
    return q[:, :, 0] + q[:, :, 1] * 8 + q[:, :, 2] * 64


@dataclass
class PModel:
    """Per-object geometric shape model (``pmodel_t``)."""
    bw: float = 0.0     # reference box width
    bh: float = 0.0     # reference box height
    hpre: float = 0.0   # head/feet ratio relative to horizon


def rects_to_pmodel_geom(rect, horiz_mean: float) -> PModel:
    return PModel(bw=rect[2] - rect[0], bh=rect[3] - rect[1],
                  hpre=(rect[1] - horiz_mean) / (rect[3] - horiz_mean))


def compute_part_rects(x, y, w, h, model) -> np.ndarray:
    """Part rects ``[x0,y0,x1,y1]`` from box origin/size and normalized model."""
    m = np.asarray(model, float)                       # NPART x 4
    out = np.empty((len(m), 4), float)
    out[:, 0] = m[:, 0] * w + x
    out[:, 1] = m[:, 1] * h + y
    out[:, 2] = m[:, 2] * w + x
    out[:, 3] = m[:, 3] * h + y
    return out


def collect_hist(bin_img: np.ndarray, rects: np.ndarray):
    """Foreground/background color histograms per part.

    ``bin_img`` is the precomputed HxW 512-bin index image.  Returns
    ``(hist_p, hist_q)`` each ``NPART x NBINS``, L1-normalized per row.
    Mirrors ``collect_hist`` in tracking_detail.hpp: an expanded box (2x width,
    1.5x height) is subsampled on a 15x30 grid; samples inside the part rect go
    to foreground, the surround to background.
    """
    H, W = bin_img.shape
    npart = rects.shape[0]
    hist_p = np.zeros((npart, NBINS), np.float64)
    hist_q = np.zeros((npart, NBINS), np.float64)

    rcx = (rects[:, 0] + rects[:, 2]) / 2.0
    rcy = (rects[:, 1] + rects[:, 3]) / 2.0
    rw = rects[:, 2] - rects[:, 0]
    rh = rects[:, 3] - rects[:, 1]

    exbb = np.empty((npart, 4))
    exbb[:, 0] = rcx - rw
    exbb[:, 1] = rcy - 0.75 * rh
    exbb[:, 2] = rcx + rw
    exbb[:, 3] = rcy + 0.75 * rh

    exbbi = np.floor(exbb + 0.5).astype(int)
    inbbi = np.floor(rects + 0.5).astype(int)
    exbbi[:, 0] = np.clip(exbbi[:, 0], 0, None)
    exbbi[:, 1] = np.clip(exbbi[:, 1], 0, None)
    exbbi[:, 2] = np.clip(exbbi[:, 2], None, W - 1)
    exbbi[:, 3] = np.clip(exbbi[:, 3], None, H - 1)

    for pp in range(npart):
        ex0, ey0, ex1, ey1 = exbbi[pp]
        if ex1 < ex0 or ey1 < ey0:
            continue
        dy = max((ey1 - ey0) / 30.0, 1.0)
        dx = max((ex1 - ex0) / 15.0, 1.0)
        ys = np.arange(ey0, ey1 + 1e-9, dy)
        xs = np.arange(ex0, ex1 + 1e-9, dx)
        if ys.size == 0 or xs.size == 0:
            continue
        YY, XX = np.meshgrid(ys, xs, indexing="ij")
        yi = np.clip((YY + 0.5).astype(int), 0, H - 1)
        xi = np.clip((XX + 0.5).astype(int), 0, W - 1)
        bins = bin_img[yi, xi].ravel()
        in0, iny0, in1, iny1 = inbbi[pp]
        fg = ((YY >= iny0) & (YY <= iny1) & (XX >= in0) & (XX <= in1)).ravel()
        hp = np.bincount(bins[fg], minlength=NBINS)[:NBINS]
        hq = np.bincount(bins[~fg], minlength=NBINS)[:NBINS]
        sp, sq = hp.sum(), hq.sum()
        if sp > 0:
            hist_p[pp] = hp / sp
        if sq > 0:
            hist_q[pp] = hq / sq
    return hist_p, hist_q


def kldivergence_rows(hp: np.ndarray, hq: np.ndarray) -> np.ndarray:
    """Per-row Sum hp*log((hp+e)/(hq+e)), clamped >= 0 (e=1e-6)."""
    v = np.sum(hp * np.log((hp + 1e-6) / (hq + 1e-6)), axis=1)
    return np.maximum(v, 0.0)


def _expected_llratio(h, ep, p, q):
    return np.sum(h * np.log((p + ep) / (q + ep)), axis=1)


def compute_consistent_score(hist_p, hist_q, p, q, ep) -> np.ndarray:
    v1 = _expected_llratio(hist_p, ep, p, q)
    v2 = _expected_llratio(p, ep, hist_p, hist_q)
    return sat((v1 + v2) / 2.0, 2.5) * 2.0


def get_cand_hist_score(bin_img, model, logp1, logp2, p, q, cand_rects):
    """Score each candidate box; returns (score_map[nc], feature_scores[nc,6]).

    Ports ``get_cand_hist_score``: contrast (KL fg/bg) + consistency (match to
    stored model ``p``/``q``) combined per part with a visible/occluded
    log-sum-exp over two hypotheses.
    """
    nc = cand_rects.shape[0]
    ep = 1e-4
    wk, wc = 0.5, 2.0
    logp1 = np.asarray(logp1, float)
    logp2 = np.asarray(logp2, float)
    score_map = np.zeros(nc)
    feature_scores = np.zeros((nc, NPART * 2))

    for ii in range(nc):
        x, y = cand_rects[ii, 0], cand_rects[ii, 1]
        w = cand_rects[ii, 2] - x
        h = cand_rects[ii, 3] - y
        rects = compute_part_rects(x, y, w, h, model)
        hist_p, hist_q = collect_hist(bin_img, rects)

        kll = kldivergence_rows(hist_p, hist_q)
        kll = sat(kll, 6.0) * 0.56
        cs = compute_consistent_score(hist_p, hist_q, p, q, ep)

        lscore1 = wk * kll + wc * cs + logp1
        lscore2 = wk * kll / 2.0 + logp2
        lscore0 = np.maximum(lscore1, lscore2)
        combined = np.log(np.exp(lscore1 - lscore0) + np.exp(lscore2 - lscore0)) + lscore0
        score_map[ii] = combined.sum()
        feature_scores[ii, :NPART] = kll
        feature_scores[ii, NPART:] = cs
    return score_map, feature_scores


class CandidateArray:
    """Scored grid over feet-positions (fy x fx) x scales (``candidate_array``)."""

    def __init__(self):
        self.fx = None
        self.fy = None
        self.grid = None            # ny x nx x ns
        self.default_value = -5.0

    def fill_fxfy(self, feetx, feety, xr, yr, ns):
        self.fx = np.asarray(xr, float) + feetx
        self.fy = np.asarray(yr, float) + feety
        self.grid = np.full((len(self.fy), len(self.fx), ns), -5.0)

    def fill_score(self, cand_score, cand_ijs, def_val=-5.0):
        self.default_value = def_val
        self.grid[:] = def_val
        for ii in range(len(cand_score)):
            y, x, s = cand_ijs[ii]
            self.grid[y, x, s] = cand_score[ii]

    def get_subpixel_score(self, y, x):
        return self.get_subpixel_score_batch(np.array([y]), np.array([x]))[0]

    def get_subpixel_score_batch(self, ys, xs):
        """Bilinearly sample the score grid at many (y,x); returns (N, ns)."""
        ys = np.asarray(ys, float)
        xs = np.asarray(xs, float)
        ns = self.grid.shape[2]
        fy, fx = self.fy, self.fx
        out = np.full((ys.size, ns), self.default_value)
        inside = (ys < fy[-1]) & (ys >= fy[0]) & (xs < fx[-1]) & (xs >= fx[0])
        if not inside.any():
            return out
        yi, xi = ys[inside], xs[inside]
        i0 = np.clip(np.searchsorted(fy, yi, side="right") - 1, 0, len(fy) - 2)
        j0 = np.clip(np.searchsorted(fx, xi, side="right") - 1, 0, len(fx) - 2)
        i1, j1 = i0 + 1, j0 + 1
        ly = ((yi - fy[i0]) / (fy[i1] - fy[i0]))[:, None]
        lx = ((xi - fx[j0]) / (fx[j1] - fx[j0]))[:, None]
        out[inside] = ((1 - ly) * (1 - lx) * self.grid[i0, j0] +
                       (1 - ly) * lx * self.grid[i0, j1] +
                       ly * (1 - lx) * self.grid[i1, j0] +
                       ly * lx * self.grid[i1, j1])
        return out


def enumerate_rects_inpoly(pmodel: PModel, feetx, feety, xr, yr, scales,
                           horiz_mean, horiz_sig, poly_im):
    """Generate candidate boxes from ground/horizon geometry.

    Returns (cand_rects[num,4], cand_scale[num], cand_ijs[num,3], cand_array).
    Rejects boxes whose implied top is outside horiz_mean +/- 6*sig, or whose
    feet fall outside the walkable polygon.
    """
    xr = np.asarray(xr, float)
    yr = np.asarray(yr, float)
    ns = len(scales)
    cand_array = CandidateArray()
    cand_array.fill_fxfy(feetx, feety, xr, yr, ns)

    rects, scale_idx, ijs = [], [], []
    hpre = pmodel.hpre
    for ii, dy in enumerate(yr):
        cur_fy = dy + feety
        cur_hy = horiz_mean + hpre * (cur_fy - horiz_mean)
        zoom = (cur_fy - cur_hy) / pmodel.bh
        horiz_ok = (horiz_mean - 6 * horiz_sig) <= cur_hy <= (horiz_mean + 6 * horiz_sig)
        for jj, dx in enumerate(xr):
            cur_fx = dx + feetx
            in_poly = horiz_ok and point_in_polygon(cur_fx, cur_fy, poly_im[0], poly_im[1])
            for kk in range(ns):
                if not (horiz_ok and in_poly):
                    continue
                ds = scales[kk] * zoom
                w = ds * pmodel.bw
                h = ds * pmodel.bh
                rects.append([cur_fx - w / 2, cur_hy, cur_fx + w / 2, cur_fy])
                scale_idx.append(kk)
                ijs.append([ii, jj, kk])
    if rects:
        return (np.array(rects, float), np.array(scale_idx, float),
                np.array(ijs, int), cand_array)
    return (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 3), int), cand_array)


@dataclass
class GroundScoreMap:
    x0: int
    y0: int
    scores: np.ndarray          # (ny, nx, ns)

    def peak(self):
        idx = np.unravel_index(np.argmax(self.scores), self.scores.shape)
        return self.y0 + idx[0], self.x0 + idx[1], idx[2]


def combine_ground_score(cand_arrays, gi, tt=0) -> GroundScoreMap:
    """Warp each camera's score grid to the ground plane and sum them.

    ``tt`` selects the per-frame homography in dynamic (moving-camera) mode.
    """
    ncam = len(cand_arrays)
    ns = cand_arrays[0].grid.shape[2]
    gl = gi.ground_lim
    maxx, minx = float(gl.xmin), float(gl.xmax)
    maxy, miny = float(gl.ymin), float(gl.ymax)

    for cam in range(ncam):
        ca = cand_arrays[cam]
        x0, x1 = ca.fx[0], ca.fx[-1]
        y0, y1 = ca.fy[0], ca.fy[-1]
        imx = np.array([x0, x1, x1, x0])
        imy = np.array([y0, y0, y1, y1])
        gx, gy = apply_homography(gi.img2grd_t(tt, cam), imx, imy)
        maxx = max(maxx, gx.max()); maxy = max(maxy, gy.max())
        minx = min(minx, gx.min()); miny = min(miny, gy.min())

    maxx = min(maxx, gl.xmax); maxy = min(maxy, gl.ymax)
    minx = max(minx, gl.xmin); miny = max(miny, gl.ymin)

    gx0, gy0 = int(minx), int(miny)
    gx1, gy1 = int(maxx + 1), int(maxy + 1)
    scores = np.zeros((gy1 - gy0, gx1 - gx0, ns))

    gxs = np.arange(gx0, gx1, dtype=float)
    gys = np.arange(gy0, gy1, dtype=float)
    GX, GY = np.meshgrid(gxs, gys)                      # (ny, nx)
    flatx, flaty = GX.ravel(), GY.ravel()
    for cam in range(ncam):
        ca = cand_arrays[cam]
        imx, imy = apply_homography(gi.grd2img_t(tt, cam), flatx, flaty)
        sc = ca.get_subpixel_score_batch(imy, imx)     # (ny*nx, ns)
        scores += sc.reshape(scores.shape)
    return GroundScoreMap(x0=gx0, y0=gy0, scores=scores)
