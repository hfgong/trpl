"""Pre-tracking: per-frame track / propose / initialize.

Ports ``pretr_main.cpp`` and its four stages.  The MPI in the original only
parallelizes candidate scoring; results are identical serially.  We follow the
MPI overloads (which -- unlike the serial ones -- also record ``trj_3d`` from
the fused ground peak).

``segment_parts`` is omitted: its only consumer is ``filter_trlet``'s
``seg_score``, which (a C++ quirk) counts every stored mask pixel, i.e. the box
area.  We recompute that directly from boxes in :mod:`trpl_track.filter`.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import features, io_utils
from .config import NCAM, NPART
from .features import (compute_part_rects, collect_hist, get_cand_hist_score,
                       enumerate_rects_inpoly, combine_ground_score,
                       rects_to_pmodel_geom, PModel)
from .geometry import apply_homography
from .tracklet import ObjectTraj


class ObjectInfo:
    """Live tracking state (``object_info_t``)."""

    def __init__(self, T, ncam, P):
        self.T = T
        self.ncam = ncam
        self.P = P
        self.curr_num_obj = 0
        self.trlet_list: list[ObjectTraj] = []
        self.model = P.part_model()
        self.pmodel = {}                 # (cam, nn) -> PModel


def _rounded_range(lo, hi, step):
    """C++ ``for(v=lo; v<=hi; v+=step)`` (float accumulation)."""
    out = []
    v = lo
    while v <= hi + 1e-9:
        out.append(v)
        v += step
    return np.array(out, float)


def _ground_peak_to_box(gi, pm, P, best_x, best_y, best_s, cam, tt):
    ix, iy = apply_homography(gi.grd2img_t(tt, cam), np.array([float(best_x)]),
                             np.array([float(best_y)]))
    cur_fx, cur_fy = float(ix[0]), float(iy[0])
    cur_hy = gi.horiz_mean + pm.hpre * (cur_fy - gi.horiz_mean)
    ds = P.scales[best_s] * (cur_fy - cur_hy) / pm.bh
    ww = ds * pm.bw
    return np.array([cur_fx - ww / 2, cur_hy, cur_fx + ww / 2, cur_fy], np.float32)


def determine_trlet_state(scores, P) -> int:
    """1 = active, 2 = lost (needs ~one strong + one medium view)."""
    if np.any(np.isnan(scores)):
        return 2
    val = 0
    for s in scores:
        val += int(s >= P.occl_thr1) + int(s >= P.occl_thr2)
    return 2 if val < 3 else 1


def _track_one(P, gi, oi, tt, cam, nn, bin_img):
    t = oi.trlet_list[nn]
    xrange, yrange = float(P.xrange), float(P.yrange)
    xstep, ystep = float(P.xstep), float(P.ystep)
    if tt > 0 and t.scores[cam, tt - 1] < P.thr:
        xrange *= 2; yrange *= 2; xstep *= 2; ystep *= 2
    xr = _rounded_range(-xrange, xrange, xstep)
    yr = _rounded_range(-yrange, yrange, ystep)

    feetx = (t.trj[cam][tt - 1, 0] + t.trj[cam][tt - 1, 2]) / 2
    feety = t.trj[cam][tt - 1, 3]
    pm = oi.pmodel[(cam, nn)]
    cr, cs, cij, ca = enumerate_rects_inpoly(pm, feetx, feety, xr, yr, P.scales,
                                             gi.horiz_mean, gi.horiz_sig,
                                             gi.polys_im_t(tt, cam))
    if cr.shape[0] == 0:
        ca.fill_score(np.zeros(0), np.zeros((0, 3), int))
        return ca
    sm, fsc = get_cand_hist_score(bin_img, oi.model, P.logp1, P.logp2,
                                  t.hist_p[cam], t.hist_q[cam], cr)
    idx = int(np.argmax(sm))
    t.trj[cam][tt] = cr[idx]
    t.scores[cam, tt] = sm[idx]
    t.fscores[cam][:, tt] = fsc[idx]
    ca.fill_score(sm, cij)
    return ca


def _update_one(P, oi, tt, cam, nn, bin_img):
    t = oi.trlet_list[nn]
    ww = t.trj[cam][tt, 2] - t.trj[cam][tt, 0]
    hh = t.trj[cam][tt, 3] - t.trj[cam][tt, 1]
    rects = compute_part_rects(t.trj[cam][tt, 0], t.trj[cam][tt, 1], ww, hh, oi.model)
    max_score = t.scores[cam, tt]
    fglr2 = P.fglr / (1 + np.exp(P.thr - max_score))
    hp, hq = collect_hist(bin_img, rects)
    t.hist_p[cam] = hp * fglr2 + t.hist_p[cam] * (1 - fglr2)
    t.hist_q[cam] = hq * P.bglr + t.hist_q[cam] * (1 - P.bglr)


def track_existed_objects(P, gi, oi, tt, bin_imgs):
    for nn in range(oi.curr_num_obj):
        t = oi.trlet_list[nn]
        if t.is_empty() or t.state >= 2:
            continue
        cand_arrays = [_track_one(P, gi, oi, tt, cam, nn, bin_imgs[cam])
                       for cam in range(oi.ncam)]
        gmap = combine_ground_score(cand_arrays, gi, tt)
        best_y, best_x, best_s = gmap.peak()
        t.trj_3d[tt] = [best_x, best_y]
        for cam in range(oi.ncam):
            t.trj[cam][tt] = _ground_peak_to_box(gi, oi.pmodel[(cam, nn)], P,
                                                 best_x, best_y, best_s, cam, tt)
        scores = t.scores[:, tt].copy()
        t.state = determine_trlet_state(scores, P)
        if t.state <= 1:
            t.endt = tt
            for cam in range(oi.ncam):
                _update_one(P, oi, tt, cam, nn, bin_imgs[cam])
        if t.state == 2:
            for cam in range(oi.ncam):
                t.scores[cam, tt] = 0
                t.fscores[cam][:, tt] = 0


def propose_new_objects(P, ds, oi, seq, tt):
    """Read refined detections; drop those already covered by a live track."""
    ncam = len(seq)
    ped_boxes = []
    for cam in range(ncam):
        name = io_utils.image_basename(seq[cam][tt])
        det = io_utils.read_detection_refine(io_utils.detection_refine_path(ds, name))
        ped_boxes.append(det.astype(float).copy())
    ndect = ped_boxes[0].shape[0]
    tracked = np.zeros(ndect, int)

    for oo in range(ndect):
        hit = False
        for cam in range(ncam):
            h = ped_boxes[cam][oo, 3] - ped_boxes[cam][oo, 1]
            ped_boxes[cam][oo, 1] += h / 10        # detection-defect fix
            r1 = ped_boxes[cam][oo, :4]
            ar = (r1[2] - r1[0]) * (r1[3] - r1[1])
            for nn in range(oi.curr_num_obj):
                t = oi.trlet_list[nn]
                if t.is_empty() or tt < t.startt or tt > t.endt:
                    continue
                r2 = t.trj[cam][tt]
                dx = min(r1[2], r2[2]) - max(r1[0], r2[0])
                dy = min(r1[3], r2[3]) - max(r1[1], r2[1])
                inar = 0.0 if (dx < 0 or dy < 0) else dx * dy
                if inar > 0.2 * ar:
                    tracked[oo] = 1
                    hit = True
                    break
            if hit:
                break

    keep = [oo for oo in range(ndect) if not tracked[oo]]
    detected_rects = [np.array([ped_boxes[cam][oo, :4] for oo in keep],
                               float).reshape(-1, 4) for cam in range(ncam)]
    return detected_rects


def initialize_new_objects(P, gi, oi, seq, tt, bin_imgs, detected_rects):
    ncam = len(seq)
    T = oi.T
    num_new = detected_rects[0].shape[0]
    for oo in range(num_new):
        nn = oi.curr_num_obj + oo
        t = ObjectTraj(T=T, ncam=ncam, startt=tt, endt=tt, state=1)
        oi.trlet_list.append(t)

        cand_arrays = []
        for cam in range(ncam):
            box = detected_rects[cam][oo]
            w, h = box[2] - box[0], box[3] - box[1]
            t.trj[cam][tt] = box
            rects = compute_part_rects(box[0], box[1], w, h, oi.model)
            oi.pmodel[(cam, nn)] = rects_to_pmodel_geom(box, gi.horiz_mean)
            hp, hq = collect_hist(bin_imgs[cam], rects)
            t.hist_p[cam] = hp
            t.hist_q[cam] = hq

            xr = _rounded_range(-P.xrange / 2, P.xrange / 2, P.xstep)
            yr = _rounded_range(-P.yrange / 2, P.yrange / 2, P.ystep)
            feetx = (t.trj[cam][tt, 0] + t.trj[cam][tt, 2]) / 2
            feety = t.trj[cam][tt, 3]
            cr, cscl, cij, ca = enumerate_rects_inpoly(
                oi.pmodel[(cam, nn)], feetx, feety, xr, yr, P.scales,
                gi.horiz_mean, gi.horiz_sig, gi.polys_im_t(tt, cam))
            if cr.shape[0] == 0:
                ca.fill_score(np.zeros(0), np.zeros((0, 3), int))
                cand_arrays.append(ca)
                continue
            sm, fsc = get_cand_hist_score(bin_imgs[cam], oi.model, P.logp1, P.logp2,
                                          t.hist_p[cam], t.hist_q[cam], cr)
            idx = int(np.argmax(sm))
            t.fscores[cam][:, tt] = fsc[idx]
            t.scores[cam, tt] = sm[idx]
            ca.fill_score(sm, cij)
            cand_arrays.append(ca)

        gmap = combine_ground_score(cand_arrays, gi, tt)
        best_y, best_x, best_s = gmap.peak()
        t.trj_3d[tt] = [best_x, best_y]
        for cam in range(ncam):
            t.trj[cam][tt] = _ground_peak_to_box(gi, oi.pmodel[(cam, nn)], P,
                                                 best_x, best_y, best_s, cam, tt)
    oi.curr_num_obj += num_new


def run_pretr(ds, P, gi, seq=None, max_frames=None, verbose=True):
    """Run pre-tracking over the sequence; return the list of raw tracklets."""
    if seq is None:
        seq = io_utils.read_sequence_list(ds)
    T = len(seq[0])
    if max_frames is not None:
        T = min(T, max_frames)
    ncam = len(seq)
    oi = ObjectInfo(len(seq[0]), ncam, P)

    for tt in range(T):
        bin_imgs = [features.rgb_bin_image(
            np.array(Image.open(seq[cam][tt]).convert("RGB"))) for cam in range(ncam)]
        track_existed_objects(P, gi, oi, tt, bin_imgs)
        detected_rects = propose_new_objects(P, ds, oi, seq, tt)
        initialize_new_objects(P, gi, oi, seq, tt, bin_imgs, detected_rects)
        if verbose:
            n_active = sum(1 for t in oi.trlet_list[:oi.curr_num_obj]
                           if not t.is_empty() and t.endt == tt)
            print(f"  frame {tt:3d}/{T}: objects={oi.curr_num_obj} active={n_active}")

    raw = [t for t in oi.trlet_list[:oi.curr_num_obj] if not t.is_empty()]
    return raw
