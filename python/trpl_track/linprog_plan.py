"""Planning affinity (``Plff``) and the full linking LP.

Ports ``linprog_plan_impl.hpp`` + the ``linprog_plan_main`` orchestration:

1. match each tracklet's motion plans to itself and to candidate successors,
2. synthesize interpolated "gap" tracklets that bridge each candidate link,
3. score the gaps by appearance against both endpoints,
4. pick the best plan per link -> ``Plff``,
5. solve ``max (Aff + 0.5*Plff) . x`` with the matching LP.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import features, io_utils
from .config import NCAM
from .features import get_cand_hist_score, sat, sat2, rgb_bin_image
from .geometry import apply_homography
from .lp import solve_linprog
from .tracklet import ObjectTraj

PLAN_ADVANCE = 7
PLFF_THR = 6.0


def nearest_point(path, pos):
    d = (path[:, 0] - pos[0]) ** 2 + (path[:, 1] - pos[1]) ** 2
    return int(np.argmin(d))


def chord_length(path):
    if len(path) == 0:
        return np.zeros(0)
    seg = np.zeros(len(path))
    seg[1:] = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
    return np.cumsum(seg)


def trj2path_distance(trj, path):
    """Sum of squared image-plane deviations of a trajectory from a path.

    Faithful to the C++ (including its arc-length index bookkeeping).
    """
    p0 = nearest_point(path, trj[0])
    pv = path[p0:]
    len_path = chord_length(pv)
    len_trj = chord_length(trj)
    p = np.zeros(len(trj), int)
    p[0] = p0
    for ii in range(1, len(trj)):
        p[ii] = int(np.argmin(np.abs(len_path - len_trj[ii])))
    trj2 = path[p]
    return float(np.sum((trj2 - trj) ** 2))


def guided_gap_interp(pos1, pos2, path, dt):
    """Smooth interior gap positions following the planned path shape."""
    if dt <= 0:
        return np.zeros((0, 2))
    p1 = nearest_point(path, pos1)
    p2 = nearest_point(path, pos2)
    len_path = chord_length(path)
    l1, l2 = len_path[p1], len_path[p2]
    p = np.zeros(dt + 2, int)
    p[0], p[-1] = p1, p2
    for ii in range(1, dt + 1):
        ll = (l1 * (dt + 1 - ii) + l2 * ii) / (dt + 1)
        p[ii] = int(np.argmin(np.abs(len_path - ll)))
    Q = path[p]                                   # (dt+2, 2)
    S = (np.diag(np.full(dt, 2.0)) + np.diag(np.full(dt - 1, -1.0), 1) +
         np.diag(np.full(dt - 1, -1.0), -1))
    b = 2 * Q[1:-1] - Q[0:-2] - Q[2:]
    b[0] = b[0] + pos1
    b[-1] = b[-1] + pos2
    return np.linalg.solve(S, b)


def trj2plan_distance(plan_item_list, trlet, gi, tt1, tt2):
    """Per-goal, per-path image-plane fit of a tracklet span to the plans.

    Computes ``Dist(r, F.)`` from Sec 4.2 for one tracklet span ``[tt1,tt2]``
    against every planned path ``r``. The deviation is SUMMED over both cameras
    (``dist[kk][pp] += ...`` inside the ``cam`` loop), so the returned value is a
    single scalar per (goal ``kk``, path ``pp``) -- no camera index survives.
    This is what makes ``dd`` in :func:`select_plan_gap_paths` camera-independent.

    In dynamic (moving-camera) mode the plan path (fixed world ground coords) is
    projected to the image with the span's MIDPOINT-frame homography
    ``grd2img((tt1+tt2)//2, cam)`` -- matching sequence5's C++.
    """
    ncam = len(trlet.trj)
    tmid = (tt1 + tt2) // 2
    dist = [np.zeros(len(pi.paths)) for pi in plan_item_list]
    for cam in range(ncam):
        tr = trlet.trj[cam][tt1:tt2 + 1]
        trf = np.column_stack([(tr[:, 0] + tr[:, 2]) / 2.0, tr[:, 3]])
        H = gi.grd2img_t(tmid, cam)
        for kk, pi in enumerate(plan_item_list):
            for pp, gpath in enumerate(pi.paths):
                imx, imy = apply_homography(H, gpath[:, 0], gpath[:, 1])
                path_img = np.column_stack([imx, imy]).astype(float)
                dist[kk][pp] += trj2path_distance(trf.astype(float), path_img)
    return dist


def match_plan_to_trlet(seq, gi, plan_advance, good, Tff, plan_time, plan_results):
    ng = len(good)
    cost_mat = [[None] * ng for _ in range(ng)]
    dist_mat = [[None] * ng for _ in range(ng)]
    reduced_paths = [[None] * ng for _ in range(ng)]
    for nn in range(ng):
        if plan_results[nn] is None:
            continue
        tt1 = plan_time[nn]
        tt2 = good[nn].endt
        dist1 = trj2plan_distance(plan_results[nn], good[nn], gi, tt1, tt2)
        for mm in range(ng):
            if not Tff[nn, mm]:
                continue
            ss1 = good[mm].startt
            ss2 = min(good[mm].startt + plan_advance, good[mm].endt)
            dist2 = trj2plan_distance(plan_results[nn], good[mm], gi, ss1, ss2)
            denom = 2 + tt2 - tt1 + ss2 - ss1
            dv, cv, gidx, pidx = [], [], [], []
            for kk, pi in enumerate(plan_results[nn]):
                for pp in range(len(pi.paths)):
                    dv.append((dist1[kk][pp] + dist2[kk][pp]) / denom)
                    cv.append(pi.dist[pp])
                    gidx.append(kk)
                    pidx.append(pp)
            dv = np.array(dv)
            if dv.size == 0:
                dist_mat[nn][mm] = np.array([1200.0])
                cost_mat[nn][mm] = np.array([800.0])
                reduced_paths[nn][mm] = []
                continue
            md = dv.min()
            idx = [i for i in range(len(dv)) if dv[i] - md < 800]
            order = sorted(idx, key=lambda i: dv[i])
            num = min(8, len(order))
            keep = order[:num]
            dist_mat[nn][mm] = np.array([np.sqrt(dv[i]) for i in keep])
            cost_mat[nn][mm] = np.array([cv[i] for i in keep])
            reduced_paths[nn][mm] = [plan_results[nn][gidx[i]].paths[pidx[i]] for i in keep]
            if len(idx) == 0:
                dist_mat[nn][mm] = np.array([1200.0])
                cost_mat[nn][mm] = np.array([800.0])
    return cost_mat, dist_mat, reduced_paths


def compute_plan_gap_trlet(gi, T, ncam, good, Tff, reduced_paths):
    ng = len(good)
    gap = [[[] for _ in range(ng)] for _ in range(ng)]
    for ii in range(ng):
        for jj in range(ng):
            if not Tff[ii, jj]:
                continue
            paths = reduced_paths[ii][jj] or []
            nr = len(paths)
            t1, t2 = good[ii].endt, good[jj].startt
            ww = np.zeros((2, ncam)); hh = np.zeros((2, ncam))
            for cam in range(ncam):
                ww[0, cam] = good[ii].trj[cam][t1, 2] - good[ii].trj[cam][t1, 0]
                ww[1, cam] = good[jj].trj[cam][t2, 2] - good[jj].trj[cam][t2, 0]
                hh[0, cam] = good[ii].trj[cam][t1, 3] - good[ii].trj[cam][t1, 1]
                hh[1, cam] = good[jj].trj[cam][t2, 3] - good[jj].trj[cam][t2, 1]
            pos1 = good[ii].trj_3d[t1]
            pos2 = good[jj].trj_3d[t2]

            def make():
                g = ObjectTraj(T=T, ncam=ncam, startt=t1 + 1, endt=t2 - 1)
                return g

            def fill(g, interp_fn):
                for tt in range(t1 + 1, t2):
                    pos = interp_fn(tt)
                    g.trj_3d[tt] = pos
                    for cam in range(ncam):
                        ix, iy = apply_homography(gi.grd2img_t(tt, cam),
                                                  np.array([pos[0]]), np.array([pos[1]]))
                        wwt = (ww[0, cam] * (t2 - tt) + ww[1, cam] * (tt - t1)) / (t2 - t1)
                        hht = (hh[0, cam] * (t2 - tt) + hh[1, cam] * (tt - t1)) / (t2 - t1)
                        g.trj[cam][tt] = [ix[0] - wwt / 2, iy[0] - hht,
                                          ix[0] + wwt / 2, iy[0]]

            if nr == 0:
                g = make()
                fill(g, lambda tt: (pos1 * (t2 - tt) + pos2 * (tt - t1)) / (t2 - t1))
                gap[ii][jj] = [g]
            else:
                lst = []
                for rr in range(nr):
                    g = make()
                    interpos = guided_gap_interp(pos1, pos2, paths[rr], t2 - t1 - 1)
                    fill(g, lambda tt, ip=interpos: ip[tt - t1 - 1])
                    lst.append(g)
                gap[ii][jj] = lst
    return gap


def compute_plan_gap_scores(seq, gi, P, good, model, Tff, gap):
    """Score each synthesized gap box by appearance -> fills ``S_Occl`` evidence.

    For every gap frame/camera, refine the interpolated box over a 3x3 offset
    grid and score it against BOTH endpoint tracklets' part-histogram models
    (``good[nn1]`` and ``good[nn2]``), take the best, and store the averaged
    score in ``g.scores[cam, tt]``. These per-(camera, frame) values are the raw
    ``S_Occl`` evidence that :func:`select_plan_gap_paths` clips to [2,8], shifts
    by -2, and sums over cameras and gap frames.
    """
    ncam = len(seq)
    T = len(seq[0])
    ng = len(good)
    dx = np.array([-4.0, 0.0, 4.0])
    dy = np.array([-4.0, 0.0, 4.0])
    for tt in range(T):
        pairs = []
        for ii in range(ng):
            for jj in range(ng):
                if not gap[ii][jj]:
                    continue
                g0 = gap[ii][jj][0]
                if g0.startt <= tt <= g0.endt:
                    pairs.append((ii, jj))
        if not pairs:
            continue
        bin_imgs = [rgb_bin_image(np.array(Image.open(seq[cam][tt]).convert("RGB")))
                    for cam in range(ncam)]
        for (ii, jj) in pairs:
            for g in gap[ii][jj]:
                for cam in range(ncam):
                    bodyr = g.trj[cam][tt]
                    cr = _enumerate_rects_refine(bodyr, dx, dy)
                    cand_sum = np.zeros(cr.shape[0])
                    for nn in (ii, jj):
                        sm, _ = get_cand_hist_score(bin_imgs[cam], model, P.logp1,
                                                    P.logp2, good[nn].hist_p[cam],
                                                    good[nn].hist_q[cam], cr)
                        sm = np.where(np.isnan(sm), -10.0, sm)
                        cand_sum += sm
                    idx = int(np.argmax(cand_sum))
                    g.scores[cam, tt] = cand_sum[idx] / 2.0
                    g.trj[cam][tt] = cr[idx]


def _enumerate_rects_refine(rect, dx, dy):
    out = []
    for yy in dy:
        for xx in dx:
            out.append([rect[0] + xx, rect[1] + yy, rect[2] + xx, rect[3] + yy])
    return np.array(out, float)


def select_plan_gap_paths(Tff, ncam, gap, dist_mat, reduced_paths):
    """Pick the best gap-filling plan per link and build the planning affinity.

    Paper mapping (Gong, Sim, Shi, ICCV 2011):

    * Eq (11): the LP maximizes ``S_App(i,j) + alpha * S_Plan(i,j)`` per link.
      This function returns ``S_Plan`` as ``Plff``; ``alpha = 0.5`` is applied
      by the caller and the acceptance offset ``beta`` is folded into
      ``plff_thr`` (subtracted at the end of :func:`prepare_plan_affinity`).
    * Sec 4.2:
        ``S_Plan(i,j) = max_{r in paths} [ -Dist(r,F_i) - Dist(r,F_j)
                                           + S_Occl(F_i,F_j,r) ]``
      where ``Dist(r,F_.)`` is a tracklet-to-path deviation and ``S_Occl`` is
      the appearance evidence along the hallucinated gap.

    Variable tracking (per candidate path ``rr`` of link ``ii -> jj``):

    * ``dd = dist_mat[ii][jj]`` : ``dd[rr] = Dist(r,F_i)+Dist(r,F_j)``, ONE
      scalar per path. Upstream (:func:`match_plan_to_trlet` /
      :func:`trj2plan_distance`) it is already summed over both tracklet ends
      AND both cameras -- it has no camera index.
    * ``g.scores[cam, tt]`` : per-camera, per-gap-frame appearance score set by
      :func:`compute_plan_gap_scores` against BOTH endpoint models -> ``S_Occl``.
    * ``score[rr]`` : the combined ``S_Plan`` for path ``rr`` (before the cap).

    Then ``Plff[i,j] = max_rr sat(score[rr], 16)`` and
    ``gap_rind[i,j] = argmax_rr``.

    BUG FIX vs. the original C++ (paper-consistent). The C++
    (``select_plan_gap_paths`` in ``linprog_plan_impl.hpp``) placed the wrap-up
    line INSIDE the ``for cam`` loop, so for a stereo pair (``ncam == 2``) the
    ``- dd/800 + 2`` term was applied twice and the ``sat(.,16)`` cap was nested
    (cam 0's partial S_Occl capped before cam 1 was added). Because ``dd[rr]``
    carries no camera dependence, that per-camera subtraction cannot come from
    the model -- it was an indentation-level slip. Here we compute S_Plan the way
    Sec 4.2 intends: accumulate S_Occl over BOTH cameras and all gap frames
    first, then apply ``-Dist``, the ``+2`` bias and the cap ONCE per path::

        s_occl    = sum_{cam,tt} ( clip(g.scores[cam,tt], 2, 8) - 2 )   # S_Occl
        score[rr] = sat(s_occl, 16) - dd[rr]/800 + 2                    # S_Plan
    """
    ng = len(gap)
    Plff = np.zeros((ng, ng))
    gap_rind = -np.ones((ng, ng), int)
    gap_paths = [[None] * ng for _ in range(ng)]
    for ii in range(ng):
        for jj in range(ng):
            if not Tff[ii, jj]:
                continue
            nr = len(gap[ii][jj])
            score = np.zeros(nr)                 # score[rr] -> S_Plan for path rr
            for rr in range(nr):
                g = gap[ii][jj][rr]
                dd = dist_mat[ii][jj]            # dd[rr] = Dist(r,F_i)+Dist(r,F_j), both cams
                # S_Occl: appearance evidence over BOTH cameras and all gap frames
                # (clip each to [2,8], shift by -2), accumulated once.
                s_occl = 0.0
                for cam in range(ncam):
                    for tt in range(g.startt, g.endt + 1):
                        s_occl += sat2(g.scores[cam, tt], 2.0, 8.0) - 2.0
                # S_Plan for this path: cap S_Occl at 16, subtract the (single)
                # plan-distance penalty, add the +2 acceptance bias -- once.
                score[rr] = sat(s_occl, 16.0) - dd[rr] / 800.0 + 2.0
            idx = int(np.argmax(score)) if nr else 0
            Plff[ii, jj] = sat(score[idx], 16.0) if nr else 0.0   # final cap at 16
            gap_rind[ii, jj] = idx
            if reduced_paths[ii][jj]:
                gap_paths[ii][jj] = reduced_paths[ii][jj][idx]
    return Plff, gap_rind, gap_paths


def prepare_plan_affinity(seq, gi, P, good, model, Tff, plan_time, plan_results,
                          plan_advance=PLAN_ADVANCE, plff_thr=PLFF_THR):
    T = len(seq[0])
    ncam = len(seq)
    cost_mat, dist_mat, reduced_paths = match_plan_to_trlet(
        seq, gi, plan_advance, good, Tff, plan_time, plan_results)
    gap = compute_plan_gap_trlet(gi, T, ncam, good, Tff, reduced_paths)
    compute_plan_gap_scores(seq, gi, P, good, model, Tff, gap)
    Plff, gap_rind, gap_paths = select_plan_gap_paths(
        Tff, ncam, gap, dist_mat, reduced_paths)
    Plff = Plff - plff_thr
    return reduced_paths, Plff, gap, gap_rind, gap_paths


def run_linprog_plan(ds, P, gi, good, Tff, Aff, seq=None, plan_grid_step=4,
                     verbose=True):
    """Full planning-based linking. Returns a result dict for finalize/vis."""
    from . import planning
    if seq is None:
        seq = io_utils.read_sequence_list(ds)
    T = len(seq[0])
    ncam = len(seq)
    model = P.part_model()

    cars = planning.load_carboxes(ds, seq)
    car_obsz = planning.prepare_car_obs(cars, gi)
    car_obs = planning.combine_car_obs(car_obsz)
    ped_obs = planning.prepare_ped_obs(good, T)

    if verbose:
        print("planning motion hypotheses ...")
    plan_time, plan_results = planning.plan_trlet_list(
        gi, Tff, good, car_obs, ped_obs, PLAN_ADVANCE, plan_grid_step, verbose)

    if verbose:
        print("preparing planning affinity ...")
    reduced_paths, Plff, gap, gap_rind, gap_paths = prepare_plan_affinity(
        seq, gi, P, good, model, Tff, plan_time, plan_results)

    C = Aff + 0.5 * Plff
    LMat, links = solve_linprog(Tff, C)
    return {
        "Plff": Plff, "Aff": Aff, "links": links, "LMat": LMat,
        "gap": gap, "gap_rind": gap_rind, "gap_paths": gap_paths,
        "reduced_paths": reduced_paths, "plan_time": plan_time,
        "plan_results": plan_results, "car_obs": car_obs, "ped_obs": ped_obs,
    }
