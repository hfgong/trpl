"""Tracklet filtering, gating (Tff) and appearance affinity (Aff).

Ports ``filter_trlet*`` and ``prepare_app_affinity``.
"""
from __future__ import annotations

import numpy as np

from .config import NCAM

# filter_trlet_main.cpp hard-coded parameters.
SEG_THRESH = 0.4
MIN_TRLET_LEN = 3
T_THRESH = 36
V_THRESH = 50
APP_MATCH_THR = -2.0


def compute_seg_score(t) -> float:
    """Mean box area over active frames/cameras.

    Faithful to ``compute_seg_score`` + ``segment_parts``: the mask is
    ``unsigned char`` so ``count_if(elem >= 0)`` counts every pixel, i.e. the
    box area ``(h+1)*(w+1)``.
    """
    ncam = t.ncam
    length = t.length
    if length <= 0:
        return 0.0
    score = 0
    for tt in range(t.startt, t.endt + 1):
        for cam in range(ncam):
            b = np.floor(t.trj[cam][tt] + 0.5).astype(int)
            score += (b[3] - b[1] + 1) * (b[2] - b[0] + 1)
    return score / ncam / length


def filter_tracklets(raw):
    """Keep tracklets with length >= 3 and seg_score >= 0.4.

    Returns (good_list, good_index) where good_index maps compacted -> raw.
    """
    good, index = [], []
    for i, t in enumerate(raw):
        if t.length >= MIN_TRLET_LEN and compute_seg_score(t) >= SEG_THRESH:
            good.append(t)
            index.append(i)
    return good, index


def prepare_valid_linkset(good) -> np.ndarray:
    """Directed gating matrix ``Tff`` over ordered tracklet pairs.

    ``Tff[i,j]=1`` iff i may precede j: temporally disjoint & ordered
    (endt_i < startt_j), gap <= T_THRESH, and required ground-x speed
    ``|dx|/dt <= V_THRESH``.
    """
    ng = len(good)
    Tff = np.zeros((ng, ng), int)
    for i in range(ng):
        endt = good[i].endt
        xg1 = good[i].trj_3d[endt, 0]
        for j in range(ng):
            startt = good[j].startt
            xg2 = good[j].trj_3d[startt, 0]
            if endt >= startt:
                continue
            if endt + T_THRESH < startt:
                continue
            dgx = xg2 - xg1
            dt = startt - endt
            if abs(dgx) / V_THRESH > dt:
                continue
            Tff[i, j] = 1
    return Tff


def appmodel_match(t1, t2, ep=1e-6) -> float:
    """Symmetric expected log-likelihood-ratio over cam x part x bin."""
    total = 0.0
    for cam in range(t1.ncam):
        hp1, hq1 = t1.hist_p[cam], t1.hist_q[cam]
        hp2, hq2 = t2.hist_p[cam], t2.hist_q[cam]
        total += np.sum(hp1 * np.log((hp2 + ep) / (hq2 + ep)) +
                        hp2 * np.log((hp1 + ep) / (hq1 + ep)))
    return float(total)


def prepare_app_affinity(Tff, good, thr=APP_MATCH_THR) -> np.ndarray:
    """Appearance affinity ``Aff[i,j] = appmodel_match - thr`` on gated pairs."""
    ng = len(good)
    Aff = np.zeros((ng, ng))
    for i in range(ng):
        for j in range(ng):
            if not Tff[i, j]:
                continue
            Aff[i, j] = appmodel_match(good[i], good[j]) - thr
    return Aff
