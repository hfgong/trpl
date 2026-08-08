"""Tracklet-linking LP (``lp.hpp`` / ``lp_impl.hpp``).

Max-weight bipartite matching on the tracklet transition graph::

    max  sum_{(i,j):Tff=1} Aff(i,j) * x_ij
    s.t. sum_j x_ij <= 1   (each tracklet <= 1 successor)
         sum_i x_ij <= 1   (each tracklet <= 1 predecessor)
         0 <= x_ij <= 1

The constraint matrix is totally unimodular, so the LP relaxation (via
``scipy.optimize.linprog(method='highs')``) is integral; GLPK's simplex is
replaced directly.  We reproduce the original's greedy row/column decode.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def solve_linprog(Tff: np.ndarray, Aff: np.ndarray):
    """Return ``(LMat, links)``: chosen-link matrix and Nx2 (from,to) pairs."""
    ng = Tff.shape[0]
    edges = [(i, j) for i in range(ng) for j in range(ng) if Tff[i, j] > 0]
    dim = len(edges)
    LMat = np.zeros((ng, ng), int)
    if dim == 0:
        return LMat, np.zeros((0, 2), int)

    affv = np.array([Aff[i, j] for (i, j) in edges], float)

    # Out-degree rows (one per source node with >=1 outgoing edge).
    out_rows, in_rows = {}, {}
    for dd, (i, j) in enumerate(edges):
        out_rows.setdefault(i, []).append(dd)
        in_rows.setdefault(j, []).append(dd)

    rows, cols = [], []
    r = 0
    for _, dds in sorted(out_rows.items()):
        for dd in dds:
            rows.append(r); cols.append(dd)
        r += 1
    for _, dds in sorted(in_rows.items()):
        for dd in dds:
            rows.append(r); cols.append(dd)
        r += 1
    A_ub = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, dim))
    b_ub = np.ones(r)

    res = linprog(-affv, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * dim,
                  method="highs")
    Lv = res.x

    # Decode: threshold then greedy 1-1 extraction over masked affinity.
    Aff2 = np.zeros((ng, ng))
    for dd, (i, j) in enumerate(edges):
        if Lv[dd] > 0.5:
            LMat[i, j] = 1
            Aff2[i, j] = Aff[i, j]

    Np = int(round(Lv.sum()))
    links = []
    for _ in range(Np):
        idx = np.unravel_index(np.argmax(Aff2), Aff2.shape)
        if Aff2[idx] > 0:
            links.append([idx[0], idx[1]])
            Aff2[idx[0], :] = 0
            Aff2[:, idx[1]] = 0
        else:
            break
    return LMat, np.array(links, int).reshape(-1, 2)
