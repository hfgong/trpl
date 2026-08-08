"""Assemble final trajectories from LP links (``finalize_trajectory_plan``).

Pure bookkeeping: walk the link graph into chains, then copy observed tracklet
frames and planned gap frames into one trajectory per chain.  ``state`` codes:
1 = observed, 0 = gap-filled (planned), -1 = absent.
"""
from __future__ import annotations

import numpy as np

from .tracklet import ObjectTraj


def finalize_trajectory_plan(ncam, T, links, good, gap_rind, gap):
    ng = len(good)
    links = np.asarray(links, int).reshape(-1, 2)

    nxt = -np.ones(ng, int)
    lfrom, lto = set(), set()
    for a, b in links:
        nxt[a] = b
        lfrom.add(int(a))
        lto.add(int(b))
    all_linked = lfrom | lto
    isola = sorted(set(range(ng)) - all_linked)
    begin = sorted(lfrom - lto)
    seeds = list(begin) + list(isola)

    trj_index = []
    for s in seeds:
        chain = [s]
        while nxt[chain[-1]] >= 0:
            chain.append(int(nxt[chain[-1]]))
        trj_index.append(chain)

    nobj = len(trj_index)
    final_list = []
    state_list = -np.ones((nobj, T), int)
    for k, chain in enumerate(trj_index):
        startt = good[chain[0]].startt
        endt = good[chain[-1]].endt
        f = ObjectTraj(T=T, ncam=ncam, startt=startt, endt=endt)
        # observed spans
        for nn in chain:
            for tt in range(good[nn].startt, good[nn].endt + 1):
                for cam in range(ncam):
                    f.trj[cam][tt] = good[nn].trj[cam][tt]
                    f.scores[cam, tt] = good[nn].scores[cam, tt]
                f.trj_3d[tt] = good[nn].trj_3d[tt]
                state_list[k, tt] = 1
        # gap spans
        for a, b in zip(chain[:-1], chain[1:]):
            rr = gap_rind[a, b]
            if rr < 0 or not gap[a][b]:
                continue
            gt = gap[a][b][rr]
            for tt in range(gt.startt, gt.endt + 1):
                for cam in range(ncam):
                    f.trj[cam][tt] = gt.trj[cam][tt]
                    f.scores[cam, tt] = gt.scores[cam, tt]
                f.trj_3d[tt] = gt.trj_3d[tt]
                state_list[k, tt] = 0
        final_list.append(f)
    return final_list, trj_index, state_list
