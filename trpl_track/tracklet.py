"""Tracklet / trajectory data structure (``object_trj_t``).

The C++ serializes these as verbose Boost XML.  Since the dataset ships **no**
``.xml`` files (tracklets are produced by the pipeline), we serialize lists of
tracklets with NumPy ``.npz`` instead -- simpler and lossless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import NCAM, NPART, NBINS


@dataclass
class ObjectTraj:
    """One tracked object over an inclusive frame window ``[startt, endt]``.

    Arrays are allocated over the full sequence length ``T``; only rows within
    ``[startt, endt]`` are meaningful.
    """

    T: int
    ncam: int = NCAM
    startt: int = -1
    endt: int = -1
    state: int = 1                                   # 1 active/visible, 2 lost
    trj: list = field(default_factory=list)          # [cam] T x 4 boxes
    trj_3d: np.ndarray = None                        # T x 2 ground positions
    scores: np.ndarray = None                        # ncam x T
    fscores: list = field(default_factory=list)      # [cam] (NPART*2) x T
    hist_p: list = field(default_factory=list)       # [cam] NPART x NBINS fg
    hist_q: list = field(default_factory=list)       # [cam] NPART x NBINS bg

    def __post_init__(self):
        if not self.trj:
            self.trj = [np.zeros((self.T, 4), np.float32) for _ in range(self.ncam)]
        if self.trj_3d is None:
            self.trj_3d = np.zeros((self.T, 2), np.float32)
        if self.scores is None:
            self.scores = np.zeros((self.ncam, self.T), np.float32)
        if not self.fscores:
            self.fscores = [np.zeros((NPART * 2, self.T), np.float32) for _ in range(self.ncam)]
        if not self.hist_p:
            self.hist_p = [np.zeros((NPART, NBINS), np.float32) for _ in range(self.ncam)]
        if not self.hist_q:
            self.hist_q = [np.zeros((NPART, NBINS), np.float32) for _ in range(self.ncam)]

    @property
    def length(self) -> int:
        return self.endt - self.startt + 1 if self.startt >= 0 else 0

    def is_empty(self) -> bool:
        return self.startt < 0 or self.endt < self.startt


# --------------------------------------------------------------------------
# (De)serialization of tracklet lists
# --------------------------------------------------------------------------

def save_tracklets(path: str | Path, trlets: list[ObjectTraj]) -> None:
    """Serialize a list of tracklets to a single ``.npz`` file."""
    blob = {"n": len(trlets)}
    for i, t in enumerate(trlets):
        p = f"t{i}_"
        blob[p + "meta"] = np.array([t.T, t.ncam, t.startt, t.endt, t.state])
        blob[p + "trj_3d"] = t.trj_3d
        blob[p + "scores"] = t.scores
        for c in range(t.ncam):
            blob[p + f"trj{c}"] = t.trj[c]
            blob[p + f"fscores{c}"] = t.fscores[c]
            blob[p + f"hist_p{c}"] = t.hist_p[c]
            blob[p + f"hist_q{c}"] = t.hist_q[c]
    np.savez_compressed(path, **blob)


def load_tracklets(path: str | Path) -> list[ObjectTraj]:
    d = np.load(path, allow_pickle=False)
    n = int(d["n"])
    out = []
    for i in range(n):
        p = f"t{i}_"
        T, ncam, startt, endt, state = [int(x) for x in d[p + "meta"]]
        t = ObjectTraj(T=T, ncam=ncam, startt=startt, endt=endt, state=state,
                       trj=[d[p + f"trj{c}"] for c in range(ncam)],
                       trj_3d=d[p + "trj_3d"], scores=d[p + "scores"],
                       fscores=[d[p + f"fscores{c}"] for c in range(ncam)],
                       hist_p=[d[p + f"hist_p{c}"] for c in range(ncam)],
                       hist_q=[d[p + f"hist_q{c}"] for c in range(ncam)])
        out.append(t)
    return out
