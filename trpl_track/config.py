"""Configuration: directory layout and algorithm parameters.

Ports ``directory_structure_t`` and ``parameter_t`` from
``sequence1/src/tracking_data_package.hpp``.  A single shared package handles
both bundled sequences (``sequence1`` = static cameras, ``sequence5`` = moving
cameras); the only per-sequence difference is the root directory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DirectoryStructure:
    """On-disk layout for one sequence (mirrors ``directory_structure_t``).

    The C++ code hard-codes paths relative to the ``src`` build dir; here we
    take the sequence root explicitly so the same code serves both sequences.
    """

    root: Path
    images: Path = field(init=False)
    workspace: Path = field(init=False)
    output: Path = field(init=False)

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.images = self.root / "images"
        self.workspace = self.root / "workspace"
        self.output = self.root / "output_py"

    def make_dirs(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)

    # Files required to run the pipeline (relative to their dirs).
    _REQUIRED_IMAGES = ("image_list_l.txt", "image_list_r.txt")
    _REQUIRED_INIT = ("horiz.txt", "KK_left_new.txt", "KK_right_new.txt",
                      "R_new.txt", "T_new.txt", "goal2d.txt", "poly2d.txt")

    def missing_inputs(self) -> list[str]:
        """Return human-readable descriptions of any missing required inputs."""
        problems = []
        if not self.root.is_dir():
            return [f"sequence directory does not exist: {self.root}"]
        if not self.images.is_dir():
            problems.append(f"missing images directory: {self.images}")
        else:
            for f in self._REQUIRED_IMAGES:
                if not (self.images / f).exists():
                    problems.append(f"missing image list: {self.images / f}")
        if not self.init_input.is_dir():
            problems.append(f"missing calibration directory: {self.init_input}")
        else:
            for f in self._REQUIRED_INIT:
                if not (self.init_input / f).exists():
                    problems.append(f"missing calibration file: {self.init_input / f}")
        return problems

    @property
    def init_input(self) -> Path:
        return self.workspace / "init_input"

    @property
    def detection_refine(self) -> Path:
        return self.workspace / "detection_refine"


@dataclass
class Parameters:
    """Motion / appearance / scoring parameters (``parameter_t`` defaults)."""

    # Motion priors (used by planning, not pretr).
    mot_param_sig1: float = 100.0     # position spread ("small")
    mot_param_sig2: float = 25.0      # velocity smoothness ("smooth")

    # Occlusion / state thresholds.
    occl_thr1: float = 6.0
    occl_thr2: float = 4.0

    # Localization search window (image pixels).
    xrange: int = 50
    yrange: int = 36
    xstep: int = 4
    ystep: int = 3

    scales: tuple = (1.05, 1.0, 0.96)          # scale search candidates
    pvp: tuple = (0.9, 0.8, 0.7)               # part visibility priors [head,torso,legs]

    # Appearance-model learning rates.
    fglr: float = 0.04
    bglr: float = 0.06

    # Score thresholds.
    thr: float = 6.0
    thr3: float = -5.0

    # Part geometry (fractions of the object box).
    head_wid_ratio: float = 0.5
    head_hi_ratio: float = 0.2
    torso_hi_ratio: float = 0.6

    @property
    def logp1(self) -> list:
        """log(pvp): per-part log prob of being visible."""
        return [math.log(p) for p in self.pvp]

    @property
    def logp2(self) -> list:
        """log(1-pvp): per-part log prob of being occluded."""
        return [math.log(1.0 - p) for p in self.pvp]

    def part_model(self) -> "list[tuple]":
        """Normalized part boxes ``[x0,y0,x1,y1]`` as fractions of the box.

        head  = [0.5-hw/2, 0, 0.5+hw/2, hh]
        torso = [0, hh, 1, th]
        legs  = [0, th, 1, 1]
        """
        hw = self.head_wid_ratio
        hh = self.head_hi_ratio
        th = self.torso_hi_ratio
        return [
            (0.5 - hw / 2.0, 0.0, 0.5 + hw / 2.0, hh),
            (0.0, hh, 1.0, th),
            (0.0, th, 1.0, 1.0),
        ]


# Global constants shared across the pipeline.
NCAM = 2                 # 0 = left (cam1), 1 = right (cam2)
IMG_H = 768
IMG_W = 1024
NPART = 3                # head, torso, legs
NBINS = 512              # 8*8*8 RGB color histogram
