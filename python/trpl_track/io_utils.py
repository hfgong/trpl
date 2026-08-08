"""File I/O for the formats shipped with the dataset.

Three text formats appear in the repo (see the data-structure spec):

* **delimiter matrices** (``read_text_array2d``): whitespace/comma/tab
  separated, one row per line.  Used by calibration + detection files.
* **key = value** pairs (``read_keyvalue_pairs``): e.g. ``horiz.txt``.
* **uBLAS bracket** format (``[m,n]((...),(...))``): used by the C++ for
  ``Tff.txt`` etc.  We provide a reader/writer so intermediate files stay
  interchangeable with the original, though the Python pipeline mainly uses
  ``.npz`` for its own state.

No Boost-XML reader is needed: the repo ships **no** ``.xml`` files -- the
tracklet lists are produced by the pipeline itself, and we serialize them with
NumPy (:mod:`trpl_track.tracklet`).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_string_list(path: str | Path) -> list[str]:
    """One non-empty token per line (trailing CR stripped)."""
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def read_text_array2d(path: str | Path) -> np.ndarray:
    """Read a 2-D array; delimiters are comma / space / tab.

    Ragged rows are zero-padded to the width of the first row (matching the
    C++ ``read_text_array2d``).
    """
    rows = []
    width = None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        parts = [p for p in re.split(r"[,\s\t]+", s) if p != ""]
        vals = [float(p) for p in parts]
        if width is None:
            width = len(vals)
        if len(vals) < width:
            vals = vals + [0.0] * (width - len(vals))
        else:
            vals = vals[:width]
        rows.append(vals)
    return np.array(rows, dtype=float)


def read_text_array1d(path: str | Path) -> np.ndarray:
    """Flatten a whitespace/comma-separated file into a 1-D array."""
    return read_text_array2d(path).ravel()


def read_keyvalue_pairs(path: str | Path) -> dict:
    """Parse ``key = value`` lines (e.g. ``horiz.txt``)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def read_sequence_list(ds) -> list[list[str]]:
    """Return ``[left_paths, right_paths]`` (full paths) for the sequence."""
    left = read_string_list(ds.images / "image_list_l.txt")
    right = read_string_list(ds.images / "image_list_r.txt")
    lp = [str(ds.images / "left_rect" / n) for n in left]
    rp = [str(ds.images / "right_rect" / n) for n in right]
    return [lp, rp]


def image_basename(path: str | Path) -> str:
    """Filename stem, e.g. ``cam1-20080702-162454-079``."""
    return Path(path).stem


def detection_refine_path(ds, name: str) -> Path | None:
    """Locate the refined-detection file for image stem ``name``.

    sequence1 uses ``<name>_3d_ped.txt``; sequence5 uses ``<name>.fmat``.
    Returns the first existing path, else ``None``.
    """
    for fn in (name + "_3d_ped.txt", name + ".fmat"):
        p = ds.detection_refine / fn
        if p.exists():
            return p
    return None


def read_detection_refine(path: str | Path | None) -> np.ndarray:
    """Refined pedestrian detections; the first 4 columns are ``x0,y0,x1,y1``.

    Returns an ``(n, C)`` array (or ``(0, 4)`` if missing/empty).
    """
    if path is None:
        return np.zeros((0, 4), dtype=float)
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 4), dtype=float)
    arr = read_text_array2d(p)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=float)
    return arr.reshape(-1, arr.shape[-1])


# --------------------------------------------------------------------------
# uBLAS bracket format:  vector "[n](a,b,c)"  matrix "[m,n]((..),(..))"
# --------------------------------------------------------------------------

def write_ublas_matrix(path: str | Path, mat: np.ndarray) -> None:
    mat = np.atleast_2d(mat)
    m, n = mat.shape
    is_int = np.issubdtype(mat.dtype, np.integer)

    def fmt(x):
        return str(int(x)) if is_int else repr(float(x))

    rows = ["(" + ",".join(fmt(mat[i, j]) for j in range(n)) + ")" for i in range(m)]
    Path(path).write_text(f"[{m},{n}](" + ",".join(rows) + ")\n")


def read_ublas_matrix(path: str | Path) -> np.ndarray:
    txt = Path(path).read_text().strip()
    mshape = re.match(r"\[(\d+),(\d+)\]", txt)
    if not mshape:
        raise ValueError(f"not a uBLAS matrix: {path}")
    m, n = int(mshape.group(1)), int(mshape.group(2))
    body = txt[mshape.end():]
    nums = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", body)
    vals = [float(x) for x in nums]
    return np.array(vals, dtype=float).reshape(m, n)


_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def read_homography_field(path: str | Path):
    """Parse a uBLAS ``matrix<matrix<double>>`` of per-frame 3x3 homographies.

    Used for sequence5's ``img2grd.txt`` / ``grd2img.txt`` (ego-motion). The
    outer matrix is ``[T,Ncam]`` in row-major order; each element is a 3x3.
    Returns ``(T, Ncam, H)`` where ``H[tt][cam]`` is a 3x3 ``np.ndarray``.
    """
    txt = Path(path).read_text()
    m = re.match(r"\s*\[(\d+),(\d+)\]", txt)
    if not m:
        raise ValueError(f"not a uBLAS matrix<matrix> field: {path}")
    T, ncam = int(m.group(1)), int(m.group(2))
    # Every inner 3x3 is delimited by a "[3,3]" token, emitted row-major.
    blocks = re.split(r"\[3,3\]", txt)[1:]
    mats = []
    for chunk in blocks:
        nums = re.findall(_NUM_RE, chunk)[:9]
        mats.append(np.array([float(x) for x in nums]).reshape(3, 3))
    if len(mats) != T * ncam:
        raise ValueError(f"{path}: expected {T*ncam} 3x3 blocks, got {len(mats)}")
    H = [[mats[tt * ncam + cam] for cam in range(ncam)] for tt in range(T)]
    return T, ncam, H


def write_ublas_vector(path: str | Path, vec: np.ndarray) -> None:
    vec = np.ravel(vec)
    is_int = np.issubdtype(vec.dtype, np.integer)

    def fmt(x):
        return str(int(x)) if is_int else repr(float(x))

    Path(path).write_text(f"[{len(vec)}](" + ",".join(fmt(x) for x in vec) + ")\n")
