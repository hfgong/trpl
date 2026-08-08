"""Scene / camera geometry.

Ports ``camera.hpp`` (camera params, plane back-projection, binocular
transform) and ``geometry.hpp`` (homography estimation / application).  The
homography is fit by the same normal-equation DLT as the C++ so the numerics
match closely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import io_utils


# --------------------------------------------------------------------------
# Homography primitives (geometry.hpp)
# --------------------------------------------------------------------------

def apply_homography(A: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Apply 3x3 homography ``A`` to points, returning (new_x, new_y)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    pts = np.vstack([x, y, np.ones_like(x)])           # 3 x N
    w = A @ pts
    # Points on the horizon give w[2]==0 -> inf, matching the C++ (unguarded).
    with np.errstate(divide="ignore", invalid="ignore"):
        return w[0] / w[2], w[1] / w[2]


def estimate_homography(px, py, nx, ny) -> np.ndarray:
    """Least-squares DLT homography mapping (px,py) -> (nx,ny).

    Uses normal equations (X^T X h = X^T a) exactly like ``estimate_homography``
    in geometry.hpp; A(2,2) is fixed to 1.
    """
    px = np.asarray(px, float).ravel()
    py = np.asarray(py, float).ravel()
    nx = np.asarray(nx, float).ravel()
    ny = np.asarray(ny, float).ravel()
    num = px.size
    X = np.zeros((2 * num, 8))
    a = np.zeros(2 * num)
    for nn in range(num):
        X[2 * nn] = [px[nn], py[nn], 1, 0, 0, 0, -px[nn] * nx[nn], -py[nn] * nx[nn]]
        a[2 * nn] = nx[nn]
        X[2 * nn + 1] = [0, 0, 0, px[nn], py[nn], 1, -px[nn] * ny[nn], -py[nn] * ny[nn]]
        a[2 * nn + 1] = ny[nn]
    h = np.linalg.solve(X.T @ X, X.T @ a)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]])


def get_plane_intersection(KK: np.ndarray, plane: np.ndarray, point2d: np.ndarray) -> np.ndarray:
    """Back-project image points (2xN) onto a 3-D plane [a,b,c,d].

    Returns 3xN 3-D points (camera.hpp ``get_plane_intersection``).
    """
    invKK = np.linalg.inv(KK)
    n = point2d.shape[1]
    p2 = np.vstack([point2d, np.ones(n)])              # 3 x N
    tmp = invKK @ p2                                   # ray directions
    planev = plane[:3]
    w = planev @ tmp
    w = -plane[3] / w
    return tmp * w[None, :]


# --------------------------------------------------------------------------
# Polygon rasterization / point-in-polygon
# --------------------------------------------------------------------------

def mask_from_polygon(h: int, w: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Binary mask (h x w, uint8) of the filled polygon."""
    img = Image.new("L", (w, h), 0)
    pts = list(zip(np.asarray(xs, float).tolist(), np.asarray(ys, float).tolist()))
    ImageDraw.Draw(img).polygon(pts, fill=1)
    return np.array(img, dtype=np.uint8)


def point_in_polygon(x: float, y: float, xs: np.ndarray, ys: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (matches mask_utils point_in_polygon)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    inside = False
    j = n - 1
    for i in range(n):
        if ((ys[i] > y) != (ys[j] > y)) and \
           (x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i] + 1e-12) + xs[i]):
            inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------
# Camera parameters
# --------------------------------------------------------------------------

@dataclass
class CameraParam:
    KK_left: np.ndarray
    KK_right: np.ndarray
    R: np.ndarray
    T: np.ndarray

    @classmethod
    def load(cls, folder: str | Path) -> "CameraParam":
        folder = Path(folder)
        return cls(
            KK_left=io_utils.read_text_array2d(folder / "KK_left_new.txt"),
            KK_right=io_utils.read_text_array2d(folder / "KK_right_new.txt"),
            R=io_utils.read_text_array1d(folder / "R_new.txt"),
            T=io_utils.read_text_array1d(folder / "T_new.txt"),
        )


@dataclass
class GroundLim:
    xmin: int
    xmax: int
    ymin: int
    ymax: int


@dataclass
class GeometricInfo:
    """Scene geometry: image<->ground homographies, goals, polygon, horizon.

    Two modes:
      * static (sequence1): one homography per camera; ``img2grd[cam]`` etc.
      * dynamic (sequence5): per-frame ego-motion homographies loaded from
        ``init_input/img2grd.txt`` / ``grd2img.txt``; ``img2grd[tt][cam]`` etc.

    Downstream code uses the ``*_t(tt, cam)`` accessors, which resolve to the
    single transform in static mode (ignoring ``tt``) and the per-frame one in
    dynamic mode.  ``poly_ground`` / ``goal_ground`` are always a single fixed
    world-frame ground map.
    """

    img2grd: list = field(default_factory=list)      # [cam] or [tt][cam] 3x3
    grd2img: list = field(default_factory=list)      # [cam] or [tt][cam] 3x3
    goals_im: list = field(default_factory=list)     # [cam] or [tt][cam] 2xNgoal
    polys_im: list = field(default_factory=list)     # [cam] or [tt][cam] 2xNpoly
    goal_ground: np.ndarray = None                   # 2xNgoal (fixed world)
    poly_ground: np.ndarray = None                   # 2xNpoly (fixed world)
    ground_lim: GroundLim = None
    poly_mask: list = field(default_factory=list)    # [cam] HxW uint8 (static only)
    cam_param: CameraParam = None
    horiz_mean: float = 0.0
    horiz_sig: float = 0.0
    dynamic: bool = False
    T: int = 1

    # Per-frame accessors (tt ignored in static mode).
    def img2grd_t(self, tt, cam):
        return self.img2grd[tt][cam] if self.dynamic else self.img2grd[cam]

    def grd2img_t(self, tt, cam):
        return self.grd2img[tt][cam] if self.dynamic else self.grd2img[cam]

    def polys_im_t(self, tt, cam):
        return self.polys_im[tt][cam] if self.dynamic else self.polys_im[cam]

    def goals_im_t(self, tt, cam):
        return self.goals_im[tt][cam] if self.dynamic else self.goals_im[cam]

    @classmethod
    def load(cls, ds, img_size=(768, 1024)) -> "GeometricInfo":
        init = ds.init_input
        horiz = io_utils.read_keyvalue_pairs(init / "horiz.txt")
        cam = CameraParam.load(init)
        goal2d = io_utils.read_text_array2d(init / "goal2d.txt")   # 2 x Ngoal
        poly2d = io_utils.read_text_array2d(init / "poly2d.txt")   # 2 x Npoly
        gi = cls(cam_param=cam,
                 horiz_mean=horiz["horiz_mean"], horiz_sig=horiz["horiz_sig"])
        if (init / "img2grd.txt").exists():
            gi._load_dynamic(init, goal2d, poly2d)
        else:
            gi._compute_binocular_transform(goal2d, poly2d, img_size)
        return gi

    def _load_dynamic(self, init, goal2d, poly2d):
        """Moving-camera case: per-frame homographies from ego-motion files.

        The ground map (poly/goal) is fixed in the world frame (frame-0
        ``img2grd``); each frame's image projections come from that frame's
        ``grd2img`` (matches sequence5's ``compute_binocular_transform``).
        """
        self.dynamic = True
        T, ncam, i2g = io_utils.read_homography_field(init / "img2grd.txt")
        _, _, g2i = io_utils.read_homography_field(init / "grd2img.txt")
        self.T = T
        self.img2grd = i2g
        self.grd2img = g2i

        px, py = apply_homography(i2g[0][0], poly2d[0], poly2d[1])
        self.poly_ground = np.vstack([px, py])
        gx, gy = apply_homography(i2g[0][0], goal2d[0], goal2d[1])
        self.goal_ground = np.vstack([gx, gy])
        self.ground_lim = GroundLim(
            xmin=int(px.min()), xmax=int(px.max()) + 1,
            ymin=int(py.min()), ymax=int(py.max()) + 1,
        )

        self.polys_im = [[None] * ncam for _ in range(T)]
        self.goals_im = [[None] * ncam for _ in range(T)]
        for tt in range(T):
            for cam in range(ncam):
                ipx, ipy = apply_homography(g2i[tt][cam], px, py)
                self.polys_im[tt][cam] = np.vstack([ipx, ipy])
                igx, igy = apply_homography(g2i[tt][cam], gx, gy)
                self.goals_im[tt][cam] = np.vstack([igx, igy])

    def _compute_binocular_transform(self, goal2d, poly2d, img_size):
        cam = self.cam_param
        H, W = img_size

        # Manually-estimated ground plane (camera.hpp constants).
        camera_height = 1800.0
        gNorm = np.array([0.0, -1.0, -0.04])
        gPlane = np.empty(4)
        gPlane[:3] = gNorm / np.linalg.norm(gNorm)
        gPlane[3] = camera_height

        self.poly_mask = [None, None]
        self.poly_mask[0] = mask_from_polygon(H, W, poly2d[0], poly2d[1])

        poly3d = get_plane_intersection(cam.KK_left, gPlane, poly2d)   # 3 x Np
        goal3d = get_plane_intersection(cam.KK_left, gPlane, goal2d)

        xmin3d, xmax3d = poly3d[0].min(), poly3d[0].max()
        zmin3d, zmax3d = poly3d[2].min(), poly3d[2].max()
        grid_size = 50.0  # 5 cm per ground pixel

        grd_x = (poly3d[0] - xmin3d) / grid_size
        grd_y = (poly3d[2] - zmin3d) / grid_size

        # Homographies from 4 polygon corners (indices 0,4,6,7).
        idx = [0, 4, 6, 7]
        ix, iy = poly2d[0][idx], poly2d[1][idx]
        gx, gy = grd_x[idx], grd_y[idx]

        self.img2grd = [None, None]
        self.grd2img = [None, None]
        self.img2grd[0] = estimate_homography(ix, iy, gx, gy)
        self.grd2img[0] = estimate_homography(gx, gy, ix, iy)

        # Right camera: shift 3-D points by baseline, reproject with KK_right.
        poly3dr = poly3d + cam.T[:, None]
        poly2drx = cam.KK_right @ poly3dr
        poly2dr = np.vstack([poly2drx[0] / poly2drx[2], poly2drx[1] / poly2drx[2]])
        self.poly_mask[1] = mask_from_polygon(H, W, poly2dr[0], poly2dr[1])

        ixr, iyr = poly2dr[0][idx], poly2dr[1][idx]
        self.img2grd[1] = estimate_homography(ixr, iyr, gx, gy)
        self.grd2img[1] = estimate_homography(gx, gy, ixr, iyr)

        goal3dr = goal3d + cam.T[:, None]
        goal2drx = cam.KK_right @ goal3dr
        goal2dr = np.vstack([goal2drx[0] / goal2drx[2], goal2drx[1] / goal2drx[2]])

        self.polys_im = [poly2d.copy(), poly2dr]
        self.goals_im = [goal2d.copy(), goal2dr]

        # Ground-space polygon + limits (from left cam poly through img2grd[0]).
        tx, ty = apply_homography(self.img2grd[0], poly2d[0], poly2d[1])
        self.poly_ground = np.vstack([tx, ty])
        self.ground_lim = GroundLim(
            xmin=int(tx.min()), xmax=int(tx.max()) + 1,
            ymin=int(ty.min()), ymax=int(ty.max()) + 1,
        )

        gx2, gy2 = apply_homography(self.img2grd[0], goal2d[0], goal2d[1])
        self.goal_ground = np.vstack([gx2, gy2])
