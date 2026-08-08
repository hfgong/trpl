"""Visualization (ports the drawing done by ``vis_linprog_plan.cpp``).

Produces, per frame:
  * an image-view overlay (per camera) with per-object bounding boxes coloured
    by state -- yellow = observed, cyan = planned/gap -- id labels, the object's
    trajectory trail, and the frame number top-left (as in the paper figures);
  * a top-down ground-plane view with the walkable polygon, goals, and each
    trajectory's ground track.

CImg draw calls map to Pillow ImageDraw (see the finalize/vis spec).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import io_utils
from .geometry import apply_homography

YELLOW = (255, 255, 0)
CYAN = (0, 255, 125)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Distinct id colors.
_ID_COLORS = [(255, 80, 80), (80, 200, 80), (80, 160, 255), (255, 180, 40),
              (200, 80, 255), (40, 220, 220), (255, 120, 200), (170, 170, 90)]


def _font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_frame_number(draw, tt, size=40):
    f = _font(size)
    txt = str(tt)
    draw.rectangle([0, 0, size * len(txt), size + 6], fill=BLACK)
    draw.text((2, 0), txt, fill=WHITE, font=f)


def vis_image_frames(ds, seq, final_list, state_list, trj_index, cam=0,
                     dim_background=True):
    """Write per-frame overlay images for one camera to ``ds.output``."""
    ds.make_dirs()
    T = len(seq[cam])
    font_id = _font(22)
    out_paths = []
    for tt in range(T):
        img = Image.open(seq[cam][tt]).convert("RGB")
        arr = np.array(img)
        base = Image.fromarray(arr)
        draw = ImageDraw.Draw(base)
        for k, f in enumerate(final_list):
            if state_list[k, tt] < 0:
                continue
            box = f.trj[cam][tt]
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            col = YELLOW if state_list[k, tt] == 1 else CYAN
            # trajectory trail (past centers, image space via box centers)
            pts = []
            for s in range(f.startt, tt + 1):
                if state_list[k, s] < 0:
                    continue
                b = f.trj[cam][s]
                pts.append(((b[0] + b[2]) / 2, b[3]))
            if len(pts) > 1:
                draw.line(pts, fill=_ID_COLORS[k % len(_ID_COLORS)], width=2)
            draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
            draw.text((x0 + 2, max(0, y0 - 22)), f"{k}", fill=col, font=font_id)
        _draw_frame_number(draw, tt)
        name = ds.output / f"track_cam{cam}_{tt:03d}.jpg"
        base.save(name, quality=90)
        out_paths.append(name)
    return out_paths


CAR_COLOR = (230, 130, 40)      # car obstacles (paper: black squares/rects)
PED_OBS_COLOR = (90, 90, 110)   # pedestrian obstacle footprints
PLAN_COLOR = (0, 230, 255)      # selected motion plan bridging a gap


def _ground_crop_box(final_list, state_list, gl, margin=30):
    """Bounding box (in ground coords) around all tracked positions.

    Cropping to where objects actually are matches the paper's Fig. 5 (a
    cropped top-view) and keeps the mostly-empty street and far-goal plan
    extensions out of frame.
    """
    pts = []
    for k, f in enumerate(final_list):
        for tt in range(f.startt, f.endt + 1):
            if state_list[k, tt] >= 0:
                pts.append((f.trj_3d[tt, 0], f.trj_3d[tt, 1]))
    if not pts:
        return 0, 0, gl.xmax, gl.ymax
    pts = np.array(pts)
    x0 = max(gl.xmin, int(pts[:, 0].min() - margin))
    x1 = min(gl.xmax, int(pts[:, 0].max() + margin))
    y0 = max(gl.ymin, int(pts[:, 1].min() - margin))
    y1 = min(gl.ymax, int(pts[:, 1].max() + margin))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def vis_ground_frames(ds, gi, final_list, state_list, trj_index=None,
                      gap_paths=None, car_obs=None, ped_obs=None, cam=0,
                      crop=True, draw_full_plans=False):
    """Write per-frame top-down ground-plane views to ``ds.output``.

    Draws the walkable polygon, goals, **car obstacles** (per frame),
    optionally pedestrian obstacle footprints, and each trajectory's ground
    track.  The track is coloured by provenance: **observed** segments in the
    object's colour, **planned/gap** segments in cyan -- so the planned bridge
    is visible inline without a separate goal-ward line.  The view is
    **cropped** to the region occupied by the tracked objects (paper Fig. 5).

    ``draw_full_plans=True`` additionally overlays the full selected plan-to-goal
    curves (paper-style, but busier); off by default since those long curves ran
    off toward far goals and dominated the view.
    """
    ds.make_dirs()
    gl = gi.ground_lim
    T = final_list[0].trj_3d.shape[0] if final_list else 0
    poly = gi.poly_ground
    goals = gi.goal_ground

    if crop and final_list:
        cx0, cy0, cx1, cy1 = _ground_crop_box(final_list, state_list, gl)
    else:
        cx0, cy0, cx1, cy1 = 0, 0, gl.xmax, gl.ymax
    W, H = cx1 - cx0, cy1 - cy0

    def P(arr_2xn):     # 2xN ground -> list of crop-local (x,y)
        return [(float(arr_2xn[0, i]) - cx0, float(arr_2xn[1, i]) - cy0)
                for i in range(arr_2xn.shape[1])]

    def Pxy(pairs):     # iterable of (x,y) ground -> crop-local
        return [(float(x) - cx0, float(y) - cy0) for x, y in pairs]

    out_paths = []
    for tt in range(T):
        canvas = Image.new("RGB", (W, H), (30, 30, 40))
        draw = ImageDraw.Draw(canvas)
        draw.polygon(P(poly), outline=(120, 120, 160))
        for g in range(goals.shape[1]):
            gx, gy = float(goals[0, g]) - cx0, float(goals[1, g]) - cy0
            draw.rectangle([gx - 6, gy - 6, gx + 6, gy + 6], outline=(150, 150, 150))

        if ped_obs is not None:
            for nn in range(len(ped_obs)):
                p = ped_obs[nn][tt] if tt < len(ped_obs[nn]) else None
                if p is not None and len(p) > 0:
                    draw.polygon(Pxy(p), outline=PED_OBS_COLOR)

        if car_obs is not None and tt < len(car_obs):
            for poly4 in car_obs[tt]:
                if poly4 is not None and len(poly4) > 0:
                    draw.polygon(Pxy(poly4), fill=CAR_COLOR, outline=(255, 200, 120))

        # Optional: full selected plan-to-goal curves (paper-style, busier).
        if draw_full_plans and trj_index is not None and gap_paths is not None:
            for k, chain in enumerate(trj_index):
                if state_list[k, tt] != 0:
                    continue
                for a, b in zip(chain[:-1], chain[1:]):
                    gp = gap_paths[a][b]
                    if gp is not None and len(gp) > 1:
                        draw.line(Pxy(gp), fill=PLAN_COLOR, width=1)

        # Trajectory track, coloured by provenance (observed vs planned gap).
        for k, f in enumerate(final_list):
            col = _ID_COLORS[k % len(_ID_COLORS)]
            prev, prev_s = None, None
            for s in range(f.startt, tt + 1):
                if state_list[k, s] < 0:
                    prev = None
                    continue
                cur = (f.trj_3d[s, 0] - cx0, f.trj_3d[s, 1] - cy0)
                if prev is not None:
                    planned = state_list[k, s] == 0 or state_list[k, prev_s] == 0
                    draw.line([prev, cur], fill=PLAN_COLOR if planned else col,
                              width=3 if planned else 2)
                prev, prev_s = cur, s
            if state_list[k, tt] >= 0:
                x, y = f.trj_3d[tt, 0] - cx0, f.trj_3d[tt, 1] - cy0
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)

        canvas = canvas.transpose(Image.FLIP_TOP_BOTTOM)   # mirror('y')
        name = ds.output / f"ground_{tt:03d}.png"
        canvas.save(name)
        out_paths.append(name)
    return out_paths


def write_results_txt(ds, final_list, state_list):
    """Bounding boxes per (object, frame) -- like ``results_plan.txt``."""
    ncam = final_list[0].ncam if final_list else 2
    lines = []
    for k, f in enumerate(final_list):
        for tt in range(f.startt, f.endt + 1):
            if state_list[k, tt] < 0:
                continue
            parts = [str(k), str(tt)]
            for cam in range(ncam):
                parts += [f"{v:.1f}" for v in f.trj[cam][tt]]
            lines.append(" ".join(parts))
    (ds.output / "results.txt").write_text("\n".join(lines) + "\n")
