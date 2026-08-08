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


def vis_ground_frames(ds, gi, final_list, state_list, trj_index=None,
                      gap_paths=None, car_obs=None, ped_obs=None, cam=0):
    """Write per-frame top-down ground-plane views to ``ds.output``.

    Draws the walkable polygon, goals, **car obstacles** (per frame),
    optionally pedestrian obstacle footprints, each trajectory's ground track
    and current position, and the **selected motion plan** used to bridge each
    linked gap (the paper's Fig. 5 bottom view).
    """
    ds.make_dirs()
    gl = gi.ground_lim
    W, H = gl.xmax, gl.ymax
    T = final_list[0].trj_3d.shape[0] if final_list else 0
    poly = gi.poly_ground
    goals = gi.goal_ground

    def _poly_pts(arr_2xn):
        return [(float(arr_2xn[0, i]), float(arr_2xn[1, i]))
                for i in range(arr_2xn.shape[1])]

    out_paths = []
    for tt in range(T):
        canvas = Image.new("RGB", (W, H), (30, 30, 40))
        draw = ImageDraw.Draw(canvas)
        draw.polygon(_poly_pts(poly), outline=(120, 120, 160))
        for g in range(goals.shape[1]):
            gx, gy = float(goals[0, g]), float(goals[1, g])
            draw.rectangle([gx - 6, gy - 6, gx + 6, gy + 6], outline=(150, 150, 150))

        # Pedestrian obstacle footprints at this frame (optional).
        if ped_obs is not None:
            for nn in range(len(ped_obs)):
                p = ped_obs[nn][tt] if tt < len(ped_obs[nn]) else None
                if p is not None and len(p) > 0:
                    draw.polygon([(float(x), float(y)) for x, y in p],
                                 outline=PED_OBS_COLOR)

        # Car obstacles at this frame (filled quads).
        if car_obs is not None and tt < len(car_obs):
            for poly4 in car_obs[tt]:
                if poly4 is not None and len(poly4) > 0:
                    draw.polygon([(float(x), float(y)) for x, y in poly4],
                                 fill=CAR_COLOR, outline=(255, 200, 120))

        # Selected motion plans bridging linked gaps (drawn during the gap span).
        if trj_index is not None and gap_paths is not None:
            for k, chain in enumerate(trj_index):
                if state_list[k, tt] != 0:       # only while in a planned gap
                    continue
                for a, b in zip(chain[:-1], chain[1:]):
                    gp = gap_paths[a][b]
                    if gp is not None and len(gp) > 1:
                        draw.line([(float(x), float(y)) for x, y in gp],
                                  fill=PLAN_COLOR, width=2)

        for k, f in enumerate(final_list):
            trail = [(f.trj_3d[s, 0], f.trj_3d[s, 1])
                     for s in range(f.startt, tt + 1) if state_list[k, s] >= 0]
            if len(trail) > 1:
                draw.line(trail, fill=_ID_COLORS[k % len(_ID_COLORS)], width=2)
            if state_list[k, tt] >= 0:
                x, y = f.trj_3d[tt, 0], f.trj_3d[tt, 1]
                draw.ellipse([x - 5, y - 5, x + 5, y + 5],
                             fill=_ID_COLORS[k % len(_ID_COLORS)])

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
