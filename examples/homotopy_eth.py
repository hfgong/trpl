"""Winding-number homotopy planning on ETH pedestrians (planner-only demo).

This is the Python counterpart of the original C++ ``homotopy_eth_main.cpp``: it
runs *only* the paper's winding-number planner (``trpl_track.planning``) -- not
the full tracker -- on the ETH pedestrian scene. Pedestrians at one moment
become ground-plane obstacles; we then enumerate the distinct *homotopy classes*
of shortest paths from a start to a goal across the crowd (go left / right of
each critical obstacle), as in Figures 2-3 of the paper.

Requires data/eth/obsmat.txt  (run: bash examples/fetch_eth_data.sh)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trpl_track.planning import (combine_obstacles, construct_obstacle_maps,
                                  construct_feature_maps, build_state_graph,
                                  edge_weights, get_legal_index, shortest_path,
                                  choose_critic_obstacles, wind_angle_planning)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                    "eth", "obsmat.txt")
CELL = 0.4                     # metres per grid cell
PED_HALF = 0.6                 # pedestrian obstacle half-size (m)
WEI = [1.61642, 3.0, 15.0]     # [path length, static-dist, dynamic(ped)-dist] weights
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)


def load_frames(path):
    arr = np.loadtxt(path)
    frames = {}
    for row in arr:
        frames.setdefault(int(row[0]), []).append((row[2], row[4]))
    return {f: np.array(v) for f, v in frames.items()}, arr


def pick_frame(frames, xlo, xhi):
    """Frame with the most pedestrians in the central x-band (good obstacles)."""
    best, bestn = None, -1
    for f, pts in frames.items():
        n = np.sum((pts[:, 0] > xlo) & (pts[:, 0] < xhi))
        if n > bestn:
            best, bestn = f, n
    return best


def square(cx, cy, h):
    return np.array([[cx - h, cy - h], [cx + h, cy - h],
                     [cx + h, cy + h], [cx - h, cy + h]], float)


def main():
    if not os.path.exists(DATA):
        print(f"ETH data not found at {DATA}\nRun: bash examples/fetch_eth_data.sh")
        return
    frames, arr = load_frames(DATA)
    allxy = arr[:, [2, 4]]
    xmin, ymin = allxy.min(0) - 2 * CELL
    xmax, ymax = allxy.max(0) + 2 * CELL

    def g(x, y):                       # world -> grid (x=col, y=row)
        return (x - xmin) / CELL, (y - ymin) / CELL

    xr = allxy[:, 0].max() - allxy[:, 0].min()
    fno = pick_frame(frames, allxy[:, 0].min() + 0.3 * xr,
                     allxy[:, 0].min() + 0.7 * xr)
    peds = frames[fno]
    # keep central pedestrians as obstacles (cap a handful)
    xlo, xhi = allxy[:, 0].min() + 0.25 * xr, allxy[:, 0].min() + 0.75 * xr
    central = peds[(peds[:, 0] > xlo) & (peds[:, 0] < xhi)]
    if len(central) > 5:
        central = central[np.argsort(np.abs(central[:, 1] - np.median(peds[:, 1])))][:5]
    ycen = float(np.median(peds[:, 1]))
    start_w = (central[:, 0].min() - 2.0, ycen)
    goal_w = (central[:, 0].max() + 2.0, ycen)

    # --- build the planner inputs in grid coordinates ---
    W = (xmax - xmin) / CELL
    H = (ymax - ymin) / CELL
    poly_ground = np.array([[1, W - 1, W - 1, 1], [1, 1, H - 1, H - 1]], float)
    goal_gx, goal_gy = g(*goal_w)
    goal_ground = np.array([[goal_gx], [goal_gy]], float)
    ped_obs = [square(*g(px, py), PED_HALF / CELL) for (px, py) in central]

    obs, obs_cent = combine_obstacles(-1, [], ped_obs, poly_ground)
    obs_map, dyn_map = construct_obstacle_maps(obs, poly_ground, goal_ground)
    # Treat pedestrians as HARD obstacles for this demo: carve their cells out
    # of the walkable graph so paths genuinely go *around* them (not merely pay
    # the soft dynamic-obstacle cost). Remove this line to get the tracker's
    # soft-obstacle behaviour instead.
    obs_map = np.where(dyn_map > 0, 1, obs_map)
    feat = construct_feature_maps(obs_map, dyn_map)
    ig2yx, yx2ig, src, dst, nbr = build_state_graph(obs_map)
    ew = edge_weights(ig2yx, src, dst, nbr, feat, WEI)

    start = get_legal_index(yx2ig, ig2yx, *g(*start_w))
    goal = get_legal_index(yx2ig, ig2yx, goal_gx, goal_gy)
    sdist, spath = shortest_path(ig2yx, src, dst, ew, start, goal)
    critic = choose_critic_obstacles(ig2yx, obs_cent, spath, sdist, thr=18.0)
    paths, dists, winds = wind_angle_planning(ig2yx, src, dst, nbr, ew, critic,
                                              -1, 0, start, goal)
    print(f"frame {fno}: {len(central)} pedestrian obstacles, "
          f"{len(critic)} critic -> {len(paths)} homotopy-class path(s)")

    def to_world(path):
        xy = np.array([(ig2yx[n, 1], ig2yx[n, 0]) for n in path], float)
        return xmin + xy[:, 0] * CELL, ymin + xy[:, 1] * CELL

    # --- plot ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for pts in list(frames.values()):
        pass
    for xy in [arr[arr[:, 1] == pid][:, [2, 4]] for pid in np.unique(arr[:, 1])]:
        ax.plot(xy[:, 0], xy[:, 1], "-", color="0.85", lw=0.5, zorder=0)
    for (px, py) in central:
        ax.add_patch(plt.Rectangle((px - PED_HALF, py - PED_HALF), 2 * PED_HALF,
                                   2 * PED_HALF, color="0.35", zorder=2))
    order = np.argsort(dists)
    cmap = plt.cm.turbo(np.linspace(0.1, 0.9, len(paths)))
    for rank, k in enumerate(order):
        wx, wy = to_world(paths[k])
        wn = winds[k] if winds.size else []
        ax.plot(wx, wy, "-", lw=2.5, color=cmap[rank], zorder=3,
                label=f"k={list(map(int, wn))}, cost={dists[k]:.0f}")
    ax.plot(*start_w, "o", color="lime", ms=13, mec="k", zorder=4, label="start")
    ax.plot(*goal_w, "*", color="red", ms=20, mec="k", zorder=4, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Winding-number homotopy planning on ETH (frame {fno})\n"
                 f"{len(paths)} distinct homotopy classes around the pedestrians")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = f"{RESULTS}/homotopy_eth.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
