"""Multi-hypothesis motion planning (``planning.hpp`` / ``planning_impl.hpp``).

This is the paper's core contribution: for each tracklet we enumerate
homotopy-distinct shortest paths to each scene goal, indexed by *winding
numbers* around the critical obstacles.  Each path is a motion hypothesis used
later to score tracklet links.

The winding-angle augmented graph edge test
``|Δwind_angle - dwa|_1 < 1e-6`` is implemented efficiently: crossing the
branch cut behind obstacle ``o`` on edge ``g->g2`` increments that obstacle's
winding-layer digit by a fixed per-edge amount, so each source layer maps to a
single target layer instead of an O(nlayer^2) search.

For tractability in pure Python the ground grid is optionally downsampled by
``plan_grid_step`` (default 4); the algorithm is unchanged and output paths are
scaled back to full ground coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import io_utils
from .geometry import apply_homography, point_in_polygon

DX = np.array([1, 0, -1, 0, 1, -1, -1, 1])
DY = np.array([0, 1, 0, -1, 1, 1, -1, -1])
TWO_PI = 2.0 * np.pi


@dataclass
class PlanItem:
    """Planning result for one (tracklet, goal): a bank of homotopy paths."""
    paths: list          # list of (P,2) float arrays, (x,y) in FULL ground coords
    dist: np.ndarray     # geodesic cost per path
    wind_num: np.ndarray # (K, no) winding numbers


# --------------------------------------------------------------------------
# Obstacles
# --------------------------------------------------------------------------

def carboxes2carobs(cars: np.ndarray, img2grd: np.ndarray) -> list:
    """Convert car image boxes into ground-plane obstacle quads."""
    out = []
    for bb in cars:
        x1, x2, y2 = bb[0], bb[2], bb[3]
        xc = (x1 + x2) / 2.0
        x1 = (x1 - xc) * 1.2 + xc
        x2 = (x2 - xc) * 1.2 + xc
        gx, gy = apply_homography(img2grd, np.array([x1, x2]), np.array([y2, y2]))
        p1 = np.array([gx[0], gy[0]])
        p4 = np.array([gx[1], gy[1]])
        dp = p4 - p1
        theta = np.arctan2(dp[1], dp[0]) + np.pi / 2.0
        thick = 2.0 * 1.2 * 20.0
        p2 = p1 + [np.cos(theta) * thick, np.sin(theta) * thick]
        p3 = p4 + [np.cos(theta) * thick, np.sin(theta) * thick]
        out.append(np.array([p1, p2, p3, p4], float))
    return out


def load_carboxes(ds, seq):
    ncam, T = len(seq), len(seq[0])
    cars = [[None] * T for _ in range(ncam)]
    for tt in range(T):
        for cam in range(ncam):
            name = io_utils.image_basename(seq[cam][tt])
            p = ds.workspace / "car_detection" / (name + ".txt")
            arr = io_utils.read_text_array2d(p) if p.exists() else np.zeros((0, 5))
            if arr.size == 0:
                cars[cam][tt] = np.zeros((0, 5))
            else:
                arr = arr.reshape(-1, arr.shape[-1])
                cars[cam][tt] = arr[arr[:, 4] >= 0][:, :5]
    return cars


def prepare_car_obs(cars, gi):
    """Ground-plane car obstacles per (cam, frame), using per-frame img2grd."""
    ncam, T = len(cars), len(cars[0])
    return [[carboxes2carobs(cars[cam][tt], gi.img2grd_t(tt, cam))
             for tt in range(T)] for cam in range(ncam)]


def combine_car_obs(car_obsz):
    """Use the right camera's car obstacles (index 1)."""
    return car_obsz[1]


def prepare_ped_obs(good, T):
    """Square obstacle around each tracklet's ground position per frame."""
    num = len(good)
    ped = [[None] * T for _ in range(num)]
    thick = 0.25 * 1.2 * 100 / 5.0
    for nn in range(num):
        for tt in range(good[nn].startt, good[nn].endt + 1):
            p0 = good[nn].trj_3d[tt]
            ped[nn][tt] = np.array([[p0[0] - thick, p0[1] - thick],
                                    [p0[0] + thick, p0[1] - thick],
                                    [p0[0] + thick, p0[1] + thick],
                                    [p0[0] - thick, p0[1] + thick]], float)
    return ped


def fix_poly_ground(poly_ground: np.ndarray) -> np.ndarray:
    """Drop a duplicated closing vertex and clamp negatives (2xN)."""
    pg = poly_ground
    dloopx = pg[0, 0] - pg[0, -1]
    dloopy = pg[1, 0] - pg[1, -1]
    if dloopx * dloopx + dloopy + dloopy < 0.5:   # replicates C++ (dloopy not squared)
        pg = pg[:, :-1]
    pg = np.maximum(pg, 0.0)
    return pg


def combine_obstacles(nn, car_obs_t, ped_obs_t, poly_ground):
    """Gather car + pedestrian (excluding self) obstacles; group touching ones.

    Returns (obs list, obs_cent (M,2)); obstacle groups attached to the region
    boundary are dropped from the centroid list (they are scene borders).
    """
    obs = list(car_obs_t)
    for pp in range(len(ped_obs_t)):
        if pp == nn:
            continue
        if ped_obs_t[pp] is not None:
            obs.append(ped_obs_t[pp])

    odx = [i for i in range(len(obs)) if obs[i] is not None and len(obs[i]) > 0]
    # Union-find over overlapping obstacle polygons.
    parent = {i: i for i in odx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for ai in range(len(odx)):
        o1 = odx[ai]
        plx, ply = obs[o1][:, 0], obs[o1][:, 1]
        for bj in range(ai + 1, len(odx)):
            o2 = odx[bj]
            for pt in obs[o2]:
                if point_in_polygon(pt[0], pt[1], plx, ply):
                    union(o1, o2)
                    break

    groups = {}
    for i in odx:
        groups.setdefault(find(i), []).append(i)

    gplx, gply = poly_ground[0], poly_ground[1]
    cents = []
    for _, members in sorted(groups.items()):
        attach_bg = False
        for oo in members:
            for pt in obs[oo]:
                if not point_in_polygon(float(pt[0]), float(pt[1]), gplx, gply):
                    attach_bg = True
                    break
            if attach_bg:
                break
        if attach_bg:
            continue
        first = obs[members[0]]
        cents.append([first[:, 0].mean(), first[:, 1].mean()])
    obs_cent = np.array(cents, float).reshape(-1, 2)
    return obs, obs_cent


def construct_obstacle_maps(obs, poly_ground, goal_ground):
    poly_x, poly_y = poly_ground[0], poly_ground[1]
    goal_x, goal_y = goal_ground[0], goal_ground[1]
    s1 = int(max(poly_y.max(), goal_y.max()) + 1.5)
    s2 = int(max(poly_x.max(), goal_x.max()) + 1.5)
    obs_map = np.ones((s1, s2), int)
    dyn = np.zeros((s1, s2), int)

    ys, xs = np.mgrid[0:s1, 0:s2]
    # Static: inside walkable polygon -> free (0).
    from .geometry import mask_from_polygon
    inside = mask_from_polygon(s1, s2, poly_x, poly_y).astype(bool)
    obs_map[inside] = 0

    for poly in obs:
        if poly is None or len(poly) == 0:
            continue
        m = mask_from_polygon(s1, s2, poly[:, 0], poly[:, 1]).astype(bool)
        dyn[m] = 1
    return obs_map, dyn


def construct_feature_maps(obs_map, dyn_obs_map):
    d_static = ndimage.distance_transform_edt(obs_map == 0)
    d_dyn = ndimage.distance_transform_edt(dyn_obs_map == 0)
    feat0 = np.ones_like(obs_map, float)
    feat1 = np.exp(-d_static / 1.0)
    feat2 = np.exp(-d_dyn / 4.0)
    return [feat0, feat1, feat2]


# --------------------------------------------------------------------------
# State graph + edge weights
# --------------------------------------------------------------------------

def build_state_graph(obs_map):
    """Return (ig2yx (ng,2), yx2ig, edges src/dst arrays)."""
    s1, s2 = obs_map.shape
    yx2ig = -np.ones((s1, s2), int)
    free = np.argwhere(obs_map == 0)
    ng = len(free)
    for ig, (yy, xx) in enumerate(free):
        yx2ig[yy, xx] = ig
    ig2yx = free.copy()

    src, dst, nbr = [], [], []
    for ig, (yy, xx) in enumerate(free):
        for k in range(8):
            y2, x2 = yy + DY[k], xx + DX[k]
            if 0 <= y2 < s1 and 0 <= x2 < s2 and yx2ig[y2, x2] >= 0:
                src.append(ig)
                dst.append(yx2ig[y2, x2])
                nbr.append(k)
    return ig2yx, yx2ig, np.array(src), np.array(dst), np.array(nbr)


def edge_weights(ig2yx, src, dst, nbr, feat, wei):
    """Feature-weighted Euclidean edge cost (compute_feat_dist)."""
    ys, xs = ig2yx[:, 0], ig2yx[:, 1]
    step = np.hypot(xs[dst] - xs[src], ys[dst] - ys[src])
    w = np.zeros(len(src))
    for ff, fmap in enumerate(feat):
        favg = 0.5 * (fmap[ys[src], xs[src]] + fmap[ys[dst], xs[dst]])
        w += wei[ff] * favg * step
    return w


def get_legal_index(yx2ig, ig2yx, x, y):
    yy, xx = int(y + 0.5), int(x + 0.5)
    if 0 <= yy < yx2ig.shape[0] and 0 <= xx < yx2ig.shape[1] and yx2ig[yy, xx] >= 0:
        return yx2ig[yy, xx]
    d = (ig2yx[:, 0] - yy) ** 2 + (ig2yx[:, 1] - xx) ** 2
    return int(np.argmin(d))


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def _reconstruct(pred, src, goal):
    path = []
    g = goal
    while g != src and g >= 0:
        path.append(g)
        g = pred[g]
    if g != src:
        return None
    path.append(src)
    path.reverse()
    return path


def is_looped(pv, ig2yx):
    seen = set()
    for n in pv:
        if n in seen:
            return True
        seen.add(n)
    edge_mid = set()
    for i in range(len(pv) - 1):
        a, b = pv[i], pv[i + 1]
        c = (ig2yx[a, 0] + ig2yx[b, 0], ig2yx[a, 1] + ig2yx[b, 1])
        if c in edge_mid:
            return True
        edge_mid.add(c)
    return False


def enclose_obstacle(path1, path2, obs_cent):
    """True if the loop (path1 + reversed interior of path2) encloses any obstacle."""
    m = np.vstack([path1, path2[1:-1][::-1]]) if len(path2) > 2 else path1
    mx, my = m[:, 0], m[:, 1]
    for oo in range(len(obs_cent)):
        if point_in_polygon(obs_cent[oo, 0], obs_cent[oo, 1], mx, my):
            return True
    return False


def check_redundancy(sel_paths_xy, sel_dist, obs_cent):
    """Keep only homotopy-distinct paths (greedy by ascending distance)."""
    order = np.argsort(sel_dist, kind="stable")
    useful = []
    for p1 in order:
        good = True
        for p2 in useful:
            if not enclose_obstacle(sel_paths_xy[p1], sel_paths_xy[p2], obs_cent):
                good = False
                break
        if good:
            useful.append(p1)
    return useful


# --------------------------------------------------------------------------
# Winding-angle planning
# --------------------------------------------------------------------------

def wind_angle_planning(ig2yx, src, dst, nbr, ew, obs_cent, wnum_l, wnum_u,
                        start, goal):
    ng = len(ig2yx)
    no = len(obs_cent)
    if no == 0:
        # Plain shortest path (single layer).
        graph = csr_matrix((ew, (src, dst)), shape=(ng, ng))
        d, pred = dijkstra(graph, directed=True, indices=start,
                           return_predecessors=True)
        path = _reconstruct(pred, start, goal)
        if path is None or is_looped(path, ig2yx):
            return [], np.zeros(0), np.zeros((0, 0), int)
        xy = np.column_stack([ig2yx[path, 1], ig2yx[path, 0]]).astype(float)
        return [path], np.array([d[goal]]), np.zeros((1, 0), int)

    nw = wnum_u - wnum_l + 1
    nlayer = nw ** no
    pw = nw ** np.arange(no)                      # place values

    # Angle of each cell around each obstacle.
    yy = ig2yx[:, 0].astype(float)[:, None]
    xx = ig2yx[:, 1].astype(float)[:, None]
    wa = np.arctan2(yy - obs_cent[:, 1][None, :], xx - obs_cent[:, 0][None, :])

    # Per base-edge wrap correction per obstacle.
    raw = wa[src] - wa[dst]                        # (E, no)
    dwa = raw.copy()
    dwa[dwa > np.pi] -= TWO_PI
    dwa[dwa <= -np.pi] += TWO_PI
    wrap = np.rint((raw - dwa) / TWO_PI).astype(int)   # (E, no) in {-1,0,1}

    # Build augmented edges.
    layer_digits = ((np.arange(nlayer)[:, None] // pw[None, :]) % nw)  # (nlayer,no)
    a_src, a_dst, a_w = [], [], []
    for ll in range(nlayer):
        w1 = layer_digits[ll]                     # (no,)
        w2 = w1[None, :] + wrap                    # (E, no)
        valid = np.all((w2 >= 0) & (w2 < nw), axis=1)
        if not valid.any():
            continue
        l2 = (w2[valid] * pw[None, :]).sum(axis=1)
        a_src.append(src[valid] * nlayer + ll)
        a_dst.append(dst[valid] * nlayer + l2)
        a_w.append(ew[valid])
    if not a_src:
        return [], np.zeros(0), np.zeros((0, no), int)
    a_src = np.concatenate(a_src); a_dst = np.concatenate(a_dst)
    a_w = np.concatenate(a_w)
    N = ng * nlayer
    graph = csr_matrix((a_w, (a_src, a_dst)), shape=(N, N))

    starts = [start * nlayer + ls for ls in range(nlayer)]
    dmat, pmat = dijkstra(graph, directed=True, indices=starts,
                          return_predecessors=True)

    sel_paths, sel_dist, sel_wind = [], [], []
    for si, ls in enumerate(range(nlayer)):
        s_node = starts[si]
        for lg in range(nlayer):
            gnode = goal * nlayer + lg
            dist = dmat[si, gnode]
            if not np.isfinite(dist):
                continue
            aug = _reconstruct(pmat[si], s_node, gnode)
            if aug is None:
                continue
            pv = [n // nlayer for n in aug]
            if is_looped(pv, ig2yx):
                continue
            wn = np.floor((wa[goal] + (wnum_l + layer_digits[lg]) * TWO_PI -
                           (wa[start] + (wnum_l + layer_digits[ls]) * TWO_PI))
                          / TWO_PI).astype(int)
            sel_paths.append(pv)
            sel_dist.append(dist)
            sel_wind.append(wn)

    if not sel_paths:
        return [], np.zeros(0), np.zeros((0, no), int)

    sel_xy = [np.column_stack([ig2yx[p, 1], ig2yx[p, 0]]).astype(float)
              for p in sel_paths]
    useful = check_redundancy(sel_xy, np.array(sel_dist), obs_cent)
    paths = [sel_paths[i] for i in useful]
    dists = np.array([sel_dist[i] for i in useful])
    winds = np.array([sel_wind[i] for i in useful]).reshape(-1, no)
    return paths, dists, winds


def shortest_path(ig2yx, src, dst, ew, start, goal):
    ng = len(ig2yx)
    graph = csr_matrix((ew, (src, dst)), shape=(ng, ng))
    d, pred = dijkstra(graph, directed=True, indices=start,
                       return_predecessors=True)
    path = _reconstruct(pred, start, goal)
    return (d[goal] if path else np.inf), (path or [])


def choose_critic_obstacles(ig2yx, obs_cent, spath, sdist, thr=150.0):
    if len(obs_cent) == 0 or len(spath) == 0:
        return np.zeros((0, 2))
    sp_xy = np.column_stack([ig2yx[spath, 1], ig2yx[spath, 0]]).astype(float)
    dist = np.zeros(len(obs_cent))
    perp = np.zeros(len(obs_cent), int)
    for cc in range(len(obs_cent)):
        d2 = (obs_cent[cc, 0] - sp_xy[:, 0]) ** 2 + (obs_cent[cc, 1] - sp_xy[:, 1]) ** 2
        idx = int(np.argmin(d2))
        dist[cc] = np.sqrt(d2[idx])
        perp[cc] = idx
    n = len(spath)
    cidx = []
    for cc in range(len(obs_cent)):
        if dist[cc] > sdist:
            continue
        if 5 <= perp[cc] < n - 5 and dist[cc] < thr:
            cidx.append(cc)
        elif (perp[cc] < 5 and dist[cc] < thr / 2) or \
             (perp[cc] >= n - 5 and dist[cc] < thr / 2):
            cidx.append(cc)
    cidx = sorted(cidx, key=lambda cc: dist[cc])[:3]
    return obs_cent[cidx] if cidx else np.zeros((0, 2))


def do_homotopy_planning(gi, nn, start_x, start_y, car_obs_t, ped_obs_t,
                         plan_grid_step=4):
    """Plan homotopy paths for tracklet ``nn`` to every goal.

    Returns a list (per goal) of :class:`PlanItem` with paths in FULL ground
    coordinates.
    """
    pf = float(plan_grid_step)
    poly_ground = fix_poly_ground(gi.poly_ground) / pf
    goal_ground = gi.goal_ground / pf
    obs_full, obs_cent_full = combine_obstacles(nn, car_obs_t, ped_obs_t,
                                                fix_poly_ground(gi.poly_ground))
    obs = [None if o is None or len(o) == 0 else o / pf for o in obs_full]
    obs_cent = obs_cent_full / pf if len(obs_cent_full) else obs_cent_full
    sx, sy = start_x / pf, start_y / pf

    obs_map, dyn_map = construct_obstacle_maps(obs, poly_ground, goal_ground)
    feat = construct_feature_maps(obs_map, dyn_map)
    ig2yx, yx2ig, src, dst, nbr = build_state_graph(obs_map)
    if len(ig2yx) == 0:
        return [PlanItem([], np.zeros(0), np.zeros((0, 0), int))
                for _ in range(goal_ground.shape[1])]
    wei = [1.61642, 3.0, 15.0]
    ew = edge_weights(ig2yx, src, dst, nbr, feat, wei)

    start = get_legal_index(yx2ig, ig2yx, sx, sy)
    results = []
    for gg in range(goal_ground.shape[1]):
        goal = get_legal_index(yx2ig, ig2yx, goal_ground[0, gg], goal_ground[1, gg])
        sdist, spath = shortest_path(ig2yx, src, dst, ew, start, goal)
        critic = choose_critic_obstacles(ig2yx, obs_cent, spath, sdist, thr=150.0 / pf)
        paths, dists, winds = wind_angle_planning(
            ig2yx, src, dst, nbr, ew, critic, -1, 0, start, goal)
        # Convert node paths -> (x,y) in FULL ground coords.
        xy_paths = []
        for p in paths:
            xy = np.column_stack([ig2yx[p, 1], ig2yx[p, 0]]).astype(float) * pf
            xy_paths.append(xy)
        results.append(PlanItem(xy_paths, dists, winds))
    return results


def plan_trlet_list(gi, Tff, good, car_obs, ped_obs, plan_advance=7,
                    plan_grid_step=4, verbose=True):
    """Plan for every tracklet that has at least one feasible successor."""
    ng = len(good)
    plan_time = np.zeros(ng, int)
    results = [None] * ng
    for ii in range(ng):
        if Tff[ii].sum() <= 0:
            continue
        endt, startt = good[ii].endt, good[ii].startt
        plant = max(endt - plan_advance, startt)
        plan_time[ii] = plant
        sx = good[ii].trj_3d[plant, 0]
        sy = good[ii].trj_3d[plant, 1]
        ped_obs_t = [ped_obs[nn][plant] for nn in range(len(ped_obs))]
        results[ii] = do_homotopy_planning(gi, ii, sx, sy, car_obs[plant],
                                           ped_obs_t, plan_grid_step)
        if verbose:
            npaths = sum(len(r.paths) for r in results[ii])
            print(f"  planned tracklet {ii}/{ng}: {npaths} paths")
    return plan_time, results
