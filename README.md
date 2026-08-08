# trpl_track — Python/NumPy port

A modern **NumPy / SciPy** reimplementation of the demo code for:

> Haifeng Gong, Jack Sim, Maxim Likhachev, Jianbo Shi,
> *"Multi-hypothesis Motion Planning for Visual Object Tracking"*, ICCV 2011.

It ports the original C++ (Boost uBLAS / CImg / GLPK / UMFPACK / MPI) core
tracking pipeline to a single, dependency-light Python package that runs
serially on the bundled `sequence1` (static cameras) and `sequence5` (moving
cameras) datasets.

## Pipeline

```
pretr → filter → appearance affinity → motion-planning LP → finalize → visualize
```

| Stage | Module | What it does |
|-------|--------|--------------|
| pre-tracking | `pretr.py` | Per-frame: track existing objects, read pre-computed detections, initialize new tracklets. Binocular fusion on the ground plane. Produces raw tracklets. |
| filtering | `filter.py` | Keep tracklets of length ≥ 3; build the gating matrix `Tff` (temporal + kinematic feasibility) and the appearance affinity `Aff`. |
| planning | `planning.py` | The paper's core: winding-number **multi-hypothesis homotopy planning** on the ground plane around car/pedestrian obstacles. |
| plan affinity | `linprog_plan.py` | Match plans to tracklet ends, synthesize interpolated gap tracklets, score them by appearance → planning affinity `Plff`. |
| linking LP | `lp.py` | Max-weight bipartite matching `max (Aff + 0.5·Plff)·x` (GLPK → `scipy.optimize.linprog`). |
| finalize | `finalize.py` | Walk the link graph into full trajectories; fill gaps with planned boxes. |
| visualize | `vis.py` | Per-frame image overlays (boxes/ids/trails) + top-down ground-plane views (full extent by default, `crop=True` for a paper-Fig.-5 crop): walkable polygon, goals, **car obstacles** as orange quads, pedestrian footprints, and trajectory tracks coloured by provenance (observed vs **planned/gap** in cyan). |

## Dependency mapping (C++ → Python)

| Original | Replacement |
|----------|-------------|
| Boost uBLAS / multi_array | `numpy` |
| GLPK (LP) | `scipy.optimize.linprog(method="highs")` |
| UMFPACK / Boost.Graph Dijkstra | `scipy.sparse` + `scipy.sparse.csgraph.dijkstra` |
| CImg (image I/O, distance transform, drawing) | `Pillow` + `scipy.ndimage` |
| Boost.MPI (`mpirun`) | dropped — runs serially |
| Boost XML serialization | `numpy` `.npz` (the dataset ships no `.xml`) |

## Install & run

Run from the repository root (the `trpl_track` package lives at top level):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install numpy scipy pillow

# Full sequence (writes overlays to <sequence>/output_py/):
python -m trpl_track.run --sequence sequence1
python -m trpl_track.run --sequence sequence5

# Quick smoke test on the first N frames:
python -m trpl_track.run --sequence sequence1 --max-frames 15
```

Options: `--max-frames N`, `--plan-grid-step K` (ground-grid downsample for the
planner; default 4), `--no-cache`, `--no-vis`.

Outputs (in `<sequence>/output_py/`): `track_cam0_###.jpg` (image overlays,
yellow = observed box, cyan = planned/gap box, with ids and trails),
`ground_###.png` (top-down views), `results.txt` (boxes per object/frame).
Pre-tracking results are cached in `<sequence>/workspace/py_cache/`.

## Faithfulness & deviations

The port reproduces the original algorithms and constants (histogram model,
scoring, gating thresholds, LP objective, winding-number homotopy enumeration,
gap interpolation, the `Aff + 0.5·Plff` mix). Deliberate deviations, all
documented in code:

- **`Plff` accumulation bug fixed** (`select_plan_gap_paths`). Cross-checked
  against the paper's Sec 4.2 `S_Plan = max_r (−Dist(r,F_i) − Dist(r,F_j) +
  S_Occl)`: `Dist` carries no camera index (it is summed over both cameras into
  the single scalar `dist_mat[i][j][r]`), so it must be applied once per path.
  The original C++ applied the `−Dist/800 + 2` penalty and the `sat(·,16)` cap
  *inside* the per-camera loop, double-counting it for the stereo pair. This
  port computes S_Plan the paper-consistent way (accumulate `S_Occl` over both
  cameras first, then apply the penalty/cap once).

- **Serial** execution (no MPI); results are identical to the single-process path.
- **Planning ground grid is downsampled** by `--plan-grid-step` (default 4) so the
  augmented winding-graph Dijkstra is tractable in pure Python. The 5 cm native
  grid yields ~110k free cells × up to 8 winding layers per goal per tracklet;
  the original relied on an MPI cluster. The homotopy logic is unchanged and
  paths are scaled back to full ground coordinates. Use `--plan-grid-step 1` for
  full resolution (slow).
- **Moving camera (sequence5)** uses per-frame ego-motion geometry: the
  ground↔image homographies are loaded from `init_input/img2grd.txt` /
  `grd2img.txt` (a `T×Ncam` field of 3×3 matrices) and indexed by frame
  `(tt, cam)` throughout, while `poly_ground`/`goal_ground` stay in one fixed
  world frame. sequence1 (static cameras) has no such files and uses a single
  homography per camera. `GeometricInfo.load` picks the mode automatically; the
  `*_t(tt, cam)` accessors resolve both uniformly.
- **Detections are pre-computed** and shipped in `<sequence>/workspace/`
  (`detection_refine/*_3d_ped.txt` for sequence1, `*.fmat` for sequence5); there
  is no in-repo detector, exactly as in the original.
- Segmentation masks (`segment_parts`) are not materialized; their only consumer
  is the filter's `seg_score`, which — because the C++ counts every mask pixel —
  equals mean box area and is recomputed directly from boxes.
```
