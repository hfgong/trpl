"""End-to-end driver: pretr -> filter -> affinity -> planning LP -> finalize -> vis.

Usage::

    python -m trpl_track.run --sequence sequence1
    python -m trpl_track.run --sequence sequence1 --max-frames 20 --plan-grid-step 4

Intermediate results are cached in ``<workspace>/py_cache`` so re-runs skip the
slow pre-tracking stage.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import filter as filt
from . import io_utils, vis
from .config import DirectoryStructure, Parameters
from .finalize import finalize_trajectory_plan
from .geometry import GeometricInfo
from .linprog_plan import run_linprog_plan
from .pretr import run_pretr
from .tracklet import save_tracklets, load_tracklets


def main(argv=None):
    ap = argparse.ArgumentParser(description="trpl_track pipeline")
    ap.add_argument("--sequence", default="sequence1",
                    help="path to a sequence root (sequence1 / sequence5)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="limit number of frames (for quick tests)")
    ap.add_argument("--plan-grid-step", type=int, default=4,
                    help="ground-grid downsample factor for planning (>=1)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached pretr result and recompute")
    ap.add_argument("--no-vis", action="store_true", help="skip writing images")
    args = ap.parse_args(argv)

    ds = DirectoryStructure(args.sequence)
    problems = ds.missing_inputs()
    if problems:
        print(f"Cannot run on sequence '{ds.root.name}' -- required input data "
              f"is missing:")
        for p in problems:
            print(f"  - {p}")
        print("\nThis repository ships input data only for 'sequence1' "
              "(static cameras) and 'sequence5' (moving cameras).")
        raise SystemExit(1)
    ds.make_dirs()
    P = Parameters()
    gi = GeometricInfo.load(ds)
    seq = io_utils.read_sequence_list(ds)
    if args.max_frames:
        seq = [seq[0][:args.max_frames], seq[1][:args.max_frames]]
    T = len(seq[0])
    print(f"sequence: {ds.root.name}  frames: {T}  cameras: {len(seq)}")

    cache = ds.workspace / "py_cache"
    cache.mkdir(exist_ok=True)
    raw_path = cache / f"raw_trlet_T{T}.npz"

    t0 = time.time()
    if raw_path.exists() and not args.no_cache:
        print(f"[pretr] loading cached tracklets from {raw_path.name}")
        raw = load_tracklets(raw_path)
    else:
        print("[pretr] running pre-tracking ...")
        raw = run_pretr(ds, P, gi, seq=seq, max_frames=args.max_frames)
        save_tracklets(raw_path, raw)
    print(f"[pretr] {len(raw)} raw tracklets  ({time.time()-t0:.1f}s)")

    good, good_index = filt.filter_tracklets(raw)
    print(f"[filter] {len(good)} good tracklets (len>=3)")
    if len(good) == 0:
        print("no good tracklets; nothing to link.")
        return

    Tff = filt.prepare_valid_linkset(good)
    Aff = filt.prepare_app_affinity(Tff, good)
    print(f"[gating] {int(Tff.sum())} candidate links")

    t1 = time.time()
    res = run_linprog_plan(ds, P, gi, good, Tff, Aff, seq=seq,
                           plan_grid_step=args.plan_grid_step)
    print(f"[LP] {len(res['links'])} links selected  ({time.time()-t1:.1f}s)")

    final_list, trj_index, state_list = finalize_trajectory_plan(
        len(seq), T, res["links"], good, res["gap_rind"], res["gap"])
    print(f"[finalize] {len(final_list)} final trajectories")
    for k, (f, chain) in enumerate(zip(final_list, trj_index)):
        print(f"   traj {k}: frames [{f.startt},{f.endt}] from tracklets {chain}")

    save_tracklets(cache / f"final_trj_T{T}.npz", final_list)
    np.save(cache / f"state_list_T{T}.npy", state_list)

    if not args.no_vis:
        print("[vis] writing overlays ...")
        vis.vis_image_frames(ds, seq, final_list, state_list, trj_index, cam=0)
        vis.vis_ground_frames(ds, gi, final_list, state_list,
                              trj_index=trj_index, gap_paths=res["gap_paths"],
                              car_obs=res["car_obs"], ped_obs=res["ped_obs"])
        vis.write_results_txt(ds, final_list, state_list)
        print(f"[vis] images written to {ds.output}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
