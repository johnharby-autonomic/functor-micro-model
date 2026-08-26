#!/usr/bin/env python3
"""
W x bandwidth sweep driver for the probabilistic functor MicroModel.

Walks the (window, bandwidth) grid and, for each point, measures:
  - per-commit and per-predict cost (drift OFF: clean timing)
  - bounded-drift signature (one extra drift-ON pass: timing ignored)

Emits one JSON row per grid point plus a compact CSV to stdout, so the
cost-vs-spillover trade-off is directly plottable.

Usage:
  python -m core.sweep_prob \
      --windows 8 16 32 64 \
      --bandwidths 0.25 0.5 1.0 \
      --updates 4000 --predictions-per-update 10 \
      --output-json sweep_results.json

Cost timing uses drift OFF (instrumentation off the timed path); the drift
figures come from a separate measure_drift=True pass on the same config whose
elapsed time is discarded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from core.prob_functor_model import (
    BenchmarkConfig,
    create_prob_model,
    run_commit_workload,
    run_lifecycle_workload,
    seed_model,
)


def _time_commit(window: int, bandwidth: float, updates: int, cap: float, warmup: int) -> Dict[str, float]:
    cfg = BenchmarkConfig("commit", updates, 0, 1, warmup, window, bandwidth, cap, measure_drift=False)
    for _ in range(warmup):
        run_commit_workload(cfg)
    t = time.perf_counter()
    model = run_commit_workload(cfg)
    elapsed = time.perf_counter() - t
    committed = max(model.commit_count, 1)
    return {"commit_us": 1e6 * elapsed / committed, "committed": model.commit_count,
            "skipped": len(model.skipped_events)}


def _time_lifecycle(window: int, bandwidth: float, updates: int, ppu: int, cap: float) -> Dict[str, float]:
    cfg = BenchmarkConfig("lifecycle", updates, ppu, 1, 0, window, bandwidth, cap, measure_drift=False)
    t = time.perf_counter()
    run_lifecycle_workload(cfg)
    elapsed = time.perf_counter() - t
    preds = max(updates * ppu, 1)
    return {"cycle_us": 1e6 * elapsed / max(updates, 1), "predict_us_approx": 1e6 * elapsed / preds}


def _drift(window: int, bandwidth: float, updates: int, cap: float) -> Dict[str, Any]:
    # Separate pass, drift ON; timing discarded.
    model = create_prob_model(window, bandwidth, cap, measure_drift=True)
    seed_model(model, updates)
    return model.get_drift_stats()


def run_sweep(
    windows: List[int],
    bandwidths: List[float],
    updates: int,
    ppu: int,
    cap: float,
    warmup: int,
    do_lifecycle: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for w in windows:
        for bw in bandwidths:
            row: Dict[str, Any] = {"window": w, "bandwidth": bw}
            row.update(_time_commit(w, bw, updates, cap, warmup))
            if do_lifecycle:
                row.update(_time_lifecycle(w, bw, updates, ppu, cap))
            row["drift"] = _drift(w, bw, updates, cap)
            rows.append(row)
    return rows


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W x bandwidth sweep for prob_functor_model.")
    p.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--bandwidths", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--predictions-per-update", type=int, default=10)
    p.add_argument("--max-cumulative-influence", type=float, default=2.0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--lifecycle", action="store_true", help="Also time lifecycle (commit+predict).")
    p.add_argument("--output-json", default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    rows = run_sweep(
        a.windows, a.bandwidths, a.updates, a.predictions_per_update,
        a.max_cumulative_influence, a.warmup, a.lifecycle,
    )

    # Compact CSV to stdout for quick plotting.
    cols = ["window", "bandwidth", "commit_us", "skipped",
            "spill_mean", "spill_max", "evict_mean", "evict_max"]
    if a.lifecycle:
        cols[3:3] = ["cycle_us"]
    print(",".join(cols))
    for r in rows:
        d = r["drift"]
        vals = [r["window"], r["bandwidth"], f"{r['commit_us']:.2f}"]
        if a.lifecycle:
            vals.append(f"{r['cycle_us']:.2f}")
        vals += [r["skipped"],
                 f"{d['mean_abs_drift']:.5f}", f"{d['max_abs_drift']:.5f}",
                 f"{d['mean_eviction_loss']:.5f}", f"{d['max_eviction_loss']:.5f}"]
        print(",".join(str(v) for v in vals))

    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as fh:
            json.dump({"grid": rows}, fh, indent=2, sort_keys=True)
            fh.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
