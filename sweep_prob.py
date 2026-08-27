#!/usr/bin/env python3
"""Route x evaluation-strategy sweep for the probabilistic functor model.

The retrieval model has no window, bandwidth, eviction, or drift parameters.
Its relevant inference choices are the requested moment-bundle route and
whether committed moments are evaluated lazily or cached eagerly.  This driver
benchmarks those choices using the timing boundaries implemented by
``core.prob_functor_model.run_workload``.

Example::

    python3 -m sweep_prob \
        --routes dist sample mean var stddev \
        --strategies lazy eager \
        --updates 400 --predictions-per-update 10 \
        --trials 3 --warmup 1 --lifecycle \
        --output-json sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.prob_functor_model import ROUTES, BenchmarkConfig, run_workload

STRATEGIES = ("lazy", "eager")


def _microseconds(elapsed_seconds: float, operations: int) -> float:
    """Return elapsed microseconds per operation.

    Sweep inputs require positive operation counts, so a zero denominator is a
    programming error rather than a value to silently normalize.
    """
    if operations <= 0:
        raise ValueError("operations must be > 0")
    return 1e6 * elapsed_seconds / operations


def _benchmark(
    *,
    mode: str,
    route: str,
    eager: bool,
    updates: int,
    predictions_per_update: int,
    trials: int,
    warmup: int,
) -> Dict[str, Any]:
    config = BenchmarkConfig(
        mode=mode,
        updates=updates,
        predictions_per_update=predictions_per_update,
        trials=trials,
        warmup=warmup,
        route=route,
        eager=eager,
    )
    return run_workload(config)


def _validate_sweep_inputs(
    routes: Sequence[str],
    strategies: Sequence[str],
    updates: int,
    predictions_per_update: int,
    trials: int,
    warmup: int,
) -> None:
    if not routes:
        raise ValueError("at least one route is required")
    unknown_routes = sorted(set(routes) - set(ROUTES))
    if unknown_routes:
        raise ValueError(f"unknown routes: {unknown_routes}; expected values from {ROUTES}")

    if not strategies:
        raise ValueError("at least one strategy is required")
    unknown_strategies = sorted(set(strategies) - set(STRATEGIES))
    if unknown_strategies:
        raise ValueError(
            f"unknown strategies: {unknown_strategies}; expected values from {STRATEGIES}"
        )

    if updates <= 0:
        raise ValueError("updates must be > 0")
    if predictions_per_update <= 0:
        raise ValueError("predictions_per_update must be > 0")
    if trials <= 0:
        raise ValueError("trials must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")


def run_sweep(
    routes: Sequence[str],
    strategies: Sequence[str],
    updates: int,
    predictions_per_update: int,
    trials: int,
    warmup: int,
    do_lifecycle: bool,
) -> List[Dict[str, Any]]:
    """Benchmark every route/strategy pair and return one flat row per pair."""
    _validate_sweep_inputs(
        routes,
        strategies,
        updates,
        predictions_per_update,
        trials,
        warmup,
    )

    rows: List[Dict[str, Any]] = []
    for strategy in strategies:
        eager = strategy == "eager"
        # Commit cost depends on lazy/eager materialization, but not on which
        # route will later be requested. Measure it once per strategy so timing
        # noise is not misrepresented as a route-dependent commit effect.
        commit = _benchmark(
            mode="commit",
            route="mean",
            eager=eager,
            updates=updates,
            predictions_per_update=0,
            trials=trials,
            warmup=warmup,
        )
        commit_benchmark = commit["benchmark"]
        commit_us = _microseconds(
            commit_benchmark["timed_elapsed_seconds"],
            commit_benchmark["measured_updates"],
        )

        for route in routes:
            inference = _benchmark(
                mode="inference",
                route=route,
                eager=eager,
                updates=updates,
                predictions_per_update=predictions_per_update,
                trials=trials,
                warmup=warmup,
            )

            inference_benchmark = inference["benchmark"]
            row: Dict[str, Any] = {
                "route": route,
                "strategy": strategy,
                "eager_moments": eager,
                "updates": updates,
                "predictions_per_update": predictions_per_update,
                "trials": trials,
                "warmup": warmup,
                "committed": commit["model"]["committed_updates"],
                "table_size": commit["model"]["table_size"],
                "state_hash": commit["model"]["state_hash"],
                "commit_us": commit_us,
                "predict_us": _microseconds(
                    inference_benchmark["timed_elapsed_seconds"],
                    inference_benchmark["measured_predictions"],
                ),
                "commit_trial_elapsed_seconds": commit_benchmark[
                    "trial_elapsed_seconds"
                ],
                "inference_trial_elapsed_seconds": inference_benchmark[
                    "trial_elapsed_seconds"
                ],
                "inference_setup_elapsed_seconds": inference_benchmark[
                    "setup_elapsed_seconds"
                ],
            }

            if do_lifecycle:
                lifecycle = _benchmark(
                    mode="lifecycle",
                    route=route,
                    eager=eager,
                    updates=updates,
                    predictions_per_update=predictions_per_update,
                    trials=trials,
                    warmup=warmup,
                )
                lifecycle_benchmark = lifecycle["benchmark"]
                row["lifecycle_us_per_update"] = _microseconds(
                    lifecycle_benchmark["timed_elapsed_seconds"],
                    lifecycle_benchmark["measured_updates"],
                )
                row["lifecycle_trial_elapsed_seconds"] = lifecycle_benchmark[
                    "trial_elapsed_seconds"
                ]

            rows.append(row)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route x lazy/eager sweep for the retrieval probabilistic functor model."
    )
    parser.add_argument("--routes", choices=ROUTES, nargs="+", default=list(ROUTES))
    parser.add_argument(
        "--strategies", choices=STRATEGIES, nargs="+", default=list(STRATEGIES)
    )
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--predictions-per-update", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--lifecycle",
        action="store_true",
        help="Also measure interleaved commit-and-predict lifecycle cost.",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def _print_csv(rows: Sequence[Dict[str, Any]], include_lifecycle: bool) -> None:
    columns = ["route", "strategy", "commit_us", "predict_us"]
    if include_lifecycle:
        columns.append("lifecycle_us_per_update")
    columns.extend(["committed", "table_size", "state_hash"])
    print(",".join(columns))

    for row in rows:
        values = [
            row["route"],
            row["strategy"],
            f"{row['commit_us']:.6f}",
            f"{row['predict_us']:.6f}",
        ]
        if include_lifecycle:
            values.append(f"{row['lifecycle_us_per_update']:.6f}")
        values.extend(
            [str(row["committed"]), str(row["table_size"]), row["state_hash"]]
        )
        print(",".join(values))


def _result_document(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "grid": list(rows),
        "runtime": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = run_sweep(
        routes=args.routes,
        strategies=args.strategies,
        updates=args.updates,
        predictions_per_update=args.predictions_per_update,
        trials=args.trials,
        warmup=args.warmup,
        do_lifecycle=args.lifecycle,
    )
    _print_csv(rows, include_lifecycle=args.lifecycle)

    if args.output_json:
        output_path = Path(args.output_json)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(_result_document(rows), handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
