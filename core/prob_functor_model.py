#!/usr/bin/env python3
"""
Retrieval-class Probabilistic Functor MicroModel - moment-bundle router.

This is the *typical* probabilistic functor, as distinct from the RBF
interpolating variant (prob_functor_model.py). Most probabilistic functors are
retrieval: predict() returns the distribution object f(x), or evaluates f at
given parameters, or reduces f to a requested moment (mean / variance / stddev)
via a bundle of morphisms over the same object. There is no soft-gate window,
no closure composition, and no drift -- committing a patch never perturbs any
other query, exactly like the deterministic hard-gate model.

The category-theory content that earns its keep here: mean, variance, and stddev
are not three separately-fitted heads (an "ensemble"); they are arrows in a
bundle over one distribution object, coherent by construction. variance and
stddev compose (stddev = sqrt o variance), so a derived quantity cannot
silently disagree with the base distribution -- it is *read off* the same
object, not separately maintained.

inference becomes a parameter-driven router:

    predict(x, "dist")    -> the whole distribution object (the parameters of f)
    predict(x, "sample")  -> f evaluated to a point estimate (the mean here)
    predict(x, "mean")    -> first moment
    predict(x, "var")     -> second central moment
    predict(x, "stddev")  -> sqrt(var), composed from the var morphism

Two moment-evaluation strategies are benchmarked, because that is the only real
cost variable in the retrieval case:

  * lazy  (default): moments computed on request from the stored distribution.
  * eager: moments precomputed at commit and cached on the patch, so a request
    is a dict read.

Expected envelope position: just above the deterministic floor (~20 microjoule
on M2), dominated by the moment arithmetic, flat in N, zero drift -- the cheap
corner that represents the bulk of the probabilistic-functor family.

State hash is reproducible SHA-256 over a canonical byte encoding (struct.pack
IEEE-754), never Python's salted hash(), matching the other models.

Author: John Harby
Status: Patent pending (November 2025)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import numpy as np

Vector: TypeAlias = np.ndarray

# A distribution is represented compactly by its parameters. Here: a diagonal
# Gaussian over k components, stored as (mean_vector, var_vector). The base
# function maps x -> such a parameter pair; patches override the parameters at
# specific anchors (exact retrieval, no interpolation).
DistParams: TypeAlias = Tuple[Vector, Vector]  # (mean, variance), variance >= 0

# Router outputs are heterogeneous (a pair for "dist", a vector for moments),
# so the router returns Any. Callers know the morphism they asked for.
ROUTES = ("dist", "sample", "mean", "var", "stddev")


@dataclass
class MomentBundle:
    """
    A bundle of morphisms over one diagonal-Gaussian distribution object.

    Each method is an arrow out of the same (mean, var) object; stddev composes
    over var (sqrt o var). This is the structural alternative to an ensemble of
    independent heads: the moments cannot disagree with the distribution because
    they are computed from it, and stddev cannot disagree with var because it is
    defined as its composition.
    """

    mean_v: Vector
    var_v: Vector

    def dist(self) -> DistParams:
        return (self.mean_v, self.var_v)

    def mean(self) -> Vector:
        return self.mean_v

    def var(self) -> Vector:
        return self.var_v

    def stddev(self) -> Vector:
        return np.sqrt(self.var_v)

    def sample_point(self) -> Vector:
        """Point estimate: the distribution's mean (deterministic 'evaluation')."""
        return self.mean_v

    def route(self, param: str) -> Any:
        if param == "dist":
            return self.dist()
        if param == "mean":
            return self.mean()
        if param == "var":
            return self.var()
        if param == "stddev":
            return self.stddev()
        if param == "sample":
            return self.sample_point()
        raise ValueError(f"Unknown route: {param!r}; expected one of {ROUTES}")


@dataclass
class RetrievalPatch:
    """
    Exact-retrieval override of the distribution parameters at one anchor.

    Stored in a hash-keyed table (like the deterministic model), so predict()
    is an O(1) lookup, not a windowed sum. `cached_moments` is populated only
    in eager mode.
    """

    anchor: Vector
    mean_v: Vector
    var_v: Vector
    version: int = 0
    cached_moments: Optional[Dict[str, Any]] = None

    def materialize_eager(self) -> None:
        b = MomentBundle(self.mean_v, self.var_v)
        self.cached_moments = {r: b.route(r) for r in ROUTES}


@dataclass
class MicroModelConfig:
    enable_audit_logging: bool = True
    eager_moments: bool = False  # precompute moments at commit (dict-read on request)


@dataclass
class MicroModel:
    """
    Retrieval-class probabilistic functor: exact-lookup distribution patches
    with a parameter-driven moment-bundle router.

    predict(x, param) looks up the distribution parameters at x (the committed
    override if present, else the base function), then routes `param` through
    the moment bundle. O(1) lookup + the requested morphism's arithmetic. No
    window, no closure chain, no drift.
    """

    f: Callable[[Vector], DistParams]  # base distribution function
    config: MicroModelConfig = field(default_factory=MicroModelConfig)

    # Exact-lookup table: x_hash -> RetrievalPatch. This *is* the learned state.
    table: Dict[str, RetrievalPatch] = field(default_factory=dict)
    version: int = 0
    commit_count: int = 0
    operation_log: List[Dict[str, Any]] = field(default_factory=list)

    # -- hashing of the lookup key (NOT the determinism hash; see get_state_hash) --
    @staticmethod
    def _key(x: Vector) -> str:
        return np.asarray(x, dtype=float).tobytes().hex()

    # ------------------------------------------------------------------
    # Prediction (the parameter-driven router)
    # ------------------------------------------------------------------

    def _bundle_at(self, x: Vector) -> Tuple[MomentBundle, Optional[RetrievalPatch]]:
        patch = self.table.get(self._key(x))
        if patch is not None:
            return MomentBundle(patch.mean_v, patch.var_v), patch
        mean_v, var_v = self.f(x)
        return MomentBundle(np.asarray(mean_v, dtype=float), np.asarray(var_v, dtype=float)), None

    def predict(self, x: Vector, param: str = "mean") -> Any:
        """
        Parameter-driven inference. Returns the distribution object, a point
        estimate, or a requested moment, depending on `param`.

        In eager mode a committed anchor's moment requests are served from the
        patch's precomputed cache (a dict read); otherwise the requested
        morphism is computed from the distribution parameters on demand.
        """
        if self.config.eager_moments:
            patch = self.table.get(self._key(x))
            if patch is not None and patch.cached_moments is not None:
                return patch.cached_moments[param]
        bundle, _ = self._bundle_at(x)
        return bundle.route(param)

    # ------------------------------------------------------------------
    # Commit (exact override; no propose/validate ceremony needed here -- the
    # retrieval case has no cumulative-influence spillover to bound)
    # ------------------------------------------------------------------

    def commit(self, x: Vector, mean_v: Vector, var_v: Vector, event_index: Optional[int] = None) -> None:
        """
        Commit an exact distribution override at anchor x. O(1): one dict
        insert (+ moment precompute in eager mode). No window eviction, no
        drift -- this anchor's override never perturbs any other anchor.
        """
        var_arr = np.asarray(var_v, dtype=float)
        if np.any(var_arr < 0):
            raise ValueError("variance must be non-negative")
        patch = RetrievalPatch(
            anchor=np.asarray(x, dtype=float).copy(),
            mean_v=np.asarray(mean_v, dtype=float).copy(),
            var_v=var_arr.copy(),
            version=self.version + 1,
        )
        if self.config.eager_moments:
            patch.materialize_eager()
        self.table[self._key(x)] = patch
        self.version += 1
        self.commit_count += 1
        if self.config.enable_audit_logging:
            self.operation_log.append(
                {"op": "commit", "version": self.version, "event_index": event_index}
            )
            if len(self.operation_log) > 500:
                del self.operation_log[: len(self.operation_log) - 500]

    # ------------------------------------------------------------------
    # State hash (determinism) -- canonical SHA-256, never salted hash()
    # ------------------------------------------------------------------

    def get_state_hash(self) -> str:
        hasher = hashlib.sha256()

        def a_str(s: str) -> None:
            hasher.update(s.encode("utf-8"))

        def a_int(i: int) -> None:
            hasher.update(struct.pack(">q", int(i)))

        def a_vec(v: Vector) -> None:
            arr = np.asarray(v, dtype=float).ravel()
            a_int(arr.size)
            for c in arr:
                hasher.update(struct.pack(">d", float(c)))

        a_str("v")
        a_int(self.version)
        a_str("n")
        a_int(len(self.table))
        # Sort by canonical key so the digest is independent of insertion order
        # but still covers full content.
        for key in sorted(self.table.keys()):
            patch = self.table[key]
            a_str("|p")
            a_int(patch.version)
            a_str("|a")
            a_vec(patch.anchor)
            a_str("|m")
            a_vec(patch.mean_v)
            a_str("|s")
            a_vec(patch.var_v)
        return hasher.hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "table_size": len(self.table),
            "commit_count": self.commit_count,
            "eager_moments": self.config.eager_moments,
            "state_hash": self.get_state_hash(),
        }


# ============================================================================
# Example base distribution function
# ============================================================================


def base_dist_gaussian(x: Vector) -> DistParams:
    """Diagonal Gaussian whose parameters vary smoothly with x (analog of
    base_model_sigmoid in the other models)."""
    mean_v = np.array([0.5 + 0.35 * np.sin(x[0] * 0.37), 0.5 + 0.35 * np.cos(x[0] * 0.41)])
    var_v = np.array([0.05 + 0.04 * (np.sin(x[0] * 0.13) ** 2), 0.05 + 0.04 * (np.cos(x[0] * 0.17) ** 2)])
    return mean_v, var_v


# ============================================================================
# Benchmark harness (commit / inference / lifecycle), JSON shape compatible
# with bench_micro.py + additive fields: --route, --eager.
# ============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    updates: int
    predictions_per_update: int
    trials: int
    warmup: int
    route: str
    eager: bool


def create_model(eager: bool = False) -> MicroModel:
    """Fresh retrieval-class model for benchmarking (audit logging off so the
    timed path is pure model work, matching the other harnesses)."""
    return MicroModel(
        f=base_dist_gaussian,
        config=MicroModelConfig(enable_audit_logging=False, eager_moments=eager),
    )


def make_input(index: int) -> Vector:
    return np.array([float(index) + 1.0], dtype=float)


def make_target_dist(index: int) -> DistParams:
    phase = float(index)
    mean_v = np.array([0.5 + 0.3 * np.sin(phase * 0.31), 0.5 + 0.3 * np.cos(phase * 0.29)])
    var_v = np.array([0.04 + 0.03 * (np.sin(phase * 0.11) ** 2), 0.04 + 0.03 * (np.cos(phase * 0.19) ** 2)])
    return mean_v, var_v


def make_prediction_input(index: int, committed_updates: int) -> Vector:
    if committed_updates <= 0:
        return np.array([1.0 + 0.001 * float(index)], dtype=float)
    anchor = (index % committed_updates) + 1
    return np.array([float(anchor)], dtype=float)


def commit_one_update(model: MicroModel, index: int) -> None:
    mean_v, var_v = make_target_dist(index)
    model.commit(make_input(index), mean_v, var_v, event_index=index)


def seed_model(model: MicroModel, updates: int) -> None:
    for index in range(updates):
        commit_one_update(model, index)


def run_commit_workload(config: BenchmarkConfig) -> MicroModel:
    model = create_model(config.eager)
    for index in range(config.updates):
        commit_one_update(model, index)
    return model


def seed_inference_model(config: BenchmarkConfig) -> MicroModel:
    model = create_model(config.eager)
    seed_model(model, config.updates)
    return model


def total_predictions(config: BenchmarkConfig) -> int:
    if config.mode == "commit":
        return 0
    return config.updates * config.predictions_per_update


def run_inference_predictions(model: MicroModel, config: BenchmarkConfig) -> MicroModel:
    prediction_count = total_predictions(config)
    route = config.route
    accumulator = 0.0
    for index in range(prediction_count):
        out = model.predict(make_prediction_input(index, config.updates), route)
        # Reduce every route shape to a float so the accumulator guard works
        # and the optimizer can't elide the call.
        if route == "dist":
            accumulator += float(np.sum(out[0])) + float(np.sum(out[1]))
        else:
            accumulator += float(np.sum(out))
    if accumulator == float("inf"):
        raise RuntimeError("Unreachable accumulator guard")
    return model


def run_lifecycle_workload(config: BenchmarkConfig) -> MicroModel:
    model = create_model(config.eager)
    route = config.route
    accumulator = 0.0
    for update_index in range(config.updates):
        commit_one_update(model, update_index)
        for prediction_index in range(config.predictions_per_update):
            absolute_index = update_index * config.predictions_per_update + prediction_index
            out = model.predict(make_prediction_input(absolute_index, update_index + 1), route)
            if route == "dist":
                accumulator += float(np.sum(out[0])) + float(np.sum(out[1]))
            else:
                accumulator += float(np.sum(out))
    if accumulator == float("inf"):
        raise RuntimeError("Unreachable accumulator guard")
    return model


def prepare_workload(config: BenchmarkConfig):
    if config.mode == "inference":
        setup_start = time.perf_counter()
        seeded = seed_inference_model(config)
        setup_elapsed = time.perf_counter() - setup_start

        def timed_fn(ctx: Optional[MicroModel]) -> MicroModel:
            assert ctx is not None
            return run_inference_predictions(ctx, config)

        return seeded, timed_fn, setup_elapsed, config.updates

    if config.mode == "commit":
        def timed_fn(_: Optional[MicroModel]) -> MicroModel:
            return run_commit_workload(config)

        return None, timed_fn, 0.0, 0

    if config.mode == "lifecycle":
        def timed_fn(_: Optional[MicroModel]) -> MicroModel:
            return run_lifecycle_workload(config)

        return None, timed_fn, 0.0, 0

    raise ValueError(f"Unsupported mode: {config.mode}")


def run_workload(config: BenchmarkConfig) -> Dict[str, Any]:
    if config.updates < 0 or config.predictions_per_update < 0:
        raise ValueError("counts must be >= 0")
    if config.trials <= 0 or config.warmup < 0:
        raise ValueError("trials must be > 0 and warmup >= 0")
    if config.route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}")

    shared_ctx, timed_fn, setup_elapsed, seeded_updates = prepare_workload(config)

    for _ in range(config.warmup):
        timed_fn(shared_ctx)

    trial_elapsed: List[float] = []
    final_model: Optional[MicroModel] = None
    timed_start = time.perf_counter()
    for _ in range(config.trials):
        trial_start = time.perf_counter()
        final_model = timed_fn(shared_ctx)
        trial_elapsed.append(time.perf_counter() - trial_start)
    timed_elapsed = time.perf_counter() - timed_start

    if final_model is None:
        final_model = shared_ctx if shared_ctx is not None else create_model(config.eager)

    prediction_count_per_trial = total_predictions(config)
    measured_predictions = prediction_count_per_trial * config.trials
    measured_updates = config.updates * config.trials if config.mode in {"commit", "lifecycle"} else 0

    return {
        "benchmark": {
            "mode": config.mode,
            "updates": config.updates,
            "predictions_per_update": config.predictions_per_update,
            "predictions": prediction_count_per_trial,
            "trials": config.trials,
            "warmup": config.warmup,
            "route": config.route,
            "eager_moments": config.eager,
            "seeded_updates": seeded_updates,
            "measured_updates": measured_updates,
            "measured_predictions": measured_predictions,
            "setup_elapsed_seconds": setup_elapsed,
            "timed_elapsed_seconds": timed_elapsed,
            "elapsed_seconds": timed_elapsed,
            "trial_elapsed_seconds": trial_elapsed,
        },
        "model": {
            "version": final_model.version,
            "table_size": len(final_model.table),
            "committed_updates": final_model.commit_count,
            "state_hash": final_model.get_state_hash(),
        },
        "runtime": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "platform_info": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_implementation": platform.python_implementation(),
            },
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval-class probabilistic functor benchmark (moment-bundle router)."
    )
    p.add_argument("--mode", choices=("commit", "inference", "lifecycle"), default="inference")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--predictions-per-update", type=int, default=1000)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--route", choices=ROUTES, default="stddev",
        help="Which morphism inference routes to (dist|sample|mean|var|stddev).",
    )
    p.add_argument(
        "--eager", action="store_true",
        help="Precompute moments at commit (dict-read on request) instead of computing lazily.",
    )
    p.add_argument("--output-json", default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(argv)
    config = BenchmarkConfig(
        mode=a.mode,
        updates=a.updates,
        predictions_per_update=a.predictions_per_update,
        trials=a.trials,
        warmup=a.warmup,
        route=a.route,
        eager=a.eager,
    )
    result = run_workload(config)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as fh:
            fh.write(output)
            fh.write("\n")
    return 0


# ============================================================================
# Tests
# ============================================================================


def test_state_hash_determinism() -> None:
    def run() -> str:
        m = create_model()
        seed_model(m, 50)
        return m.get_state_hash()
    assert run() == run()


def test_no_drift_on_commit() -> None:
    """Committing a new anchor must not change predict() at any other anchor --
    the retrieval model's zero-drift property (unlike the RBF variant)."""
    m = create_model()
    seed_model(m, 20)
    probe = make_input(5)
    before = np.asarray(m.predict(probe, "stddev"), dtype=float).copy()
    commit_one_update(m, 999)  # a different, far anchor
    after = np.asarray(m.predict(probe, "stddev"), dtype=float)
    assert np.allclose(before, after)


def test_stddev_composes_over_var() -> None:
    """stddev morphism must equal sqrt(var morphism) exactly -- coherence by
    construction, not separate fitting."""
    m = create_model()
    seed_model(m, 10)
    x = make_input(3)
    var = np.asarray(m.predict(x, "var"), dtype=float)
    sd = np.asarray(m.predict(x, "stddev"), dtype=float)
    assert np.allclose(sd, np.sqrt(var))


def test_eager_matches_lazy() -> None:
    """Eager and lazy moment strategies must return identical values."""
    ml = create_model(eager=False)
    me = create_model(eager=True)
    seed_model(ml, 30)
    seed_model(me, 30)
    for idx in range(30):
        x = make_input(idx)
        for r in ROUTES:
            a = ml.predict(x, r)
            b = me.predict(x, r)
            if r == "dist":
                assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
            else:
                assert np.allclose(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
    raise SystemExit(main())
