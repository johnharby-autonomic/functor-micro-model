"""
Probabilistic Functor MicroModel - Bounded-Window Soft-Gate Streaming Learning

This is the probabilistic counterpart to core.functor_model.MicroModel. It trades
the deterministic model's exact-lookup / zero-drift property for RBF
interpolation with bounded drift:

  - The deterministic model materializes hard gates into an exact-lookup table:
    predict() either hits the table exactly or falls through to the untouched
    base function, so committing a patch never perturbs any other input.
  - This model instead evaluates, every predict() call:

        f'(x) = f(x) + Sum_i [ Pi_i(x) * Delta_f_i(x) ]

    over a flat list of active patches, where Pi_i(x) is a soft RBF gate with
    *infinite support* (it is exp(-||x - a_i||^2 / (2*bw^2)), so it is never
    exactly zero away from its anchor a_i). That means committing a new patch
    can measurably shift predictions at *other* already-committed anchors --
    "bounded drift" rather than the deterministic model's "zero drift". See
    `get_drift_stats()`.

Reference semantics ported from BioGuard/MicroModel.swift (Swift):
  - Flat windowed summation in predict(), not closure composition. There is no
    f_n(f_{n-1}(...f_0...)) chain, so predict() is O(W) and commit() is O(1)
    extra work, independent of the total number of updates ever streamed in
    (N) -- only the configured window size W matters.
  - A bounded sliding window of at most `config.window` patches with FIFO
    eviction (oldest patch dropped) instead of raising at capacity.
  - Validation includes a cumulative-influence cap (mean |f'(x) - f(x)| over
    samples, simulating the candidate patch added to the window) in addition
    to the per-patch max-delta bound.
  - predict() clamps to [0, 1] (matching Swift's min(max(score,0),1)), so the
    output is always a valid probability even when stacked patch contributions
    would otherwise push it out of range.
  - A reproducible SHA-256 state hash built from a canonical byte encoding
    (struct.pack IEEE-754 big-endian), never Python's salted hash() -- exactly
    the trap Swift's own comment calls out about its seeded hashValue.

Measurement notes (added during audit):
  - Drift measurement is OFF the timed commit path by default for benchmarking.
    Building prior-anchor lists and walking them is instrumentation, not model
    work, and would inflate commit/lifecycle timing the same way audit logging
    and snapshot tables did in the deterministic model. Enable it via
    config.measure_drift_on_commit (or --measure-drift) for a separate
    characterization pass whose *timing* you ignore.
  - The drift metric reports RBF *spillover* (the new patch's tail at prior
    anchors still in the window) separately from *eviction loss* (magnitude
    removed when the oldest patch ages out). The earlier version conflated the
    two by including the about-to-be-evicted anchor in the drift set, where the
    evicted patch's full-magnitude contribution at its own anchor swamped the
    tail-spillover the metric is meant to capture.

Author: John Harby
Status: Patent pending (November 2025)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, TypeAlias

import numpy as np

# Types
Vector: TypeAlias = np.ndarray
Function: TypeAlias = Callable[[Vector], Vector]


@dataclass
class TrainingEvent:
    """Structured training event with metadata."""

    x: Vector
    y: Vector
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class DeltaPatch:
    """
    Soft (RBF) or hard (tolerance) delta patch contributing Pi(x)*Delta_f to the
    flat windowed sum f'(x) = f(x) + Sum_i Pi_i(x)*Delta_f_i(x).

    Unlike the deterministic model's DeltaPatch, this is never composed into f
    via closure-wrapping (`delta.apply(old_f)`). predict() evaluates every
    active patch's gate/delta directly against its stored anchor each time, so
    cost scales with the window size W, never with how many patches have ever
    been committed.
    """

    anchor: Vector
    delta_value: Vector
    gate_type: str = "soft"  # "soft" (RBF, infinite support) or "hard" (tolerance indicator)
    gate_param: float = 0.5  # bandwidth for "soft"; tolerance radius for "hard"
    version: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def gate(self, x: Vector) -> float:
        """Pi(x). The soft RBF gate has infinite support (never exactly zero
        away from the anchor); the hard gate is a tolerance-radius indicator."""
        distance = float(np.linalg.norm(np.asarray(x, dtype=float) - self.anchor))
        if self.gate_type == "hard":
            return 1.0 if distance <= self.gate_param else 0.0
        bw = self.gate_param
        return float(np.exp(-(distance**2) / (2.0 * bw * bw)))

    def contribution(self, x: Vector) -> Vector:
        """Pi(x)*Delta_f(x); Delta_f is the constant committed target delta at this anchor."""
        return self.gate(x) * self.delta_value

    def validate_gate_bounds(self, x: Vector) -> bool:
        """Verify gate returns a value in [0, 1]."""
        g = self.gate(x)
        return 0.0 <= g <= 1.0


@dataclass
class Invariants:
    """Container for invariants used to validate proposed patches."""

    max_delta: float  # per-component epsilon bound on a single patch's delta
    scope_fn: Callable[[Vector], bool]  # returns True if x is in scope S
    # Cumulative-influence cap: mean |f'(x) - f(x)| over validation samples,
    # simulating the candidate patch added to the active window. Because soft
    # gates have infinite support, several nearby committed patches can stack
    # at a given x even though no single patch exceeds max_delta there. This
    # mirrors Swift's invariants.isCumulativeInfluenceValid check, which the
    # deterministic Python model has no equivalent for (it only checks the
    # per-delta bound).
    max_cumulative_influence: float = 2.0
    check_monotonicity: bool = False
    check_calibration: bool = False
    custom_validators: List[Callable] = field(default_factory=list)

    def is_delta_valid(self, delta_value: Vector) -> bool:
        """Per-patch magnitude bound: worst-component |delta| <= max_delta."""
        return bool(np.max(np.abs(delta_value)) <= self.max_delta)

    def is_cumulative_influence_valid(self, influence: float) -> bool:
        """Cumulative spillover bound across the (simulated) active window."""
        return influence <= self.max_cumulative_influence


@dataclass
class MicroModelConfig:
    """Configuration for the probabilistic MicroModel."""

    enable_skip_training: bool = True
    enable_delay_training: bool = True
    enable_audit_logging: bool = True
    enable_profiling: bool = False
    learning_enabled: bool = True  # mirrors Swift's config.learningEnabled gate on propose()

    # Bounded sliding window: at most `window` active patches. On overflow the
    # oldest patch is evicted (FIFO), matching Swift's maxPatches/removeFirst
    # behavior -- not the deterministic model's "raise at capacity". This is
    # what keeps predict()/commit() at O(W) regardless of how many updates
    # have streamed through the model.
    window: int = 32

    # Default RBF bandwidth used by propose()/propose_soft() when the caller
    # doesn't supply one explicitly.
    bandwidth: float = 0.5

    # Drift measurement is instrumentation, not model work. When False
    # (the benchmark default for timing runs), commit() does no prior-anchor
    # bookkeeping at all, keeping the timed path purely propose/validate/commit.
    # Enable it for a separate drift-characterization pass whose timing you
    # ignore. See commit() and get_drift_stats().
    measure_drift_on_commit: bool = True

    # Unlike the deterministic model, the bounded window already keeps
    # commit() at O(1) extra memory/work per call regardless of this flag --
    # there is no growing per-version snapshot table to avoid here. This flag
    # is now a pure feature gate on rollback(): rollback() raises when False.
    # Kept for interface parity with the deterministic model's config surface.
    retain_version_snapshots: bool = True


@dataclass
class MicroModel:
    """
    Probabilistic functor micromodel: bounded-window soft-gate delta patches.

    predict(x) = clamp(f(x) + Sum_i Pi_i(x) * Delta_f_i(x), 0, 1), summed over a
    flat list of at most `config.window` active patches (oldest evicted on
    overflow). There is no closure composition, so predict() is O(W) and
    commit() is O(1) extra work, independent of total commit count N.

    Example:
        >>> inv = Invariants(max_delta=1.0, scope_fn=lambda x: True)
        >>> model = MicroModel(f=base_model_sigmoid, invariants=inv)
        >>> event = TrainingEvent(x=np.array([2.0]), y=np.array([0.9, 0.1]))
        >>> patch = model.propose(event, gate_type="soft")
        >>> if patch is not None and model.validate(patch, [event.x]):
        ...     model.commit(patch, event=event, samples=[event.x])
    """

    f: Function  # base function, never mutated by commit()
    invariants: Invariants
    config: MicroModelConfig = field(default_factory=MicroModelConfig)

    # Bounded FIFO window of active patches, oldest first. This *is* the
    # model's learned state -- there is no separate materialized table and no
    # closure chain.
    patches: Deque[DeltaPatch] = field(default_factory=deque)
    version: int = 0
    proposal_count: int = 0
    commit_count: int = 0

    skipped_events: List[TrainingEvent] = field(default_factory=list)
    delayed_events: List[TrainingEvent] = field(default_factory=list)

    # Audit trail, bounded to the last 500 entries (mirrors Swift's
    # operationLog cap) regardless of how long enable_audit_logging has been on.
    operation_log: List[Dict[str, Any]] = field(default_factory=list)

    # Most recent commit's drift measurement; see get_drift_stats() for the
    # running aggregate. O(1) extra state, independent of N.
    last_drift: Optional[Dict[str, float]] = field(default=None, repr=False)
    _drift_abs_sum: float = field(default=0.0, repr=False)
    _drift_abs_count: int = field(default=0, repr=False)
    _drift_abs_max: float = field(default=0.0, repr=False)
    # Eviction loss is tracked separately from spillover (see _measure_drift).
    _eviction_loss_sum: float = field(default=0.0, repr=False)
    _eviction_loss_max: float = field(default=0.0, repr=False)
    _eviction_count: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(score: Vector) -> Vector:
        """Clamp to [0, 1], matching Swift's min(max(score,0),1) on predict()."""
        return np.clip(score, 0.0, 1.0)

    def predict(self, x: Vector) -> Vector:
        """
        f'(x) = clamp(f(x) + Sum_i Pi_i(x)*Delta_f_i(x), 0, 1), over the window.

        O(W): evaluates the base function once, then walks the bounded patch
        window once. Never reconstructs f by composing closures, so cost does
        not grow with how many patches have ever been committed -- only with
        the current window size W.
        """
        score = np.array(self.f(x), dtype=float, copy=True)
        for patch in self.patches:
            score = score + patch.contribution(x)
        return self._clamp(score)

    def _predict_with(self, x: Vector, extra_patches: Iterable[DeltaPatch]) -> Vector:
        """predict(x) as if `extra_patches` were also active, without mutating state.

        Used by validate() to simulate cumulative influence before committing.
        Clamped consistently with predict().
        """
        score = np.array(self.f(x), dtype=float, copy=True)
        for patch in self.patches:
            score = score + patch.contribution(x)
        for patch in extra_patches:
            score = score + patch.contribution(x)
        return self._clamp(score)

    def cumulative_influence(
        self, samples: Iterable[Vector], extra_patches: Iterable[DeltaPatch] = ()
    ) -> float:
        """
        Mean over `samples` of the worst-component |f'(x) - f(x)|, with
        `extra_patches` simulated as additionally active. Mirrors Swift's
        cumulativeInfluence(samples:withPatches:).
        """
        sample_list = list(samples)
        if not sample_list:
            return 0.0
        total = 0.0
        for x in sample_list:
            base = self._clamp(np.array(self.f(x), dtype=float))
            with_patches = self._predict_with(x, extra_patches)
            total += float(np.max(np.abs(with_patches - base)))
        return total / len(sample_list)

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose(
        self,
        event: TrainingEvent,
        gate_type: str = "soft",
        bandwidth: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> Optional[DeltaPatch]:
        """
        Propose a delta patch nudging predict(event.x) toward event.y.

        Returns None if learning is disabled or the clamped delta is
        negligible (mirrors Swift's propose() returning nil rather than an
        empty/no-op patch).
        """
        if not self.config.learning_enabled:
            self._log_operation("propose", {"skipped": "learning_disabled"})
            return None

        x0 = event.x
        y0 = event.y
        current = self.predict(x0)
        raw_delta = np.clip(y0 - current, -self.invariants.max_delta, self.invariants.max_delta)

        if float(np.max(np.abs(raw_delta))) <= 1e-6:
            self._log_operation("propose", {"skipped": "delta_too_small"})
            return None

        if gate_type == "soft":
            gate_param = bandwidth if bandwidth is not None else self.config.bandwidth
        elif gate_type == "hard":
            gate_param = tolerance if tolerance is not None else 1e-9
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        patch = DeltaPatch(
            anchor=np.array(x0, copy=True),
            delta_value=np.array(raw_delta, copy=True),
            gate_type=gate_type,
            gate_param=float(gate_param),
            version=self.version + 1,
        )

        self.proposal_count += 1
        self._log_operation(
            "propose", {"version": patch.version, "gate_type": gate_type, "gate_param": gate_param}
        )
        return patch

    def propose_soft(self, event: TrainingEvent, bandwidth: Optional[float] = None) -> Optional[DeltaPatch]:
        """Convenience wrapper: propose() with gate_type="soft"."""
        return self.propose(event, gate_type="soft", bandwidth=bandwidth)

    def propose_bias_patch(
        self, anchor: Vector, current: Vector, target: Vector, bandwidth: float = 1.5
    ) -> DeltaPatch:
        """
        Wide-bandwidth global bias correction, mirroring Swift's
        proposeBiasPatch(currentScore:targetScore:).

        Uses a much wider bandwidth than typical local patches so the soft RBF
        gate stays close to 1.0 across most of the input space, acting as a
        near-global correction rather than a localized one -- the equivalent
        of the deterministic model's bias/global-correction path.
        """
        raw_delta = np.clip(
            np.asarray(target, dtype=float) - np.asarray(current, dtype=float),
            -self.invariants.max_delta,
            self.invariants.max_delta,
        )
        patch = DeltaPatch(
            anchor=np.array(anchor, copy=True),
            delta_value=np.array(raw_delta, copy=True),
            gate_type="soft",
            gate_param=float(bandwidth),
            version=self.version + 1,
        )
        self._log_operation(
            "propose", {"version": patch.version, "gate_type": "bias", "gate_param": bandwidth}
        )
        return patch

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, patch: DeltaPatch, samples: Optional[List[Vector]] = None) -> bool:
        """
        Validate a proposed patch against all invariants:
          1. delta is finite
          2. per-patch magnitude bound (|delta| <= max_delta)
          3. cumulative-influence cap, simulating the patch added to the
             active window, sampled at (by default) off-anchor points -- the
             RBF tail reaches points other than the anchor, so anchor-only
             sampling under-measures spillover from this and other patches.
          4. scope_fn holds for every sample
        """
        if not np.all(np.isfinite(patch.delta_value)):
            self._log_operation(
                "validate_fail", {"reason": "delta_non_finite", "patch_version": patch.version}
            )
            return False

        if not self.invariants.is_delta_valid(patch.delta_value):
            self._log_operation(
                "validate_fail",
                {
                    "reason": "delta_exceeds_max",
                    "patch_version": patch.version,
                    "max_abs_delta": float(np.max(np.abs(patch.delta_value))),
                    "max_delta": self.invariants.max_delta,
                },
            )
            return False

        sample_list = list(samples) if samples else [patch.anchor]
        influence = self.cumulative_influence(sample_list, extra_patches=[patch])
        if not self.invariants.is_cumulative_influence_valid(influence):
            self._log_operation(
                "validate_fail",
                {
                    "reason": "cumulative_influence_cap",
                    "patch_version": patch.version,
                    "influence": influence,
                    "cap": self.invariants.max_cumulative_influence,
                },
            )
            return False

        out_of_scope = [x for x in sample_list if not self.invariants.scope_fn(x)]
        if out_of_scope:
            self._log_operation(
                "validate_fail",
                {
                    "reason": "out_of_scope",
                    "patch_version": patch.version,
                    "count": len(out_of_scope),
                },
            )
            return False

        self._log_operation(
            "validate_ok", {"patch_version": patch.version, "samples": len(sample_list)}
        )
        return True

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _measure_drift(
        self,
        surviving_anchors: List[Vector],
        new_patch: DeltaPatch,
        evicted_patch: Optional[DeltaPatch],
    ) -> None:
        """
        Update the running drift aggregate (see get_drift_stats()) for this
        commit, separating two distinct effects:

          * RBF spillover -- the new patch's tail evaluated at prior anchors
            that REMAIN in the window. This is the quantity that is identically
            zero in the deterministic hard-gate model and is the characteristic
            signature of the probabilistic one: committing a patch perturbs
            predictions at other live anchors. Reported as mean/max_abs_drift.

          * Eviction loss -- the magnitude removed at the evicted patch's own
            anchor when the oldest patch ages out (gate == 1 there, so this is
            ~the full delta). This is a window-management effect, not RBF
            spillover, and is reported separately so it does not swamp the
            spillover figure (the bug in the prior version was including the
            evicted anchor in the spillover set).

        Exact and O(W): every other patch is unchanged by this commit, so the
        new patch's contribution at each surviving anchor is computed directly
        with no re-prediction.
        """
        spillover_diffs = [
            float(np.max(np.abs(new_patch.contribution(a)))) for a in surviving_anchors
        ]
        if spillover_diffs:
            mean_spill = float(np.mean(spillover_diffs))
            max_spill = float(np.max(spillover_diffs))
        else:
            mean_spill = 0.0
            max_spill = 0.0

        eviction_loss = 0.0
        if evicted_patch is not None:
            eviction_loss = float(
                np.max(np.abs(evicted_patch.contribution(evicted_patch.anchor)))
            )
            self._eviction_loss_sum += eviction_loss
            self._eviction_loss_max = max(self._eviction_loss_max, eviction_loss)
            self._eviction_count += 1

        self._drift_abs_sum += mean_spill
        self._drift_abs_count += 1
        self._drift_abs_max = max(self._drift_abs_max, max_spill)

        self.last_drift = {
            "mean_abs_drift": mean_spill,
            "max_abs_drift": max_spill,
            "eviction_loss": eviction_loss,
            "num_prior_anchors": len(surviving_anchors),
        }

    def commit(
        self,
        patch: DeltaPatch,
        event: Optional[TrainingEvent] = None,
        samples: Optional[List[Vector]] = None,
        profile: bool = False,
        measure_drift: bool = True,
    ) -> None:
        """
        Commit a validated patch: FIFO-evict at window capacity, append, bump
        version. O(1) extra work per call -- independent of total commit
        count N, bounded only by the configured window size W.

        Drift measurement (bounded-drift signature of this commit) runs only
        when BOTH `measure_drift` is True AND config.measure_drift_on_commit is
        True. It is OFF by default for benchmark models so the timed commit
        path is pure model work; enable it for a separate characterization pass
        whose timing you ignore. When on it is O(W): the new patch's
        contribution at each surviving prior anchor, each O(1).
        """
        if profile or self.config.enable_profiling:
            start_time = datetime.now().timestamp()

        if not np.all(np.isfinite(patch.delta_value)):
            raise ValueError("Cannot commit a patch with a non-finite delta")
        if not self.invariants.is_delta_valid(patch.delta_value):
            raise ValueError("Cannot commit a patch whose delta exceeds max_delta")

        do_drift = measure_drift and self.config.measure_drift_on_commit

        # Capture the patches present BEFORE this commit. The one about to be
        # evicted (leftmost) is excluded from the spillover set: at its own
        # anchor the evicted patch's contribution is full-magnitude and would
        # dominate the tail-spillover the drift metric is meant to capture.
        prior_patches = list(self.patches) if do_drift else []

        evicted: Optional[DeltaPatch] = None
        if len(self.patches) >= self.config.window:
            evicted = self.patches.popleft()

        self.patches.append(patch)
        self.version += 1
        self.commit_count += 1

        if do_drift:
            surviving = [p.anchor for p in prior_patches if p is not evicted]
            self._measure_drift(surviving, patch, evicted)

        # Guard the payload construction itself: when audit logging is
        # disabled the event dict (event.to_dict()) must never be built, since
        # Python evaluates call arguments eagerly before _log_operation's own
        # early-return would discard them. Building it unconditionally would
        # contaminate per-commit timing/energy measurements.
        if self.config.enable_audit_logging:
            self._log_operation(
                "commit",
                {
                    "version": self.version,
                    "patch_version": patch.version,
                    "evicted_version": evicted.version if evicted is not None else None,
                    "event": event.to_dict() if event else None,
                    "validation_sample_count": len(samples) if samples else 0,
                },
            )

        if profile or self.config.enable_profiling:
            elapsed = datetime.now().timestamp() - start_time
            self._log_operation("commit_profile", {"elapsed_seconds": elapsed})

    def commit_bias_patch(self, patch: DeltaPatch) -> None:
        """Commit a patch produced by propose_bias_patch() (no drift measurement;
        the bias patch is intentionally near-global rather than anchor-local)."""
        self.commit(patch, measure_drift=False)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, to_version: Optional[int] = None) -> None:
        """
        Roll back by dropping patches with patch.version > to_version, mirroring
        Swift's rollback(toVersion:): a simple filter over the *current* window,
        not a snapshot restore. There is no separate snapshot history to
        maintain, because the bounded window already retains every
        not-yet-evicted patch's version -- rollback can only reach versions
        still present in the window (older, evicted patches are gone for good,
        exactly as in Swift, where evicted patches are also unrecoverable).

        Raises if config.retain_version_snapshots is False: this flag is a
        pure feature gate on rollback() for this model (see MicroModelConfig),
        since the bounded window already keeps commit() at O(1) regardless.
        """
        if not self.config.retain_version_snapshots:
            raise RuntimeError(
                "rollback() requires retain_version_snapshots=True; this model "
                "instance was configured with rollback support disabled."
            )

        if to_version is None:
            to_version = max(0, self.version - 1)
        if to_version < 0 or to_version > self.version:
            raise ValueError(f"Invalid target version: {to_version} (current: {self.version})")

        before = len(self.patches)
        self.patches = deque(p for p in self.patches if p.version <= to_version)
        removed = before - len(self.patches)

        if removed > 0:
            self.version = to_version
            self._log_operation(
                "rollback", {"to_version": to_version, "patches_removed": removed}
            )

    # ------------------------------------------------------------------
    # Skip / delay
    # ------------------------------------------------------------------

    def skip(self, event: TrainingEvent, reason: str = "") -> None:
        """Skip a training event (log but don't apply)."""
        if not self.config.enable_skip_training:
            raise ValueError("Skip training is disabled in config")
        self.skipped_events.append(event)
        self._log_operation("skip", {"event": event.to_dict(), "reason": reason})

    def delay(self, event: TrainingEvent, reason: str = "") -> None:
        """Queue a training event for later application."""
        if not self.config.enable_delay_training:
            raise ValueError("Delay training is disabled in config")
        self.delayed_events.append(event)
        self._log_operation("delay", {"event": event.to_dict(), "reason": reason})

    def apply_delayed(
        self, event: TrainingEvent, samples: List[Vector], gate_type: str = "soft"
    ) -> bool:
        """Apply a previously delayed training event."""
        if event not in self.delayed_events:
            raise ValueError("Event not in delayed queue")

        patch = self.propose(event, gate_type=gate_type)
        if patch is not None and self.validate(patch, samples):
            self.commit(patch, event=event, samples=samples)
            self.delayed_events.remove(event)
            self._log_operation("apply_delayed", {"event": event.to_dict(), "success": True})
            return True

        self._log_operation(
            "apply_delayed",
            {"event": event.to_dict(), "success": False, "reason": "validation_failed"},
        )
        return False

    # ------------------------------------------------------------------
    # State hash (determinism)
    # ------------------------------------------------------------------

    def get_state_hash(self) -> str:
        """
        Deterministic SHA-256 digest of model state, covering the ordered
        window (each patch's anchor, delta value, gate type, and gate
        parameter) plus version/window size.

        Built from a canonical byte encoding -- IEEE-754 big-endian bytes for
        floats via struct.pack, big-endian bytes for ints -- never from
        Python's built-in hash() (salted per-process via PYTHONHASHSEED and
        therefore not reproducible across runs) and never from json.dumps of
        floats (whose textual repr is not a canonical encoding). This mirrors
        Swift's own SHA256()-over-absorbed-bytes approach exactly, which its
        source comments call out for avoiding the equivalent trap with
        Swift's seeded hashValue.
        """
        hasher = hashlib.sha256()

        def absorb_str(s: str) -> None:
            hasher.update(s.encode("utf-8"))

        def absorb_int(i: int) -> None:
            hasher.update(struct.pack(">q", int(i)))

        def absorb_float(v: float) -> None:
            hasher.update(struct.pack(">d", float(v)))

        def absorb_vector(v: Vector) -> None:
            arr = np.asarray(v, dtype=float).ravel()
            absorb_int(arr.size)
            for component in arr:
                absorb_float(float(component))

        absorb_str("v")
        absorb_int(self.version)
        absorb_str("w")
        absorb_int(self.config.window)
        absorb_str("n")
        absorb_int(len(self.patches))
        for patch in self.patches:  # ordered window, oldest -> newest
            absorb_str("|p")
            absorb_int(patch.version)
            absorb_str("|d")
            absorb_vector(patch.delta_value)
            absorb_str("|g")
            absorb_str(patch.gate_type)
            absorb_float(patch.gate_param)
            absorb_str("|a")
            absorb_vector(patch.anchor)

        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Drift (the probabilistic model's characteristic metric)
    # ------------------------------------------------------------------

    def get_drift_stats(self) -> Dict[str, Any]:
        """
        Running aggregate of bounded drift across all measured commits: O(1)
        extra state (running sums/counts/max), independent of N.

        `mean_abs_drift` / `max_abs_drift` are the RBF *spillover* of each new
        commit at prior anchors still in the window -- the quantity that is
        identically zero in the deterministic hard-gate model. `eviction_loss`
        figures report, separately, the magnitude removed when the oldest patch
        ages out of the bounded window (a window-management effect, not
        spillover). `num_commits_measured` is 0 when drift measurement was off
        (the benchmark timing default) -- enable config.measure_drift_on_commit
        for a characterization pass.
        """
        if self._drift_abs_count == 0:
            return {
                "mean_abs_drift": 0.0,
                "max_abs_drift": 0.0,
                "mean_eviction_loss": 0.0,
                "max_eviction_loss": 0.0,
                "num_commits_measured": 0,
                "num_evictions_measured": 0,
            }
        mean_evict = (
            self._eviction_loss_sum / self._eviction_count if self._eviction_count else 0.0
        )
        return {
            "mean_abs_drift": self._drift_abs_sum / self._drift_abs_count,
            "max_abs_drift": self._drift_abs_max,
            "mean_eviction_loss": mean_evict,
            "max_eviction_loss": self._eviction_loss_max,
            "num_commits_measured": self._drift_abs_count,
            "num_evictions_measured": self._eviction_count,
        }

    # ------------------------------------------------------------------
    # Validation sample generation
    # ------------------------------------------------------------------

    def generate_validation_samples(
        self, x0: Vector, n_samples: int = 5, radius: Optional[float] = None, seed: Optional[int] = None
    ) -> List[Vector]:
        """
        Generate validation samples around `x0`, including off-anchor points.

        The anchor is the gate's peak (Pi(x0) == 1 for soft gates); sampling
        only the anchor under-measures how far the RBF tail's influence
        reaches. Defaults `radius` to half the configured bandwidth so the
        samples land where the gate's contribution is still meaningful but
        not at its peak.
        """
        if radius is None:
            radius = max(self.config.bandwidth * 0.5, 1e-3)
        rng = np.random.default_rng(seed)
        samples = [x0]
        for _ in range(max(n_samples - 1, 0)):
            samples.append(x0 + rng.normal(scale=radius, size=np.asarray(x0).shape))
        return samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Internal logging of operations, bounded to the last 500 entries
        (mirrors Swift's operationLog cap) regardless of how long audit
        logging has been enabled."""
        if not self.config.enable_audit_logging:
            return
        self.operation_log.append(
            {
                "timestamp": datetime.now().timestamp(),
                "operation": operation,
                "details": details,
            }
        )
        if len(self.operation_log) > 500:
            del self.operation_log[: len(self.operation_log) - 500]

    def get_stats(self) -> Dict[str, Any]:
        """Return model statistics."""
        return {
            "version": self.version,
            "num_patches": len(self.patches),
            "window": self.config.window,
            "proposal_count": self.proposal_count,
            "commit_count": self.commit_count,
            "num_skipped": len(self.skipped_events),
            "num_delayed": len(self.delayed_events),
            "state_hash": self.get_state_hash(),
            "drift": self.get_drift_stats(),
        }


# ============================================================================
# Example Base Functions
# ============================================================================


def base_model_sigmoid(x: Vector) -> Vector:
    """Example: independent probabilities (sigmoid outputs)."""
    logits = np.array([np.sin(x[0]), np.cos(x[0])])
    return 1 / (1 + np.exp(-logits))


def base_model_linear(x: Vector) -> Vector:
    """Example: simple linear model."""
    weights = np.array([1.0, 0.5])
    return weights * x[0]


def base_model_lookup(lookup_table: Dict[str, Vector]) -> Function:
    """Example: lookup table base function."""

    def lookup(x: Vector) -> Vector:
        key = str(x.tolist())
        return lookup_table.get(key, np.zeros_like(x))

    return lookup


# ============================================================================
# Benchmark harness (commit / inference / lifecycle)
#
# JSON output shape is kept compatible with bench_micro.py (same top-level
# "benchmark"/"model"/"runtime" keys) with additive fields: --window,
# --bandwidth, and a "drift" block under "model" reporting the bounded-drift
# signature that distinguishes this model from the deterministic one.
#
# Timing runs default to drift measurement OFF (instrumentation off the timed
# path). Pass --measure-drift for a separate characterization pass whose timing
# you ignore but whose "drift" block is meaningful.
# ============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    updates: int
    predictions_per_update: int
    trials: int
    warmup: int
    window: int
    bandwidth: float
    max_cumulative_influence: float
    measure_drift: bool = False


def scope_all(_: Vector) -> bool:
    """Benchmark invariant scope: all benchmark anchors are in scope."""
    return True


def create_prob_model(
    window: int = 32,
    bandwidth: float = 0.5,
    max_cumulative_influence: float = 2.0,
    measure_drift: bool = False,
) -> MicroModel:
    """Create a fresh probabilistic MicroModel for benchmarking.

    Audit logging and rollback-snapshot support are disabled: neither affects
    the measured propose/validate/commit/predict work or the state hash, but
    both add bookkeeping that would contaminate timing/energy measurement.

    Drift measurement is OFF by default for the same reason -- it is
    instrumentation, not model work. Pass measure_drift=True only for a
    characterization pass whose timing you do not report.
    """
    config = MicroModelConfig(
        enable_audit_logging=False,
        enable_profiling=False,
        retain_version_snapshots=False,
        measure_drift_on_commit=measure_drift,
        window=window,
        bandwidth=bandwidth,
    )
    invariants = Invariants(
        max_delta=1.0,
        scope_fn=scope_all,
        max_cumulative_influence=max_cumulative_influence,
    )
    return MicroModel(f=base_model_sigmoid, invariants=invariants, config=config)


def make_input(index: int) -> Vector:
    """Deterministic one-dimensional benchmark input."""
    return np.array([float(index) + 1.0], dtype=float)


def make_target(index: int) -> Vector:
    """Deterministic target with bounded delta against base_model_sigmoid."""
    phase = float(index)
    return np.array(
        [0.5 + 0.35 * np.sin(phase * 0.37), 0.5 + 0.35 * np.cos(phase * 0.41)],
        dtype=float,
    )


def make_prediction_input(index: int, committed_updates: int) -> Vector:
    """Deterministic prediction input cycling across committed anchors."""
    if committed_updates <= 0:
        return np.array([1.0 + 0.001 * float(index)], dtype=float)
    anchor = (index % committed_updates) + 1
    offset = 0.0 if index % 2 == 0 else 0.125
    return np.array([float(anchor) + offset], dtype=float)


def commit_one_update(model: MicroModel, index: int) -> bool:
    """
    Run one propose/validate/commit update. Returns True if a patch was
    committed, False if it was proposed-away or failed validation.

    Validation samples include off-anchor points seeded deterministically by
    `index` (never Python's hash()), so the cumulative-influence check
    actually measures RBF tail spillover rather than just the anchor (the
    gate's peak).

    Validation failure is recorded as a skip rather than raised: a (window,
    bandwidth) sweep point that trips the cumulative-influence cap should be
    measured, not crash the whole run.
    """
    event = TrainingEvent(
        x=make_input(index),
        y=make_target(index),
        metadata={"benchmark_index": index},
    )
    patch = model.propose(event, gate_type="soft")
    if patch is None:
        return False
    samples = model.generate_validation_samples(event.x, n_samples=5, seed=index)
    if not model.validate(patch, samples):
        model.skip(event, reason="validation_failed")
        return False
    model.commit(patch, event=event, samples=samples, profile=False)
    return True


def seed_model(model: MicroModel, updates: int) -> None:
    """Preload a model with committed benchmark updates."""
    for index in range(updates):
        commit_one_update(model, index)


def run_commit_workload(config: BenchmarkConfig) -> MicroModel:
    """Measure commit/propose/validate/commit workload."""
    model = create_prob_model(
        config.window, config.bandwidth, config.max_cumulative_influence, config.measure_drift
    )
    for index in range(config.updates):
        commit_one_update(model, index)
    return model


def seed_inference_model(config: BenchmarkConfig) -> MicroModel:
    """Untimed setup for inference mode: create and seed before the timed region."""
    model = create_prob_model(
        config.window, config.bandwidth, config.max_cumulative_influence, config.measure_drift
    )
    seed_model(model, config.updates)
    print(f"Updates complete ({config.updates} committed); starting inference...")
    return model


def run_inference_predictions(model: MicroModel, config: BenchmarkConfig) -> MicroModel:
    """Timed region for inference mode: model.predict() calls only."""
    prediction_count = total_predictions(config)
    accumulator = 0.0
    for index in range(prediction_count):
        y = model.predict(make_prediction_input(index, config.updates))
        accumulator += float(np.sum(y))
    if accumulator == float("inf"):
        raise RuntimeError("Unreachable accumulator guard")
    return model


def run_lifecycle_workload(config: BenchmarkConfig) -> MicroModel:
    """Measure N updates with M model.predict() calls per update."""
    model = create_prob_model(
        config.window, config.bandwidth, config.max_cumulative_influence, config.measure_drift
    )
    accumulator = 0.0
    for update_index in range(config.updates):
        commit_one_update(model, update_index)
        for prediction_index in range(config.predictions_per_update):
            absolute_index = update_index * config.predictions_per_update + prediction_index
            y = model.predict(make_prediction_input(absolute_index, update_index + 1))
            accumulator += float(np.sum(y))
    if accumulator == float("inf"):
        raise RuntimeError("Unreachable accumulator guard")
    return model


def total_predictions(config: BenchmarkConfig) -> int:
    if config.mode == "commit":
        return 0
    return config.updates * config.predictions_per_update


def prepare_workload(config: BenchmarkConfig):
    """Build the (untimed) setup context and the timed-region callable for a mode."""
    if config.mode == "inference":
        setup_start = time.perf_counter()
        seeded_model = seed_inference_model(config)
        setup_elapsed = time.perf_counter() - setup_start

        def timed_fn(ctx: Optional[MicroModel]) -> MicroModel:
            assert ctx is not None
            return run_inference_predictions(ctx, config)

        return seeded_model, timed_fn, setup_elapsed, config.updates

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
    """Run untimed setup, warmup, and measured benchmark trials."""
    if config.updates < 0:
        raise ValueError("updates must be >= 0")
    if config.predictions_per_update < 0:
        raise ValueError("predictions_per_update must be >= 0")
    if config.trials <= 0:
        raise ValueError("trials must be > 0")
    if config.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if config.window <= 0:
        raise ValueError("window must be > 0")

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
        final_model = shared_ctx if shared_ctx is not None else create_prob_model(
            config.window, config.bandwidth, config.max_cumulative_influence, config.measure_drift
        )

    prediction_count_per_trial = total_predictions(config)
    measured_predictions = prediction_count_per_trial * config.trials
    measured_updates = (
        config.updates * config.trials if config.mode in {"commit", "lifecycle"} else 0
    )

    return {
        "benchmark": {
            "mode": config.mode,
            "updates": config.updates,
            "predictions_per_update": config.predictions_per_update,
            "predictions": prediction_count_per_trial,
            "trials": config.trials,
            "warmup": config.warmup,
            "window": config.window,
            "bandwidth": config.bandwidth,
            "max_cumulative_influence": config.max_cumulative_influence,
            "measure_drift": config.measure_drift,
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
            "patch_count": len(final_model.patches),
            "committed_updates": final_model.commit_count,
            "skipped_updates": len(final_model.skipped_events),
            "state_hash": final_model.get_state_hash(),
            "drift": final_model.get_drift_stats(),
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
    parser = argparse.ArgumentParser(
        description="Standalone benchmark harness for core.prob_functor_model.MicroModel."
    )
    parser.add_argument(
        "--mode", choices=("commit", "inference", "lifecycle"), default="lifecycle"
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--predictions-per-update", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--window", type=int, default=32, help="Bounded sliding-window size W (FIFO eviction)."
    )
    parser.add_argument(
        "--bandwidth", type=float, default=0.5, help="Default RBF bandwidth for soft gates."
    )
    parser.add_argument(
        "--max-cumulative-influence",
        type=float,
        default=2.0,
        help="Validation cap on mean |f'(x)-f(x)| over samples.",
    )
    parser.add_argument(
        "--measure-drift",
        action="store_true",
        help="Enable drift measurement on the commit path. This is "
        "instrumentation; use it for a drift-characterization pass and ignore "
        "the timing from that run.",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = BenchmarkConfig(
        mode=args.mode,
        updates=args.updates,
        predictions_per_update=args.predictions_per_update,
        trials=args.trials,
        warmup=args.warmup,
        window=args.window,
        bandwidth=args.bandwidth,
        max_cumulative_influence=args.max_cumulative_influence,
        measure_drift=args.measure_drift,
    )

    result = run_workload(config)
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")

    return 0


# ============================================================================
# Tests
# ============================================================================


def test_state_hash_determinism() -> None:
    """Two identical commit sequences (fresh models, same inputs) must
    produce identical state hashes -- the core determinism guarantee."""

    def run() -> str:
        model = create_prob_model(window=8, bandwidth=0.5)
        seed_model(model, 20)
        return model.get_state_hash()

    assert run() == run()


def test_window_eviction_is_bounded() -> None:
    """Committing far more updates than the window size must never grow the
    active patch list past the window."""
    model = create_prob_model(window=5, bandwidth=0.5)
    seed_model(model, 50)
    assert len(model.patches) <= 5


def test_predict_cost_independent_of_update_count() -> None:
    """predict() must walk only the bounded window, not full commit history."""
    model = create_prob_model(window=4, bandwidth=0.5)
    seed_model(model, 100)
    assert len(model.patches) == 4


def test_drift_excludes_eviction_loss() -> None:
    """Spillover (new patch's tail at surviving anchors) must be reported
    separately from, and much smaller than, eviction loss on inputs whose
    anchor spacing is large relative to bandwidth."""
    model = create_prob_model(window=4, bandwidth=0.5, measure_drift=True)
    seed_model(model, 50)
    stats = model.get_drift_stats()
    # Anchors are spaced 1.0 apart, bandwidth 0.5 -> nearest-neighbour tail is
    # exp(-1/0.5) ~ 0.135 of delta, while eviction loss is ~the full delta.
    assert stats["max_abs_drift"] < stats["max_eviction_loss"]


if __name__ == "__main__":
    raise SystemExit(main())
