"""Tests for the retrieval-class probabilistic functor model."""

import numpy as np
import pytest

from core.prob_functor_model import (
    ROUTES,
    BenchmarkConfig,
    MicroModel,
    MicroModelConfig,
    MomentBundle,
    base_dist_gaussian,
    commit_one_update,
    create_model,
    make_input,
    make_target_dist,
    run_workload,
    seed_model,
)


def assert_route_outputs_equal(route, actual, expected):
    """Compare either a vector route or the two-vector ``dist`` route."""
    if route == "dist":
        assert len(actual) == len(expected) == 2
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
    else:
        np.testing.assert_allclose(actual, expected)


def test_moment_bundle_routes_are_coherent():
    mean = np.array([0.25, 0.75])
    variance = np.array([0.04, 0.09])
    bundle = MomentBundle(mean, variance)

    dist_mean, dist_variance = bundle.route("dist")
    np.testing.assert_array_equal(dist_mean, mean)
    np.testing.assert_array_equal(dist_variance, variance)
    np.testing.assert_array_equal(bundle.route("sample"), mean)
    np.testing.assert_array_equal(bundle.route("mean"), mean)
    np.testing.assert_array_equal(bundle.route("var"), variance)
    np.testing.assert_allclose(bundle.route("stddev"), np.sqrt(variance))


def test_moment_bundle_rejects_unknown_route():
    bundle = MomentBundle(np.array([0.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="Unknown route"):
        bundle.route("median")


def test_uncommitted_input_uses_base_distribution_for_every_route():
    model = create_model()
    x = np.array([2.5])
    mean, variance = base_dist_gaussian(x)
    expected = {
        "dist": (mean, variance),
        "sample": mean,
        "mean": mean,
        "var": variance,
        "stddev": np.sqrt(variance),
    }

    for route in ROUTES:
        assert_route_outputs_equal(route, model.predict(x, route), expected[route])


@pytest.mark.parametrize("eager", [False, True])
def test_commit_exactly_overrides_distribution_at_anchor(eager):
    model = create_model(eager=eager)
    x = np.array([7.0])
    mean = np.array([0.2, 0.8])
    variance = np.array([0.01, 0.16])

    model.commit(x, mean, variance, event_index=12)

    np.testing.assert_array_equal(model.predict(x, "mean"), mean)
    np.testing.assert_array_equal(model.predict(x, "var"), variance)
    np.testing.assert_allclose(model.predict(x, "stddev"), [0.1, 0.4])
    assert model.version == 1
    assert model.commit_count == 1
    assert len(model.table) == 1


def test_commit_does_not_change_a_different_anchor():
    model = create_model()
    seed_model(model, 20)
    probe = make_input(5)
    before = model.predict(probe, "stddev").copy()

    commit_one_update(model, 999)

    np.testing.assert_array_equal(model.predict(probe, "stddev"), before)


def test_commit_copies_caller_owned_arrays():
    model = create_model()
    x = np.array([3.0])
    mean = np.array([0.1, 0.9])
    variance = np.array([0.04, 0.25])
    model.commit(x, mean, variance)

    x[0] = 99.0
    mean[:] = -1.0
    variance[:] = 100.0

    np.testing.assert_array_equal(model.predict(np.array([3.0]), "mean"), [0.1, 0.9])
    np.testing.assert_array_equal(model.predict(np.array([3.0]), "var"), [0.04, 0.25])


def test_negative_variance_is_rejected_without_mutating_state():
    model = create_model()
    original_hash = model.get_state_hash()

    with pytest.raises(ValueError, match="variance must be non-negative"):
        model.commit(np.array([1.0]), np.array([0.5]), np.array([-0.01]))

    assert model.version == 0
    assert model.commit_count == 0
    assert not model.table
    assert model.get_state_hash() == original_hash


def test_recommitting_an_anchor_replaces_value_but_records_both_commits():
    model = create_model()
    x = np.array([1.0])
    model.commit(x, np.array([0.2]), np.array([0.04]))
    model.commit(x, np.array([0.8]), np.array([0.09]))

    assert len(model.table) == 1
    assert model.version == model.commit_count == 2
    np.testing.assert_array_equal(model.predict(x, "mean"), [0.8])
    np.testing.assert_array_equal(model.predict(x, "var"), [0.09])


def test_eager_and_lazy_models_match_for_all_routes():
    lazy = create_model(eager=False)
    eager = create_model(eager=True)
    seed_model(lazy, 30)
    seed_model(eager, 30)

    for index in range(30):
        x = make_input(index)
        for route in ROUTES:
            assert_route_outputs_equal(
                route, eager.predict(x, route), lazy.predict(x, route)
            )

    assert all(patch.cached_moments is None for patch in lazy.table.values())
    assert all(
        set(patch.cached_moments or {}) == set(ROUTES)
        for patch in eager.table.values()
    )


def test_identical_commit_sequences_have_identical_state_hashes():
    first = create_model()
    second = create_model(eager=True)
    seed_model(first, 20)
    seed_model(second, 20)

    # Cache strategy is an evaluation detail, not learned model state.
    assert first.get_state_hash() == second.get_state_hash()
    assert len(first.get_state_hash()) == 64


def test_state_hash_changes_when_distribution_changes():
    first = create_model()
    second = create_model()
    mean, variance = make_target_dist(0)
    first.commit(make_input(0), mean, variance)
    second.commit(make_input(0), mean + 0.001, variance)

    assert first.get_state_hash() != second.get_state_hash()


def test_audit_log_is_bounded_to_most_recent_500_entries():
    model = MicroModel(
        f=base_dist_gaussian,
        config=MicroModelConfig(enable_audit_logging=True),
    )
    seed_model(model, 505)

    assert len(model.operation_log) == 500
    assert model.operation_log[0]["event_index"] == 5
    assert model.operation_log[-1]["event_index"] == 504


@pytest.mark.parametrize("mode", ["commit", "inference", "lifecycle"])
def test_benchmark_workload_contract(mode):
    config = BenchmarkConfig(
        mode=mode,
        updates=4,
        predictions_per_update=2,
        trials=1,
        warmup=0,
        route="stddev",
        eager=False,
    )

    result = run_workload(config)

    assert result["benchmark"]["mode"] == mode
    assert result["benchmark"]["route"] == "stddev"
    assert result["benchmark"]["timed_elapsed_seconds"] >= 0
    assert len(result["model"]["state_hash"]) == 64
    if mode == "inference":
        assert result["benchmark"]["seeded_updates"] == 4
        assert result["benchmark"]["measured_predictions"] == 8
    else:
        assert result["benchmark"]["measured_updates"] == 4
        assert result["model"]["committed_updates"] == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"updates": -1}, "counts must be >= 0"),
        ({"predictions_per_update": -1}, "counts must be >= 0"),
        ({"trials": 0}, "trials must be > 0"),
        ({"warmup": -1}, "trials must be > 0"),
        ({"route": "median"}, "route must be one of"),
    ],
)
def test_benchmark_rejects_invalid_configuration(overrides, message):
    values = {
        "mode": "commit",
        "updates": 1,
        "predictions_per_update": 1,
        "trials": 1,
        "warmup": 0,
        "route": "mean",
        "eager": False,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_workload(BenchmarkConfig(**values))
