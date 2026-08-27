"""Tests for the route x evaluation-strategy benchmark sweep."""

import json

import pytest

from sweep_prob import main, parse_args, run_sweep


def test_run_sweep_covers_each_route_and_strategy():
    rows = run_sweep(
        routes=["mean", "stddev"],
        strategies=["lazy", "eager"],
        updates=4,
        predictions_per_update=2,
        trials=1,
        warmup=0,
        do_lifecycle=True,
    )

    assert {(row["route"], row["strategy"]) for row in rows} == {
        ("mean", "lazy"),
        ("stddev", "lazy"),
        ("mean", "eager"),
        ("stddev", "eager"),
    }
    for row in rows:
        assert row["committed"] == 4
        assert row["table_size"] == 4
        assert row["commit_us"] >= 0
        assert row["predict_us"] >= 0
        assert row["lifecycle_us_per_update"] >= 0
        assert len(row["state_hash"]) == 64

    # Commit has no route parameter; it is measured once per strategy rather
    # than allowing timing noise to suggest a route-dependent commit cost.
    for strategy in ("lazy", "eager"):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        assert strategy_rows[0]["commit_us"] == strategy_rows[1]["commit_us"]


def test_lazy_and_eager_sweeps_produce_the_same_learned_state():
    rows = run_sweep(
        routes=["dist"],
        strategies=["lazy", "eager"],
        updates=3,
        predictions_per_update=1,
        trials=1,
        warmup=0,
        do_lifecycle=False,
    )

    assert len(rows) == 2
    assert rows[0]["state_hash"] == rows[1]["state_hash"]
    assert "lifecycle_us_per_update" not in rows[0]
    assert "lifecycle_us_per_update" not in rows[1]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"routes": []}, "at least one route"),
        ({"routes": ["median"]}, "unknown routes"),
        ({"strategies": []}, "at least one strategy"),
        ({"strategies": ["cached"]}, "unknown strategies"),
        ({"updates": 0}, "updates must be > 0"),
        ({"predictions_per_update": 0}, "predictions_per_update must be > 0"),
        ({"trials": 0}, "trials must be > 0"),
        ({"warmup": -1}, "warmup must be >= 0"),
    ],
)
def test_run_sweep_rejects_invalid_inputs(overrides, message):
    values = {
        "routes": ["mean"],
        "strategies": ["lazy"],
        "updates": 1,
        "predictions_per_update": 1,
        "trials": 1,
        "warmup": 0,
        "do_lifecycle": False,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_sweep(**values)


def test_parse_args_uses_retrieval_model_dimensions():
    args = parse_args(
        [
            "--routes",
            "mean",
            "var",
            "--strategies",
            "eager",
            "--updates",
            "8",
            "--predictions-per-update",
            "3",
            "--lifecycle",
        ]
    )

    assert args.routes == ["mean", "var"]
    assert args.strategies == ["eager"]
    assert args.updates == 8
    assert args.predictions_per_update == 3
    assert args.lifecycle is True


def test_main_prints_csv_and_writes_json(tmp_path, capsys):
    output = tmp_path / "sweep.json"

    exit_code = main(
        [
            "--routes",
            "var",
            "--strategies",
            "lazy",
            "--updates",
            "2",
            "--predictions-per-update",
            "1",
            "--trials",
            "1",
            "--warmup",
            "0",
            "--output-json",
            str(output),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out.splitlines()
    assert stdout[0].startswith("route,strategy,commit_us,predict_us")
    assert stdout[1].startswith("var,lazy,")

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["grid"][0]["route"] == "var"
    assert document["grid"][0]["strategy"] == "lazy"
    assert document["runtime"]["python_version"]
