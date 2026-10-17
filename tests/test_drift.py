"""Drift monitoring on synthetic reference and served data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.monitoring.drift import DriftError, load_current, main, run


def _frame(n: int, seed: int, temp_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tyre_age": rng.integers(0, 35, n).astype(float),
            "compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n),
            "stint_number": rng.integers(1, 4, n).astype(float),
            "lap_number": rng.integers(2, 60, n).astype(float),
            "race_progress": rng.uniform(0, 1, n),
            "track_temperature": rng.normal(35, 5, n) + temp_shift,
            "air_temperature": rng.normal(25, 3, n),
            "humidity": rng.uniform(20, 80, n),
            "rainfall": np.zeros(n),
            "wind_speed": rng.uniform(0, 6, n),
            "track_temp_delta": rng.normal(0, 1, n),
            "prev_lap_delta": rng.normal(0, 0.3, n),
            "pace_decay_slope": rng.normal(0.05, 0.05, n),
            "prev_gap_to_best": np.abs(rng.normal(0.3, 0.3, n)),
            "recent_gap_to_best": np.abs(rng.normal(0.3, 0.3, n)),
        }
    )[list(FEATURE_COLUMNS)]


def _write_log(path: Path, frame: pd.DataFrame, extra_lines: tuple[str, ...] = ()) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(frame.to_dict("records")):
            clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
            fh.write(json.dumps({"ts": "t", "request_id": str(i), "endpoint": "/predict", "inputs": clean}) + "\n")
        for line in extra_lines:
            fh.write(line + "\n")


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "train.parquet"
    _frame(3000, seed=1).to_parquet(path, index=False)
    return path


def test_no_drift_when_served_data_matches_training(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(400, seed=2))
    summary = run(log, reference, tmp_path / "out")
    assert summary["status"] == "ok" and summary["drift_detected"] is False
    assert summary["drifted_columns"] == []
    assert "rainfall" in summary["skipped_constant_columns"]
    assert (tmp_path / "out" / "drift_report.html").stat().st_size > 1000
    assert json.loads((tmp_path / "out" / "drift_summary.json").read_text())["n_current"] == 400


def test_a_clear_shift_is_detected_and_named(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    shifted = _frame(400, seed=3, temp_shift=20.0)
    shifted["humidity"] = shifted["humidity"] + 60  # a second shifted column keeps the share at 2/14
    _write_log(log, shifted)
    summary = run(log, reference, tmp_path / "out", drift_share=0.1)
    assert summary["drift_detected"] is True
    assert {"track_temperature", "humidity"} <= set(summary["drifted_columns"])
    assert summary["columns"]["track_temperature"]["drifted"] is True


def test_threshold_decides_whether_a_partial_drift_counts(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(400, seed=3, temp_shift=20.0))
    assert run(log, reference, tmp_path / "a", drift_share=0.5)["drift_detected"] is False
    assert run(log, reference, tmp_path / "b", drift_share=0.05)["drift_detected"] is True


def test_too_few_requests_reports_insufficient_data_not_drift(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(20, seed=3, temp_shift=50.0))
    summary = run(log, reference, tmp_path / "out", min_rows=200)
    assert summary["status"] == "insufficient_data" and summary["drift_detected"] is False
    assert (tmp_path / "out" / "drift_summary.json").is_file()


def test_unreadable_log_lines_are_skipped_and_counted(tmp_path):
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(5, seed=4), extra_lines=("not json", '{"inputs": null}', '{"no_inputs": 1}', ""))
    frame, skipped = load_current(log)
    assert len(frame) == 5 and skipped == 3
    assert list(frame.columns) == list(FEATURE_COLUMNS)


def test_null_lag_features_are_kept_as_missing(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    data = _frame(300, seed=5)
    data.loc[:149, ["prev_lap_delta", "pace_decay_slope", "prev_gap_to_best", "recent_gap_to_best"]] = np.nan
    _write_log(log, data)
    summary = run(log, reference, tmp_path / "out")
    assert summary["missing_share"]["prev_gap_to_best"]["current"] == pytest.approx(0.5)
    assert summary["missing_share"]["prev_gap_to_best"]["reference"] == 0.0


def test_missing_files_raise_a_clear_error(tmp_path, reference):
    with pytest.raises(DriftError, match="Prediction log"):
        run(tmp_path / "none.jsonl", reference, tmp_path / "out")
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(300, seed=2))
    with pytest.raises(DriftError, match="Reference data"):
        run(log, tmp_path / "none.parquet", tmp_path / "out")


def test_invalid_drift_share_is_rejected(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(300, seed=2))
    with pytest.raises(ValueError):
        run(log, reference, tmp_path / "out", drift_share=0.0)


def test_cli_exit_codes(tmp_path, reference):
    log = tmp_path / "p.jsonl"
    _write_log(log, _frame(400, seed=3, temp_shift=20.0))
    base = ["--log", str(log), "--reference", str(reference), "--out", str(tmp_path / "o"), "--drift-share", "0.05"]
    assert main(base) == 0
    assert main([*base, "--fail-on-drift"]) == 2
    assert main(["--log", str(tmp_path / "missing"), "--reference", str(reference), "--out", str(tmp_path / "o")]) == 1
