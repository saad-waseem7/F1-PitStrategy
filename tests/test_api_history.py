"""The service computes lag features itself; it must agree with the preprocessing pipeline."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.api.history import history_features
from src.config import load_params
from src.data.preprocess import _add_targets_and_features


def _stint_frame(n: int = 16, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    laps = np.arange(5, 5 + n)
    duration = 92.0 + 0.08 * np.arange(n) - 0.05 * laps + rng.normal(0, 0.25, n)
    return pd.DataFrame(
        {
            "session_key": 1, "driver_number": 7, "stint_number": 1, "lap_number": laps,
            "lap_duration": duration, "race_laps": 60,
        }
    )


def test_history_features_match_the_preprocessing_pipeline_row_by_row():
    p = copy.deepcopy(load_params()["preprocess"])
    p.update(min_laps_per_driver=1, min_stint_laps=1)
    raw = _stint_frame()
    out = _add_targets_and_features(raw, p).reset_index(drop=True)
    assert len(out) == len(raw) - (int(p["pit_rolling_window"]) - 1)  # label needs a full rolling window
    for _, row in out.iterrows():
        earlier = raw[raw["lap_number"] < row["lap_number"]]
        got = history_features(
            earlier["lap_number"].tolist(), earlier["lap_duration"].tolist(), int(row["lap_number"]),
            p["fuel_correction_s_per_lap"], int(p["decay_window"]), int(p["recent_gap_window"]),
        )
        for name, value in got.items():
            expected = row[name]
            if value is None:
                assert np.isnan(expected), (name, row["lap_number"])
            else:
                assert value == pytest.approx(expected), (name, row["lap_number"])


def test_features_need_enough_laps():
    one = history_features([5], [92.0], 6, 0.05, 5, 2)
    assert one["prev_lap_delta"] is None and one["pace_decay_slope"] is None
    assert one["prev_gap_to_best"] == pytest.approx(0.0) and one["recent_gap_to_best"] is None
    two = history_features([5, 6], [92.0, 92.4], 7, 0.05, 5, 2)
    assert two["prev_lap_delta"] == pytest.approx(0.4 + 0.05) and two["recent_gap_to_best"] is not None
    assert two["pace_decay_slope"] is None  # needs 5 laps
    assert history_features([], [], 3, 0.05, 5, 2) == dict.fromkeys(two)


@pytest.mark.parametrize(
    "laps, current",
    [([6, 5], 8), ([5, 5], 8), ([5, 9], 9), ([5, 6], 6)],
)
def test_rejects_unordered_or_future_laps(laps, current):
    with pytest.raises(ValueError):
        history_features(laps, [92.0] * len(laps), current, 0.05, 5, 2)
