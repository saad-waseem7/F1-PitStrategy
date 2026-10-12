from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_CLASSIFICATION, TARGET_REGRESSION
from src.data.preprocess import (
    build_dataset,
    safety_car_laps,
    split_by_year,
    trailing_slope,
)
from tests.conftest import PIT_LAP, SC_DEPLOYED_LAP, SC_END_LAP


@pytest.fixture(scope="module")
def built(raw_frames, params):
    return build_dataset(params=params, **raw_frames)


def test_trailing_slope_recovers_linear_trend():
    y = 2.0 * np.arange(10) + 5
    out = trailing_slope(y, 5)
    assert np.isnan(out[:4]).all()
    assert np.allclose(out[4:], 2.0)


def test_trailing_slope_short_input_and_nan_propagation():
    assert np.isnan(trailing_slope(np.array([1.0, 2.0]), 5)).all()
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    out = trailing_slope(y, 3)
    assert np.isnan(out[2:5]).all()
    assert np.isclose(out[5], 1.0)


def test_safety_car_laps_include_restart_lap(raw_frames):
    laps = safety_car_laps(raw_frames["race_control"])
    expected = {(1001, lap) for lap in range(SC_DEPLOYED_LAP, SC_END_LAP + 2)}
    assert expected <= laps


def test_safety_car_laps_empty_input():
    assert safety_car_laps(pd.DataFrame(columns=["category"])) == set()


def test_filters_remove_unusable_laps(built):
    df, _ = built
    laps_by_session = df[df["session_key"] == 1001]["lap_number"]
    assert 1 not in laps_by_session.values
    assert not laps_by_session.between(SC_DEPLOYED_LAP, SC_END_LAP + 1).any()
    assert PIT_LAP not in laps_by_session.values        # pit-in lap
    assert PIT_LAP + 1 not in laps_by_session.values    # pit-out lap


def test_no_missing_values_in_model_inputs(built):
    df, _ = built
    # NaN is allowed only in lagged pace features (first laps of a stint), LightGBM handles it.
    strict = [c for c in FEATURE_COLUMNS if c not in ("prev_lap_delta", "pace_decay_slope", "track_temp_delta")]
    assert not df[strict].isna().any().any()
    assert not df[TARGET_REGRESSION].isna().any()


def test_tyre_age_resets_after_pit_stop(built):
    df, _ = built
    d = df[(df["session_key"] == 1001) & (df["driver_number"] == 1)]
    first_after_stop = d[d["stint_number"] == 2].sort_values("lap_number").iloc[0]
    assert first_after_stop["tyre_age"] <= 5
    assert (d[d["stint_number"] == 1]["tyre_age"] == d[d["stint_number"] == 1]["lap_number"] - 1).all()


def test_pace_loss_increases_with_tyre_age(built):
    df, _ = built
    stint1 = df[df["stint_number"] == 1]
    corr = stint1["tyre_age"].corr(stint1[TARGET_REGRESSION])
    assert corr > 0.9


def test_pit_flag_is_boolean_and_monotone_with_loss(built):
    df, _ = built
    assert df[TARGET_CLASSIFICATION].dtype == bool
    assert df[df[TARGET_CLASSIFICATION]][TARGET_REGRESSION].mean() > df[~df[TARGET_CLASSIFICATION]][TARGET_REGRESSION].mean()


def test_lagged_features_do_not_leak_current_lap(raw_frames, params):
    """Changing the current lap's time must not change its own lagged features."""
    base, _ = build_dataset(params=params, **raw_frames)
    mutated = {k: v.copy() for k, v in raw_frames.items()}
    mask = (mutated["laps"]["session_key"] == 1001) & (mutated["laps"]["driver_number"] == 1) & (mutated["laps"]["lap_number"] == 30)
    mutated["laps"].loc[mask, "lap_duration"] += 2.0
    changed, _ = build_dataset(params=params, **mutated)
    key = lambda d: d[(d["session_key"] == 1001) & (d["driver_number"] == 1) & (d["lap_number"] == 30)].iloc[0]
    for col in ("prev_lap_delta", "pace_decay_slope"):
        assert key(base)[col] == pytest.approx(key(changed)[col])


def test_report_counts_are_monotonic(built):
    _, report = built
    steps = ["after_dropna", "after_drop_wet", "after_drop_lap1", "after_drop_pit_laps", "after_drop_safety_car", "after_drop_slow_outliers"]
    values = [report[s] for s in steps]
    assert values == sorted(values, reverse=True)
    assert report["final_rows"] > 0


def test_split_by_year_is_disjoint_and_chronological(built, params):
    df, _ = built
    train, test = split_by_year(df, [2023, 2024], [2025])
    assert set(train["year"]) == {2023, 2024}
    assert set(test["year"]) == {2025}


def test_split_raises_on_empty_partition(built):
    df, _ = built
    with pytest.raises(ValueError):
        split_by_year(df, [2023], [2030])


# ----------------------------------------------------------------- wet filter, reference scope
def test_wet_session_is_dropped(raw_frames, params):
    frames = {k: v.copy() for k, v in raw_frames.items()}
    frames["weather"].loc[frames["weather"]["session_key"] == 1002, "rainfall"] = 1
    df, report = build_dataset(params=params, **frames)
    assert 1002 not in set(df["session_key"])
    assert {1001, 1003} <= set(df["session_key"])
    assert report["wet_sessions_dropped"] == 1
    assert report["after_drop_wet"] < report["after_dropna"]


def test_wet_tyre_laps_removed_even_when_session_is_mostly_dry(raw_frames, params):
    frames = {k: v.copy() for k, v in raw_frames.items()}
    mask = (frames["stints"]["session_key"] == 1001) & (frames["stints"]["driver_number"] == 44) & (frames["stints"]["stint_number"] == 2)
    frames["stints"].loc[mask, "compound"] = "INTERMEDIATE"
    p = {**params, "preprocess": {**params["preprocess"], "wet_session_fraction": 0.9}}
    df, _ = build_dataset(params=p, **frames)
    assert "INTERMEDIATE" not in set(df["compound"].astype(str))


def test_dry_only_false_keeps_wet_sessions(raw_frames, params):
    frames = {k: v.copy() for k, v in raw_frames.items()}
    frames["weather"].loc[frames["weather"]["session_key"] == 1002, "rainfall"] = 1
    p = {**params, "preprocess": {**params["preprocess"], "dry_only": False}}
    df, report = build_dataset(params=p, **frames)
    assert 1002 in set(df["session_key"]) and report["wet_sessions_dropped"] == 0


def test_invalid_reference_scope_raises(raw_frames, params):
    p = {**params, "preprocess": {**params["preprocess"], "reference_scope": "nonsense"}}
    with pytest.raises(ValueError, match="reference_scope"):
        build_dataset(params=p, **raw_frames)


def test_min_stint_laps_filters_short_stints(raw_frames, params):
    p = {**params, "preprocess": {**params["preprocess"], "min_stint_laps": 1000}}
    with pytest.raises(ValueError, match="empty"):
        build_dataset(params=p, **raw_frames)


@pytest.mark.parametrize("scope", ["stint", "driver_race"])
def test_both_reference_scopes_produce_valid_data(raw_frames, params, scope):
    p = {**params, "preprocess": {**params["preprocess"], "reference_scope": scope}}
    df, _ = build_dataset(params=p, **raw_frames)
    assert len(df) > 0 and df[TARGET_REGRESSION].notna().all()


def test_stint_reference_pace_loss_is_near_zero_at_best_laps(built):
    df, _ = built
    per_stint_min = df.groupby(["session_key", "driver_number", "stint_number"])[TARGET_REGRESSION].min()
    assert per_stint_min.abs().median() < 0.5