"""Unit tests for src.data.preprocess using a synthetic, fully known race."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import preprocess as pp

TOTAL_LAPS = 50
BASE_TIME = 90.0
DEG = {"SOFT": 0.05, "HARD": 0.03}
PIT_LAP = 25  # in-lap; out-lap is PIT_LAP + 1


def make_raw(drivers: tuple[str, ...] = ("VER", "HAM")) -> pd.DataFrame:
    """Two-stint race with linear tyre wear and exact fuel effect."""
    rows = []
    for driver in drivers:
        for lap in range(1, TOTAL_LAPS + 1):
            stint = 1 if lap <= PIT_LAP else 2
            compound = "SOFT" if stint == 1 else "HARD"
            stint_lap = lap if stint == 1 else lap - PIT_LAP
            fuel = max(pp.FUEL_START_KG * (1 - (lap - 1) / TOTAL_LAPS), 0.0)
            lap_time = (
                BASE_TIME
                + pp.FUEL_EFFECT_S_PER_KG * fuel
                + DEG[compound] * (stint_lap - 1)
            )
            rows.append(
                {
                    "Year": 2024,
                    "RoundNumber": 5,
                    "EventName": "Test GP",
                    "TotalLaps": TOTAL_LAPS,
                    "Driver": driver,
                    "LapNumber": float(lap),
                    "LapTime": lap_time,
                    "Stint": float(stint),
                    "Compound": compound,
                    "TyreLife": float(stint_lap),
                    "TrackStatus": "1",
                    "IsInLap": lap == PIT_LAP,
                    "IsOutLap": lap == PIT_LAP + 1,
                    "IsAccurate": True,
                    "Deleted": False,
                    "AirTemp": 25.0,
                    "TrackTemp": 40.0,
                    "Rainfall": False,
                }
            )
    return pd.DataFrame(rows)


def test_output_schema_and_nonempty() -> None:
    out = pp.build_dataset(make_raw())
    assert list(out.columns) == pp.OUTPUT_COLUMNS
    assert len(out) > 0
    assert out[pp.MODEL_FEATURES + [pp.TARGET]].notna().all().all()


def test_target_recovers_known_degradation() -> None:
    out = pp.build_dataset(make_raw())
    soft = out[(out["Driver"] == "VER") & (out["tire_compound"] == "SOFT")]
    # Baseline = median of stint laps 2,3,4 -> DEG * 2, so loss = DEG * (k - 3).
    row = soft[soft["lap_number"] == 10].iloc[0]
    assert row[pp.TARGET] == pytest.approx(DEG["SOFT"] * 7, abs=1e-9)
    hard = out[(out["Driver"] == "VER") & (out["tire_compound"] == "HARD")]
    row = hard[hard["lap_number"] == 10].iloc[0]
    assert row[pp.TARGET] == pytest.approx(DEG["HARD"] * 7, abs=1e-9)


def test_in_out_and_first_laps_removed() -> None:
    out = pp.build_dataset(make_raw())
    laps = set(out["race_lap"])
    assert 1 not in laps
    assert PIT_LAP not in laps
    assert PIT_LAP + 1 not in laps


def test_safety_car_and_rain_laps_removed() -> None:
    raw = make_raw()
    raw.loc[raw["LapNumber"].between(10, 12), "TrackStatus"] = "4"
    raw.loc[raw["LapNumber"] == 15, "TrackStatus"] = "12"
    raw.loc[raw["LapNumber"] == 16, "Rainfall"] = True
    laps = set(pp.build_dataset(raw)["race_lap"])
    assert laps.isdisjoint({10, 11, 12, 15, 16})


def test_non_dry_compounds_removed() -> None:
    raw = make_raw()
    raw.loc[raw["LapNumber"] == 20, "Compound"] = "INTERMEDIATE"
    raw.loc[raw["LapNumber"] == 21, "Compound"] = None
    laps = set(pp.build_dataset(raw)["race_lap"])
    assert laps.isdisjoint({20, 21})


def test_slow_outlier_lap_removed() -> None:
    raw = make_raw()
    raw.loc[raw["LapNumber"] == 18, "LapTime"] += 30.0
    assert 18 not in set(pp.build_dataset(raw)["race_lap"])


def test_short_stints_dropped() -> None:
    raw = make_raw(("VER",))
    # Stint 2 shrinks to 5 laps -> below MIN_STINT_LAPS even before cleaning.
    raw = raw[raw["LapNumber"] <= PIT_LAP + 5]
    out = pp.build_dataset(raw)
    assert set(out["Stint"]) == {1.0}


def test_fuel_estimate_monotonic_and_bounded() -> None:
    out = pp.build_dataset(make_raw(("VER",))).sort_values("race_lap")
    assert out["fuel_load_est"].is_monotonic_decreasing
    assert out["fuel_load_est"].between(0, pp.FUEL_START_KG).all()


def test_all_laps_dirty_raises() -> None:
    raw = make_raw()
    raw["TrackStatus"] = "4"
    with pytest.raises(ValueError, match="No usable laps"):
        pp.build_dataset(raw)


def test_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        pp.build_dataset(make_raw().drop(columns=["TrackTemp"]))


def test_missing_weather_rows_removed() -> None:
    raw = make_raw()
    raw.loc[raw["LapNumber"] == 30, "TrackTemp"] = np.nan
    assert 30 not in set(pp.build_dataset(raw)["race_lap"])


def test_track_status_helper() -> None:
    status = pd.Series(["1", "12", "4", "67", None, ""])
    assert pp.is_clean_track_status(status).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_load_raw_empty_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        pp.load_raw(tmp_path)


def test_fuel_effect_parameter_changes_target_as_expected() -> None:
    raw = make_raw(("VER",))
    default = pp.build_dataset(raw)
    explicit = pp.build_dataset(raw, fuel_effect=pp.FUEL_EFFECT_S_PER_KG)
    pd.testing.assert_frame_equal(default, explicit)

    # A larger fuel constant credits earlier (heavy) laps with more fuel
    # penalty, so later laps look relatively slower and measured loss grows.
    heavy = pp.build_dataset(raw, fuel_effect=0.04)
    key = ["Driver", "Stint", "race_lap"]
    merged = default.merge(heavy, on=key, suffixes=("_base", "_heavy"))
    late = merged[merged["lap_number_base"] >= 10]
    assert len(late) > 0
    assert (late["lap_time_loss_heavy"] > late["lap_time_loss_base"]).all()


def test_negative_fuel_effect_rejected() -> None:
    with pytest.raises(ValueError, match="fuel_effect"):
        pp.build_dataset(make_raw(), fuel_effect=-0.01)


def test_duplicate_laps_are_dropped_not_double_counted() -> None:
    raw = make_raw()
    once = pp.build_dataset(raw)
    twice = pp.build_dataset(pd.concat([raw, raw], ignore_index=True))
    pd.testing.assert_frame_equal(once, twice)


def test_validate_output_rejects_contract_violations() -> None:
    good = pp.build_dataset(make_raw())
    pp.validate_output(good)
    for column, value in [
        ("tire_age", -1),
        ("lap_number", 0),
        ("fuel_load_est", -5.0),
        ("lap_time_loss", 99.0),
        ("track_temp", np.nan),
    ]:
        bad = good.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            pp.validate_output(bad)
    dup = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        pp.validate_output(dup)


def test_thin_races_are_warned_about(caplog) -> None:
    out = pp.build_dataset(make_raw())  # the synthetic race keeps < MIN_RACE_ROWS rows
    with caplog.at_level("WARNING"):
        pp.log_race_coverage(out)
    assert "fewer than" in caplog.text


def test_main_writes_atomically(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_raw().to_parquet(raw_dir / "2024_05_R.parquet")
    out = tmp_path / "out" / "laps.parquet"
    assert pp.main(["--raw-dir", str(raw_dir), "--out", str(out)]) == 0
    assert [p.name for p in out.parent.iterdir()] == ["laps.parquet"]
