"""Tests for src.data.fetch_openf1 using mocked HTTP and hand-built OpenF1 tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from src.data import fetch_data as fd
from src.data import fetch_openf1 as of
from src.data import preprocess as pp

BASE = pd.Timestamp("2024-05-05 13:00", tz="UTC")
LAPS = 60


def make_tables():
    """Two drivers, SOFT laps 1-30 (new) then HARD laps 31-60 (3 laps old)."""
    laps, stints = [], []
    for number in (1, 44):
        for lap in range(1, LAPS + 1):
            stint_lap = lap if lap <= 30 else lap - 30
            laps.append(
                {
                    "driver_number": number,
                    "lap_number": lap,
                    "lap_duration": 90.0 + 0.05 * stint_lap,
                    "is_pit_out_lap": lap == 31,
                    "date_start": None
                    if lap == 1
                    else (BASE + pd.Timedelta(f"{90 * lap}s")).isoformat(),
                }
            )
        stints.append(
            {"driver_number": number, "stint_number": 1, "compound": "soft",
             "lap_start": 1, "lap_end": 30, "tyre_age_at_start": 0}
        )  # fmt: skip
        stints.append(
            {"driver_number": number, "stint_number": 2, "compound": "HARD",
             "lap_start": 31, "lap_end": None, "tyre_age_at_start": 3}
        )  # fmt: skip
    weather = [
        {"date": (BASE + pd.Timedelta(f"{m}min")).isoformat(), "air_temperature": 25.0,
         "track_temperature": 40.0, "rainfall": 0}
        for m in range(120)
    ]  # fmt: skip
    drivers = [
        {"driver_number": 1, "name_acronym": "VER"},
        {"driver_number": 44, "name_acronym": "HAM"},
    ]
    control = [
        {"category": "Flag", "flag": "YELLOW", "lap_number": 10, "message": "YELLOW IN TRACK SECTOR 3", "date": "2024-05-05T13:10:00"},
        {"category": "SafetyCar", "flag": None, "lap_number": 20, "message": "SAFETY CAR DEPLOYED", "date": "2024-05-05T13:20:00"},
        {"category": "SafetyCar", "flag": None, "lap_number": 22, "message": "SAFETY CAR IN THIS LAP", "date": "2024-05-05T13:22:00"},
    ]  # fmt: skip
    return (
        pd.DataFrame(laps), pd.DataFrame(stints), pd.DataFrame(drivers),
        pd.DataFrame(weather), pd.DataFrame(control),
    )  # fmt: skip


@pytest.fixture()
def frame() -> pd.DataFrame:
    return of.build_session_frame(
        *make_tables(), year=2024, round_number=5, event_name="Miami"
    )


def lap_row(frame: pd.DataFrame, driver: str, lap: int) -> pd.Series:
    return frame[(frame["Driver"] == driver) & (frame["LapNumber"] == lap)].iloc[0]


def test_schema_matches_fastf1_raw_contract(frame) -> None:
    assert set(pp.REQUIRED_RAW_COLUMNS) <= set(frame.columns)
    assert len(frame) == 2 * LAPS


def test_tyre_life_counts_laps_on_the_set(frame) -> None:
    assert lap_row(frame, "VER", 1)["TyreLife"] == 1
    assert lap_row(frame, "VER", 30)["TyreLife"] == 30
    assert lap_row(frame, "VER", 31)["TyreLife"] == 4  # age 3 at start, +1 for this lap
    assert (
        lap_row(frame, "VER", 60)["TyreLife"] == 33
    )  # null lap_end filled to last lap


def test_compound_stint_and_driver_mapping(frame) -> None:
    row = lap_row(frame, "HAM", 5)
    assert row["Compound"] == "SOFT" and row["Stint"] == 1
    assert lap_row(frame, "HAM", 45)["Compound"] == "HARD"


def test_in_and_out_laps(frame) -> None:
    in_laps = frame.loc[frame["IsInLap"], "LapNumber"].unique().tolist()
    out_laps = frame.loc[frame["IsOutLap"], "LapNumber"].unique().tolist()
    assert in_laps == [30.0]  # final stint never has an in lap
    assert out_laps == [31.0]


def test_track_status_codes(frame) -> None:
    status = {
        int(r.LapNumber): r.TrackStatus
        for r in frame[frame["Driver"] == "VER"].itertuples()
    }
    assert status[5] == "1"
    assert status[10] == "2"
    assert all("4" in status[lap] for lap in (20, 21, 22, 23))  # SC plus restart lap
    assert status[24] == "1"
    assert pp.is_clean_track_status(
        pd.Series([status[5], status[10], status[21]])
    ).tolist() == [True, False, False]


def test_weather_is_filled_even_when_a_lap_has_no_start_time(frame) -> None:
    assert frame[["AirTemp", "TrackTemp"]].notna().all().all()
    assert (frame["TrackTemp"] == 40.0).all()
    assert (frame["Rainfall"] == False).all()


def test_weather_outside_tolerance_stays_missing() -> None:
    laps, stints, drivers, weather, control = make_tables()
    weather = weather.iloc[
        :1
    ]  # one reading at race start; later laps are > 15 min away
    out = of.build_session_frame(
        laps, stints, drivers, weather, control, 2024, 5, "Miami"
    )
    assert out.loc[out["LapNumber"] == 60, "TrackTemp"].isna().all()


def test_output_flows_through_preprocess(frame) -> None:
    out = pp.build_dataset(frame)
    assert list(out.columns) == pp.OUTPUT_COLUMNS
    assert len(out) > 0
    assert (out["tire_age"] >= 1).all()
    assert set(out["tire_compound"]) <= {"SOFT", "HARD"}


def test_empty_tables_raise() -> None:
    laps, stints, drivers, weather, control = make_tables()
    with pytest.raises(ValueError, match="no laps or no stints"):
        of.build_session_frame(
            laps.iloc[0:0], stints, drivers, weather, control, 2024, 5, "X"
        )


def test_unterminated_safety_car_runs_to_the_end() -> None:
    control = pd.DataFrame(
        [{"category": "SafetyCar", "flag": None, "lap_number": 55, "message": "SAFETY CAR DEPLOYED", "date": "2024-05-05T14:00:00"}]
    )  # fmt: skip
    status = of.build_track_status(control, 60)
    assert status[54] == "1" and all(status[lap] == "4" for lap in range(55, 61))


def test_vsc_and_red_flag_codes() -> None:
    control = pd.DataFrame(
        [
            {"category": "SafetyCar", "flag": None, "lap_number": 5, "message": "VIRTUAL SAFETY CAR DEPLOYED", "date": "1"},
            {"category": "SafetyCar", "flag": None, "lap_number": 7, "message": "VIRTUAL SAFETY CAR ENDING", "date": "2"},
            {"category": "Flag", "flag": "RED", "lap_number": 30, "message": "RED FLAG", "date": "3"},
        ]
    )  # fmt: skip
    status = of.build_track_status(control, 40)
    assert "6" in status[5] and "6" in status[6] and "7" in status[7]
    assert "5" in status[30] and "5" in status[31] and status[32] == "1"


def test_no_race_control_means_all_green() -> None:
    assert (of.build_track_status(pd.DataFrame(), 10) == "1").all()


# ---------------------------------------------------------------- HTTP


class FakeResponse:
    def __init__(self, status: int, payload=None, headers=None) -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.ok = status < 400

    def json(self):
        return self._payload


@pytest.fixture()
def fast_http(monkeypatch):
    monkeypatch.setattr(of.time, "sleep", lambda s: None)
    monkeypatch.setattr(of, "_last_call", 0.0)


def test_get_json_retries_429_then_succeeds(monkeypatch, fast_http) -> None:
    replies = iter(
        [FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, [{"a": 1}])]
    )
    monkeypatch.setattr(of.requests, "get", lambda *a, **k: next(replies))
    assert of.get_json("laps", {"session_key": 1}) == [{"a": 1}]


def test_get_json_retries_connection_errors(monkeypatch, fast_http) -> None:
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("reset by peer")
        return FakeResponse(200, [{"ok": True}])

    monkeypatch.setattr(of.requests, "get", flaky)
    assert of.get_json("laps", {}) == [{"ok": True}]
    assert calls["n"] == 3


def test_get_json_gives_up_after_max_attempts(monkeypatch, fast_http) -> None:
    calls = {"n": 0}

    def down(*a, **k):
        calls["n"] += 1
        return FakeResponse(503)

    monkeypatch.setattr(of.requests, "get", down)
    with pytest.raises(RuntimeError, match="failed after"):
        of.get_json("laps", {})
    assert calls["n"] == of.MAX_ATTEMPTS


def test_get_json_empty_result_shapes(monkeypatch, fast_http) -> None:
    monkeypatch.setattr(of.requests, "get", lambda *a, **k: FakeResponse(404))
    assert of.get_json("laps", {}) == []
    monkeypatch.setattr(
        of.requests,
        "get",
        lambda *a, **k: FakeResponse(200, {"detail": "No results found."}),
    )
    assert of.get_json("laps", {}) == []


def test_get_json_does_not_retry_client_errors(monkeypatch, fast_http) -> None:
    calls = {"n": 0}

    def bad(*a, **k):
        calls["n"] += 1
        return FakeResponse(422)

    monkeypatch.setattr(of.requests, "get", bad)
    with pytest.raises(RuntimeError, match="HTTP 422"):
        of.get_json("laps", {})
    assert calls["n"] == 1


# ---------------------------------------------------------------- schedule and orchestration


def sessions_table() -> pd.DataFrame:
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta("30D")).isoformat()
    return pd.DataFrame(
        [
            {"session_key": 30, "date_start": "2024-03-10T15:00:00+00:00", "is_cancelled": False, "location": "B"},
            {"session_key": 10, "date_start": "2024-03-01T15:00:00+00:00", "is_cancelled": False, "location": "A"},
            {"session_key": 20, "date_start": "2024-03-05T15:00:00+00:00", "is_cancelled": True, "location": "X"},
            {"session_key": 99, "date_start": future, "is_cancelled": False, "location": "Future"},
        ]
    )  # fmt: skip


def test_list_race_sessions_orders_and_drops_cancelled_and_future(monkeypatch) -> None:
    monkeypatch.setattr(of, "fetch_table", lambda endpoint, params: sessions_table())
    out = of.list_race_sessions(2024)
    assert out["session_key"].tolist() == [10, 30]
    assert out["RoundNumber"].tolist() == [1, 2]
    assert out["EventName"].tolist() == ["A", "B"]


def test_list_race_sessions_empty(monkeypatch) -> None:
    monkeypatch.setattr(of, "fetch_table", lambda endpoint, params: pd.DataFrame())
    assert of.list_race_sessions(2024).empty


def test_fetch_all_writes_skips_and_records_failures(
    tmp_path, frame, monkeypatch
) -> None:
    monkeypatch.setattr(
        of,
        "list_race_sessions",
        lambda year: pd.DataFrame(
            {"session_key": [10, 30], "RoundNumber": [1, 2], "EventName": ["A", "B"]}
        ),
    )

    def fake(session_key, year, rnd, name):
        if rnd == 2:
            raise RuntimeError("no stints")
        return frame

    monkeypatch.setattr(of, "fetch_session", fake)
    first = of.fetch_all([2024], None, tmp_path, force=False)
    assert first.written == ["2024_01_R.parquet"]
    assert first.failed == {"2024_02_R.parquet": "no stints"}
    second = of.fetch_all([2024], None, tmp_path, force=False)
    assert second.skipped == ["2024_01_R.parquet"]
    assert fd.is_valid_parquet(tmp_path / "2024_01_R.parquet")


def test_fetch_all_schedule_failure_is_recorded(tmp_path, monkeypatch) -> None:
    def boom(year):
        raise RuntimeError("offline")

    monkeypatch.setattr(of, "list_race_sessions", boom)
    report = of.fetch_all([2024, 2025], None, tmp_path, force=False)
    assert set(report.failed) == {"2024_schedule", "2025_schedule"}


def test_parse_args_rejects_seasons_before_2023() -> None:
    with pytest.raises(SystemExit):
        of.parse_args(["--years", "2022"])
    assert of.parse_args(["--years", "2023", "2024"]).years == [2023, 2024]


def test_main_exit_codes(tmp_path, frame, monkeypatch) -> None:
    monkeypatch.setattr(
        of,
        "list_race_sessions",
        lambda year: pd.DataFrame(
            {
                "session_key": list(range(10)),
                "RoundNumber": list(range(1, 11)),
                "EventName": ["x"] * 10,
            }
        ),
    )
    monkeypatch.setattr(of, "fetch_session", lambda *a: frame)
    assert of.main(["--years", "2024", "--out-dir", str(tmp_path)]) == 0

    def failing(*a):
        raise RuntimeError("down")

    monkeypatch.setattr(of, "fetch_session", failing)
    empty = tmp_path / "other"
    assert of.main(["--years", "2024", "--out-dir", str(empty)]) == 1


def test_fetch_session_validates_thin_data(monkeypatch) -> None:
    laps, stints, drivers, weather, control = make_tables()
    laps["lap_duration"] = np.nan
    tables = {
        "laps": laps,
        "stints": stints,
        "drivers": drivers,
        "weather": weather,
        "race_control": control,
    }
    monkeypatch.setattr(of, "fetch_table", lambda endpoint, params: tables[endpoint])
    with pytest.raises(ValueError, match="timed laps"):
        of.fetch_session(1, 2024, 5, "Miami")


def test_mixed_iso_timestamp_formats_are_parsed() -> None:
    """OpenF1 mixes timestamps with and without fractional seconds."""
    laps, stints, drivers, weather, control = make_tables()
    laps["date_start"] = laps["date_start"].map(
        lambda v: (
            v
            if v is None
            else v.replace("+00:00", ".123456+00:00")
            if len(v) % 2
            else v
        )
    )
    weather["date"] = weather["date"].map(
        lambda v: v.replace("+00:00", ".5+00:00") if v.endswith("0:00+00:00") else v
    )
    out = of.build_session_frame(
        laps, stints, drivers, weather, control, 2024, 5, "Miami"
    )
    assert out["TrackTemp"].notna().all()
