"""Tests for src.data.fetch_data with FastF1 replaced by stubs (no network)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data import fetch_data as fd

from .test_preprocessing import make_raw


@pytest.fixture()
def frame() -> pd.DataFrame:
    return make_raw()  # 100 timed laps, enough to pass validation


@pytest.fixture()
def stub_schedule(monkeypatch):
    def fake(year: int, session_code: str = "R") -> pd.DataFrame:
        return pd.DataFrame({"RoundNumber": [1, 2, 3], "EventName": ["A", "B", "C"]})

    monkeypatch.setattr(fd, "list_completed_rounds", fake)


def test_resolve_years() -> None:
    today = date(2026, 5, 1)
    assert fd.resolve_years([2024, 2023, 2023], None, today) == [2023, 2024]
    assert fd.resolve_years(None, 2024, today) == [2024, 2025, 2026]
    with pytest.raises(ValueError, match="--years or --since"):
        fd.resolve_years(None, None, today)
    with pytest.raises(ValueError, match="outside"):
        fd.resolve_years([2010], None, today)
    with pytest.raises(ValueError, match="outside"):
        fd.resolve_years([2027], None, today)


def test_parse_args_requires_exactly_one_of_years_since() -> None:
    with pytest.raises(SystemExit):
        fd.parse_args([])
    with pytest.raises(SystemExit):
        fd.parse_args(["--years", "2024", "--since", "2020"])
    assert fd.parse_args(["--since", "2020"]).since == 2020


def test_validate_session_frame_rejects_thin_data(frame) -> None:
    fd.validate_session_frame(frame)
    thin = frame.copy()
    thin["LapTime"] = pd.NA
    with pytest.raises(ValueError, match="timed laps"):
        fd.validate_session_frame(thin)


def test_atomic_write_leaves_no_temp_file_and_no_partial_file(
    tmp_path, frame, monkeypatch
) -> None:
    path = tmp_path / "x.parquet"
    fd.write_parquet_atomic(frame, path)
    assert fd.is_valid_parquet(path)
    assert [p.name for p in tmp_path.iterdir()] == ["x.parquet"]

    target = tmp_path / "y.parquet"

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        fd.write_parquet_atomic(frame, target)
    assert not target.exists()
    assert not list(tmp_path.glob("y*"))


def test_is_valid_parquet_detects_corruption(tmp_path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    assert not fd.is_valid_parquet(bad)
    assert not fd.is_valid_parquet(tmp_path / "missing.parquet")


def test_fetch_all_writes_skips_and_records_failures(
    tmp_path, frame, stub_schedule, monkeypatch
) -> None:
    def fake_fetch(year, rnd, code, retries, delay):
        if rnd == 3:
            raise RuntimeError("cancelled")
        return frame

    monkeypatch.setattr(fd, "fetch_with_retry", fake_fetch)
    first = fd.fetch_all([2024], "R", None, tmp_path, force=False)
    assert sorted(first.written) == ["2024_01_R.parquet", "2024_02_R.parquet"]
    assert first.failed == {"2024_03_R.parquet": "cancelled"}
    assert first.failure_rate == pytest.approx(1 / 3)

    second = fd.fetch_all([2024], "R", None, tmp_path, force=False)
    assert sorted(second.skipped) == ["2024_01_R.parquet", "2024_02_R.parquet"]
    assert second.written == []


def test_fetch_all_redownloads_corrupt_file(
    tmp_path, frame, stub_schedule, monkeypatch
) -> None:
    monkeypatch.setattr(fd, "fetch_with_retry", lambda *a: frame)
    (tmp_path / "2024_01_R.parquet").write_bytes(b"truncated")
    report = fd.fetch_all([2024], "R", [1], tmp_path, force=False)
    assert report.written == ["2024_01_R.parquet"]
    assert fd.is_valid_parquet(tmp_path / "2024_01_R.parquet")


def test_fetch_all_respects_rounds_filter(
    tmp_path, frame, stub_schedule, monkeypatch
) -> None:
    monkeypatch.setattr(fd, "fetch_with_retry", lambda *a: frame)
    report = fd.fetch_all([2024], "R", [2], tmp_path, force=False)
    assert report.written == ["2024_02_R.parquet"]


def test_schedule_failure_is_recorded_not_raised(tmp_path, monkeypatch) -> None:
    def broken(year, session_code="R"):
        raise ConnectionError("offline")

    monkeypatch.setattr(fd, "list_completed_rounds", broken)
    report = fd.fetch_all([2024], "R", None, tmp_path, force=False)
    assert "2024_schedule" in report.failed


def test_retry_succeeds_after_transient_failure(monkeypatch, frame) -> None:
    calls = {"n": 0}

    def flaky(year, rnd, code):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return frame

    monkeypatch.setattr(fd, "fetch_session", flaky)
    monkeypatch.setattr(fd.time, "sleep", lambda s: None)
    assert fd.fetch_with_retry(2024, 1, "R", retries=2, delay_s=0).equals(frame)
    assert calls["n"] == 3


def test_retry_gives_up_and_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        fd, "fetch_session", lambda *a: (_ for _ in ()).throw(ValueError("no"))
    )
    monkeypatch.setattr(fd.time, "sleep", lambda s: None)
    with pytest.raises(ValueError, match="no"):
        fd.fetch_with_retry(2024, 1, "R", retries=1, delay_s=0)


@pytest.mark.parametrize(("n_fail", "expected"), [(0, 0), (1, 0), (3, 1)])
def test_main_exit_code_follows_failure_rate(
    tmp_path, frame, monkeypatch, n_fail, expected
) -> None:
    monkeypatch.setattr(fd, "enable_cache", lambda d: None)
    monkeypatch.setattr(
        fd,
        "list_completed_rounds",
        lambda year, code="R": pd.DataFrame(
            {"RoundNumber": list(range(1, 11)), "EventName": ["x"] * 10}
        ),
    )

    def fake_fetch(year, rnd, code, retries, delay):
        if rnd <= n_fail:
            raise RuntimeError("fail")
        return frame

    monkeypatch.setattr(fd, "fetch_with_retry", fake_fetch)
    code = fd.main(
        ["--years", "2024", "--out-dir", str(tmp_path), "--max-failure-rate", "0.15"]
    )
    assert code == expected


def test_sprint_schedule_keeps_only_sprint_weekends(monkeypatch) -> None:
    schedule = pd.DataFrame(
        {
            "RoundNumber": [1, 2],
            "EventName": ["A", "B"],
            "EventDate": [pd.Timestamp("2020-01-01")] * 2,
            "EventFormat": ["conventional", "sprint_qualifying"],
        }
    )
    monkeypatch.setattr(
        fd.fastf1, "get_event_schedule", lambda y, include_testing=False: schedule
    )
    assert fd.list_completed_rounds(2024, "S")["RoundNumber"].tolist() == [2]
    assert fd.list_completed_rounds(2024, "R")["RoundNumber"].tolist() == [1, 2]
