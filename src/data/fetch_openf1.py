"""Download raw F1 race data from the OpenF1 API (2023 onwards, no key needed).

An alternative to ``fetch_data`` (FastF1). It writes the SAME raw Parquet schema,
so ``preprocess`` and training work unchanged. OpenF1 serves plain JSON over
HTTPS, which avoids FastF1's livetiming mirror failures and its 500 calls/hour
limit. Historical data starts in 2023.

Per race it calls five endpoints: ``sessions`` (once per year), ``laps``,
``stints``, ``drivers``, ``weather`` and ``race_control``.

Mapping to the raw schema:

* ``TyreLife``    = ``tyre_age_at_start + (lap - stint lap_start) + 1``. A new set
  has age 0 at the start, and the first lap on it counts as 1 (as in FastF1).
* ``IsOutLap``    = OpenF1's pit-out flag, or the first lap of any later stint.
* ``IsInLap``     = last lap of every stint except the driver's final one.
* ``Weather``     = the reading at or just before each lap's start time.
* ``TrackStatus`` = FastF1-style codes built from race-control messages:
  ``2`` yellow, ``4`` safety car, ``5`` red flag, ``6`` VSC, ``7`` VSC ending,
  ``1`` otherwise. The lap after a safety car or VSC period is also flagged
  because it is a restart lap. This is a reconstruction from text messages,
  so compare it with FastF1 on one race before trusting it (see README).
* ``RoundNumber`` is the order of the season's non-cancelled race sessions. It
  can differ from the official round number after a cancellation, so keep this
  data in its own folder instead of mixing it with FastF1 files.

Usage:
    python -m src.data.fetch_openf1 --years 2023 2024 2025 --out-dir data/raw_openf1
    python -m src.data.preprocess --raw-dir data/raw_openf1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.data.fetch_data import (
    DEFAULT_MAX_FAILURE_RATE,
    FetchReport,
    is_valid_parquet,
    output_path,
    validate_session_frame,
    write_parquet_atomic,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
FIRST_OPENF1_SEASON = 2023
DEFAULT_OUT_DIR = Path("data/raw_openf1")
REQUEST_TIMEOUT_S = 60
MAX_ATTEMPTS = 5
MIN_INTERVAL_S = 0.4  # politeness delay between calls
WEATHER_TOLERANCE = pd.Timedelta("15min")
SESSION_CODE = "R"

_last_call = 0.0


# ------------------------------------------------------------ HTTP


def get_json(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """GET ``{BASE_URL}/{endpoint}`` and return the rows as a list of dicts.

    Retries connection errors, HTTP 429 and 5xx with exponential backoff
    (honouring ``Retry-After``). OpenF1 answers an empty result with HTTP 404 or
    a ``{"detail": ...}`` object; both become an empty list.

    Raises:
        RuntimeError: after ``MAX_ATTEMPTS`` failed attempts or on another
            HTTP error.
    """
    global _last_call
    url = f"{BASE_URL}/{endpoint}"
    last_error = "unknown error"
    for attempt in range(MAX_ATTEMPTS):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 404:
                return []
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    time.sleep(min(int(retry_after), 120))
            elif response.ok:
                data = response.json()
                return data if isinstance(data, list) else []
            else:
                raise RuntimeError(f"{url} returned HTTP {response.status_code}")
        if attempt < MAX_ATTEMPTS - 1:
            delay = 2.0 * 2**attempt
            logger.warning(
                "%s failed (%s); retrying in %.0fs", endpoint, last_error, delay
            )
            time.sleep(delay)
    raise RuntimeError(f"{url} failed after {MAX_ATTEMPTS} attempts: {last_error}")


def fetch_table(endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
    """``get_json`` as a DataFrame (empty if there are no rows)."""
    return pd.DataFrame(get_json(endpoint, params))


# ------------------------------------------------------------ schedule


def list_race_sessions(year: int) -> pd.DataFrame:
    """Completed, non-cancelled race sessions of a season with their round number.

    Returns columns ``session_key, RoundNumber, EventName, date_start``.
    """
    sessions = fetch_table("sessions", {"year": year, "session_name": "Race"})
    if sessions.empty:
        return pd.DataFrame(
            columns=["session_key", "RoundNumber", "EventName", "date_start"]
        )
    if "is_cancelled" in sessions.columns:
        sessions = sessions[~sessions["is_cancelled"].fillna(False).astype(bool)]
    sessions = sessions.assign(
        date_start=pd.to_datetime(sessions["date_start"], utc=True, format="ISO8601")
    )
    sessions = sessions[sessions["date_start"] < pd.Timestamp.now(tz="UTC")]
    sessions = sessions.sort_values("date_start").reset_index(drop=True)
    sessions["RoundNumber"] = np.arange(1, len(sessions) + 1)
    name = (
        sessions["location"]
        if "location" in sessions.columns
        else sessions["country_name"]
    )
    sessions["EventName"] = name.astype(str)
    return sessions[["session_key", "RoundNumber", "EventName", "date_start"]]


# ------------------------------------------------------------ building blocks


def stint_laps(stints: pd.DataFrame, last_lap: pd.Series) -> pd.DataFrame:
    """Expand stint rows into one row per (driver, lap).

    Args:
        stints: OpenF1 stints (``lap_start``, ``lap_end``, ``compound``, ...).
        last_lap: each driver's final lap number, used when ``lap_end`` is null.

    Returns columns ``driver_number, lap_number, Stint, Compound, TyreLife,
    IsInLap, IsOutLapStint``. Stints without a start lap are dropped. If stints
    overlap on a lap, the later stint wins.
    """
    rows: list[pd.DataFrame] = []
    s = stints.dropna(subset=["lap_start", "driver_number"]).copy()
    s["lap_end"] = s["lap_end"].fillna(s["driver_number"].map(last_lap))
    s = s.dropna(subset=["lap_end"]).sort_values(["driver_number", "stint_number"])
    final_stint = s.groupby("driver_number")["stint_number"].transform("max")
    for row, is_final in zip(
        s.itertuples(index=False), s["stint_number"] == final_stint, strict=True
    ):
        laps = np.arange(int(row.lap_start), int(row.lap_end) + 1)
        age0 = 0 if pd.isna(row.tyre_age_at_start) else int(row.tyre_age_at_start)
        rows.append(
            pd.DataFrame(
                {
                    "driver_number": int(row.driver_number),
                    "lap_number": laps,
                    "Stint": float(row.stint_number),
                    "Compound": str(row.compound).upper()
                    if pd.notna(row.compound)
                    else None,
                    "TyreLife": float(age0) + (laps - int(row.lap_start)) + 1.0,
                    "IsInLap": (laps == int(row.lap_end)) & (not is_final),
                    "IsOutLapStint": (laps == int(row.lap_start))
                    & (row.stint_number > 1),
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "driver_number", "lap_number", "Stint", "Compound",
                "TyreLife", "IsInLap", "IsOutLapStint",
            ]
        )  # fmt: skip
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["driver_number", "lap_number"], keep="last")


def build_track_status(race_control: pd.DataFrame, total_laps: int) -> pd.Series:
    """FastF1-style track status string for every lap 1..``total_laps``.

    Laps are labelled from race-control messages. Safety-car and VSC periods run
    from their ``DEPLOYED`` message to their end message, plus one restart lap.
    """
    codes: dict[int, set[str]] = {lap: set() for lap in range(1, total_laps + 1)}

    def mark(first: int, last: int, code: str) -> None:
        for lap in range(max(first, 1), min(last, total_laps) + 1):
            codes[lap].add(code)

    if not race_control.empty and "lap_number" in race_control.columns:
        rc = race_control.dropna(subset=["lap_number"]).sort_values("date")
        sc_start: int | None = None
        vsc_start: int | None = None
        for row in rc.itertuples(index=False):
            lap = int(row.lap_number)
            message = str(getattr(row, "message", "") or "").upper()
            flag = str(getattr(row, "flag", "") or "").upper()
            category = str(getattr(row, "category", "") or "")
            if flag in {"YELLOW", "DOUBLE YELLOW"}:
                mark(lap, lap, "2")
            elif flag == "RED":
                mark(lap, lap + 1, "5")
            if category != "SafetyCar":
                continue
            if "VIRTUAL SAFETY CAR" in message or message.startswith("VSC"):
                if "DEPLOYED" in message:
                    vsc_start = lap
                elif "ENDING" in message:
                    mark(vsc_start if vsc_start is not None else lap, lap, "6")
                    mark(lap, lap + 1, "7")
                    vsc_start = None
            elif "SAFETY CAR" in message:
                if "DEPLOYED" in message:
                    sc_start = lap
                elif "IN THIS LAP" in message:
                    mark(sc_start if sc_start is not None else lap, lap + 1, "4")
                    sc_start = None
        if sc_start is not None:
            mark(sc_start, total_laps, "4")
        if vsc_start is not None:
            mark(vsc_start, total_laps, "6")

    return pd.Series(
        {lap: "".join(sorted(c)) if c else "1" for lap, c in codes.items()},
        name="TrackStatus",
    )


def attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Add AirTemp, TrackTemp and Rainfall from the reading at or before each lap.

    Laps without a start time get the nearest known time of the same driver.
    Laps with no reading within ``WEATHER_TOLERANCE`` keep missing weather and
    are removed later by preprocessing.
    """
    out = laps.copy()
    air = np.full(len(out), np.nan)
    track = np.full(len(out), np.nan)
    rain = np.full(len(out), np.nan)
    if not weather.empty and "date" in weather.columns:
        w = weather.assign(
            date=pd.to_datetime(weather["date"], utc=True, format="ISO8601")
        ).sort_values("date")
        readings = pd.DataFrame(
            {
                "date": w["date"],
                "air": pd.to_numeric(w["air_temperature"], errors="coerce"),
                "track": pd.to_numeric(w["track_temperature"], errors="coerce"),
                "rain": (
                    pd.to_numeric(w.get("rainfall"), errors="coerce").fillna(0) > 0
                ).astype(float),
            }
        )
        times = pd.to_datetime(out["date_start"], utc=True, format="ISO8601")
        times = times.groupby(out["driver_number"]).transform(
            lambda s: s.ffill().bfill()
        )
        left = pd.DataFrame({"pos": np.arange(len(out)), "t": times.to_numpy()})
        left = left.dropna(subset=["t"]).sort_values("t")
        merged = pd.merge_asof(
            left,
            readings,
            left_on="t",
            right_on="date",
            direction="backward",
            tolerance=WEATHER_TOLERANCE,
        )
        pos = merged["pos"].to_numpy()
        air[pos] = merged["air"].to_numpy()
        track[pos] = merged["track"].to_numpy()
        rain[pos] = merged["rain"].to_numpy()
    out["AirTemp"] = air
    out["TrackTemp"] = track
    out["Rainfall"] = pd.Series(rain, index=out.index).map(
        lambda v: None if pd.isna(v) else bool(v)
    )
    return out


def build_session_frame(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    drivers: pd.DataFrame,
    weather: pd.DataFrame,
    race_control: pd.DataFrame,
    year: int,
    round_number: int,
    event_name: str,
) -> pd.DataFrame:
    """Join the five OpenF1 tables into the raw schema ``preprocess`` expects.

    Raises:
        ValueError: if laps or stints are empty.
    """
    if laps.empty or stints.empty:
        raise ValueError("OpenF1 returned no laps or no stints for this session")
    laps = laps.dropna(subset=["driver_number", "lap_number"]).copy()
    laps["driver_number"] = laps["driver_number"].astype(int)
    laps["lap_number"] = laps["lap_number"].astype(int)
    laps = laps.drop_duplicates(["driver_number", "lap_number"], keep="last")
    total_laps = int(laps["lap_number"].max())

    last_lap = laps.groupby("driver_number")["lap_number"].max()
    tyres = stint_laps(stints, last_lap)
    frame = laps.merge(tyres, on=["driver_number", "lap_number"], how="left")
    frame = attach_weather(frame, weather)

    acronyms = (
        drivers.drop_duplicates("driver_number").set_index("driver_number")[
            "name_acronym"
        ]
        if not drivers.empty
        else pd.Series(dtype=str)
    )
    frame["Driver"] = (
        frame["driver_number"].map(acronyms).fillna(frame["driver_number"].astype(str))
    )
    status = build_track_status(race_control, total_laps)
    pit_out = frame.get("is_pit_out_lap", pd.Series(False, index=frame.index))
    pit_out = pit_out.eq(True)

    result = pd.DataFrame(
        {
            "Year": year,
            "RoundNumber": round_number,
            "EventName": event_name,
            "SessionName": "Race",
            "TotalLaps": total_laps,
            "Driver": frame["Driver"],
            "LapNumber": frame["lap_number"].astype(float),
            "LapTime": pd.to_numeric(frame["lap_duration"], errors="coerce"),
            "Stint": frame["Stint"],
            "Compound": frame["Compound"],
            "TyreLife": frame["TyreLife"],
            "TrackStatus": frame["lap_number"].map(status).fillna("1"),
            "IsInLap": frame["IsInLap"].eq(True),
            "IsOutLap": pit_out | frame["IsOutLapStint"].eq(True),
            "AirTemp": frame["AirTemp"],
            "TrackTemp": frame["TrackTemp"],
            "Rainfall": frame["Rainfall"],
        }
    )
    return result.reset_index(drop=True)


# ------------------------------------------------------------ fetching


def fetch_session(
    session_key: int, year: int, round_number: int, event_name: str
) -> pd.DataFrame:
    """Download and assemble one race. Raises on missing or thin data."""
    key = {"session_key": session_key}
    frame = build_session_frame(
        fetch_table("laps", key),
        fetch_table("stints", key),
        fetch_table("drivers", key),
        fetch_table("weather", key),
        fetch_table("race_control", key),
        year,
        round_number,
        event_name,
    )
    validate_session_frame(frame)
    return frame


def fetch_all(
    years: list[int], rounds: list[int] | None, out_dir: Path, force: bool
) -> FetchReport:
    """Fetch every requested race and return a ``FetchReport`` (never raises per race)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = FetchReport()
    for year in years:
        try:
            races = list_race_sessions(year)
        except Exception as exc:  # noqa: BLE001  a failed season must not stop the rest
            logger.error("Could not load schedule for %s: %s", year, exc)
            report.failed[f"{year}_schedule"] = str(exc)
            continue
        for race in races.itertuples(index=False):
            rnd = int(race.RoundNumber)
            if rounds and rnd not in rounds:
                continue
            path = output_path(out_dir, year, rnd, SESSION_CODE)
            if path.exists() and not force:
                if is_valid_parquet(path):
                    logger.info("Skip existing %s", path.name)
                    report.skipped.append(path.name)
                    continue
                logger.warning("Corrupt or empty %s; downloading again", path.name)
            try:
                frame = fetch_session(
                    int(race.session_key), year, rnd, str(race.EventName)
                )
                write_parquet_atomic(frame, path)
                logger.info("Wrote %s (%d laps)", path.name, len(frame))
                report.written.append(path.name)
            except Exception as exc:  # noqa: BLE001  record and continue
                logger.error("Failed %s round %s: %s", year, rnd, exc)
                report.failed[path.name] = str(exc)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download files")
    parser.add_argument(
        "--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE
    )
    args = parser.parse_args(argv)
    early = [y for y in args.years if y < FIRST_OPENF1_SEASON]
    if early:
        parser.error(
            f"OpenF1 historical data starts in {FIRST_OPENF1_SEASON}; got {early}"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    report = fetch_all(sorted(set(args.years)), args.rounds, args.out_dir, args.force)
    logger.info(
        "Done: %d written, %d already present, %d failed",
        len(report.written), len(report.skipped), len(report.failed),
    )  # fmt: skip
    for name, reason in sorted(report.failed.items()):
        logger.warning("MISSING %s: %s", name, reason)
    if report.failure_rate > args.max_failure_rate:
        logger.error(
            "Failure rate %.0f%% exceeds %.0f%%; the dataset is incomplete",
            100 * report.failure_rate, 100 * args.max_failure_rate,
        )  # fmt: skip
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
