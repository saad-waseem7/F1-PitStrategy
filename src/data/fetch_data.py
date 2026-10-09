"""Download raw F1 lap and weather data via FastF1 and store it as Parquet.

One Parquet file is written per session (``{year}_{round:02d}_{session}.parquet``)
so that new races can be added incrementally and tracked cleanly by DVC.

Usage:
    python -m src.data.fetch_data --years 2023 2024 --session R
    python -m src.data.fetch_data --years 2024 --rounds 5 6 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_OUT_DIR = Path("data/raw")

REQUIRED_LAP_COLUMNS: tuple[str, ...] = (
    "Driver",
    "LapNumber",
    "LapTime",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
)
OPTIONAL_LAP_COLUMNS: tuple[str, ...] = (
    "DriverNumber",
    "Team",
    "FreshTyre",
    "IsAccurate",
    "Deleted",
    "PitInTime",
    "PitOutTime",
)
WEATHER_COLUMNS: tuple[str, ...] = (
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "WindSpeed",
    "WindDirection",
)


def enable_cache(cache_dir: Path) -> None:
    """Enable the FastF1 on-disk cache. The directory must exist first."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def output_path(out_dir: Path, year: int, rnd: int, session: str) -> Path:
    """Return the deterministic Parquet path for one session."""
    return out_dir / f"{year}_{rnd:02d}_{session}.parquet"


def list_completed_rounds(year: int) -> pd.DataFrame:
    """Return the completed, non-testing events of a season."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    done = schedule[schedule["EventDate"] < pd.Timestamp.now()]
    return done[["RoundNumber", "EventName"]].reset_index(drop=True)


def build_session_frame(session: fastf1.core.Session, year: int) -> pd.DataFrame:
    """Flatten a loaded session into a lap-level frame with weather attached.

    Raises:
        ValueError: if the session lacks columns the pipeline depends on.
    """
    laps = session.laps
    missing = [c for c in REQUIRED_LAP_COLUMNS if c not in laps.columns]
    if missing:
        raise ValueError(f"Session is missing required lap columns: {missing}")

    keep = list(REQUIRED_LAP_COLUMNS) + [
        c for c in OPTIONAL_LAP_COLUMNS if c in laps.columns
    ]
    frame = pd.DataFrame(laps[keep]).reset_index(drop=True)

    # Weather aligned to each lap by timestamp (FastF1 handles the join).
    weather = laps.get_weather_data().reset_index(drop=True)
    for col in WEATHER_COLUMNS:
        frame[col] = weather[col] if col in weather.columns else pd.NA

    # Timedelta -> float seconds / booleans so Parquet and pandas stay simple.
    frame["LapTime"] = frame["LapTime"].dt.total_seconds()
    frame["IsInLap"] = frame["PitInTime"].notna() if "PitInTime" in frame else False
    frame["IsOutLap"] = frame["PitOutTime"].notna() if "PitOutTime" in frame else False
    frame = frame.drop(columns=["PitInTime", "PitOutTime"], errors="ignore")

    # Total laps drives the fuel-load estimate downstream.
    total_laps = getattr(session, "total_laps", None)
    if not total_laps:
        total_laps = int(frame["LapNumber"].max())

    frame["Year"] = year
    frame["RoundNumber"] = int(session.event["RoundNumber"])
    frame["EventName"] = str(session.event["EventName"])
    frame["SessionName"] = str(session.name)
    frame["TotalLaps"] = int(total_laps)
    return frame


def fetch_session(year: int, rnd: int, session_code: str) -> pd.DataFrame:
    """Load one session from FastF1 (laps + weather only, no telemetry)."""
    session = fastf1.get_session(year, rnd, session_code)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    # FastF1 logs network failures instead of raising, then fails on first
    # access. Surface that as one clear error.
    try:
        _ = session.laps
    except fastf1.exceptions.DataNotLoadedError as exc:
        raise RuntimeError(
            f"FastF1 returned no lap data for {year} round {rnd}. "
            "Check network access to livetiming.formula1.com."
        ) from exc
    return build_session_frame(session, year)


def fetch_all(
    years: list[int],
    session_code: str,
    rounds: list[int] | None,
    out_dir: Path,
    force: bool,
) -> tuple[int, int]:
    """Fetch every requested session. Returns (n_written, n_failed).

    A single bad session is logged and skipped; the caller decides whether the
    overall failure rate is acceptable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    for year in years:
        try:
            events = list_completed_rounds(year)
        except Exception:  # network/schedule failure: skip the season, keep going
            logger.exception("Could not load schedule for %s", year)
            failed += 1
            continue
        for rnd in events["RoundNumber"].astype(int):
            if rounds and rnd not in rounds:
                continue
            path = output_path(out_dir, year, rnd, session_code)
            if path.exists() and not force:
                logger.info("Skip existing %s", path.name)
                continue
            try:
                frame = fetch_session(year, rnd, session_code)
                frame.to_parquet(path, index=False)
                logger.info("Wrote %s (%d laps)", path.name, len(frame))
                written += 1
            except Exception:
                logger.exception("Failed %s round %s", year, rnd)
                failed += 1
    return written, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--session", default="R", help="R=race, S=sprint")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    enable_cache(args.cache_dir)
    written, failed = fetch_all(
        args.years, args.session, args.rounds, args.out_dir, args.force
    )
    logger.info("Done: %d written, %d failed", written, failed)
    # Non-zero only if nothing worked, so one flaky race does not break the DAG.
    return 1 if failed and not written else 0


if __name__ == "__main__":
    sys.exit(main())
