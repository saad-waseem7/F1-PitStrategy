"""Download raw F1 lap and weather data via FastF1 and store it as Parquet.

One Parquet file is written per session (``{year}_{round:02d}_{session}.parquet``)
so new races can be added incrementally and tracked by DVC.

Reliability features, because a half-downloaded dataset silently biases a model:

* **Atomic writes.** Each file is written to a temporary name and renamed, so a
  crash never leaves a truncated file that later runs would skip as "done".
* **Corrupt-file recovery.** Existing files are checked and re-downloaded if
  unreadable or empty.
* **Retries.** Transient network errors are retried with a growing delay.
* **Validation.** A session with too few timed laps is rejected, not saved.
* **Completeness report.** After every run the missing sessions are listed and
  the exit code is non-zero when too many failed. A few failures are normal
  (cancelled races, such as Emilia-Romagna 2023, have no data).

Usage:
    python -m src.data.fetch_data --since 2018            # everything available
    python -m src.data.fetch_data --years 2023 2024 --session R
    python -m src.data.fetch_data --years 2024 --rounds 5 6 --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import fastf1
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_OUT_DIR = Path("data/raw")
FIRST_RELIABLE_SEASON = 2018  # FastF1 lap and tyre data is dependable from here
MIN_TIMED_LAPS = 50  # a real race has hundreds; fewer means a broken download
DEFAULT_RETRIES = 2
RETRY_DELAY_S = 5.0
DEFAULT_MAX_FAILURE_RATE = 0.15

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


@dataclass
class FetchReport:
    """Outcome of one ``fetch_all`` call."""

    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # file name -> reason

    @property
    def attempted(self) -> int:
        return len(self.written) + len(self.failed)

    @property
    def total(self) -> int:
        return len(self.written) + len(self.skipped) + len(self.failed)

    @property
    def failure_rate(self) -> float:
        return len(self.failed) / self.total if self.total else 0.0


def enable_cache(cache_dir: Path) -> None:
    """Enable the FastF1 on-disk cache. The directory must exist first."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def output_path(out_dir: Path, year: int, rnd: int, session: str) -> Path:
    """Return the deterministic Parquet path for one session."""
    return out_dir / f"{year}_{rnd:02d}_{session}.parquet"


def resolve_years(
    years: list[int] | None, since: int | None, today: date | None = None
) -> list[int]:
    """Turn ``--years`` / ``--since`` into a sorted list of seasons.

    Raises:
        ValueError: if neither is given, or a year is before the first reliable
            season or in the future.
    """
    current = (today or datetime.now(tz=UTC).date()).year
    if since is not None:
        years = list(range(since, current + 1))
    if not years:
        raise ValueError("Give --years or --since")
    bad = [y for y in years if y < FIRST_RELIABLE_SEASON or y > current]
    if bad:
        raise ValueError(
            f"Seasons {bad} are outside {FIRST_RELIABLE_SEASON}..{current}"
        )
    return sorted(set(years))


def is_valid_parquet(path: Path) -> bool:
    """True if ``path`` is a readable Parquet file with at least one row."""
    try:
        return pq.ParquetFile(path).metadata.num_rows > 0
    except Exception:  # noqa: BLE001  truncated, empty or not Parquet at all
        return False


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` so that ``path`` is either complete or absent."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def list_completed_rounds(year: int, session_code: str = "R") -> pd.DataFrame:
    """Return the completed, non-testing events of a season.

    For sprint sessions (``S``) only weekends that actually hold a sprint are
    returned, so ordinary weekends are not counted as failures.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    done = schedule[schedule["EventDate"] < pd.Timestamp.now()]
    if session_code.upper() == "S" and "EventFormat" in done.columns:
        done = done[done["EventFormat"].astype(str).str.contains("sprint", case=False)]
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


def validate_session_frame(frame: pd.DataFrame) -> None:
    """Reject a download that is too thin to be a real race.

    Raises:
        ValueError: if fewer than ``MIN_TIMED_LAPS`` laps have a lap time.
    """
    timed = int(frame["LapTime"].notna().sum())
    if timed < MIN_TIMED_LAPS:
        raise ValueError(f"Only {timed} timed laps (need {MIN_TIMED_LAPS})")


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
    frame = build_session_frame(session, year)
    validate_session_frame(frame)
    return frame


def fetch_with_retry(
    year: int,
    rnd: int,
    session_code: str,
    retries: int = DEFAULT_RETRIES,
    delay_s: float = RETRY_DELAY_S,
) -> pd.DataFrame:
    """``fetch_session`` with ``retries`` extra attempts and a growing delay.

    The last exception is re-raised. A cancelled race fails every attempt, which
    only costs a few seconds.
    """
    for attempt in range(retries + 1):
        try:
            return fetch_session(year, rnd, session_code)
        except Exception as exc:
            if attempt == retries:
                raise
            wait = delay_s * (attempt + 1)
            logger.warning(
                "%s round %s attempt %d failed (%s); retrying in %.0fs",
                year,
                rnd,
                attempt + 1,
                exc,
                wait,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")  # pragma: no cover


def fetch_all(
    years: list[int],
    session_code: str,
    rounds: list[int] | None,
    out_dir: Path,
    force: bool,
    retries: int = DEFAULT_RETRIES,
    delay_s: float = RETRY_DELAY_S,
) -> FetchReport:
    """Fetch every requested session and return a ``FetchReport``.

    A bad session is logged and recorded, never fatal; the caller decides if
    the overall failure rate is acceptable (see ``main``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = FetchReport()
    for year in years:
        try:
            events = list_completed_rounds(year, session_code)
        except Exception as exc:  # network/schedule failure: skip the season
            logger.exception("Could not load schedule for %s", year)
            report.failed[f"{year}_schedule"] = str(exc)
            continue
        for rnd in events["RoundNumber"].astype(int):
            if rounds and rnd not in rounds:
                continue
            path = output_path(out_dir, year, rnd, session_code)
            if path.exists() and not force:
                if is_valid_parquet(path):
                    logger.info("Skip existing %s", path.name)
                    report.skipped.append(path.name)
                    continue
                logger.warning("Corrupt or empty %s; downloading again", path.name)
            try:
                frame = fetch_with_retry(year, rnd, session_code, retries, delay_s)
                write_parquet_atomic(frame, path)
                logger.info("Wrote %s (%d laps)", path.name, len(frame))
                report.written.append(path.name)
            except Exception as exc:  # noqa: BLE001  record and continue
                logger.error("Failed %s round %s: %s", year, rnd, exc)
                report.failed[path.name] = str(exc)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument(
        "--since",
        type=int,
        default=None,
        help=f"First season; fetches every season up to the current one "
        f"(earliest {FIRST_RELIABLE_SEASON})",
    )
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--session", default="R", help="R=race, S=sprint")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download files")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=DEFAULT_MAX_FAILURE_RATE,
        help="Exit non-zero if more than this share of sessions failed",
    )
    args = parser.parse_args(argv)
    if (args.years is None) == (args.since is None):
        parser.error("give exactly one of --years or --since")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    years = resolve_years(args.years, args.since)
    enable_cache(args.cache_dir)
    report = fetch_all(
        years, args.session, args.rounds, args.out_dir, args.force, args.retries
    )
    logger.info(
        "Done: %d written, %d already present, %d failed",
        len(report.written),
        len(report.skipped),
        len(report.failed),
    )
    for name, reason in sorted(report.failed.items()):
        logger.warning("MISSING %s: %s", name, reason)
    if report.failure_rate > args.max_failure_rate:
        logger.error(
            "Failure rate %.0f%% exceeds %.0f%%; the dataset is incomplete",
            100 * report.failure_rate,
            100 * args.max_failure_rate,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
