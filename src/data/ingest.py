"""Download OpenF1 race sessions to per-session parquet files.

Layout: data/raw/<endpoint>/<session_key>.parquet
Existing files are skipped, so reruns (and the Airflow batch DAG later) are incremental.

Usage:
    python -m src.data.ingest
    python -m src.data.ingest --years 2025
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import ENDPOINTS, RAW_DIR, load_params

LOG = logging.getLogger(__name__)

# Keep only columns we use. Drops nested list columns (e.g. lap segments) that bloat parquet.
KEEP_COLUMNS: dict[str, list[str]] = {
    "laps": ["session_key", "driver_number", "lap_number", "lap_duration", "date_start", "is_pit_out_lap"],
    "stints": ["session_key", "driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"],
    "weather": ["session_key", "date", "track_temperature", "air_temperature", "humidity", "rainfall", "wind_speed"],
    "race_control": ["session_key", "date", "lap_number", "category", "flag", "message"],
    "sessions": ["session_key", "session_name", "session_type", "circuit_short_name", "country_name", "year", "date_start"],
}


def _retry_after_s(resp: requests.Response) -> float:
    """Seconds requested by the server's Retry-After header, 0 if absent or not numeric."""
    try:
        return max(0.0, float(resp.headers.get("Retry-After", 0)))
    except (TypeError, ValueError):
        return 0.0


class OpenF1Client:
    def __init__(
        self,
        base_url: str,
        delay_s: float,
        max_retries: int,
        timeout_s: int,
        retry_delay_s: float = 3.0,
        empty_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._delay_s = delay_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._retry_delay_s = retry_delay_s
        self._empty_retries = empty_retries
        self._session = requests.Session()

    def get(self, endpoint: str, **query: Any) -> pd.DataFrame:
        """GET one endpoint. Rate limits, 5xx, network errors, truncated or malformed bodies
        are all retried after a fixed delay (never skipped). Raises after max_retries so a
        persistent outage fails loudly instead of silently producing an incomplete dataset."""
        url = f"{self._base_url}/{endpoint}"
        time.sleep(self._delay_s)  # polite spacing between requests
        last_error = "unknown"
        for attempt in range(1, self._max_retries + 1):
            wait = self._retry_delay_s
            try:
                resp = self._session.get(url, params=query, timeout=self._timeout_s)
            except requests.RequestException as exc:  # timeouts, resets, truncated transfers
                last_error = repr(exc)
            else:
                if resp.status_code == 404:
                    return pd.DataFrame()  # OpenF1 uses 404 for "no results"
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                    wait = max(wait, _retry_after_s(resp))  # never retry sooner than the server asks
                else:
                    resp.raise_for_status()  # other 4xx are our bug, not transient
                    try:
                        payload = resp.json()
                    except ValueError:
                        last_error = "response body is not valid JSON (incomplete?)"
                    else:
                        if isinstance(payload, list):
                            return pd.DataFrame(payload)
                        last_error = f"unexpected payload: {str(payload)[:120]}"
            LOG.warning(
                "%s %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                endpoint, query, attempt, self._max_retries, last_error, wait,
            )
            time.sleep(wait)
        raise RuntimeError(f"OpenF1 request failed after {self._max_retries} attempts: {url} {query} ({last_error})")

    def get_nonempty(self, endpoint: str, **query: Any) -> pd.DataFrame:
        """Like get(), but for data that must exist (laps, stints, weather). An empty result is
        retried `empty_retries` times after the fixed delay before it is accepted as empty,
        because a throttled backend can answer with nothing."""
        df = self.get(endpoint, **query)
        for attempt in range(1, self._empty_retries + 1):
            if not df.empty:
                break
            LOG.warning("%s %s returned no rows (retry %d/%d in %.1fs)", endpoint, query, attempt, self._empty_retries, self._retry_delay_s)
            time.sleep(self._retry_delay_s)
            df = self.get(endpoint, **query)
        return df


def _write(df: pd.DataFrame, endpoint: str, path: Path) -> None:
    cols = [c for c in KEEP_COLUMNS[endpoint] if c in df.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df[cols].to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: a crash never leaves a half-written parquet


def ingest(years: list[int] | None = None) -> list[int]:
    params = load_params()["ingest"]
    years = years or list(params["years"])
    client = OpenF1Client(
        params["base_url"],
        params["request_delay_s"],
        params["max_retries"],
        params["timeout_s"],
        params["retry_delay_s"],
        params["empty_retries"],
    )
    new_sessions: list[int] = []

    for year in years:
        sessions = client.get_nonempty("sessions", year=year, session_name=params["session_name"])
        if sessions.empty:
            LOG.warning("No sessions for %d", year)
            continue
        for key in sessions["session_key"].astype(int):
            marker = RAW_DIR / "laps" / f"{key}.parquet"
            if marker.exists():
                LOG.info("Skip session %d (already downloaded)", key)
                continue
            frames = {ep: client.get_nonempty(ep, session_key=key) for ep in ("laps", "stints", "weather")}
            frames["race_control"] = client.get("race_control", session_key=key)  # may be legitimately empty
            if frames["laps"].empty or frames["stints"].empty or frames["weather"].empty:
                LOG.warning("Session %d incomplete, skipping", key)
                continue
            _write(sessions[sessions["session_key"] == key], "sessions", RAW_DIR / "sessions" / f"{key}.parquet")
            for ep, df in frames.items():
                if ep == "race_control" and df.empty:
                    df = pd.DataFrame(columns=KEEP_COLUMNS["race_control"])
                # laps written last: it is the completeness marker checked above
                if ep != "laps":
                    _write(df, ep, RAW_DIR / ep / f"{key}.parquet")
            _write(frames["laps"], "laps", RAW_DIR / "laps" / f"{key}.parquet")
            new_sessions.append(key)
            LOG.info("Downloaded session %d", key)
    return new_sessions


def load_raw(endpoint: str) -> pd.DataFrame:
    """Concatenate all per-session parquet files for an endpoint."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    files = sorted((RAW_DIR / endpoint).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {RAW_DIR / endpoint}. Run ingest first.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    new = ingest(args.years)
    LOG.info("Done. %d new sessions.", len(new))


if __name__ == "__main__":
    main()