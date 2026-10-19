"""Read-only sample of races for the stint replay page.

The sample is a gzipped JSON file built by `python -m src.replay.build_sample` from the processed data. It holds
the model's input features and the true labels, so the page can show what the model says next to what happened.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_CLASSIFICATION, TARGET_REGRESSION

LOG = logging.getLogger(__name__)

ID_COLUMNS: tuple[str, ...] = ("session_key", "driver_number", "year", "circuit_short_name")
SAMPLE_COLUMNS: tuple[str, ...] = (*ID_COLUMNS, *FEATURE_COLUMNS, TARGET_REGRESSION, TARGET_CLASSIFICATION)
MAX_SAMPLE_BYTES = 25 * 1024 * 1024
FORMAT_VERSION = 1


class ReplayError(RuntimeError):
    """The sample file is missing, too large or malformed."""


def write_sample(frame: pd.DataFrame, path: Path, meta: dict[str, Any]) -> None:
    """Write `frame` (already restricted to SAMPLE_COLUMNS) as a deterministic gzipped JSON sample."""
    data: dict[str, list[Any]] = {}
    for col in SAMPLE_COLUMNS:
        series = frame[col]
        if col in ("session_key", "driver_number", "year", "stint_number", "lap_number"):
            data[col] = [int(v) for v in series]
        elif col in ("compound", "circuit_short_name"):
            data[col] = [str(v) for v in series]
        elif col == TARGET_CLASSIFICATION:
            data[col] = [bool(v) for v in series]
        else:
            data[col] = [None if pd.isna(v) else round(float(v), 4) for v in series]
    payload = {"version": FORMAT_VERSION, "meta": meta, "columns": list(SAMPLE_COLUMNS), "data": data}
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh, gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:  # no name, no time: same input, same bytes
        gz.write(raw)
    tmp.replace(path)


class ReplayStore:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.meta: dict[str, Any] = {}
        self._frame: pd.DataFrame | None = None
        self._index: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self._frame is not None

    def load(self) -> None:
        """Load the sample. Raises ReplayError for a missing, oversized or malformed file."""
        if self.path is None or not self.path.is_file():
            raise ReplayError(f"No replay sample at {self.path}")
        if self.path.stat().st_size > MAX_SAMPLE_BYTES:
            raise ReplayError(f"Replay sample is larger than {MAX_SAMPLE_BYTES} bytes")
        try:
            with gzip.open(self.path, "rb") as fh:
                payload = json.loads(fh.read(MAX_SAMPLE_BYTES * 8))
            if payload["version"] != FORMAT_VERSION:
                raise ReplayError(f"Unsupported replay sample version {payload['version']}")
            frame = pd.DataFrame({col: payload["data"][col] for col in SAMPLE_COLUMNS})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReplayError(f"Replay sample is unreadable: {exc}") from exc
        if frame.empty:
            raise ReplayError("Replay sample has no rows")
        frame = frame.sort_values(["session_key", "driver_number", "stint_number", "lap_number"], kind="stable")
        self._frame = frame.reset_index(drop=True)
        self.meta = payload.get("meta") or {}
        self._index = self._build_index(self._frame)

    @staticmethod
    def _build_index(frame: pd.DataFrame) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for key, sess in frame.groupby("session_key", sort=True):
            drivers = []
            for driver, drv in sess.groupby("driver_number", sort=True):
                stints = [
                    {
                        "stint_number": int(num),
                        "compound": str(grp["compound"].iloc[0]),
                        "laps": int(len(grp)),
                        "first_lap": int(grp["lap_number"].min()),
                        "last_lap": int(grp["lap_number"].max()),
                    }
                    for num, grp in drv.groupby("stint_number", sort=True)
                ]
                drivers.append({"driver_number": int(driver), "stints": stints})
            sessions.append({
                "session_key": int(key),
                "circuit": str(sess["circuit_short_name"].iloc[0]),
                "year": int(sess["year"].iloc[0]),
                "drivers": drivers,
            })
        return sessions

    def rows(self) -> pd.DataFrame | None:
        """Every sampled lap, in session, driver, stint, lap order."""
        return self._frame

    def sessions(self) -> list[dict[str, Any]]:
        return self._index

    def stint(self, session_key: int, driver_number: int, stint_number: int) -> pd.DataFrame | None:
        """Laps of one stint in lap order, or None when the combination is not in the sample."""
        if self._frame is None:
            return None
        f = self._frame
        rows = f[(f["session_key"] == session_key) & (f["driver_number"] == driver_number) & (f["stint_number"] == stint_number)]
        return rows.reset_index(drop=True) if len(rows) else None
