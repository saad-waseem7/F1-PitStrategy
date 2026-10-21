"""Latest drift summary, as written by `python -m src.monitoring.drift`."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

KEYS = (
    "status", "generated_at", "drift_detected", "drifted_columns", "share_drifted", "drift_share_threshold",
    "n_current", "n_reference", "min_rows", "columns", "skipped_constant_columns",
)


def _finite(value: Any) -> Any:
    """Replace NaN and infinity with None so the JSON response is always valid."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def read_drift_summary(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "no_report"}
    except (OSError, ValueError):
        return {"status": "unreadable"}
    if not isinstance(raw, dict):
        return {"status": "unreadable"}
    return {k: _finite(raw[k]) for k in KEYS if k in raw}
