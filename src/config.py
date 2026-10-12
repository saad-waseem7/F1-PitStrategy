"""Central configuration: filesystem paths and params.yaml loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
METRICS_DIR: Path = PROJECT_ROOT / "metrics"
MODELS_DIR: Path = PROJECT_ROOT / "models"
PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"

ENDPOINTS: tuple[str, ...] = ("sessions", "laps", "stints", "weather", "race_control")

COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")

FEATURE_COLUMNS: tuple[str, ...] = (
    "tyre_age",
    "compound",
    "stint_number",
    "lap_number",
    "race_progress",
    "track_temperature",
    "air_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "track_temp_delta",
    "prev_lap_delta",
    "pace_decay_slope",
)

TARGET_REGRESSION: str = "pace_loss_s"
TARGET_CLASSIFICATION: str = "pit_recommended"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    if not isinstance(params, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return params
