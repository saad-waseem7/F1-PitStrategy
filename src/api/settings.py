"""Runtime configuration from environment variables, so the same image runs anywhere."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config import METRICS_DIR, MODELS_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    model_dir: Path = MODELS_DIR / "champion"
    metrics_path: Path = METRICS_DIR / "train_metrics.json"
    params_path: Path = PROJECT_ROOT / "params.yaml"
    prediction_log: Path | None = PROJECT_ROOT / "logs" / "predictions.jsonl"
    max_batch: int = 500

    @classmethod
    def from_env(cls) -> Settings:
        default = cls()
        log = os.environ.get("PREDICTION_LOG")
        return cls(
            model_dir=Path(os.environ.get("MODEL_DIR", default.model_dir)),
            metrics_path=Path(os.environ.get("METRICS_PATH", default.metrics_path)),
            params_path=Path(os.environ.get("PARAMS_PATH", default.params_path)),
            # An empty PREDICTION_LOG disables request logging.
            prediction_log=default.prediction_log if log is None else (Path(log) if log else None),
            max_batch=int(os.environ.get("MAX_BATCH", default.max_batch)),
        )
