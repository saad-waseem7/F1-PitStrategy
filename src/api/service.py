"""Model loading and prediction. Holds no HTTP concepts so it is easy to test."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import yaml

from src.config import FEATURE_COLUMNS

LOG = logging.getLogger(__name__)

_SMOKE_ROW: dict[str, Any] = {
    "tyre_age": 10, "compound": "MEDIUM", "stint_number": 1, "lap_number": 15, "race_progress": 0.3,
    "track_temperature": 38.0, "air_temperature": 26.0, "humidity": 45.0, "rainfall": 0.0, "wind_speed": 2.0,
    "track_temp_delta": 0.5, "prev_lap_delta": 0.1, "pace_decay_slope": 0.05,
    "prev_gap_to_best": 0.3, "recent_gap_to_best": 0.35,
}


class ModelNotReady(RuntimeError):
    """Raised when a prediction is requested before the model is loaded."""


class ModelService:
    def __init__(self, model_dir: Path, metrics_path: Path | None, params_path: Path) -> None:
        self.model_dir = model_dir
        self.metrics_path = metrics_path
        self.params_path = params_path
        self._model: mlflow.pyfunc.PyFuncModel | None = None
        self.threshold: float = 0.5
        self.preprocess_params: dict[str, Any] = {}

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model and prove it works with one smoke prediction. Raises on any problem."""
        if not (self.model_dir / "MLmodel").is_file():
            raise FileNotFoundError(f"No MLflow model at {self.model_dir} (expected an MLmodel file)")
        params = yaml.safe_load(self.params_path.read_text(encoding="utf-8"))
        self.preprocess_params = params["preprocess"]
        model = mlflow.pyfunc.load_model(str(self.model_dir))
        out = model.predict(pd.DataFrame([_SMOKE_ROW])[list(FEATURE_COLUMNS)])
        if list(out.columns) != ["pace_loss_s", "pit_probability", "pit_recommended"]:
            raise RuntimeError(f"Unexpected model output columns: {list(out.columns)}")
        inner = getattr(model, "unwrap_python_model", lambda: None)()
        if inner is None or not hasattr(inner, "threshold"):
            raise RuntimeError("Model has no decision threshold; was it exported by src.models.train?")
        self.threshold = float(inner.threshold)
        self._model = model

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """ValueError (bad compound, missing column) propagates to the caller as a client error."""
        if self._model is None:
            raise ModelNotReady("model is not loaded")
        return self._model.predict(frame[list(FEATURE_COLUMNS)])

    def info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "ready": self.ready,
            "model_dir": str(self.model_dir),
            "threshold": self.threshold,
            "features": list(FEATURE_COLUMNS),
        }
        meta = self._read_mlmodel()
        info["mlflow_model_uuid"] = meta.get("model_uuid")
        info["created_utc"] = meta.get("utc_time_created")
        report = self._read_report()
        info["champion_version"] = report.get("champion_version") or report.get("model_version")
        info["algorithm"] = report.get("selected_algorithm")
        info["test_metrics"] = {
            "test_regression": report.get("test_regression"),
            "test_classification": report.get("test_classification"),
        } if report else {}
        return info

    def _read_mlmodel(self) -> dict[str, Any]:
        try:
            return yaml.safe_load((self.model_dir / "MLmodel").read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}

    def _read_report(self) -> dict[str, Any]:
        if self.metrics_path is None:
            return {}
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
