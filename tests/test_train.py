"""End-to-end training run on synthetic processed data with a throwaway MLflow store."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from src.config import FEATURE_COLUMNS, TARGET_CLASSIFICATION, TARGET_REGRESSION, load_params
from src.models.train import run


def _write_processed(directory: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for session in range(1, 15):
        year = 2023 if session <= 5 else 2024 if session <= 10 else 2025
        for driver in range(1, 7):
            for lap in range(2, 40):
                age = float(rng.integers(0, 30))
                compound = rng.choice(["SOFT", "MEDIUM", "HARD"])
                temp = float(rng.uniform(25, 50))
                loss = 0.06 * age * (1.6 if compound == "SOFT" else 1.0) + 0.02 * (temp - 35) + rng.normal(0, 0.15)
                rows.append(
                    dict(session_key=session, driver_number=driver, year=year, circuit_short_name="T", tyre_age=age,
                         compound=compound, stint_number=float(rng.integers(1, 4)), lap_number=float(lap),
                         race_progress=lap / 40, track_temperature=temp, air_temperature=25.0, humidity=50.0,
                         rainfall=0.0, wind_speed=2.0, track_temp_delta=float(rng.normal()),
                         prev_lap_delta=float(rng.normal(0, 0.3)) if rng.random() > 0.2 else np.nan,
                         pace_decay_slope=float(rng.normal(0.05, 0.05)) if rng.random() > 0.2 else np.nan,
                         pace_loss_s=loss, pit_recommended=bool(loss > 1.0))
                )
    df = pd.DataFrame(rows)
    df["compound"] = pd.Categorical(df["compound"], categories=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
    directory.mkdir(parents=True, exist_ok=True)
    df[df["year"] <= 2024].to_parquet(directory / "train.parquet", index=False)
    df[df["year"] == 2025].to_parquet(directory / "test.parquet", index=False)


@pytest.fixture
def workspace(tmp_path: Path):
    _write_processed(tmp_path / "processed")
    params = copy.deepcopy(load_params())
    tp = params["train"]
    tp.update(cv_folds=3, lgbm_trials=2, latency_samples=5, permutation_sample=200)
    tp["rf"].update(n_estimators=20)
    tp["lgbm_classifier"].update(n_estimators=30)
    tp["mlflow"].update(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=str(tmp_path / "artifacts"))
    yield params, tmp_path
    mlflow.end_run()


def _run(params, root: Path) -> dict:
    return run(data_dir=root / "processed", metrics_dir=root / "metrics", models_dir=root / "models", params=params)


def test_end_to_end_registers_champion_and_exports_loadable_model(workspace):
    params, root = workspace
    report = _run(params, root)

    assert report["promoted"] is True and report["model_version"] == "1"
    assert report["lift_over_baseline"] >= params["train"]["min_lift_over_baseline"]
    assert report["selected_algorithm"] in ("random_forest", "lightgbm")
    assert json.loads((root / "metrics" / "train_metrics.json").read_text())["promoted"] is True

    model = mlflow.pyfunc.load_model(str(root / "models" / "champion"))
    test = pd.read_parquet(root / "processed" / "test.parquet")
    frame = test[list(FEATURE_COLUMNS)].head(8).assign(compound=lambda d: d["compound"].astype(str))
    out = model.predict(frame)
    assert list(out.columns) == ["pace_loss_s", "pit_probability", "pit_recommended"] and len(out) == 8
    assert out["pit_probability"].between(0, 1).all()

    client = MlflowClient(tracking_uri=params["train"]["mlflow"]["tracking_uri"])
    version = client.get_model_version_by_alias(params["train"]["mlflow"]["model_name"], "champion")
    assert str(version.version) == "1" and float(version.tags["test_mae"]) == pytest.approx(report["test_regression"]["mae"], rel=1e-4)


def test_equal_model_is_registered_but_does_not_replace_champion(workspace):
    params, root = workspace
    first = _run(params, root)
    second = _run(params, root)
    assert first["promoted"] is True
    assert second["model_version"] == "2" and second["promoted"] is False
    client = MlflowClient(tracking_uri=params["train"]["mlflow"]["tracking_uri"])
    assert str(client.get_model_version_by_alias(params["train"]["mlflow"]["model_name"], "champion").version) == "1"


def test_run_fails_loudly_when_model_does_not_beat_baseline(workspace):
    params, root = workspace
    params["train"]["min_lift_over_baseline"] = 0.999
    with pytest.raises(RuntimeError, match="Model rejected"):
        _run(params, root)
    assert (root / "metrics" / "train_metrics.json").exists()
    assert not (root / "models" / "champion").exists()


def test_too_few_sessions_for_cv_raises(workspace):
    params, root = workspace
    params["train"]["cv_folds"] = 50
    with pytest.raises(ValueError, match="race sessions"):
        _run(params, root)
