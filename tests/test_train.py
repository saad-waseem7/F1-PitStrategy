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
from mlflow.exceptions import MlflowException

from src.config import FEATURE_COLUMNS, load_params
from src.models.evaluate import persistence_proba
from src.models.train import maybe_promote, run


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
                         prev_gap_to_best=float(abs(rng.normal(0.3, 0.3))), recent_gap_to_best=float(abs(rng.normal(0.3, 0.3))),
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
    assert report["baseline_test_classification"]["pr_auc"] < report["test_classification"]["pr_auc"]
    assert report["pr_auc_lift_over_baseline"] >= params["train"]["min_pr_auc_lift_over_baseline"]
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


def test_rerun_on_same_data_promotes_so_export_matches_current_run(workspace):
    params, root = workspace
    first = _run(params, root)
    second = _run(params, root)
    assert first["promoted"] is True
    assert second["model_version"] == "2" and second["promoted"] is True and second["champion_version"] == "2"
    client = MlflowClient(tracking_uri=params["train"]["mlflow"]["tracking_uri"])
    assert str(client.get_model_version_by_alias(params["train"]["mlflow"]["model_name"], "champion").version) == "2"


class _Run:
    def __init__(self, tags: dict[str, str]) -> None:
        self.data = type("D", (), {"tags": tags})()


class _Version:
    def __init__(self, tags: dict[str, str]) -> None:
        self.tags = {k: v for k, v in tags.items() if k == "cv_mae"}
        self.run_id = tags.get("data_md5", "none")


class _StubClient:
    def __init__(self, champion: _Version | None) -> None:
        self.champion = champion
        self.aliased: str | None = None

    def get_run(self, run_id: str) -> _Run:
        md5 = run_id.split(":")
        return _Run({"train_data_md5": md5[0], "test_data_md5": md5[1]} if len(md5) == 2 else {})

    def get_model_version_by_alias(self, name: str, alias: str) -> _Version:
        if self.champion is None:
            raise MlflowException("no alias")
        return self.champion

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliased = version


@pytest.mark.parametrize(
    "champion_tags, mae, expected",
    [
        (None, 1.0, True),                                     # no champion
        ({"cv_mae": "1.0", "data_md5": "a:a"}, 0.98, True),    # clearly better, same data
        ({"cv_mae": "1.0", "data_md5": "a:a"}, 1.0, True),     # tie on same data
        ({"cv_mae": "1.0", "data_md5": "a:a"}, 1.01, False),   # worse on same data
        ({"cv_mae": "1.0", "data_md5": "a:a"}, 0.995, True),   # slightly better on same data
        ({"cv_mae": "1.0", "data_md5": "b:b"}, 0.995, False),  # marginal gain on different data
        ({"cv_mae": "1.0", "data_md5": "b:b"}, 0.98, True),    # clear gain on different data
        ({"cv_mae": "1.0"}, 1.0, False),                     # champion run without data tags
        ({}, 5.0, True),                                       # champion without MAE tag
    ],
)
def test_promotion_policy(champion_tags, mae, expected):
    client = _StubClient(None if champion_tags is None else _Version(champion_tags))
    assert maybe_promote(client, "m", "7", mae, 0.01, "a:a") is expected
    assert client.aliased == ("7" if expected else None)


def test_report_includes_persistence_baseline_and_cv_lifts(workspace):
    params, root = workspace
    report = _run(params, root)
    assert 0.0 <= report["persistence_test_classification"]["pr_auc"] <= 1.0
    assert np.isfinite(report["pr_auc_lift_over_persistence"]) and np.isfinite(report["cv_pr_auc_lift_over_persistence"])
    assert report["cv_lift_over_baseline"] >= params["train"]["min_lift_over_baseline"]


def test_persistence_gate_fails_when_required_lift_is_impossible(workspace):
    params, root = workspace
    params["train"]["min_pr_auc_lift_over_persistence"] = 1.0
    with pytest.raises(RuntimeError, match="persistence"):
        _run(params, root)


def test_persistence_proba_uses_only_earlier_laps_of_the_same_stint():
    df = pd.DataFrame(
        {
            "session_key": [1, 1, 1, 1, 1],
            "driver_number": [7, 7, 7, 7, 9],
            "stint_number": [1, 1, 1, 2, 1],
            "lap_number": [5, 3, 4, 6, 3],
            "pace_loss_s": [9.0, 1.0, 3.0, 8.0, 5.0],
        },
        index=[10, 11, 12, 13, 14],
    )
    out = pd.Series(persistence_proba(df, window=2), index=df.index)
    squash = lambda x: 1.0 - np.exp(-x)  # noqa: E731
    assert out[11] == 0.0 and out[13] == 0.0 and out[14] == 0.0      # no history in their own stint
    assert out[12] == pytest.approx(squash(1.0))                       # lap 4 sees lap 3 only
    assert out[10] == pytest.approx(squash(2.0))                       # lap 5 sees laps 3, 4: mean 2.0; own 9.0 unused


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


def test_run_fails_when_classifier_does_not_beat_compound_age_baseline(workspace):
    params, root = workspace
    params["train"]["min_pr_auc_lift_over_baseline"] = 1.0  # impossible: PR-AUC cannot rise by 1.0
    with pytest.raises(RuntimeError, match="PR-AUC"):
        _run(params, root)
    assert not (root / "models" / "champion").exists()
    report = json.loads((root / "metrics" / "train_metrics.json").read_text())
    assert report["promoted"] is False and "baseline_test_classification" in report


def test_exported_model_rejects_wet_compounds(workspace):
    params, root = workspace
    _run(params, root)
    model = mlflow.pyfunc.load_model(str(root / "models" / "champion"))
    test = pd.read_parquet(root / "processed" / "test.parquet")
    frame = test[list(FEATURE_COLUMNS)].head(3).assign(compound="INTERMEDIATE")
    with pytest.raises(Exception, match="dry races only"):
        model.predict(frame)


def test_exported_model_accepts_integers_and_missing_lag_features(workspace):
    """What a JSON client sends: ints for whole numbers, nulls for lag features, lowercase compound."""
    params, root = workspace
    _run(params, root)
    model = mlflow.pyfunc.load_model(str(root / "models" / "champion"))
    row = pd.DataFrame([{
        "tyre_age": 18, "compound": "medium", "stint_number": 1, "lap_number": 22, "race_progress": 0.38,
        "track_temperature": 41.0, "air_temperature": 28.0, "humidity": 45, "rainfall": 0, "wind_speed": 2.5,
        "track_temp_delta": None, "prev_lap_delta": None, "pace_decay_slope": None,
        "prev_gap_to_best": None, "recent_gap_to_best": 0.4,
    }])
    out = model.predict(row)
    assert list(out.columns) == ["pace_loss_s", "pit_probability", "pit_recommended"]
    assert len(out) == 1 and 0.0 <= float(out["pit_probability"].iloc[0]) <= 1.0
    with pytest.raises(Exception, match="Missing feature columns"):
        model.predict(row.drop(columns=["tyre_age"]))
