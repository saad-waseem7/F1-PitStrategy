"""HTTP behaviour of the service, against a model trained on synthetic data by the real training code."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mlflow
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.service import ModelService
from src.api.settings import Settings
from src.config import PROJECT_ROOT, load_params
from src.models.train import run
from tests.test_train import _write_processed

ROW = {
    "tyre_age": 18, "compound": "MEDIUM", "stint_number": 1, "lap_number": 22, "race_progress": 0.38,
    "track_temperature": 41.0, "air_temperature": 28.0, "humidity": 45.0, "rainfall": 0.0, "wind_speed": 2.5,
    "track_temp_delta": 1.2, "prev_lap_delta": 0.15, "pace_decay_slope": 0.08,
    "prev_gap_to_best": 0.4, "recent_gap_to_best": 0.5,
}


@pytest.fixture(scope="session")
def champion_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api_model")
    _write_processed(root / "processed")
    params = copy.deepcopy(load_params())
    tp = params["train"]
    tp.update(cv_folds=3, lgbm_trials=2, latency_samples=5, permutation_sample=200)
    tp["rf"].update(n_estimators=20)
    tp["lgbm_classifier"].update(n_estimators=30)
    tp["mlflow"].update(tracking_uri=f"sqlite:///{root / 'mlflow.db'}", artifact_root=str(root / "artifacts"))
    run(data_dir=root / "processed", metrics_dir=root / "metrics", models_dir=root / "models", params=params)
    mlflow.end_run()
    return root / "models" / "champion"


def _settings(champion: Path, tmp_path: Path, **kw) -> Settings:
    return Settings(
        model_dir=champion, metrics_path=champion.parent.parent / "metrics" / "train_metrics.json",
        params_path=PROJECT_ROOT / "params.yaml", prediction_log=tmp_path / "logs" / "predictions.jsonl", **kw,
    )


@pytest.fixture
def client(champion_dir: Path, tmp_path: Path):
    app = create_app(_settings(champion_dir, tmp_path))
    with TestClient(app) as c:
        c.log_path = tmp_path / "logs" / "predictions.jsonl"
        yield c


def test_health_ready_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"ready": True}
    info = client.get("/model-info").json()
    assert info["ready"] is True and len(info["features"]) == 15 and 0 < info["threshold"] < 1
    assert info["algorithm"] in {"lightgbm", "random_forest", "baseline"} and "test_regression" in info["test_metrics"]


def test_predict_returns_valid_output_and_request_id(client):
    r = client.post("/predict", json=ROW)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"pace_loss_s", "pit_probability", "pit_recommended", "threshold", "request_id", "warnings"}
    assert 0.0 <= body["pit_probability"] <= 1.0 and isinstance(body["pit_recommended"], bool)
    assert body["pit_recommended"] == (body["pit_probability"] >= body["threshold"])
    assert r.headers["X-Request-ID"] == body["request_id"]
    assert body["warnings"] == []


def test_lowercase_compound_ints_and_nulls_are_accepted(client):
    row = {**ROW, "compound": "soft", "humidity": 45, "rainfall": 0,
           "track_temp_delta": None, "prev_lap_delta": None, "pace_decay_slope": None,
           "prev_gap_to_best": None, "recent_gap_to_best": None}
    r = client.post("/predict", json=row)
    assert r.status_code == 200, r.text
    assert any("No lap history" in w for w in r.json()["warnings"])


@pytest.mark.parametrize("compound", ["INTERMEDIATE", "wet", "ULTRASOFT", ""])
def test_unsupported_compound_is_422(client, compound):
    r = client.post("/predict", json={**ROW, "compound": compound})
    assert r.status_code == 422
    assert "compound" in json.dumps(r.json())


@pytest.mark.parametrize(
    "patch",
    [
        {"tyre_age": -1}, {"race_progress": 1.5}, {"lap_number": 0}, {"humidity": 150},
        {"tyre_age": "abc"}, {"typo_field": 1}, {"prev_lap_delta": 1e9},
    ],
)
def test_invalid_values_and_unknown_fields_are_422(client, patch):
    assert client.post("/predict", json={**ROW, **patch}).status_code == 422


def test_non_finite_numbers_are_rejected(client):
    raw = json.dumps({**ROW, "tyre_age": 1}).replace('"tyre_age": 1', '"tyre_age": NaN')
    r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "NaN" not in r.text and "input" not in r.json()["detail"][0]


def test_missing_required_field_is_422(client):
    row = dict(ROW)
    del row["tyre_age"]
    assert client.post("/predict", json=row).status_code == 422


def test_rainfall_adds_a_warning_but_still_predicts(client):
    r = client.post("/predict", json={**ROW, "rainfall": 1.0})
    assert r.status_code == 200 and any("dry races only" in w for w in r.json()["warnings"])


def test_batch_matches_single_predictions_and_preserves_order(client):
    rows = [ROW, {**ROW, "tyre_age": 3, "compound": "SOFT"}, {**ROW, "tyre_age": 30, "compound": "HARD"}]
    batch = client.post("/predict/batch", json={"laps": rows})
    assert batch.status_code == 200, batch.text
    got = batch.json()["predictions"]
    assert len(got) == 3
    for row, pred in zip(rows, got, strict=True):
        single = client.post("/predict", json=row).json()
        assert pred["pace_loss_s"] == pytest.approx(single["pace_loss_s"])
        assert pred["pit_probability"] == pytest.approx(single["pit_probability"])


def test_batch_validation_and_size_limit(champion_dir, tmp_path):
    app = create_app(_settings(champion_dir, tmp_path, max_batch=2))
    with TestClient(app) as c:
        assert c.post("/predict/batch", json={"laps": []}).status_code == 422
        assert c.post("/predict/batch", json={"laps": [ROW, {**ROW, "compound": "WET"}]}).status_code == 422
        assert c.post("/predict/batch", json={"laps": [ROW] * 3}).status_code == 413


def _history(n: int):
    return [{"lap_number": 5 + i, "lap_duration": 92.0 + 0.1 * i} for i in range(n)]


def test_from_history_computes_the_lag_features(client):
    base = {k: v for k, v in ROW.items() if k not in {"prev_lap_delta", "pace_decay_slope", "prev_gap_to_best", "recent_gap_to_best"}}
    r = client.post("/predict/from-history", json={**base, "lap_number": 22, "previous_laps": _history(8)})
    assert r.status_code == 200, r.text
    feats = r.json()["computed_features"]
    assert all(v is not None for v in feats.values())
    assert feats["prev_lap_delta"] == pytest.approx(0.1 + 0.05)  # +0.1 s raw, +0.05 s fuel correction per lap
    assert r.json()["warnings"] == []


def test_from_history_with_no_laps_still_works_with_a_warning(client):
    base = {k: v for k, v in ROW.items() if k not in {"prev_lap_delta", "pace_decay_slope", "prev_gap_to_best", "recent_gap_to_best"}}
    r = client.post("/predict/from-history", json={**base, "previous_laps": []})
    assert r.status_code == 200 and any("No lap history" in w for w in r.json()["warnings"])


def test_from_history_rejects_bad_history(client):
    base = {k: v for k, v in ROW.items() if k not in {"prev_lap_delta", "pace_decay_slope", "prev_gap_to_best", "recent_gap_to_best"}}
    unordered = [{"lap_number": 9, "lap_duration": 92.0}, {"lap_number": 8, "lap_duration": 92.1}]
    assert client.post("/predict/from-history", json={**base, "previous_laps": unordered}).status_code == 422
    future = [{"lap_number": 30, "lap_duration": 92.0}]
    assert client.post("/predict/from-history", json={**base, "previous_laps": future}).status_code == 422
    too_long = _history(101)
    assert client.post("/predict/from-history", json={**base, "previous_laps": too_long}).status_code == 422
    bad = [{"lap_number": 9, "lap_duration": -5}]
    assert client.post("/predict/from-history", json={**base, "previous_laps": bad}).status_code == 422


def test_predictions_are_logged_as_jsonl(client):
    client.post("/predict", json=ROW)
    client.post("/predict/batch", json={"laps": [ROW, ROW]})
    lines = [json.loads(x) for x in client.log_path.read_text().splitlines()]
    assert len(lines) == 3
    assert set(lines[0]) >= {"ts", "request_id", "endpoint", "inputs", "outputs", "latency_ms"}
    assert lines[0]["inputs"]["compound"] == "MEDIUM" and len(lines[0]["inputs"]) == 15
    assert lines[1]["request_id"] == lines[2]["request_id"]


def test_log_failure_never_breaks_a_prediction(champion_dir, tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    app = create_app(_settings(champion_dir, tmp_path).__class__(
        model_dir=champion_dir, metrics_path=None, params_path=PROJECT_ROOT / "params.yaml",
        prediction_log=blocked / "predictions.jsonl",
    ))
    with TestClient(app) as c:
        assert c.post("/predict", json=ROW).status_code == 200


def test_logging_can_be_disabled(champion_dir, tmp_path):
    app = create_app(Settings(model_dir=champion_dir, metrics_path=None, params_path=PROJECT_ROOT / "params.yaml", prediction_log=None))
    with TestClient(app) as c:
        assert c.post("/predict", json=ROW).status_code == 200
        assert c.get("/model-info").json()["test_metrics"] == {}


def test_missing_model_keeps_service_up_but_not_ready(tmp_path):
    app = create_app(Settings(model_dir=tmp_path / "nope", metrics_path=None, params_path=PROJECT_ROOT / "params.yaml", prediction_log=None))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=ROW).status_code == 503
        assert c.post("/predict/from-history", json={**{k: v for k, v in ROW.items() if "gap" not in k and k not in {"prev_lap_delta", "pace_decay_slope"}}, "previous_laps": []}).status_code == 503
        assert c.get("/model-info").json()["ready"] is False


def test_service_rejects_a_directory_that_is_not_a_model(tmp_path):
    svc = ModelService(tmp_path, None, PROJECT_ROOT / "params.yaml")
    with pytest.raises(FileNotFoundError):
        svc.load()
    assert svc.ready is False


REAL_MODEL = PROJECT_ROOT / "models" / "champion"


@pytest.mark.skipif(not (REAL_MODEL / "MLmodel").is_file(), reason="models/champion not present (needs dvc pull or the zip)")
def test_real_champion_serves_the_readme_example(tmp_path):
    app = create_app(Settings(model_dir=REAL_MODEL, metrics_path=PROJECT_ROOT / "metrics" / "train_metrics.json",
                              params_path=PROJECT_ROOT / "params.yaml", prediction_log=None))
    with TestClient(app) as c:
        assert c.get("/ready").status_code == 200
        body = c.post("/predict", json=ROW).json()
        assert 0.0 <= body["pit_probability"] <= 1.0 and -1.0 < body["pace_loss_s"] < 10.0
        assert c.get("/model-info").json()["champion_version"] is not None
