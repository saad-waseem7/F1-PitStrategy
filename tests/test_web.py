"""The web page, the replay endpoints and the drift summary endpoint."""
from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.monitoring import read_drift_summary
from src.api.settings import Settings
from src.config import FEATURE_COLUMNS, PROJECT_ROOT
from src.replay import build_sample
from src.replay.store import ReplayError, ReplayStore
from tests.test_train import _write_processed

WEB_DIR = PROJECT_ROOT / "web"


@pytest.fixture(scope="session")
def sample_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("replay")
    _write_processed(root / "processed")
    out = root / "sample.json.gz"
    build_sample.build(root / "processed" / "test.parquet", out, races=2)
    return out


def _settings(champion: Path, tmp_path: Path, **kw) -> Settings:
    base = dict(
        model_dir=champion, metrics_path=champion.parent.parent / "metrics" / "train_metrics.json",
        params_path=PROJECT_ROOT / "params.yaml", prediction_log=tmp_path / "logs" / "predictions.jsonl",
        web_dir=WEB_DIR, replay_path=None, drift_summary_path=tmp_path / "drift_summary.json",
    )
    return Settings(**{**base, **kw})


def _client(champion: Path, tmp_path: Path, **kw):
    return TestClient(create_app(_settings(champion, tmp_path, **kw)))


# ------------------------------------------------------------------ page
def test_home_serves_the_page_with_security_headers(champion_dir, tmp_path):
    with _client(champion_dir, tmp_path) as c:
        r = c.get("/")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "<title>Pit Wall</title>" in r.text
        assert "default-src 'self'" in r.headers["content-security-policy"]
        assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"
        js = c.get("/assets/app.js")
        assert js.status_code == 200 and "javascript" in js.headers["content-type"]
        assert "content-security-policy" in js.headers
        assert c.get("/assets/style.css").status_code == 200
        assert c.get("/assets/../index.html").status_code in (200, 404)  # never escapes the assets directory


def test_api_responses_get_nosniff_but_docs_keep_their_own_policy(champion_dir, tmp_path):
    with _client(champion_dir, tmp_path) as c:
        assert c.get("/health").headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" not in c.get("/health").headers
        assert "content-security-policy" not in c.get("/docs").headers  # Swagger UI needs inline scripts


def test_missing_web_directory_returns_404_not_a_crash(champion_dir, tmp_path):
    with _client(champion_dir, tmp_path, web_dir=tmp_path / "nope") as c:
        assert c.get("/").status_code == 404
        assert c.get("/health").status_code == 200


def test_page_code_never_parses_strings_as_html():
    code = (WEB_DIR / "assets" / "app.js").read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in code, banned
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html), "inline scripts are blocked by the CSP"
    assert "style=" not in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_script_has_valid_syntax():
    result = subprocess.run(["node", "--check", str(WEB_DIR / "assets" / "app.js")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------ replay
def test_replay_reports_unavailable_without_a_sample(champion_dir, tmp_path):
    with _client(champion_dir, tmp_path, replay_path=tmp_path / "missing.json.gz") as c:
        assert c.get("/replay/sessions").json() == {"available": False, "meta": {}, "sessions": []}
        assert c.get("/replay/stint", params=dict(session_key=1, driver_number=1, stint_number=1)).status_code == 404


def test_corrupt_sample_disables_replay_but_the_service_stays_up(champion_dir, tmp_path):
    bad = tmp_path / "bad.json.gz"
    bad.write_bytes(b"not gzip")
    with _client(champion_dir, tmp_path, replay_path=bad) as c:
        assert c.get("/replay/sessions").json()["available"] is False
        assert c.get("/ready").status_code == 200


def test_replay_lists_sessions_and_scores_a_stint_like_the_model(champion_dir, tmp_path, sample_path):
    with _client(champion_dir, tmp_path, replay_path=sample_path) as c:
        listing = c.get("/replay/sessions").json()
        assert listing["available"] and listing["meta"]["split"] == "test" and len(listing["sessions"]) == 2
        sess = listing["sessions"][0]
        driver = sess["drivers"][0]
        stint = driver["stints"][0]
        r = c.get("/replay/stint", params=dict(
            session_key=sess["session_key"], driver_number=driver["driver_number"], stint_number=stint["stint_number"]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["laps"]) == stint["laps"] and body["circuit"] == sess["circuit"]
        # The page must show exactly what the model says, so recompute through the service.
        store = ReplayStore(sample_path)
        store.load()
        rows = store.stint(sess["session_key"], driver["driver_number"], stint["stint_number"])
        expected = c.app.state.service.predict(rows[list(FEATURE_COLUMNS)])
        got = pd.DataFrame(body["laps"])
        np.testing.assert_allclose(got["predicted_pace_loss_s"], expected["pace_loss_s"], atol=1e-9)
        np.testing.assert_allclose(got["pit_probability"], expected["pit_probability"], atol=1e-9)
        assert list(got["pit_recommended"]) == list(expected["pit_recommended"])
        assert list(got["lap_number"]) == sorted(got["lap_number"])
        assert body["threshold"] == c.app.state.service.threshold


def test_replay_does_not_pollute_the_prediction_log(champion_dir, tmp_path, sample_path):
    with _client(champion_dir, tmp_path, replay_path=sample_path) as c:
        sess = c.get("/replay/sessions").json()["sessions"][0]
        d, st = sess["drivers"][0], sess["drivers"][0]["stints"][0]
        assert c.get("/replay/stint", params=dict(
            session_key=sess["session_key"], driver_number=d["driver_number"], stint_number=st["stint_number"])).status_code == 200
    assert not (tmp_path / "logs" / "predictions.jsonl").exists()


def test_replay_rejects_unknown_stints_and_bad_parameters(champion_dir, tmp_path, sample_path):
    with _client(champion_dir, tmp_path, replay_path=sample_path) as c:
        assert c.get("/replay/stint", params=dict(session_key=999999, driver_number=1, stint_number=1)).status_code == 404
        assert c.get("/replay/stint", params=dict(session_key=-1, driver_number=1, stint_number=1)).status_code == 422
        assert c.get("/replay/stint", params=dict(session_key="x", driver_number=1, stint_number=1)).status_code == 422
        assert c.get("/replay/stint").status_code == 422


# ------------------------------------------------------------------ sample builder
def test_sample_is_deterministic_and_keeps_nulls(tmp_path):
    _write_processed(tmp_path / "p")
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    meta = build_sample.build(tmp_path / "p" / "test.parquet", a, races=2)
    build_sample.build(tmp_path / "p" / "test.parquet", b, races=2)
    assert a.read_bytes() == b.read_bytes()
    assert meta["split"] == "test" and meta["years"] == [2025] and len(meta["sessions"]) == 2
    payload = json.loads(gzip.decompress(a.read_bytes()))
    assert any(v is None for v in payload["data"]["prev_lap_delta"])  # NaN is stored as null, never as NaN
    store = ReplayStore(a)
    store.load()
    assert store.available and store._frame["prev_lap_delta"].isna().any()


def test_sample_builder_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_sample.build(tmp_path / "nope.parquet", tmp_path / "o.json.gz", races=1)
    _write_processed(tmp_path / "p")
    with pytest.raises(ValueError):
        build_sample.build(tmp_path / "p" / "test.parquet", tmp_path / "o.json.gz", races=0)
    assert build_sample.main(["--input", str(tmp_path / "nope.parquet"), "--output", str(tmp_path / "o.json.gz")]) == 1


def test_store_rejects_a_sample_with_missing_columns(tmp_path):
    path = tmp_path / "s.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"version": 1, "meta": {}, "columns": [], "data": {}}).encode()))
    with pytest.raises(ReplayError):
        ReplayStore(path).load()


# ------------------------------------------------------------------ monitoring
def test_drift_summary_states(champion_dir, tmp_path):
    summary = tmp_path / "drift_summary.json"
    with _client(champion_dir, tmp_path) as c:
        assert c.get("/monitoring/summary").json()["status"] == "no_report"
        summary.write_text("{broken", encoding="utf-8")
        assert c.get("/monitoring/summary").json()["status"] == "unreadable"
        summary.write_text(json.dumps({
            "status": "ok", "drift_detected": True, "drifted_columns": ["humidity"], "share_drifted": 0.25,
            "n_current": 400, "n_reference": 20000, "secret": "not exposed",
            "columns": {"humidity": {"method": "wasserstein", "value": float("nan"), "threshold": 0.2, "drifted": True}},
        }), encoding="utf-8")  # json.dumps writes NaN, which a JSON response must not echo
        r = c.get("/monitoring/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["drift_detected"] is True and body["drifted_columns"] == ["humidity"]
        assert body["columns"]["humidity"]["value"] is None and "secret" not in body


def test_read_drift_summary_rejects_non_objects(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("[1, 2]", encoding="utf-8")
    assert read_drift_summary(path) == {"status": "unreadable"}


# ------------------------------------------------------------------ replay summary
def _scored(actual, pred, label, prob, flag, compound="MEDIUM", session=1):
    return pd.DataFrame({
        "session_key": session, "circuit_short_name": "T", "year": 2025, "compound": compound,
        "pace_loss_s": actual, "predicted_pace_loss_s": pred, "pit_recommended": label,
        "pit_probability": prob, "predicted_flag": flag,
    })


def test_metrics_match_hand_computed_values():
    from src.replay.summary import metrics

    m = metrics(_scored([1.0, 2.0, 3.0, 4.0], [1.5, 2.0, 2.0, 5.0], [False, False, True, True],
                        [0.1, 0.6, 0.4, 0.9], [False, True, False, True]))
    assert m["laps"] == 4 and m["mae"] == pytest.approx(0.625) and m["bias"] == pytest.approx(0.125)
    assert m["rmse"] == pytest.approx(np.sqrt((0.25 + 0 + 1 + 1) / 4))
    assert m["r2"] == pytest.approx(1 - 2.25 / 5.0)
    assert m["flag_accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == pytest.approx(0.5)
    assert m["prevalence"] == 0.5 and 0 < m["pr_auc"] <= 1


def test_metrics_report_none_when_the_data_cannot_support_them():
    from src.replay.summary import metrics

    m = metrics(_scored([1.0, 1.0], [1.0, 1.2], [False, False], [0.1, 0.2], [False, False]))
    assert m["r2"] is None and m["pr_auc"] is None and m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["flag_accuracy"] == 1.0 and m["bias"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        metrics(_scored([], [], [], [], []))


def test_summary_endpoint_matches_a_direct_computation(champion_dir, tmp_path, sample_path):
    from src.replay.summary import metrics

    with _client(champion_dir, tmp_path, replay_path=sample_path) as c:
        r = c.get("/replay/summary")
        assert r.status_code == 200, r.text
        body = r.json()
        svc = c.app.state.service
        store = ReplayStore(sample_path)
        store.load()
        rows = store.rows()
        preds = svc.predict(rows[list(FEATURE_COLUMNS)])
        expected = metrics(rows.assign(
            predicted_pace_loss_s=preds["pace_loss_s"].to_numpy(), pit_probability=preds["pit_probability"].to_numpy(),
            predicted_flag=preds["pit_recommended"].to_numpy(dtype=bool)))
        assert body["available"] and body["threshold"] == svc.threshold
        assert body["overall"]["laps"] == len(rows) == sum(s["metrics"]["laps"] for s in body["by_session"])
        assert body["overall"]["mae"] == pytest.approx(expected["mae"]) and body["overall"]["bias"] == pytest.approx(expected["bias"])
        assert len(body["by_session"]) == 2 and {x["compound"] for x in body["by_compound"]} <= {"SOFT", "MEDIUM", "HARD"}
        assert c.get("/replay/summary").json() == body  # cached, identical


def test_summary_is_unavailable_without_a_sample_and_not_logged(champion_dir, tmp_path):
    with _client(champion_dir, tmp_path, replay_path=tmp_path / "missing.json.gz") as c:
        assert c.get("/replay/summary").json() == {
            "available": False, "threshold": None, "overall": None, "by_session": [], "by_compound": []}
    assert not (tmp_path / "logs" / "predictions.jsonl").exists()


def test_builder_keeps_every_race_by_default(tmp_path):
    _write_processed(tmp_path / "p")
    meta = build_sample.build(tmp_path / "p" / "test.parquet", tmp_path / "all.json.gz")
    every = pd.read_parquet(tmp_path / "p" / "test.parquet")
    assert meta["sessions"] == sorted(int(k) for k in every["session_key"].unique()) and meta["rows"] == len(every)
    assert build_sample.build(tmp_path / "p" / "test.parquet", tmp_path / "big.json.gz", races=999)["rows"] == len(every)
    assert build_sample.main(["--input", str(tmp_path / "p" / "test.parquet"), "--output", str(tmp_path / "cli.json.gz")]) == 0


# ------------------------------------------------------------------ readable names and rebuild-free edits
APP_JS = (WEB_DIR / "assets" / "app.js").read_text(encoding="utf-8")


def _labels() -> dict[str, str]:
    block = re.search(r"const FEATURE_LABELS = \{(.*?)\n\};", APP_JS, flags=re.S).group(1)
    return dict(re.findall(r'(\w+): "([^"]+)"', block))


def test_every_model_feature_has_a_readable_label():
    labels = _labels()
    assert set(FEATURE_COLUMNS) <= set(labels), set(FEATURE_COLUMNS) - set(labels)
    for name, label in labels.items():
        assert "_" not in label and label[0].isupper(), (name, label)
        assert all(w[0].isupper() or w in {"To", "Of", "And", "The", "to", "of", "and", "the"} for w in label.split()), label


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_label_helpers_never_leak_variable_names():
    start = APP_JS.index("// ---------------------------------------------------------------- readable names")
    end = APP_JS.index("// ---------------------------------------------------------------- DOM helpers")
    script = APP_JS[start:end] + """
      const out = [featureLabel("tyre_age"), featureLabel("recent_gap_to_best"), featureLabel("some_new_feature"),
        humanizeText("lap_number: Input should be less than or equal to 120"),
        humanizeText("previous_laps must be ordered by strictly increasing lap_number"), humanizeText("no names here")];
      console.log(JSON.stringify(out));
    """
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "Tyre Age", "Recent Gap To Best", "Some New Feature",
        "Lap Number: Input should be less than or equal to 120",
        "Earlier Laps must be ordered by strictly increasing Lap Number", "no names here",
    ]


def test_ui_is_revalidated_on_every_load_so_edits_show_without_a_rebuild(champion_dir, tmp_path):
    with _client(champion_dir, tmp_path) as c:
        for path in ("/", "/assets/app.js", "/assets/style.css"):
            assert c.get(path).headers["cache-control"] == "no-cache", path
        assert "cache-control" not in c.get("/health").headers


def test_edited_page_files_are_served_as_they_are_on_disk(champion_dir, tmp_path):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<title>one</title>", encoding="utf-8")
    (web / "assets" / "app.js").write_text("// one", encoding="utf-8")
    with _client(champion_dir, tmp_path, web_dir=web) as c:
        assert "one" in c.get("/").text and "one" in c.get("/assets/app.js").text
        (web / "index.html").write_text("<title>two</title>", encoding="utf-8")
        (web / "assets" / "app.js").write_text("// two, longer", encoding="utf-8")
        assert "two" in c.get("/").text and "two" in c.get("/assets/app.js").text  # no restart, no rebuild


def test_compose_mounts_the_page_and_the_replay_sample_from_the_checkout():
    import yaml

    volumes = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]["api"]["volumes"]
    assert "./web:/app/web:ro" in volumes and "./replay:/app/replay:ro" in volumes


def test_wet_warning_names_no_variable():
    from src.api.app import NO_HISTORY_WARNING, WET_WARNING

    assert not re.search(r"\b[a-z]+_[a-z_]+\b", WET_WARNING + NO_HISTORY_WARNING)
