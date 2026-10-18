"""Synthetic OpenF1-shaped data so tests never touch the network."""
from __future__ import annotations

import copy
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import load_params
from src.models.train import run
from tests.test_train import _write_processed

T0 = pd.Timestamp("2024-05-05 14:00:00", tz="UTC")
LAP_S = 90.0
N_LAPS = 40
PIT_LAP = 20
SC_DEPLOYED_LAP = 10
SC_END_LAP = 12


@pytest.fixture(scope="session")
def params() -> dict:
    p = load_params()
    p["preprocess"]["train_years"] = [2023, 2024]
    p["preprocess"]["test_years"] = [2025]
    return p


def _build_session(session_key: int, year: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    laps, stints = [], []
    for driver in (1, 44):
        t = T0
        for lap in range(1, N_LAPS + 1):
            stint_lap = lap if lap <= PIT_LAP else lap - PIT_LAP
            duration = LAP_S + 0.08 * stint_lap - 0.03 * lap + rng.normal(0, 0.15)
            if SC_DEPLOYED_LAP <= lap <= SC_END_LAP:
                duration += 25
            if lap == PIT_LAP:
                duration += 3
            if lap == PIT_LAP + 1:
                duration += 20
            laps.append(
                dict(
                    session_key=session_key, driver_number=driver, lap_number=lap,
                    lap_duration=duration, date_start=t, is_pit_out_lap=(lap == PIT_LAP + 1),
                )
            )
            t += pd.Timedelta(seconds=duration)
        stints.append(dict(session_key=session_key, driver_number=driver, stint_number=1,
                           lap_start=1, lap_end=PIT_LAP, compound="MEDIUM", tyre_age_at_start=0))
        stints.append(dict(session_key=session_key, driver_number=driver, stint_number=2,
                           lap_start=PIT_LAP + 1, lap_end=N_LAPS, compound="HARD", tyre_age_at_start=0))

    minutes = pd.date_range(T0 - pd.Timedelta(minutes=2), periods=80, freq="min")
    weather = pd.DataFrame(
        dict(session_key=session_key, date=minutes, track_temperature=35 + 0.1 * np.arange(80),
             air_temperature=25.0, humidity=50.0, rainfall=0, wind_speed=2.0)
    )
    race_control = pd.DataFrame(
        [
            dict(session_key=session_key, date=T0 + pd.Timedelta(minutes=15), lap_number=SC_DEPLOYED_LAP,
                 category="SafetyCar", flag=None, message="SAFETY CAR DEPLOYED"),
            dict(session_key=session_key, date=T0 + pd.Timedelta(minutes=18), lap_number=SC_END_LAP,
                 category="SafetyCar", flag=None, message="SAFETY CAR IN THIS LAP"),
        ]
    )
    sessions = pd.DataFrame(
        [dict(session_key=session_key, session_name="Race", session_type="Race",
              circuit_short_name="Test", country_name="Testland", year=year, date_start=T0)]
    )
    return dict(laps=pd.DataFrame(laps), stints=pd.DataFrame(stints), weather=weather,
                race_control=race_control, sessions=sessions)


@pytest.fixture(scope="session")
def raw_frames() -> dict[str, pd.DataFrame]:
    parts = [_build_session(1001, 2023, 1), _build_session(1002, 2024, 2), _build_session(1003, 2025, 3)]
    return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in parts[0]}


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
