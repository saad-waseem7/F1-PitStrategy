"""The CI smoke test depends on this builder, so it must keep working."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.api.service import ModelService
from src.config import PROJECT_ROOT
from tests.make_ci_champion import build


def test_ci_champion_is_a_loadable_model(tmp_path: Path):
    out = build(tmp_path / "champion")
    assert (out / "MLmodel").is_file() and (out / "requirements.txt").is_file()
    svc = ModelService(out, None, PROJECT_ROOT / "params.yaml")
    svc.load()
    assert svc.ready and 0 < svc.threshold < 1
    assert isinstance(svc.predict(pd.DataFrame([{
        "tyre_age": 5, "compound": "SOFT", "stint_number": 1, "lap_number": 8, "race_progress": 0.1,
        "track_temperature": 40, "air_temperature": 25, "humidity": 50, "rainfall": 0, "wind_speed": 1,
        "track_temp_delta": None, "prev_lap_delta": None, "pace_decay_slope": None,
        "prev_gap_to_best": None, "recent_gap_to_best": None,
    }])), pd.DataFrame)
