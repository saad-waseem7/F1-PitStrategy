from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.diagnose_target import stint_slopes, summarise
from src.data.preprocess import build_dataset


def _stint(session: int, driver: int, stint: int, compound: str, slope: float, n: int = 12) -> pd.DataFrame:
    age = np.arange(n)
    return pd.DataFrame(
        dict(session_key=session, driver_number=driver, stint_number=stint, compound=compound,
             tyre_age=age, pace_loss_s=0.4 + slope * age, pit_recommended=False)
    )


def test_stint_slopes_recovers_known_slopes():
    df = pd.concat([_stint(1, 1, 1, "SOFT", 0.10), _stint(1, 1, 2, "HARD", 0.04)], ignore_index=True)
    out = stint_slopes(df).set_index("compound")["slope"]
    assert out["SOFT"] == pytest.approx(0.10)
    assert out["HARD"] == pytest.approx(0.04)


def test_stint_slopes_skips_short_stints():
    df = _stint(1, 1, 1, "SOFT", 0.1, n=4)
    assert stint_slopes(df).empty


def test_summarise_on_pipeline_output(raw_frames, params):
    df, _ = build_dataset(params=params, **raw_frames)
    result = summarise(df)
    assert result["slope_all"] > 0
    assert 0.0 <= result["pit_rate"] <= 1.0


def _staggered_pit_frame(seed: int, fuel: float, deg: float) -> pd.DataFrame:
    """Drivers pit on different laps, one compound, known fuel and degradation effects."""
    rng = np.random.default_rng(seed)
    frames = []
    for session in range(1, 6):
        for driver in range(1, 11):
            pit = int(rng.integers(14, 30))
            for lap in range(2, 46):
                age = lap if lap <= pit else lap - pit
                frames.append(
                    dict(session_key=session, driver_number=driver, compound="MEDIUM", lap_number=lap,
                         tyre_age=age, pace_loss_s=deg * age - fuel * lap + rng.normal(0, 0.1) + 0.3 * driver)
                )
    return pd.DataFrame(frames)


def test_estimate_time_trend_recovers_known_effects():
    from src.data.diagnose_target import estimate_time_trend

    out = estimate_time_trend(_staggered_pit_frame(0, fuel=0.05, deg=0.07))
    assert out["fuel_per_lap"] == pytest.approx(0.05, abs=0.01)
    assert out["deg_per_lap"] == pytest.approx(0.07, abs=0.01)
    assert out["fuel_se"] > 0 and out["clusters"] == 50


def test_estimate_time_trend_rejects_collinear_data():
    from src.data.diagnose_target import estimate_time_trend

    rows = [dict(session_key=1, driver_number=1, compound="SOFT", lap_number=lap, tyre_age=lap, pace_loss_s=0.1 * lap)
            for lap in range(2, 30)]
    with pytest.raises(ValueError, match="collinear"):
        estimate_time_trend(pd.DataFrame(rows))