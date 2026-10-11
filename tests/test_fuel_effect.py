"""The fuel-effect estimator must recover a known effect from synthetic races."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import fuel_effect as fe
from src.data import preprocess as pp

TOTAL = 60
DEG = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}
PACE = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}


def make_race(race: int, true_fuel: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in range(10):
        pit = int(rng.integers(15, 40))
        first, second = rng.choice(list(DEG), size=2, replace=False)
        start_age = int(rng.integers(0, 6))  # used second set varies the age/lap split
        driver_pace = float(rng.normal(90, 0.5))
        for lap in range(1, TOTAL + 1):
            stint = 1 if lap <= pit else 2
            comp = first if stint == 1 else second
            age = lap if stint == 1 else lap - pit + start_age
            fuel = max(pp.FUEL_START_KG * (1 - (lap - 1) / TOTAL), 0.0)
            rows.append(
                {
                    "Year": 2024, "RoundNumber": race + 1, "EventName": f"GP{race}",
                    "TotalLaps": TOTAL, "Driver": f"D{d}", "LapNumber": float(lap),
                    "LapTime": driver_pace + PACE[comp] + DEG[comp] * age + true_fuel * fuel + rng.normal(0, 0.1),
                    "Stint": float(stint), "Compound": comp, "TyreLife": float(age),
                    "TrackStatus": "1", "IsInLap": lap == pit, "IsOutLap": lap == pit + 1,
                    "IsAccurate": True, "Deleted": False,
                    "AirTemp": 25.0, "TrackTemp": 40.0, "Rainfall": False,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.parametrize("true_fuel", [0.02, 0.035, 0.05])
def test_recovers_known_fuel_effect(true_fuel) -> None:
    rng = np.random.default_rng(1)
    raw = pd.concat([make_race(r, true_fuel, rng) for r in range(8)], ignore_index=True)
    out = fe.estimate_fuel_effect(raw)
    assert out["fuel_s_per_kg"] == pytest.approx(true_fuel, abs=0.004)
    assert out["fuel_s_per_kg_se"] > 0
    assert out["total_burn_gain_s"] == pytest.approx(
        true_fuel * pp.FUEL_START_KG, abs=0.4
    )
    for comp, slope in DEG.items():
        assert out["wear_s_per_lap"][comp] == pytest.approx(slope, abs=0.01)


def test_too_little_data_raises() -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="clean laps"):
        fe.estimate_fuel_effect(make_race(0, 0.03, rng).head(100))


def test_unseparable_design_raises() -> None:
    """One compound, one pit lap, no used tyres: age and lap are the same variable."""
    rows = []
    for race in range(6):
        for d in range(6):
            for lap in range(1, TOTAL + 1):
                fuel = max(pp.FUEL_START_KG * (1 - (lap - 1) / TOTAL), 0.0)
                rows.append(
                    {
                        "Year": 2024, "RoundNumber": race + 1, "EventName": "G", "TotalLaps": TOTAL,
                        "Driver": f"D{d}", "LapNumber": float(lap), "LapTime": 90 + 0.03 * fuel + 0.05 * lap,
                        "Stint": 1.0, "Compound": "SOFT", "TyreLife": float(lap), "TrackStatus": "1",
                        "IsInLap": False, "IsOutLap": False, "AirTemp": 25.0, "TrackTemp": 40.0, "Rainfall": False,
                    }
                )  # fmt: skip
    with pytest.raises(ValueError, match="not separable"):
        fe.estimate_fuel_effect(pd.DataFrame(rows))
