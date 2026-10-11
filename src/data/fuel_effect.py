"""Estimate the fuel effect (seconds per kg) from raw lap times.

Why not pick it by model score: fuel burns off in step with the race lap, so an
over-large correction manufactures a tyre-age-like trend in the label and the
wear model's R2 rises even though the physics is wrong. The effect has to be
identified from data where tyre age and fuel load vary independently, which
happens ACROSS stints (different pit laps, used tyres).

Method (fixed-effects regression, never uses the label):

    lap_time = driver_race_effect + compound_pace[c] + wear[c] * tire_age
               + beta * fuel_load_est + error

The driver-race effect is removed by demeaning within each (race, driver), and
the standard error is clustered by race. ``beta`` is the NET effect of fuel burn
and track evolution (the track gets faster as rubber goes down), in seconds per
kg. If it differs a lot from the 0.03 default, pass the estimate to
``preprocess --fuel-effect``.

Usage:
    python -m src.data.fuel_effect --raw-dir data/raw_openf1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import preprocess as pp

logger = logging.getLogger(__name__)
COMPOUNDS = list(pp.DRY_COMPOUNDS)
MIN_ROWS = 500


def estimate_fuel_effect(raw: pd.DataFrame) -> dict[str, Any]:
    """Return the fuel coefficient, its clustered standard error and wear slopes.

    Raises:
        ValueError: if too few clean laps remain or the design is singular
            (tyre age and fuel are not separable in this data).
    """
    base = pp.add_base_columns(pp.drop_duplicate_laps(raw), fuel_effect=0.0)
    clean = pp.drop_pace_outliers(base[pp.clean_mask(base)])
    if len(clean) < MIN_ROWS:
        raise ValueError(f"Only {len(clean)} clean laps; need at least {MIN_ROWS}")

    onehot = np.stack(
        [(clean["tire_compound"] == c).to_numpy(float) for c in COMPOUNDS], axis=1
    )
    age = clean["tire_age"].to_numpy(float)[:, None] * onehot
    fuel = clean["fuel_load_est"].to_numpy(float)[:, None]
    # Dummies sum to 1, so after demeaning they are collinear: drop the first compound.
    X = np.hstack([age, onehot[:, 1:], fuel])
    y = clean["LapTime"].to_numpy(float)
    group = (clean["race_id"] + "|" + clean["Driver"].astype(str)).to_numpy()

    frame = pd.DataFrame(np.column_stack([y, X]))
    demeaned = frame - frame.groupby(group).transform("mean")
    yd, Xd = demeaned.iloc[:, 0].to_numpy(), demeaned.iloc[:, 1:].to_numpy()
    if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
        raise ValueError("Tyre age and fuel load are not separable in this data")

    gram_inv = np.linalg.inv(Xd.T @ Xd)
    beta = gram_inv @ Xd.T @ yd
    resid = yd - Xd @ beta
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for race in np.unique(clean["race_id"]):
        sel = (clean["race_id"] == race).to_numpy()
        score = Xd[sel].T @ resid[sel]
        meat += np.outer(score, score)
    cov = gram_inv @ meat @ gram_inv
    se = np.sqrt(np.diag(cov))
    return {
        "fuel_s_per_kg": float(beta[-1]),
        "fuel_s_per_kg_se": float(se[-1]),
        "wear_s_per_lap": {c: float(beta[i]) for i, c in enumerate(COMPOUNDS)},
        "n_laps": len(clean),
        "n_races": int(clean["race_id"].nunique()),
        "total_burn_gain_s": float(beta[-1] * pp.FUEL_START_KG),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--min-year", type=int, default=2022)
    args = parser.parse_args(argv)
    raw = pp.load_raw(args.raw_dir)
    raw = raw[raw["Year"] >= args.min_year]
    result = estimate_fuel_effect(raw)
    lo = result["fuel_s_per_kg"] - 1.96 * result["fuel_s_per_kg_se"]
    hi = result["fuel_s_per_kg"] + 1.96 * result["fuel_s_per_kg_se"]
    logger.info(
        "Fuel effect: %.4f s/kg (95%% CI %.4f to %.4f)", result["fuel_s_per_kg"], lo, hi
    )
    logger.info(
        "Implied gain over a full %d kg load: %.2f s",
        pp.FUEL_START_KG,
        result["total_burn_gain_s"],
    )
    logger.info("Wear slopes by compound (s/lap): %s", result["wear_s_per_lap"])
    logger.info("Based on %d laps from %d races", result["n_laps"], result["n_races"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
