"""Compare target definitions on the real raw data, in memory (no DVC outputs are touched).

For each (reference_scope, fuel correction) pair it rebuilds the dataset and reports:
  pit_rate      share of laps flagged pit_recommended
  pooled_corr   Pearson correlation of tyre_age and pace_loss_s over all laps
  slope_*       median per-stint OLS slope of pace_loss_s vs tyre_age, in seconds per lap
                (tyre degradation; expect SOFT > MEDIUM > HARD, all clearly positive)

Within one stint, tyre age and lap number are collinear, so the data cannot identify the fuel
coefficient by itself. The table shows sensitivity; the choice rests on the physical prior.

Usage:
    python -m src.data.diagnose_target
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from src.config import TARGET_CLASSIFICATION, TARGET_REGRESSION, load_params
from src.data.ingest import load_raw
from src.data.preprocess import build_dataset

STINT_KEY = ["session_key", "driver_number", "stint_number"]
MIN_STINT_POINTS = 8
FUEL_GRID = (0.03, 0.05, 0.07)
SCOPES = ("driver_race", "stint")


def stint_slopes(df: pd.DataFrame, min_points: int = MIN_STINT_POINTS) -> pd.DataFrame:
    """OLS slope of pace_loss_s on tyre_age for each stint, plus the stint's compound."""
    work = df[[*STINT_KEY, "compound", "tyre_age", TARGET_REGRESSION]].copy()
    work["compound"] = work["compound"].astype(str)
    grp = work.groupby(STINT_KEY)
    x_dev = work["tyre_age"] - grp["tyre_age"].transform("mean")
    y_dev = work[TARGET_REGRESSION] - grp[TARGET_REGRESSION].transform("mean")
    work["xy"] = x_dev * y_dev
    work["xx"] = x_dev**2
    agg = work.groupby(STINT_KEY).agg(xy=("xy", "sum"), xx=("xx", "sum"), n=("xx", "size"), compound=("compound", "first"))
    agg = agg[(agg["n"] >= min_points) & (agg["xx"] > 0)]
    agg["slope"] = agg["xy"] / agg["xx"]
    return agg[["slope", "compound"]].reset_index()


def summarise(df: pd.DataFrame) -> dict[str, float]:
    slopes = stint_slopes(df)
    out: dict[str, float] = {
        "rows": float(len(df)),
        "pit_rate": float(df[TARGET_CLASSIFICATION].mean()),
        "pooled_corr": float(df["tyre_age"].corr(df[TARGET_REGRESSION])),
        "slope_all": float(slopes["slope"].median()) if len(slopes) else float("nan"),
    }
    for compound in ("SOFT", "MEDIUM", "HARD"):
        sub = slopes.loc[slopes["compound"] == compound, "slope"]
        out[f"slope_{compound}"] = float(sub.median()) if len(sub) else float("nan")
    return out


def estimate_time_trend(df: pd.DataFrame) -> dict[str, float]:
    """Estimate tyre degradation and the per-lap time gain from fuel burn plus track evolution.

    Regression of lap time on tyre_age and lap_number with fixed effects per driver-and-race
    (within transformation) and compound dummies. Identification comes from drivers pitting on
    different laps, which decouples tyre age from lap number. Standard errors are clustered by
    driver-and-race. Use a dataset built with fuel_correction_s_per_lap = 0, so the target holds
    uncorrected lap time (minus a per-driver-race constant, which the fixed effects remove).

    Returns deg_per_lap, fuel_per_lap (= minus the lap_number coefficient) and their SEs.
    Raises ValueError if tyre age and lap number cannot be separated.
    """
    keys = ["session_key", "driver_number"]
    work = df[[*keys, "compound", "tyre_age", "lap_number", TARGET_REGRESSION]].copy()
    dummies = pd.get_dummies(work["compound"].astype(str), prefix="c", drop_first=True, dtype=float)
    regressors = pd.concat([work[["tyre_age", "lap_number"]].astype(float), dummies], axis=1)
    by_group = [work[k] for k in keys]
    x = (regressors - regressors.groupby(by_group).transform("mean")).to_numpy()
    y_series = work[TARGET_REGRESSION].astype(float)
    y = (y_series - y_series.groupby(by_group).transform("mean")).to_numpy()

    xtx = x.T @ x
    if np.linalg.matrix_rank(xtx) < xtx.shape[0]:
        raise ValueError("tyre_age and lap_number are collinear in this data; cannot separate them")
    bread = np.linalg.inv(xtx)
    beta = bread @ x.T @ y
    scores = x * (y - x @ beta)[:, None]
    codes = work.groupby(keys).ngroup().to_numpy()
    cluster_scores = np.zeros((codes.max() + 1, x.shape[1]))
    np.add.at(cluster_scores, codes, scores)
    cov = bread @ (cluster_scores.T @ cluster_scores) @ bread
    se = np.sqrt(np.diag(cov))
    return {
        "deg_per_lap": float(beta[0]),
        "deg_se": float(se[0]),
        "fuel_per_lap": float(-beta[1]),
        "fuel_se": float(se[1]),
        "n": float(len(work)),
        "clusters": float(codes.max() + 1),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    params = load_params()
    raw = {ep: load_raw(ep) for ep in ("laps", "stints", "weather", "race_control", "sessions")}

    trial = copy.deepcopy(params)
    trial["preprocess"].update(reference_scope="driver_race", fuel_correction_s_per_lap=0.0)
    uncorrected, _ = build_dataset(params=trial, **raw)
    est = estimate_time_trend(uncorrected)
    print("Estimated from data (fixed effects per driver-race, clustered SE):")
    print(f"  fuel burn + track evolution: {est['fuel_per_lap']:.3f} s/lap  (95% CI +/- {1.96 * est['fuel_se']:.3f})")
    print(f"  mean tyre degradation:       {est['deg_per_lap']:.3f} s/lap  (95% CI +/- {1.96 * est['deg_se']:.3f})")
    print(f"  laps: {int(est['n'])}, driver-races: {int(est['clusters'])}\n")

    rows = []
    for scope in SCOPES:
        for fuel in FUEL_GRID:
            trial = copy.deepcopy(params)
            trial["preprocess"]["reference_scope"] = scope
            trial["preprocess"]["fuel_correction_s_per_lap"] = fuel
            df, _ = build_dataset(params=trial, **raw)
            rows.append({"scope": scope, "fuel_s_per_lap": fuel, **summarise(df)})
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    if np.isnan(table["slope_all"]).any():
        print("WARNING: some configurations had no stints with enough laps")


if __name__ == "__main__":
    main()