"""Turn raw OpenF1 parquet files into a model-ready lap-level dataset.

Pipeline (one row per clean race lap):
  1. Drop unusable laps: lap 1, pit-in/pit-out laps, safety car / VSC laps, slow outliers.
  2. Join stint info (compound, tyre age) and weather (as-of join on lap start time).
  3. Build targets:
       pace_loss_s      = fuel-corrected lap time - driver's reference pace in that race
       pit_recommended  = trailing mean of pace_loss_s over N laps >= threshold
  4. Build serving-safe features. Every rolling feature uses only PAST laps (shifted by 1),
     so the same inputs can be supplied by the API at prediction time.

Usage:
    python -m src.data.preprocess
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import (
    COMPOUNDS,
    FEATURE_COLUMNS,
    METRICS_DIR,
    PROCESSED_DIR,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    load_params,
)
from src.data.ingest import load_raw

LOG = logging.getLogger(__name__)

ID_COLUMNS: tuple[str, ...] = ("session_key", "driver_number", "year", "circuit_short_name")
WET_COMPOUNDS: tuple[str, ...] = ("INTERMEDIATE", "WET")
REFERENCE_SCOPES: tuple[str, ...] = ("stint", "driver_race")
LAP_KEY: list[str] = ["session_key", "driver_number"]
WEATHER_TOLERANCE = pd.Timedelta(minutes=10)


# --------------------------------------------------------------------------- helpers
def _to_utc_us(series: pd.Series) -> pd.Series:
    """Normalise to datetime64[us, UTC] so merge_asof never sees mismatched dtypes."""
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[us, UTC]")


def trailing_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope over the trailing `window` points ending at each index (NaN if not enough data
    or any NaN in the window). Output has same length as input."""
    n = len(values)
    out = np.full(n, np.nan)
    if window < 2 or n < window:
        return out
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    denom = float((xc**2).sum())
    windows = sliding_window_view(values.astype(float), window)  # (n-window+1, window)
    out[window - 1 :] = windows @ xc / denom  # NaN propagates automatically
    return out


def safety_car_laps(race_control: pd.DataFrame) -> set[tuple[int, int]]:
    """Return {(session_key, lap_number)} for laps under SC/VSC.

    A neutralisation starts at a 'DEPLOYED' message and ends at 'IN THIS LAP' (SC) or 'ENDING'
    (VSC). The lap after the end is included too: it is the slow restart lap.
    Unmatched starts only mark their own lap; the outlier filter handles the rest.
    """
    if race_control.empty:
        return set()
    rc = race_control[race_control["category"].astype("string") == "SafetyCar"].copy()
    if rc.empty:
        return set()
    rc["date"] = _to_utc_us(rc["date"])
    rc = rc.dropna(subset=["date", "lap_number"]).sort_values(["session_key", "date"])
    rc["message"] = rc["message"].astype("string").str.upper().fillna("")

    result: set[tuple[int, int]] = set()
    for session_key, grp in rc.groupby("session_key"):
        start: int | None = None
        for lap, msg in zip(grp["lap_number"].astype(int), grp["message"]):
            if "DEPLOYED" in msg:
                start = lap if start is None else start
            elif "IN THIS LAP" in msg or "ENDING" in msg:
                if start is not None:
                    result.update((int(session_key), x) for x in range(start, lap + 2))
                    start = None
        if start is not None:
            result.add((int(session_key), start))
    return result


# --------------------------------------------------------------------------- steps
def _prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    df["lap_number"] = pd.to_numeric(df["lap_number"], errors="coerce")
    df["lap_duration"] = pd.to_numeric(df["lap_duration"], errors="coerce")
    df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
    df = df.dropna(subset=["lap_number", "driver_number", "date_start"]).copy()
    df[["lap_number", "driver_number", "session_key"]] = df[["lap_number", "driver_number", "session_key"]].astype(int)
    df["date_start"] = _to_utc_us(df["date_start"])
    df = df.dropna(subset=["date_start"])
    df["is_pit_out_lap"] = df["is_pit_out_lap"].fillna(False).astype(bool)
    df = df.drop_duplicates(subset=[*LAP_KEY, "lap_number"], keep="last")
    df = df.sort_values([*LAP_KEY, "lap_number"]).reset_index(drop=True)
    # Pit-in lap = the lap right before a pit-out lap for the same driver.
    next_is_out = df.groupby(LAP_KEY)["is_pit_out_lap"].shift(-1)
    next_lap = df.groupby(LAP_KEY)["lap_number"].shift(-1)
    df["is_pit_in_lap"] = (next_is_out.fillna(False).astype(bool)) & (next_lap == df["lap_number"] + 1)
    df["race_laps"] = df.groupby("session_key")["lap_number"].transform("max")
    return df


def _attach_stints(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    st = stints.copy()
    for col in ("driver_number", "stint_number", "lap_start", "lap_end", "tyre_age_at_start"):
        st[col] = pd.to_numeric(st[col], errors="coerce")
    st = st.dropna(subset=["driver_number", "stint_number"]).copy()
    st["lap_start"] = st["lap_start"].fillna(1)
    st["lap_end"] = st["lap_end"].fillna(10_000)  # open-ended final stint
    st["tyre_age_at_start"] = st["tyre_age_at_start"].fillna(0)
    st["compound"] = st["compound"].astype("string").str.upper()
    st = st[st["compound"].isin(COMPOUNDS)]
    st["driver_number"] = st["driver_number"].astype(int)
    st["session_key"] = st["session_key"].astype(int)

    merged = laps.merge(st, on=LAP_KEY, how="inner")
    merged = merged[(merged["lap_number"] >= merged["lap_start"]) & (merged["lap_number"] <= merged["lap_end"])]
    # Overlapping/duplicated stints in raw data: keep the latest stint per lap.
    merged = merged.sort_values([*LAP_KEY, "lap_number", "stint_number"]).drop_duplicates(
        subset=[*LAP_KEY, "lap_number"], keep="last"
    )
    merged["tyre_age"] = merged["tyre_age_at_start"] + (merged["lap_number"] - merged["lap_start"])
    merged["stint_number"] = merged["stint_number"].astype(int)
    return merged.drop(columns=["lap_start", "lap_end", "tyre_age_at_start"])


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["date"] = _to_utc_us(w["date"])
    num_cols = ["track_temperature", "air_temperature", "humidity", "rainfall", "wind_speed"]
    for c in num_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["date"]).sort_values("date")
    left = laps.sort_values("date_start")
    out = pd.merge_asof(
        left,
        w[["session_key", "date", *num_cols]],
        left_on="date_start",
        right_on="date",
        by="session_key",
        direction="backward",
        tolerance=WEATHER_TOLERANCE,
    )
    return out.drop(columns=["date"]).sort_values([*LAP_KEY, "lap_number"]).reset_index(drop=True)


def _add_track_temp_delta(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    """track_temperature now minus track_temperature `lag` laps earlier (same driver)."""
    past = df[[*LAP_KEY, "lap_number", "track_temperature"]].copy()
    past["lap_number"] += lag
    past = past.rename(columns={"track_temperature": "track_temp_lagged"})
    out = df.merge(past, on=[*LAP_KEY, "lap_number"], how="left")
    out["track_temp_delta"] = out["track_temperature"] - out["track_temp_lagged"]
    return out.drop(columns=["track_temp_lagged"])


def _drop_wet(df: pd.DataFrame, max_fraction: float) -> pd.DataFrame:
    """Drop sessions where wet tyres or rain cover >= max_fraction of laps, then any stray
    wet-tyre laps left in dry sessions."""
    wet_tyre = df["compound"].isin(WET_COMPOUNDS)
    rainy = pd.to_numeric(df["rainfall"], errors="coerce").fillna(0) > 0
    wet_share = (wet_tyre | rainy).groupby(df["session_key"]).transform("mean")
    return df[(wet_share < max_fraction) & ~wet_tyre].copy()


def _filter_clean_laps(df: pd.DataFrame, sc_laps: set[tuple[int, int]], p: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    report: dict[str, int] = {"laps_before_filtering": len(df)}
    df = df.dropna(subset=["lap_duration", "compound", "track_temperature"]).copy()
    report["after_dropna"] = len(df)

    sessions_before = int(df["session_key"].nunique())
    if p.get("dry_only", True):
        df = _drop_wet(df, float(p["wet_session_fraction"]))
    report["after_drop_wet"] = len(df)
    report["wet_sessions_dropped"] = sessions_before - int(df["session_key"].nunique())

    df = df[df["lap_number"] > 1]
    report["after_drop_lap1"] = len(df)

    df = df[~(df["is_pit_out_lap"] | df["is_pit_in_lap"])]
    report["after_drop_pit_laps"] = len(df)

    keys = pd.MultiIndex.from_arrays([df["session_key"], df["lap_number"]])
    sc_index = pd.MultiIndex.from_tuples(sorted(sc_laps), names=["session_key", "lap_number"]) if sc_laps else None
    df = df[~keys.isin(sc_index)] if sc_index is not None else df
    report["after_drop_safety_car"] = len(df)

    median = df.groupby(LAP_KEY)["lap_duration"].transform("median")
    df = df[df["lap_duration"] <= p["outlier_ratio"] * median]
    report["after_drop_slow_outliers"] = len(df)
    return df.copy(), report


def _add_targets_and_features(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    df = df.sort_values([*LAP_KEY, "lap_number"]).copy()
    df["fuel_corrected_s"] = df["lap_duration"] + p["fuel_correction_s_per_lap"] * df["lap_number"]

    scope = p.get("reference_scope", "stint")
    if scope not in REFERENCE_SCOPES:
        raise ValueError(f"reference_scope must be one of {REFERENCE_SCOPES}, got {scope!r}")
    stint_key = [*LAP_KEY, "stint_number"]

    counts = df.groupby(LAP_KEY)["lap_number"].transform("count")
    df = df[counts >= p["min_laps_per_driver"]].copy()
    stint_counts = df.groupby(stint_key)["lap_number"].transform("count")
    df = df[stint_counts >= p["min_stint_laps"]].copy()

    ref_key = stint_key if scope == "stint" else LAP_KEY
    reference = df.groupby(ref_key)["fuel_corrected_s"].transform(lambda s: s.quantile(p["reference_quantile"]))
    df[TARGET_REGRESSION] = df["fuel_corrected_s"] - reference

    window = int(p["pit_rolling_window"])
    rolling_loss = df.groupby(stint_key)[TARGET_REGRESSION].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df[TARGET_CLASSIFICATION] = rolling_loss >= p["pit_loss_threshold_s"]
    df = df[rolling_loss.notna()].copy()  # label undefined for the first laps of a stint

    # Serving-safe features: use laps strictly before the current one, within the same stint.
    df["prev_lap_delta"] = df.groupby(stint_key)["fuel_corrected_s"].transform(lambda s: s.shift(1) - s.shift(2))
    decay_w = int(p["decay_window"])
    df["pace_decay_slope"] = df.groupby(stint_key)["fuel_corrected_s"].transform(
        lambda s: pd.Series(trailing_slope(s.to_numpy(), decay_w), index=s.index).shift(1)
    )
    df["race_progress"] = df["lap_number"] / df["race_laps"]
    return df


# --------------------------------------------------------------------------- public API
def build_dataset(
    laps: pd.DataFrame,
    stints: pd.DataFrame,
    weather: pd.DataFrame,
    race_control: pd.DataFrame,
    sessions: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    p = params["preprocess"]
    df = _prepare_laps(laps)
    df = _attach_stints(df, stints)
    df = _attach_weather(df, weather)
    df = _add_track_temp_delta(df, int(p["temp_delta_lag_laps"]))

    df, report = _filter_clean_laps(df, safety_car_laps(race_control), p)
    df = _add_targets_and_features(df, p)
    report["final_rows"] = len(df)

    meta = sessions[["session_key", "year", "circuit_short_name"]].drop_duplicates("session_key")
    meta = meta.assign(session_key=meta["session_key"].astype(int))
    df = df.merge(meta, on="session_key", how="left")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df["compound"] = pd.Categorical(df["compound"].astype(str), categories=list(COMPOUNDS))
    df["rainfall"] = df["rainfall"].fillna(0.0)
    df["wind_speed"] = df["wind_speed"].fillna(df["wind_speed"].median())
    df["humidity"] = df["humidity"].fillna(df["humidity"].median())

    columns = [*ID_COLUMNS, *FEATURE_COLUMNS, TARGET_REGRESSION, TARGET_CLASSIFICATION]
    out = df[columns].reset_index(drop=True)
    out[TARGET_CLASSIFICATION] = out[TARGET_CLASSIFICATION].astype(bool)
    validate(out)
    return out, report


def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Preprocessing produced an empty dataset")
    if df[TARGET_REGRESSION].isna().any():
        raise ValueError("NaN in regression target")
    if (df["tyre_age"] < 0).any():
        raise ValueError("Negative tyre_age found")
    if not df["race_progress"].between(0, 1).all():
        raise ValueError("race_progress outside [0, 1]")
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")


def split_by_year(df: pd.DataFrame, train_years: list[int], test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].isin(train_years)].reset_index(drop=True)
    test = df[df["year"].isin(test_years)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(
            f"Empty split: train={len(train)} test={len(test)}. Years present: {sorted(df['year'].unique())}"
        )
    return train, test


def run(output_dir: Path = PROCESSED_DIR, metrics_dir: Path = METRICS_DIR) -> None:
    params = load_params()
    raw = {ep: load_raw(ep) for ep in ("laps", "stints", "weather", "race_control", "sessions")}
    dataset, report = build_dataset(params=params, **raw)
    p = params["preprocess"]
    train, test = split_by_year(dataset, p["train_years"], p["test_years"])

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_dir / "features.parquet", index=False)
    train.to_parquet(output_dir / "train.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    report.update(
        train_rows=len(train),
        test_rows=len(test),
        sessions=int(dataset["session_key"].nunique()),
        pit_flag_rate_train=round(float(train[TARGET_CLASSIFICATION].mean()), 4),
    )
    # Metrics live outside data/processed: DVC forbids a tracked output nested inside another.
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOG.info("Wrote %d train / %d test rows to %s", len(train), len(test), output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()