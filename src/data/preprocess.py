"""Clean raw lap data and build the model feature matrix.

Target (``lap_time_loss``, seconds): fuel-corrected lap time minus the
fuel-corrected baseline pace of the same driver stint, where the baseline is the
median of the stint's first clean laps. It isolates tyre-wear loss from compound
pace, fuel burn-off and driver/car pace.

Usage:
    python -m src.data.preprocess --raw-dir data/raw --out data/processed/laps.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Physical / cleaning constants. Change them here, nowhere else.
FUEL_START_KG = 100.0  # FIA race fuel limit is 100 kg (110 kg from 2026 rules)
FUEL_EFFECT_S_PER_KG = 0.03  # commonly cited ~0.03 s/lap per kg of fuel
DRY_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
MIN_STINT_LAPS = 8  # stints with fewer clean laps give an unstable baseline
BASELINE_LAPS = 3  # first N clean laps of a stint define its baseline pace
OUTLIER_RATIO = 1.07  # drop laps >107% of the driver's median (traffic, errors)
LOSS_BOUNDS = (-2.0, 5.0)  # seconds; outside this the label is noise
BAD_TRACK_STATUS = "24567"  # yellow, safety car, red flag, VSC deployed/ending

STINT_KEYS: list[str] = ["Year", "RoundNumber", "Driver", "Stint"]
MODEL_FEATURES: list[str] = [
    "tire_compound",
    "tire_age",
    "track_temp",
    "air_temp",
    "lap_number",  # lap within the current stint, per the team's input contract
    "fuel_load_est",
]
TARGET = "lap_time_loss"
META_COLUMNS: list[str] = [
    "race_id",
    "Year",
    "RoundNumber",
    "EventName",
    "Driver",
    "Stint",
    "race_lap",
]
OUTPUT_COLUMNS: list[str] = META_COLUMNS + MODEL_FEATURES + [TARGET]

REQUIRED_RAW_COLUMNS: tuple[str, ...] = (
    "Year",
    "RoundNumber",
    "EventName",
    "TotalLaps",
    "Driver",
    "LapNumber",
    "LapTime",
    "Stint",
    "Compound",
    "TyreLife",
    "TrackStatus",
    "IsInLap",
    "IsOutLap",
    "AirTemp",
    "TrackTemp",
    "Rainfall",
)


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Concatenate every Parquet file in ``raw_dir``.

    Raises:
        FileNotFoundError: if the directory holds no Parquet files.
    """
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {raw_dir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def is_clean_track_status(status: pd.Series) -> pd.Series:
    """True where the lap ran fully under green flag conditions.

    Missing or empty status is treated as unknown, hence not clean.
    """
    text = status.fillna("").astype(str)
    has_bad = text.str.contains(f"[{BAD_TRACK_STATUS}]", regex=True)
    return (text != "") & ~has_bad


def add_base_columns(
    raw: pd.DataFrame, fuel_effect: float = FUEL_EFFECT_S_PER_KG
) -> pd.DataFrame:
    """Add race id, stint lap, fuel estimate and fuel-corrected lap time.

    These need the *unfiltered* laps (stint lap counts dirty laps too), so this
    runs before any cleaning. ``fuel_effect`` is the assumed lap-time gain in
    seconds per kg of fuel burned; the label depends strongly on it.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")

    df = raw.dropna(subset=["LapNumber", "Stint", "Driver", "TyreLife"]).copy()
    df["race_lap"] = df["LapNumber"].astype(int)
    df["race_id"] = (
        df["Year"].astype(int).astype(str)
        + "_"
        + df["RoundNumber"].astype(int).astype(str).str.zfill(2)
    )
    stint_first_lap = df.groupby(STINT_KEYS)["race_lap"].transform("min")
    df["lap_number"] = df["race_lap"] - stint_first_lap + 1

    laps_done = (df["race_lap"] - 1) / df["TotalLaps"]
    df["fuel_load_est"] = (FUEL_START_KG * (1.0 - laps_done)).clip(lower=0.0)
    df["fuel_corrected_time"] = df["LapTime"] - fuel_effect * df["fuel_load_est"]
    df["tire_compound"] = df["Compound"].astype("string").str.upper()
    df["tire_age"] = df["TyreLife"].round().astype(int)
    df["track_temp"] = df["TrackTemp"].astype(float)
    df["air_temp"] = df["AirTemp"].astype(float)
    return df


def _is_true(series: pd.Series) -> pd.Series:
    """True only where the value is exactly True; NaN/None count as False.

    Avoids the deprecated object-dtype ``fillna(False)`` downcast.
    """
    return series.eq(True)


def clean_mask(df: pd.DataFrame, drop_rain: bool = True) -> pd.Series:
    """Boolean mask of laps usable for tyre-degradation modelling."""
    mask = df["LapTime"].notna()
    mask &= ~_is_true(df["IsInLap"])
    mask &= ~_is_true(df["IsOutLap"])
    mask &= df["tire_compound"].isin(DRY_COMPOUNDS).fillna(False)
    mask &= is_clean_track_status(df["TrackStatus"])
    mask &= df["race_lap"] > 1  # standing start distorts lap 1
    mask &= df[["track_temp", "air_temp"]].notna().all(axis=1)
    if "Deleted" in df.columns:
        mask &= ~_is_true(df["Deleted"])
    if "IsAccurate" in df.columns:
        mask &= _is_true(df["IsAccurate"])
    if drop_rain:
        mask &= ~_is_true(df["Rainfall"])
    return mask


def drop_pace_outliers(clean: pd.DataFrame) -> pd.DataFrame:
    """Drop laps much slower than the driver's median pace in that race."""
    median = clean.groupby(["Year", "RoundNumber", "Driver"])[
        "fuel_corrected_time"
    ].transform("median")
    return clean[clean["fuel_corrected_time"] <= OUTLIER_RATIO * median]


def add_target(clean: pd.DataFrame) -> pd.DataFrame:
    """Compute ``lap_time_loss`` against each stint's early-lap baseline."""
    ordered = clean.sort_values([*STINT_KEYS, "lap_number"]).copy()
    ordered["_rank"] = ordered.groupby(STINT_KEYS).cumcount()
    baseline = (
        ordered[ordered["_rank"] < BASELINE_LAPS]
        .groupby(STINT_KEYS)["fuel_corrected_time"]
        .median()
        .rename("baseline_time")
        .reset_index()
    )
    counts = ordered.groupby(STINT_KEYS).size().rename("n_clean").reset_index()
    out = ordered.merge(baseline, on=STINT_KEYS, how="left").merge(
        counts, on=STINT_KEYS, how="left"
    )
    out = out[out["n_clean"] >= MIN_STINT_LAPS].copy()
    out[TARGET] = out["fuel_corrected_time"] - out["baseline_time"]
    low, high = LOSS_BOUNDS
    return out[out[TARGET].between(low, high)]


def drop_duplicate_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated laps (the same raw file loaded twice, overlapping fetches).

    A duplicate would silently double-weight that race and leak between the
    train and validation folds. Rows lacking a lap number are left for
    ``add_base_columns`` to drop.
    """
    key = ["Year", "RoundNumber", "Driver", "LapNumber"]
    present = [c for c in key if c in raw.columns]
    if len(present) < len(key):
        return raw  # add_base_columns reports the missing columns
    deduped = raw.drop_duplicates(subset=key, keep="first")
    if len(deduped) < len(raw):
        logger.warning("Dropped %d duplicate laps", len(raw) - len(deduped))
    return deduped


def validate_output(df: pd.DataFrame) -> None:
    """Fail loudly if the feature matrix breaks the training contract.

    Raises:
        ValueError: on NaN/inf, negative tyre age, a stint lap below 1, negative
            fuel, or a label outside ``LOSS_BOUNDS``.
    """
    numeric = df[
        ["tire_age", "track_temp", "air_temp", "lap_number", "fuel_load_est", TARGET]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Output contains NaN or infinite values")
    if (df["tire_age"] < 0).any() or (df["lap_number"] < 1).any():
        raise ValueError("Output has a negative tyre age or a stint lap below 1")
    if (df["fuel_load_est"] < 0).any():
        raise ValueError("Output has negative fuel load")
    if not df[TARGET].between(*LOSS_BOUNDS).all():
        raise ValueError(f"Label outside {LOSS_BOUNDS}")
    if df.duplicated(subset=["race_id", "Driver", "race_lap"]).any():
        raise ValueError("Output contains duplicate laps")


MIN_RACE_ROWS = 150  # a typical race keeps about 700; far fewer suggests bad data


def log_race_coverage(df: pd.DataFrame) -> None:
    """Log the race count and warn about races that kept suspiciously few laps."""
    per_race = df.groupby("race_id").size()
    logger.info("Dataset covers %d races", len(per_race))
    thin = per_race[per_race < MIN_RACE_ROWS]
    if not thin.empty:
        logger.warning(
            "Races with fewer than %d usable laps: %s",
            MIN_RACE_ROWS,
            ", ".join(f"{r} ({n})" for r, n in thin.items()),
        )


def build_dataset(
    raw: pd.DataFrame,
    drop_rain: bool = True,
    fuel_effect: float = FUEL_EFFECT_S_PER_KG,
) -> pd.DataFrame:
    """Full pipeline: raw laps -> model-ready feature matrix.

    Raises:
        ValueError: if ``fuel_effect`` is negative, or cleaning leaves no rows
            (almost always a data problem).
    """
    if fuel_effect < 0:
        raise ValueError(f"fuel_effect must be >= 0, got {fuel_effect}")
    raw = drop_duplicate_laps(raw)
    base = add_base_columns(raw, fuel_effect=fuel_effect)
    clean = base[clean_mask(base, drop_rain=drop_rain)]
    clean = drop_pace_outliers(clean)
    dataset = add_target(clean)
    if dataset.empty:
        raise ValueError(
            f"No usable laps left after cleaning ({len(raw)} raw laps). "
            "Check track status, compounds and stint lengths."
        )
    logger.info(
        "Kept %d of %d laps (%.1f%%)",
        len(dataset),
        len(raw),
        100.0 * len(dataset) / len(raw),
    )
    result = dataset[OUTPUT_COLUMNS].reset_index(drop=True)
    result["tire_compound"] = result["tire_compound"].astype(str)
    validate_output(result)
    log_race_coverage(result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/laps.parquet"))
    parser.add_argument(
        "--keep-rain", action="store_true", help="Keep laps flagged as rainy"
    )
    parser.add_argument(
        "--fuel-effect",
        type=float,
        default=FUEL_EFFECT_S_PER_KG,
        help="Assumed lap-time gain in s per kg of fuel burned",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    dataset = build_dataset(
        load_raw(args.raw_dir),
        drop_rain=not args.keep_rain,
        fuel_effect=args.fuel_effect,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    dataset.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)  # never leave a half-written file behind
    logger.info("Wrote %s (%d rows)", args.out, len(dataset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
