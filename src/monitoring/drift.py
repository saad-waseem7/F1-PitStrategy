"""Data drift report: served requests (prediction log) versus the training data, using Evidently.

    python -m src.monitoring.drift --log logs/predictions.jsonl --reference data/processed/train.parquet

Writes `drift_report.html` (human) and `drift_summary.json` (machines, e.g. the Airflow DAG) to the output
directory. The summary's `drift_detected` is true when at least `drift_share` of the monitored columns drifted.

Only INPUT drift is measured. Labels (pace loss, pit flag) exist only after a stint is over, so model
performance cannot be monitored live; drifted inputs are the early warning.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import FEATURE_COLUMNS, PROJECT_ROOT

LOG = logging.getLogger(__name__)

DEFAULT_LOG = PROJECT_ROOT / "logs" / "predictions.jsonl"
DEFAULT_REFERENCE = PROJECT_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "drift"
CATEGORICAL = ("compound",)


class DriftError(RuntimeError):
    """Inputs for the drift check are missing or unusable."""


def load_current(log_path: Path) -> tuple[pd.DataFrame, int]:
    """Feature rows from the prediction log. Returns (frame, number of unreadable lines skipped)."""
    if not log_path.is_file():
        raise DriftError(f"Prediction log not found: {log_path}")
    rows: list[dict[str, Any]] = []
    skipped = 0
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                inputs = json.loads(line)["inputs"]
                rows.append({c: inputs.get(c) for c in FEATURE_COLUMNS})
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                skipped += 1
    frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    return _typed(frame), skipped


def load_reference(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    if not path.is_file():
        raise DriftError(f"Reference data not found: {path}")
    frame = pd.read_parquet(path, columns=list(FEATURE_COLUMNS))
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    return _typed(frame.reset_index(drop=True))


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL:
            out[col] = out[col].astype("object").where(out[col].notna(), None).astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _parse_columns(snapshot: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Per-column drift verdicts and the overall drifted-column count from an Evidently result dict."""
    per_column: dict[str, dict[str, Any]] = {}
    overall: dict[str, float] = {}
    for metric in snapshot["metrics"]:
        cfg = metric["config"]
        kind = cfg.get("type", "")
        if kind.endswith("DriftedColumnsCount"):
            overall = {"count": float(metric["value"]["count"]), "share": float(metric["value"]["share"])}
        elif kind.endswith("ValueDrift"):
            method = str(cfg["method"])
            value = float(metric["value"])
            threshold = float(cfg["threshold"])
            # p-value tests drift when p is below the threshold; distance measures when it reaches it.
            drifted = value < threshold if "p_value" in method else value >= threshold
            per_column[cfg["column"]] = {"method": method, "value": value, "threshold": threshold, "drifted": drifted}
    return per_column, overall


def run(
    log_path: Path = DEFAULT_LOG,
    reference_path: Path = DEFAULT_REFERENCE,
    out_dir: Path = DEFAULT_OUT,
    min_rows: int = 200,
    drift_share: float = 0.2,
    max_reference_rows: int = 20000,
    seed: int = 42,
    num_threshold: float = 0.2,
    cat_threshold: float = 0.1,
) -> dict[str, Any]:
    """Build the report and summary. Raises DriftError when an input file is missing.

    `drift_share` is the fraction of monitored columns that must drift to count as drift. Evidently's default
    of 0.5 needs seven of fourteen columns; the weather columns move together, so a real change in conditions
    shifts three or four at once. 0.2 means three columns. Identical distributions flagged no overall drift
    in 15 of 15 trials at 200 rows.

    `num_threshold` is a normalised Wasserstein distance (in reference standard deviations) for numeric
    columns, `cat_threshold` a Jensen-Shannon distance for categorical ones. Evidently's own default of 0.1
    is below the sampling noise of a few hundred requests (identical distributions measured 0.05 to 0.12 at
    400 rows), so it flags columns that did not move. 0.2 means the typical value moved by a fifth of a
    standard deviation.
    """
    if not 0.0 < drift_share <= 1.0:
        raise ValueError("drift_share must be in (0, 1]")
    current, skipped = load_current(log_path)
    reference = load_reference(reference_path, max_reference_rows, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "unreadable_log_lines": skipped,
        "min_rows": min_rows,
        "drift_share_threshold": drift_share,
        "num_threshold": num_threshold,
        "cat_threshold": cat_threshold,
    }
    if len(current) < min_rows:
        # Too few requests to say anything: report it and do not claim drift.
        summary.update(status="insufficient_data", drift_detected=False, drifted_columns=[], share_drifted=0.0, columns={})
        _write(summary, out_dir)
        return summary

    constant = [c for c in FEATURE_COLUMNS if reference[c].nunique(dropna=True) <= 1]
    columns = [c for c in FEATURE_COLUMNS if c not in constant]
    numeric = [c for c in columns if c not in CATEGORICAL]
    categorical = [c for c in columns if c in CATEGORICAL]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    report = Report([DataDriftPreset(drift_share=drift_share, num_threshold=num_threshold, cat_threshold=cat_threshold)])
    snapshot = report.run(
        Dataset.from_pandas(current[columns], data_definition=definition),
        Dataset.from_pandas(reference[columns], data_definition=definition),
    )
    snapshot.save_html(str(out_dir / "drift_report.html"))
    per_column, overall = _parse_columns(snapshot.dict())
    drifted = sorted(c for c, v in per_column.items() if v["drifted"])
    share = float(overall.get("share", len(drifted) / max(len(per_column), 1)))
    summary.update(
        status="ok",
        drift_detected=bool(share >= drift_share),
        drifted_columns=drifted,
        share_drifted=share,
        columns=per_column,
        skipped_constant_columns=constant,
        missing_share={
            c: {"reference": float(reference[c].isna().mean()), "current": float(current[c].isna().mean())}
            for c in FEATURE_COLUMNS
            if reference[c].isna().any() or current[c].isna().any()
        },
    )
    _write(summary, out_dir)
    return summary


def _write(summary: dict[str, Any], out_dir: Path) -> None:
    tmp = out_dir / "drift_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(out_dir / "drift_summary.json")  # atomic: a reader never sees half a file


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--drift-share", type=float, default=0.2)
    parser.add_argument("--num-threshold", type=float, default=0.2, help="normalised Wasserstein distance for numeric columns")
    parser.add_argument("--cat-threshold", type=float, default=0.1, help="Jensen-Shannon distance for categorical columns")
    parser.add_argument("--fail-on-drift", action="store_true", help="exit with status 2 when drift is detected")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        summary = run(
            args.log, args.reference, args.out, args.min_rows, args.drift_share,
            num_threshold=args.num_threshold, cat_threshold=args.cat_threshold,
        )
    except DriftError as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info(
        "status=%s drift_detected=%s drifted=%s (%d current rows)",
        summary["status"], summary["drift_detected"], summary["drifted_columns"], summary["n_current"],
    )
    return 2 if args.fail_on_drift and summary["drift_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
