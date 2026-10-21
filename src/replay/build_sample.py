"""Build the replay sample from processed data.

    python -m src.replay.build_sample --input data/processed/test.parquet --output replay/sample.json.gz --races 4

Races are picked at evenly spaced positions in session order, so the same input always gives the same sample.
By default the input is the held-out test split, which the model never trained on and never used to choose
between models. The sample records which split it came from and the page says so.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.replay.store import SAMPLE_COLUMNS, write_sample

LOG = logging.getLogger(__name__)

DEFAULT_INPUT = PROCESSED_DIR / "test.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "replay" / "sample.json.gz"


def build(input_path: Path, output_path: Path, races: int) -> dict[str, object]:
    if races < 1:
        raise ValueError("races must be at least 1")
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed data not found: {input_path} (run `dvc repro` or `dvc pull` first)")
    frame = pd.read_parquet(input_path, columns=list(SAMPLE_COLUMNS))
    keys = np.array(sorted(frame["session_key"].unique()))
    if len(keys) == 0:
        raise ValueError(f"{input_path} has no sessions")
    picks = np.unique(np.linspace(0, len(keys) - 1, min(races, len(keys))).round().astype(int))
    chosen = keys[picks]
    sample = frame[frame["session_key"].isin(chosen)]
    meta = {
        "split": input_path.stem,
        "years": sorted(int(y) for y in sample["year"].unique()),
        "sessions": [int(k) for k in chosen],
        "rows": int(len(sample)),
    }
    write_sample(sample, output_path, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--races", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        meta = build(args.input, args.output, args.races)
    except (FileNotFoundError, ValueError) as exc:
        LOG.error("%s", exc)
        return 1
    LOG.info("Wrote %s: %s rows from sessions %s (%s split)", args.output, meta["rows"], meta["sessions"], meta["split"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
