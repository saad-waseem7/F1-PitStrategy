"""Train a THROWAWAY champion on synthetic data, for CI smoke tests of the serving image.

    python -m tests.make_ci_champion --out models/champion

The model is meaningless for F1. It exists so CI can prove that the Dockerfile builds, the image starts, the
model loads and the API answers, without needing the real model (which lives in DVC). Never ship its output.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path

import mlflow

from src.config import PROJECT_ROOT, load_params
from src.models.train import run
from tests.test_train import _write_processed


def build(out: Path) -> Path:
    params = copy.deepcopy(load_params())
    tp = params["train"]
    tp.update(cv_folds=3, lgbm_trials=2, latency_samples=5, permutation_sample=200)
    tp["rf"].update(n_estimators=20)
    tp["lgbm_classifier"].update(n_estimators=30)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_processed(root / "processed")
        tp["mlflow"].update(tracking_uri=f"sqlite:///{root / 'mlflow.db'}", artifact_root=str(root / "artifacts"))
        run(data_dir=root / "processed", metrics_dir=root / "metrics", models_dir=root / "models", params=params)
        mlflow.end_run()
        shutil.rmtree(out, ignore_errors=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / "models" / "champion", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "champion")
    args = parser.parse_args()
    print(f"Synthetic champion written to {build(args.out)}")


if __name__ == "__main__":
    main()
