# Replay sample

`sample.json.gz` feeds the Stint replay page: a few races of model inputs and true labels. It is built from the
processed data and committed, so the image can ship it.

    python -m src.replay.build_sample                  # held-out test split, 4 races, deterministic
    python -m src.replay.build_sample --races 6

Rebuild it after `dvc repro` changes `data/processed/test.parquet`, then commit the new file.
