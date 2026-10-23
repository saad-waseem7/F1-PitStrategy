# Replay sample

`sample.json.gz` feeds the Stint Replay page: model inputs and true labels for every race of the held-out 2025 test split. It is built from the processed data and committed, so the Docker image ships it and CI can test against it.

    python -m src.replay.build_sample                  # held-out test split, every race
    python -m src.replay.build_sample --races 6        # six evenly spaced races

Rebuild it after `dvc repro` changes `data/processed/test.parquet`, commit the new file, then run `docker compose restart api`. The folder is mounted into the container, so no image rebuild is needed.

The sample records which split it came from. The page calls the races held-out only when that split is `test`.
