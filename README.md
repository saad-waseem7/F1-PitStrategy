# F1 Pit Strategy

End-to-end pipeline that predicts how much lap time a driver loses to tyre wear
and recommends whether to pit. This README covers the data and model parts
(Phases 1 and 2). Serving, CI and drift monitoring come later.

```
FastF1 API --fetch_data--> data/raw/*.parquet --preprocess--> data/processed/laps.parquet
                                                                      |
                                       train (MLflow, registry) <-----+
                                                |
                                         models_local/  +  MLflow registry
```

## Setup

Python 3.11 or 3.12 (DVC does not install on 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc init            # once, then: dvc remote add -d localstore ~/Documents/dvc-storage/f1-pit-strategy
```

## 1. Get the data

```bash
python -m src.data.fetch_data --since 2018        # every season up to now (slow the first time, cached after)
python -m src.data.fetch_data --years 2024 --rounds 5 6 --force   # re-download specific races
```

* One Parquet file per race in `data/raw/`; existing valid files are skipped, so
  re-running only fetches what is missing.
* Files are written atomically, corrupt files are re-downloaded, network errors
  are retried, and a session with fewer than 50 timed laps is rejected.
* A summary lists every missing session. The exit code is non-zero if more than
  15 percent failed (`--max-failure-rate`). A few failures are normal: cancelled
  races such as Emilia-Romagna 2023 have no data.
* FastF1 lap data is dependable from 2018 onwards.

## 2. Preprocess

```bash
python -m src.data.preprocess                     # data/raw -> data/processed/laps.parquet
python -m src.data.preprocess --fuel-effect 0.03  # assumed seconds per kg of fuel
```

The label `lap_time_loss` (seconds) is the fuel-corrected lap time minus the
median of the stint's first 3 clean laps. It isolates tyre wear from car pace,
compound pace and fuel burn.

Laps are dropped when they are lap 1, in or out laps, wet, under yellow, safety
car or VSC, inaccurate or deleted, slower than 107 percent of the driver's
median, or in a stint with fewer than 8 clean laps. Duplicate laps are removed
and the output is validated (no NaN, sane ranges, label within -2..5 s).

**Assumption to know about:** the label magnitude depends on the fuel effect
(0.03 s per kg, 100 kg start). At 0.02, 0.03 and 0.04 the median loss for
tyres 20+ laps old is 0.45, 0.78 and 1.13 s. Report it as an assumption.

Output columns: `race_id, Year, RoundNumber, EventName, Driver, Stint,
race_lap, tire_compound, tire_age, track_temp, air_temp, lap_number,
fuel_load_est, lap_time_loss`.

Version data with DVC (`dvc add data/raw data/processed`, `dvc push`).

## 3. Train

```bash
python -m src.models.train --tune --promote
python -m src.models.train --model lightgbm --no-register   # comparison model
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

* **Split:** the last 20 percent of races (chronological) are the test set. Tuning
  uses race-grouped K-fold CV on the rest, so no race is in both train and
  validation.
* **Baselines:** predicting the mean, and plain linear regression.
* **Promotion gate:** a model is registered as `staging`. It becomes `production`
  only if it beats the mean baseline on the test set and, if a production model
  exists, beats it by at least 1 percent MAE on the same test races.
* Everything is logged to MLflow: params, CV and test metrics, train metrics,
  coefficients, data version.

### Model choice

The default model is a small purpose-built wear model (`src/models/wear.py`),
not a generic learner:

```
loss = intercept[compound] + slope[compound] * tire_age + w1 * track_temp + w2 * air_temp
```

* **Why:** lap-level loss is mostly noise. About 48 percent of its variance is
  between stints and the best achievable R2 from the contract features is
  about 0.08. A 8 to 9 parameter model cannot overfit that, and every
  coefficient can be read and defended. LightGBM (monotone constrained) was
  tried and scored worse on every real-data metric (test R2 -0.18 against 0.10).
* **Overfitting:** few parameters, slopes constrained to be non-negative, ridge
  on the temperature terms only, race-grouped CV, a held-out test set of later
  races, and logged train-versus-test gaps.
* **Underfitting:** the gate requires beating the mean baseline, and a plain
  linear model is reported alongside. The binned metrics (race x compound x age
  bucket means) show the wear trend that the pit decision depends on.
* **Two outputs:** `predict` gives the pooled lap-level loss. The pit decision
  uses the stint wear rate (median per-stint slope per compound, about 0.05 to
  0.08 s per lap), because pooled slopes are biased low: long stints survive
  only when tyres wear slowly.

### Pit rule

Pit when `rate * (age * laps_left + T) - rate_new * T - pit_penalty > min_gain`
with `T = laps_left * (laps_left + 1) / 2` and a 22 s pit penalty. For a stop
onto the same compound this reduces to `rate * age * laps_left > 22`.

### Known limits

* CV R2 is about 0. Lap-level prediction is barely better than the mean; trust
  the binned and stint-rate results for the wear trend.
* Linear wear misses late-stint cliffs. The pit rule ignores traffic, undercut,
  safety cars and compound pace gaps. Pit loss is constant, though it varies by
  circuit.
* With about 9 test races, test metrics are noisy. CV is the steadier signal.

## Tests and lint

```bash
ruff check src tests
pytest -q tests
```

The tests use synthetic data with known wear laws, so they need no network.
`tests/test_pipeline_integration.py` runs raw data through preprocess, training
and a pit recommendation.

## Workflow

One branch per piece of work, pull requests into `main`, a teammate reviews and
merges. `saad/data-pipeline` holds fetch and preprocess; `saad/train-model`
holds training and inference.
