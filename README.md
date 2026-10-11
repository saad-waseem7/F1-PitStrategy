# F1 Pit Strategy

End-to-end pipeline that measures how much lap time a driver loses to tyre wear
and recommends whether to pit.

```
OpenF1 API --fetch_data--> data/raw/*.parquet --preprocess--> data/processed/laps.parquet
```

## Setup

Python 3.11 or 3.12 (DVC does not install on 3.13).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc init   # once, then: dvc remote add -d localstore ~/Documents/dvc-storage/f1-pit-strategy
```

## 1. Get the data

Data comes from the [OpenF1](https://openf1.org) API: free, no key, plain JSON.
Historical data starts in 2023.

```bash
python -m src.data.fetch_data --years 2023 2024 2025 2026
python -m src.data.fetch_data --years 2025 --rounds 5 6 --force   # re-download specific races
```

* One Parquet file per race in `data/raw/`. Existing valid files are skipped, so
  re-running only fetches what is missing.
* Files are written atomically, corrupt files are re-downloaded, connection
  errors, HTTP 429 and 5xx are retried with backoff, and a race with fewer than
  50 timed laps is rejected.
* A summary lists every missing race. The exit code is non-zero if more than 15
  percent failed (`--max-failure-rate`).
* `TyreLife` is `tyre_age_at_start + laps on the set`. `TrackStatus` is rebuilt
  from race-control messages (yellow, safety car, VSC, red flag, plus the restart
  lap). `RoundNumber` is the order of the season's non-cancelled races, so it can
  differ from the official number after a cancellation.

## 2. Preprocess

```bash
python -m src.data.preprocess                     # data/raw -> data/processed/laps.parquet
python -m src.data.preprocess --fuel-effect 0.03  # assumed seconds per kg of fuel
```

The label `lap_time_loss` (seconds) is the fuel-corrected lap time minus the
median of the stint's first 3 clean laps. It isolates tyre wear from car pace,
compound pace and fuel burn.

Laps are dropped when they are lap 1, in or out laps, wet, under yellow, safety
car, VSC or red flag, inaccurate or deleted, slower than 107 percent of the
driver's median, or in a stint with fewer than 8 clean laps. Duplicate laps are
removed and the output is validated (no NaN, sane ranges, label within -2 to 5 s).
Wet races therefore keep few laps, and the run logs a warning for any race with
fewer than 150 usable laps.

Output columns: `race_id, Year, RoundNumber, EventName, Driver, Stint,
race_lap, tire_compound, tire_age, track_temp, air_temp, lap_number,
fuel_load_est, lap_time_loss`.

### Calibrating the fuel effect

The label depends on an assumed fuel effect (0.03 s per kg, 100 kg start). Do not
pick it by model R2: fuel burns off in step with the lap, so an over-large value
creates a tyre-age-like trend that the wear model then "explains". Estimate it
from raw lap times instead, where tyre age and fuel vary independently across
stints:

```bash
python -m src.data.fuel_effect
```

It prints the net effect (fuel burn plus track evolution) in s/kg with a 95
percent interval. Measured on 2023 to 2026 (84 races, 71,590 laps): **0.0323
s/kg, 95 percent CI 0.0307 to 0.0340**, about 0.23 s over a full race away from
the 0.03 default, so the default is kept.

### Versioning data

```bash
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc && git commit -m "Track data with DVC"
dvc push
```


## Tests and lint

```bash
ruff check src tests
ruff format --check src tests
pytest -q tests
```

The tests use synthetic data with known laws and mocked HTTP, so they need no
network.

## Workflow

One branch per piece of work, pull requests into `main`, a teammate reviews and
merges.
