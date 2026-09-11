# Usage

## Install and run

```bash
uv run pytest                 # the suite
forecast                      # command listing
forecast train --help         # every flag
```

`forecast` dispatches to one command's `main()`, leaving its arguments
untouched. A result is expected to be traceable to the command that produced it,
and a front end that rewrote arguments would make the recorded command and the
real one diverge. `train` also runs standalone as `python -m forecast.train`.

## The published configuration

```bash
forecast train \
  --catalog-path catalogs/catalog_current.csv \
  --catalog-span 2000-01-01 2026-08-12 \
  --horizon-days 14 --cv-folds 2 --bg-min-mag 3.0 --batch-size 128 \
  --keep-features log1p_dsp mean_mag_30d cv_interevent_90d mag_deficit_90d \
  --ensemble-seeds 42,43,44,45,46
```

Of the fourteen forecasters the parent project tried, this is the one that
clears its own fold's floor.

## Flags

Defaults are pinned by `tests/test_defaults.py`, because a default is part of
the model: the published configuration passes eight flags and inherits the rest,
so a default that drifts is a different model reported under the same name.

### Input

| flag | default | meaning |
|---|---|---|
| `--catalog-path` | required | catalogue CSV |
| `--catalog-span START END` | required | the hourly index to build |
| `--threshold` | 4.5 | magnitude defining a positive label |
| `--bg-min-mag` | 3.0 | completeness threshold of the background catalogue |
| `--horizon-days` | 14.0 | how far the label looks forward |
| `--seq-hours` | 24 | hours per input window |
| `--keep-features` | all | restrict to a named subset, in the order given |

`--catalog-span` is required because with no waveform archive nothing else
defines how long the record is.

### Study region

Mutually exclusive; default is the Aegean box. See
[study-regions.md](study-regions.md).

| flag | meaning |
|---|---|
| `--catalog-bbox LAT0 LAT1 LON0 LON1` | forecast inside this box |
| `--catalog-radius LAT LON KM` | a true great-circle disc |
| `--station-radius STATION KM` | the same disc, centred on a station |
| `--station-catalog CSV` | inventory that `--station-radius` looks up |

### Label mode

| flag | default | meaning |
|---|---|---|
| `--label-mode` | `event` | `event` or `rate` |
| `--rate-min-mag` | 3.0 | events defining the rate target |
| `--rate-baseline-days` | None | trailing window; defaults to the horizon |
| `--rate-features` | off | append the eight trailing-rate features |

Rate mode without `--rate-features` is trying to beat a persistence floor built
from precisely the number it was never given.

### Model

| flag | default | meaning |
|---|---|---|
| `--cat-hidden` | 16 | branch width |
| `--fusion-hidden` | 32 | head width |
| `--dropout` | 0.2 | dropout throughout |

### Training

| flag | default | meaning |
|---|---|---|
| `--epochs` | 40 | maximum epochs |
| `--batch-size` | 16 | raise this for pooled runs |
| `--lr` | 3e-4 | AdamW learning rate |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--patience` | 8 | early-stopping patience |
| `--checkpoint-metric` | `auc` | `auc` or `loss`, selects the kept epoch |
| `--ensemble-seeds` | `42,43,44` | fixed seeds |
| `--random-seeds` | None | draw N seeds instead, printed for replay |
| `--train-stride-hours` | 1 | keep every Nth training sample |
| `--eval-stride-hours` | 1 | keep every Nth val and test sample |
| `--out-dir` | None | save checkpoints here |

Training uses `AdamW` with cosine annealing over `--epochs`, gradient clipping
at norm 1.0, and `BCEWithLogitsLoss` weighted by the training split's positive
rate. The best epoch by `--checkpoint-metric` is restored before scoring.

Fixed seeds hide run-to-run variance behind one sample of it, and per-seed
spread reaches 0.17 here, so prefer `--random-seeds`.

### Transfer

| flag | default | meaning |
|---|---|---|
| `--train-regions N` | 0 | pool N similar same-size regions into training |
| `--train-region-centers LAT,LON ...` | None | name them explicitly |
| `--train-region-grid-deg` | 1.2 | candidate centre spacing |

### Evaluation

| flag | default | meaning |
|---|---|---|
| `--cv-folds` | 1 | walk-forward folds |
| `--train-frac` | 0.70 | single-split training fraction |
| `--val-frac` | 0.15 | single-split validation fraction |
| `--balanced-folds` | off | boundaries by positive mass, not hour count |
| `--skip` | none | fold numbers to skip |
| `--region-split` | `none` | `lat` or `lon` spatial holdout |
| `--region-split-value` | median | boundary coordinate |
| `--region-test-side` | `high` | which half is held out |

## Worked example

A regional forecast around one station, evaluated on weekly non-overlapping
samples, with training pooled across similar regions.

```bash
forecast train \
  --catalog-path ../seismic_cli/catalogs/catalog_current.csv \
  --catalog-span 2007-01-01 2026-08-29 \
  --threshold 4 --bg-min-mag 3.0 --horizon-days 14 \
  --cv-folds 3 --random-seeds 3 --batch-size 128 \
  --keep-features log1p_dsp mean_mag_30d cv_interevent_90d mag_deficit_90d \
  --station-catalog ../seismic_cli/catalogs/istasyon_katalog.csv \
  --station-radius TU.ELBA 150 \
  --eval-stride-hours 168 --train-regions 12 \
  --out-dir trained_model_transfer
```

Choices worth explaining:

- **`--threshold 4` with `--bg-min-mag 3.0`.** Keeping them apart is what lets
  the distribution features be fitted at all. At M3 the label is positive in over
  70% of hours, which is not a forecasting problem.
- **Four features.** Thirteen inputs against a few dozen effective positives is
  too many, and this subset also skips the nearest-neighbour precompute.
- **`--batch-size 128`.** A pooled run carries roughly thirteen times the
  training data; at the default 16 that is tens of thousands of steps per epoch.
- **`--eval-stride-hours 168`.** Scores weekly non-overlapping samples, so the
  reported `n` is an honest count.

## Choosing a threshold

Lowering the threshold buys events and costs meaning. Measured on a 150 km disc
over roughly nineteen years:

| threshold | events | hourly positive rate at 14 d |
|---|---|---|
| 3.0 | 1390 | 0.725 |
| 3.5 | 317 | 0.330 |
| 4.0 | 90 | 0.118 |
| 4.5 | 31 | 0.039 |

Above about 0.5 the label is closer to a constant than a forecast; below about
30 events the trainer warns that the result rests on a handful of earthquakes.

## Reading the output

A run prints, per fold, the split sizes, the positive rate over time, each
split's distinct event count, the per-epoch validation curve, the floors, and
the full metric report. Then a walk-forward summary.

What to read, in order: how many folds cleared their own floor, the per-seed
spread, and the Brier skill against persistence. See
[evaluation.md](evaluation.md#reading-a-result).

## Running one experiment at a time

Per-fold AUC at these event counts carries roughly ±0.10 to ±0.17, so a single
run cannot separate a real effect from noise, and two changed variables cannot
be attributed. Change one thing per run and keep the evaluation side fixed. If
the floors move between two runs you are comparing, something reached the
evaluation that should not have, and the comparison is void.
