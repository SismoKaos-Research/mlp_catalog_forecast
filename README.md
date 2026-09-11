# forecast — catalog_mlp

Forecasting whether an M≥4.5 earthquake occurs in the Aegean within the next
14 days, from the earthquake catalogue alone.

```bash
forecast train \
    --catalog-path catalogs/catalog_current.csv \
    --catalog-span 2000-01-01 2026-08-12 \
    --horizon-days 14 --cv-folds 2 --bg-min-mag 3.0 --batch-size 128 \
    --keep-features log1p_dsp mean_mag_30d cv_interevent_90d mag_deficit_90d \
    --ensemble-seeds 42,43,44,45,46
```

That is the published configuration. Of the fourteen forecasters the parent
project tried, this is the one that clears its own fold's floor.

## Why there is no waveform here

The original module was `cnn_lstm_catalog_waveform_fusion.py`, and its waveform
arm is the negative half of the result. On the corrected catalogue, neither the
raw-waveform CNN, nor hand-crafted continuous features, nor chaotic features
beat the persistence floor — across every architecture tried, in 0 of 10 sweep
cells for the chaos suite. The 2026-08-30 experiment concluded:

> The forecasting signal in this project comes from the earthquake catalogue,
> not from the seismogram. That is worth stating directly, because it is a
> negative result with a clear boundary rather than an absence of one.

This repo is the half that survived that boundary. The ablation itself, and its
figures, stay in `cnn_earthquake` — the finding is preserved by the report, not
by carrying the code. So `--channels`, `--data-root` and the transfer-learning
flags are gone, and `--catalog-span` is now required: with no archive, nothing
else defines how long the record is.

The input is one catalogue CSV.

## What the model is

A two-layer MLP — `Linear → GELU → Dropout`, twice, hidden width 16 — reading
the window's **last hour** and nothing else. That is deliberate. The original
put an LSTM over the 24-hour window and it sat at chance for the entire
catalogue-only run, because catalogue features barely move inside a window:
within-window std over overall std measured 0.055 / 0.056 / 0.025 / 0.011 for
`log1p_dsp` / `count_7d` / `count_30d` / `count_90d`. What looks like a time
series is a single point-in-time tabular reading, and a direct MLP is the
better-matched tool.

The substance is in the **features**, not the network: thirteen derived from the
catalogue, eight describing the magnitude distribution and its timing from a
lower-completeness background catalogue (M≥`--bg-min-mag`), two spatial
(Zaliapin–Ben-Zion nearest-neighbour distance and a spatial Shannon entropy,
after Convertito et al. 2024). The published run keeps four of them.

## Every number comes with its floor

`evaluate.fold_result` is the only thing that produces a result, and it computes
the model's AUC, the floor it had to clear on *that fold*, and the per-seed
spread together. `summarise` raises `TypeError` if handed a bare AUC.

That is structural because the parent project's central methodological finding
is that a pooled number misleads here:

- fold SD runs 0.07–0.16, which dwarfs every gap between models
- `feature_lstm` beat its own fold's floor in 2 of 5 folds while losing to
  persistence on average
- per-seed AUC spread reaches 0.17, so a single fixed seed hides run-to-run
  variance behind one sample of it

**The floor is oriented.** An anti-predictive rule is inverted for free, so the
achievable baseline is `max(auc, 1 − auc)`. Event mode did not do this
originally, which collapsed the floor to chance whenever persistence landed
below 0.5 — that is what once made an n=4 result look like it cleared a 0.5000
bar when the properly oriented bar was ~0.58.

**Labels are purged at fold boundaries.** The label at hour H is decided by
events in [H, H+horizon], so without an embargo of `seq_hours − 1 + horizon` the
last ~14 days of every block carry labels determined by events in the *next*
block — train labels encoding what happens in val. That is ~9% of samples at
horizon 14 d, seq 24 h (Lopez de Prado, *Advances in Financial Machine
Learning*, Ch. 7).

`--region-split lat|lon` holds out half the study region instead of a stretch of
time: a patch of crust whose events the model has never seen. Off by default,
kept because it is the harder generalisation question. Its boundary is the
median of the events actually loaded, so it follows whatever region is in force.

## Choosing the study region

```bash
forecast train ... --catalog-bbox 37.0 38.5 26.0 28.0
forecast train ... --catalog-radius 37.5 27.5 120
forecast train ... --station-catalog stations.csv --station-radius KO.BODT 120
```

The default is the Aegean box, `lat 36–40, lon 25–30`, which every published
number was produced over. The three flags override it and are mutually
exclusive. `--catalog-radius` is a true great-circle disc, not its bounding box;
at this latitude the box reaches about 40% further into its corners than the
radius.

`--station-radius` is the same disc with its centre looked up by station code,
in an inventory of this shape:

```
Network,Code,Longitude,Latitude,Height,Province,District
"TU","ABT",31.3208,40.6058,1794,"Bolu","Mudurnu"
"KO","BODT",27.3103,37.0622,120,"Muğla","Bodrum"
```

Longitude precedes latitude in that format, the reverse of everywhere else here,
so the columns are read by name. Nothing reads the station's data: this is a way
of saying *where*, not a return of the station filter the port removed. A bare
code works when only one network carries it, and an inventory that lists one
station at two different sites is refused rather than resolved by row order.

Narrowing the region changes the question rather than sharpening it. Features,
labels, the persistence floor and the fold boundaries are all rebuilt from the
events inside it, and the spatial entropy grid is re-cast over it, so a run over
a 120 km disc forecasts a different thing than a run over the Aegean. It is also
not free: the M≥4.5 set is ~10² events across the whole box, and the trainer
warns when a region leaves fewer than 30 of them, because that is the number
every AUC below it rests on however many hourly rows there are.

The region is recorded in each checkpoint next to the horizon and the threshold.
The same weights answer a different question in a different patch of crust, and
nothing in the weights says which.

## Training on more regions than you forecast

A fold in a single 150 km region can fit a model on ten earthquakes. That is the
binding constraint, not the architecture.

```bash
forecast train ... --station-radius TU.ELBA 150 --train-regions 12
```

This pools twelve further regions of the same size and shape into **training
only**. Validation, test, the persistence floor and the reported `n` still come
from the study region alone, so a pooled run is directly comparable to one
without it. Measured on the Turkish catalogue, it takes the training pool from
90 events to about 1470.

Regions are chosen by measured similarity in positive rate and Aki b-value, over
the **training window only**. Ranking that used the whole record would pick
training regions partly for what happens in the block the model is later scored
on. Candidates within two radii of the target are excluded, so no earthquake can
be in both the pool and the test block. The selection is printed as
`--train-region-centers` so the run replays exactly.

A geographic rule was tried first and does not survive checking:

| training region | M≥4 events | positive rate |
|---|---|---|
| target, 150 km disc | 90 | 0.122 |
| a North Anatolian corridor | 413 | 0.377 |
| Turkey-wide box | 4039 | **0.971** |

A national label is positive 97% of the time, so it teaches "always yes". And a
hand-drawn fault corridor captures under a fifth of the seismicity in most
longitude bins, because Turkish seismicity is not one latitude band. Statistical
similarity is checkable against the catalogue; a drawn boundary is not.

Selecting twelve regions takes about a minute. Building their features is the
slow part, since the feature builder loops over every hour of every region, so
expect a pooled run to spend considerably longer before the first epoch.
Restricting `--keep-features` to columns that exclude `nnd_log_eta_90d` and
`shannon_entropy_90d` skips the nearest-neighbour precompute entirely.

## Hourly rows are not observations

The label at hour H looks `--horizon-days` forward, so at a 14-day horizon one
qualifying earthquake turns 336 consecutive rows positive. Measured on a 150 km
disc around one station at M≥4.0:

| test block | rows reported | distinct events |
|---|---|---|
| fold 1 | 32448 | 10 |
| fold 2 | 32448 | 19 |
| fold 3 | 32448 | 26 |

An AUC over ten events carries roughly ±0.16, which is larger than every gap
this project reports between a model and its floor. The reported `n` is not a
sample size.

```bash
forecast train ... --eval-stride-hours 168     # score weekly, non-overlapping
forecast train ... --train-stride-hours 168    # and train on them too
```

Both default to 1, which is how every published number was produced. They are
separate flags because they answer separate questions. The evaluation stride
fixes the measurement, which is the standard treatment for autocorrelated
scoring. The training stride asks whether the 336-fold duplication is part of
why validation loss climbs from the first epoch.

Neither raises the AUC. They make the count honest and its uncertainty visible,
which is what has to happen before any other change can be judged. The split
diagnostics now print each split's distinct event count beside its sample count
for the same reason.

Thinning happens after the fold boundaries are cut, never before. The embargo is
applied in hour-index units and the single-split path expresses it as a position
offset, so the two agree only at stride 1. Cutting densely and thinning
afterwards leaves that arithmetic alone, and can only widen a purge, never
narrow one.

## Keeping a trained model

```bash
forecast train ... --out-dir ./trained_model
```

One file per fold per seed, plus a `manifest.json`. Off by default.

A checkpoint carries three things that are useless apart: the weights, the
**training split's** normalization mean and sd, and the **feature order** those
weights were fit to. Neither of the last two is recoverable afterwards. Val and
test are scored through the train split's stats precisely so the later splits'
distribution never reaches the model, and new hours have no training split to
recompute from; `--keep-features` fixes the column order, and a right-width
vector in the wrong order loads without raising and then forecasts from
`mean_mag_30d` as though it were `log1p_dsp`.

Seeds stay in separate files. The published run averages five of them because
per-seed AUC spread reaches 0.17 here, and the reported number is the average of
their *scores*, not a model built from averaged weights. The manifest records
each fold's AUC beside the floor it had to clear and the command that produced
it — a directory of weights is no more a result on this data than a bare AUC is.

```python
from forecast.checkpoint import load_checkpoint

model, (cat_mu, cat_sd), meta = load_checkpoint("trained_model/fold_1_of_2_seed_42.pt")
meta["feature_names"]  # check these against the array you built
meta["target"]         # the horizon and threshold these weights answer
```

## Layout

| module | what it holds |
|---|---|
| `features.py` | the thirteen catalogue features — the substance |
| `catalog.py` | loading the catalogue, the study region, and the hourly labels |
| `model.py` | `CatalogMLPBranch` and its head |
| `evaluate.py` | the floor, the folds, and the one function that reports them |
| `splits.py` | walk-forward CV and its distinct-event diagnostics |
| `data.py` | feature windows, normalized against the training split |
| `regions.py` | the spatial holdout |
| `stations.py` | naming the study region by station code |
| `checkpoint.py` | saving a trained model, and reading one back |
| `train.py` | the harness |

```bash
uv run pytest
```

## Documentation

[`docs/`](docs/) holds the technical reference:
[architecture](docs/architecture.md), [features](docs/features.md),
[data pipeline](docs/data-pipeline.md), [evaluation](docs/evaluation.md),
[study regions](docs/study-regions.md), [usage](docs/usage.md) and
[checkpoints](docs/checkpoints.md).

Ported from `cnn_earthquake`'s
`src/sismokaos/forecasting/cnn_lstm_catalog_waveform_fusion.py`. The feature
builder is verified bit-identical across all thirteen columns; the floor
reproduces to four decimals.
