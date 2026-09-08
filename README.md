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

`--region-split lat|lon` holds out half the Aegean bbox instead of a stretch of
time: a patch of crust whose events the model has never seen. Off by default,
kept because it is the harder generalisation question.

## Layout

| module | what it holds |
|---|---|
| `features.py` | the thirteen catalogue features — the substance |
| `catalog.py` | loading the catalogue, and the hourly labels |
| `model.py` | `CatalogMLPBranch` and its head |
| `evaluate.py` | the floor, the folds, and the one function that reports them |
| `splits.py` | walk-forward CV and its distinct-event diagnostics |
| `data.py` | feature windows, normalized against the training split |
| `regions.py` | the spatial holdout |
| `train.py` | the harness |

```bash
uv run pytest
```

Ported from `cnn_earthquake`'s
`src/sismokaos/forecasting/cnn_lstm_catalog_waveform_fusion.py`. The feature
builder is verified bit-identical across all thirteen columns; the floor
reproduces to four decimals.
