# Checkpoints

`src/forecast/checkpoint.py`. Off by default; `--out-dir DIR` turns it on.

A run writes one file per fold per seed, named from the fold label and the seed,
plus a `manifest.json`.

```
trained_model/
  fold_1_of_3_seed_42.pt
  fold_1_of_3_seed_43.pt
  fold_2_of_3_seed_42.pt
  ...
  manifest.json
```

## Why one file per seed

The published configuration averages several seeds because per-seed AUC spread
reaches 0.17. An average of *weights* is not the model that produced the
reported number; the average of their *scores* is. Keeping seeds separate is
what lets that average be reproduced rather than approximated.

## What a checkpoint carries

Three things that are useless apart, and none recoverable from the others.

**The weights.**

**The training split's normalisation statistics.** Validation and test are
scored through the training split's mean and standard deviation, precisely so
the later splits' distribution never reaches the model. New hours arriving after
the run have no training split to recompute from, so a checkpoint without its
statistics can only be fed z-scores it was never trained on.

**The feature order.** `--keep-features` changes both the input width and the
column order. A wrong width raises on `load_state_dict`; a right width in the
wrong order loads cleanly and forecasts from `mean_mag_30d` as though it were
`log1p_dsp`. That is the quiet failure, so the names travel with the file and
callers are expected to check them.

## Schema

Saved with `torch.save`. Statistics are stored as tensors and metadata as plain
Python values, so the file loads under `weights_only=True`. A checkpoint should
not need to be trusted code to be read.

| key | contents |
|---|---|
| `format` | `CHECKPOINT_FORMAT`, currently 1 |
| `state_dict` | the model weights |
| `cat_mu`, `cat_sd` | training statistics, shape `(1, n_features)` |
| `feature_names` | column order the model reads |
| `arch` | `catalog_dim`, `cat_hidden`, `fusion_hidden`, `dropout`, `seq_hours` |
| `target` | `label_mode`, `threshold`, `horizon_days`, `bg_min_mag`, `rate_min_mag`, `region` |
| `sampling` | `train_stride_hours`, `eval_stride_hours` |
| `train_regions` | centres of any pooled training regions, empty otherwise |
| `fold`, `seed` | provenance |
| `checkpoint_metric`, `val_metric` | which metric selected the epoch, and its best value |

`target.region` records `bbox`, `center`, `radius_km`, `station` and a
human-readable `described`. The same weights answer a different question in a
different patch of crust, at a different horizon, or at a different threshold,
and nothing in the weights says which.

`sampling` and `train_regions` exist for the same reason: a model fit on weekly
samples pooled across twelve regions is not the model the same command produced
without them.

Readers should use `.get()` for `sampling` and `train_regions`. Files written
before those fields existed are still valid at this format version, because no
existing field changed meaning.

`load_checkpoint` refuses a file written under a different `format` rather than
loading it into a model that reads a field differently.

## Loading one

```python
from forecast.checkpoint import load_checkpoint

model, (cat_mu, cat_sd), meta = load_checkpoint("trained_model/fold_1_of_3_seed_42.pt")

meta["feature_names"]   # check against the array you built, in this order
meta["target"]          # horizon, threshold and region these weights answer
meta["arch"]["seq_hours"]
```

The model comes back in eval mode, so dropout is disabled and repeated calls on
the same input agree. The statistics come back as float32 arrays shaped
`(1, n_features)`, ready to pass to `CatalogSeqDataset(..., stats=...)`.

To reproduce a fold's reported number, load every seed for that fold, score each
one, and average the **probabilities**.

## The manifest

A directory of weights is not a result any more than a bare AUC is, so
`manifest.json` carries each fold's AUC beside the floor it had to clear, the
per-seed spread, the full argument set, the study region, the sampling strides,
and the command that produced the run.

```json
{
  "format": 1,
  "command": "forecast train --catalog-path ... --train-regions 12",
  "args": { "...": "every parsed argument" },
  "feature_names": ["log1p_dsp", "mean_mag_30d", "cv_interevent_90d", "mag_deficit_90d"],
  "region": { "described": "150km around KO.BODT (37.0622, 27.3103)" },
  "sampling": { "train_stride_hours": 1, "eval_stride_hours": 168 },
  "seeds": [42, 43, 44],
  "checkpoints": ["fold_1_of_3_seed_42.pt"],
  "folds": [
    { "fold": "fold 1/3", "auc": 0.5805, "floor": 0.5379,
      "beats_floor": true, "per_seed_aucs": [0.58], "seed_spread": 0.0, "n": 194 }
  ]
}
```

## A caution about output directories

Re-running into a directory that already holds checkpoints overwrites the
manifest but leaves older `.pt` files in place, and drawn seeds differ between
runs. The directory then mixes two runs while the manifest describes only one.
Use a fresh `--out-dir` per run.
