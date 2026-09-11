# catalog_mlp documentation

Technical reference for the forecaster in `src/forecast`. The top-level
[`README.md`](../README.md) is the short version; these pages are the detail
behind it.

| page | what it covers |
|---|---|
| [architecture.md](architecture.md) | the network, why it is not a sequence model, parameter counts |
| [features.md](features.md) | all thirteen catalogue features, their fallbacks, the rate set |
| [data-pipeline.md](data-pipeline.md) | catalogue to hourly index to windows to splits |
| [evaluation.md](evaluation.md) | floors, walk-forward folds, the embargo, effective sample size |
| [study-regions.md](study-regions.md) | choosing where to forecast, and training on more regions than that |
| [usage.md](usage.md) | the CLI, every flag, worked runs |
| [checkpoints.md](checkpoints.md) | what a saved model carries and how to load it |

## Orientation

The project forecasts whether an earthquake of at least `--threshold` occurs
within `--horizon-days` of a given hour, in a given patch of crust, using the
earthquake catalogue as its only input. There is no waveform archive, and that
absence is a result rather than a limitation: see
[architecture.md](architecture.md#what-is-deliberately-absent).

Three claims shape every design decision in the codebase, and each has its own
page:

1. **The signal is in the features, not the network.** The model is under 1200
   parameters and is the smallest interesting part of the system.
2. **An AUC is not a result without the floor it had to clear on its own fold.**
   Nothing in `evaluate.py` can produce one without the other.
3. **Hourly rows are not observations.** A test block of 32448 rows has been
   measured at ten distinct earthquakes.
