# Data pipeline

From a catalogue CSV to batched tensors. Modules: `catalog.py`, `features.py`,
`data.py`, `splits.py`.

## Input format

One CSV with `Date`, `Latitude`, `Longitude`, `Magnitude`. Dates parse as
`%d/%m/%Y %H:%M:%S`; unparseable rows are dropped. Extra columns are ignored.

```
Date,Longitude,Latitude,Depth,Rms,Type,Magnitude,Location,EventID
01/01/2007 00:21:44,36.1235,37.0982,6.9,0.3,Md,2.6,Toprakkale (Osmaniye),50516
```

The file is parsed once per path and cached (`catalog._read_catalog`, an LRU of
4). Region selection measures hundreds of candidate discs, and re-parsing a
576k-row file for each one turned a geometry question into minutes of work. The
cached frame is shared, so callers filter it and never mutate it.

## Stage 1: the hourly index

`--catalog-span START END` builds a `DatetimeIndex` at hourly frequency. It is
required, because with no waveform archive nothing else defines how long the
record is.

`truncate_to_reliable_catalog_end` then trims the tail. The label looks forward
by `--horizon-days`, so the final stretch of the index would carry labels
decided by events that have not been recorded yet. Those hours are dropped with
a `buffer_days` equal to the horizon, and the run prints how many.

This is why a span ending 2026-08-29 against a catalogue whose last event is
2025-10-02 reports the index ending 2025-09-18.

## Stage 2: events inside the region

`load_aegean_events` returns sorted times above a magnitude. Its companion
`load_aegean_events_with_location` also returns magnitudes, latitudes and
longitudes, which features 11 and 12 need.

Both take a `Region`. See [study-regions.md](study-regions.md).

## Stage 3: features and labels

`build_catalog_features` produces a `(n_hours, 13)` float32 array. Every feature
is backward-looking, computed from events strictly before the hour in question.
See [features.md](features.md).

Labels come from one of two functions:

- `label_hours` for `--label-mode event`. Positive when a qualifying event falls
  within `--horizon-days` of the hour.
- `label_hours_rate_change` for `--label-mode rate`. Positive when the next
  window holds more events above `--rate-min-mag` than the trailing one. This is
  an acceleration target, driven by roughly 10^3 events instead of 10^1.

Rate mode is the honest target when the event label is near-constant. At M3 over
a 14-day horizon in an active region, "will an earthquake occur" is answered yes
in over 70% of hours and above 95% in active years, which is not a forecasting
problem.

## Stage 4: windows

`CatalogSeqDataset` turns the arrays into samples. One sample is `seq_hours`
consecutive hours of features, plus the label at the window's last hour. The
model reads only that last hour, but the window is kept whole because it defines
the sample's boundary and the purge is expressed in those terms.

### Normalisation

Per-feature standardisation. The critical rule: **validation and test must be
given the training split's statistics**. Computing their own would let the later
splits' distribution into the model. `CatalogSeqDataset` takes a `stats`
argument for exactly this, and `train.py` always passes the training stats down.

Training statistics are sampled from up to 500 windows spread across the *whole*
training split, not the first 500. The archive's opening hours are
unrepresentative for any trailing-window feature, whose lookback is still
filling up. Measured on the rate features, a standard deviation taken from the
first 50 windows understated the true training value by up to 52x
(`rate_log1p_count_90d`, 0.0157 against 0.827). Dividing by it produced z-scores
up to 156, saturated the GELU layers, and made the untrained model the best
checkpoint.

### Pooled normalisation

With `--train-regions`, statistics come from every training region at once, via
`train.pooled_stats`. One shared mean and standard deviation, not one per
region: normalising each separately would erase what distinguishes a quiet
region from an active one, which is the signal the pool exists to supply.

The pooled standard deviation is

```
sqrt(mean(sd_i^2) + var(mu_i))
```

and not the mean of the regions' standard deviations. The naive version counts
only the spread inside each region and drops the spread between their means, so
it always understates and always inflates z-scores. Measured on twelve Turkish
regions the error was only 1.01x to 1.06x, but it is the same class of mistake
as the 52x failure above, so it is computed correctly rather than nearly.

## Stage 5: sampling and splits

Sample end-indices start as every hour from `seq_hours - 1` onward. Folds are cut
from that dense index, and only then are the splits thinned by
`--train-stride-hours` and `--eval-stride-hours`.

The order matters. See
[evaluation.md](evaluation.md#why-thinning-happens-after-the-cut).
