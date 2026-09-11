# Features

`src/forecast/features.py`. Thirteen catalogue features, plus eight more when
`--rate-features` is passed. This is where the substance of the project is; the
network is the small part.

## Two source catalogues

Features split by which event set they read.

- **The target set**, magnitude at least `--threshold`. These are the same
  events the label is built from.
- **The background set**, magnitude at least `--bg-min-mag`, default 3.0. A
  lower completeness threshold and therefore a much larger set.

The split exists because the target set is too small to fit a distribution to.
A 150 km region might hold 90 events above M4 and 1390 above M3. A b-value or an
inter-event coefficient of variation estimated from 90 points spread over twenty
years is noise.

**Setting `--threshold` equal to `--bg-min-mag` collapses this distinction.**
Both groups then read one identical event set and the design is defeated without
anything raising. If you lower the threshold, lower the background with it.

## The thirteen

Index order matters: it is the column order of the feature array, and
`--keep-features` selects by name against it.

### Timing, from the target set

| # | name | definition |
|---|---|---|
| 0 | `log1p_dsp` | `log1p(days since previous qualifying event)`, NaN filled with 3650 |
| 1 | `count_7d` | qualifying events in the trailing 7 days |
| 2 | `count_30d` | trailing 30 days |
| 3 | `count_90d` | trailing 90 days |

### Magnitude distribution, from the background set

| # | name | definition |
|---|---|---|
| 4 | `mean_mag_30d` | mean magnitude over the trailing 30 days |
| 5 | `max_mag_90d` | maximum magnitude over the trailing 90 days |
| 6 | `b_value_90d` | Aki maximum-likelihood b, `(1/ln10) / mean excess magnitude` |
| 7 | `energy_sqrt_30d` | `sqrt(sum(10^(1.5 M)))` over 30 days |
| 8 | `mean_interevent_90d` | mean gap between events, in days |
| 9 | `cv_interevent_90d` | standard deviation over mean of those gaps |
| 10 | `mag_deficit_90d` | `bg_min_mag + log10(n)/b` minus the observed maximum |

`mag_deficit_90d` is a Gutenberg-Richter extrapolation of the magnitude the
window should have produced, minus what it did produce. It is a proxy for
overdue release.

`cv_interevent_90d` is the strongest of the original eleven by recursive feature
elimination, and Convertito et al. 2024 (Sci. Rep. 14:2964) arrived at the same
quantity independently.

### Spatial, from the background set with coordinates

| # | name | definition |
|---|---|---|
| 11 | `nnd_log_eta_90d` | mean Zaliapin-Ben-Zion nearest-neighbour `log10(eta)` |
| 12 | `shannon_entropy_90d` | energy-weighted entropy over a 10x10 spatial grid |

Both follow Convertito et al. 2024. The nearest-neighbour distance separates
triggered events from background ones:

```
eta_ij = T_ij * R_ij
T_ij   = (t_j - t_i) * 10^(-0.5 b m_i)
R_ij   = haversine_km(i, j)^df * 10^(-0.5 b m_i)
```

with `NND_FRACTAL_DIM = 1.6`, the standard literature default. The search is
bounded to the `NND_LOOKBACK = 500` most recent candidate parents per event,
because the full pairwise computation is intractable at catalogue scale. True
nearest neighbours are overwhelmingly recent for local clustering, so this is a
tractable approximation rather than an exact result.

**These two are expensive.** The nearest-neighbour precompute is an O(n x 500)
haversine in a Python loop and fires whenever coordinates are supplied.
`train.py` skips loading coordinates entirely when `--keep-features` excludes
both of them, which is a large saving on multi-region runs.

**The entropy grid follows the study region.** `build_catalog_features` takes a
`grid_bbox`, defaulting to the Aegean box that produced the published figures.
A run restricted to a smaller region must pass that region's box, or every event
lands in two or three of the hundred cells and the column reports the same
number every hour. `train.py` and `regions.py` both pass `region.bbox`.

## Fallbacks when a window is thin

Several features have hardcoded defaults when the trailing window holds too few
background events. These are not errors, but they do mean the column is a
constant for those hours.

| condition | affected | value |
|---|---|---|
| fewer than 5 events in 90 d | `b_value_90d` | 1.0, the global tectonic average |
| fewer than 3 events in 90 d | `mean_interevent_90d` | 90.0 |
| fewer than 3 events in 90 d | `cv_interevent_90d` | 1.0, Poisson-like |
| no events in 90 d | `max_mag_90d` | `bg_min_mag` |
| no events in 30 d | `mean_mag_30d`, `energy_sqrt_30d` | 0.0 |
| fewer than 2 events in 90 d | `shannon_entropy_90d` | 0.0 |
| no usable neighbour distances | `nnd_log_eta_90d` | 0.0 |
| no coordinates supplied | both spatial features | 0.0 |

This matters more than it looks. Measured on a 150 km region around one station
with a background catalogue of 1390 events, `b_value_90d` fell back to its
default in 21.4% of hours overall, but the rate was uneven: 0.0% in 2009 and
2013, 60.1% in 2017, 86.5% in 2021. Through a quiet decade, three of thirteen
inputs are frequently constant.

Check this before concluding a feature is uninformative.

## The rate features

Appended only with `--rate-features`, and appended **before** `--keep-features`
subsets, so they can be selected by name alongside the originals.

Five log counts of events above `--rate-min-mag` over windows of 3, 7, 14, 30
and 90 days, named `rate_log1p_count_<w>d`. Counts are log1p'd because raw
counts are heavy-tailed, with a median of 9 against a maximum of 297 at 14 days.

Three log ratios of per-day rates over the pairs (3, 14), (7, 30) and (14, 90),
named `rate_logratio_<a>_<b>`. Ratios are logged so acceleration and
deceleration are symmetric around zero.

They exist because none of the thirteen encodes the trailing count of the
low-magnitude events that define the rate target: `count_7d` and friends count
events above `--threshold`, not above `--rate-min-mag`. Without them a rate-mode
run is trying to beat a persistence floor built from precisely the number it was
never given. Measured, that lost fold 1 at 0.6573 against a floor of 0.7991
while scoring a *positive* Brier skill of +0.129, which is a model learning
something real but unable to rank without the rate signal.

## Selecting a subset

`--keep-features` takes names and preserves the order you give them. That order
becomes part of what a trained model means, which is why it travels inside every
checkpoint. See [checkpoints.md](checkpoints.md).

The published configuration keeps four:

```
--keep-features log1p_dsp mean_mag_30d cv_interevent_90d mag_deficit_90d
```

None of those is spatial, so this subset also skips the nearest-neighbour
precompute.
