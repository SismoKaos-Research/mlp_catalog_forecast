# Evaluation

`src/forecast/evaluate.py`, `splits.py`, `metrics.py`. This is the part of the
project with the strongest opinions, because the parent project's central
methodological finding is that a pooled number misleads on this data.

## An AUC is not a result

`fold_result` computes the model's AUC, the floor it had to clear on *that*
fold, and the per-seed spread, all at once. There is no code path that produces
one without the others, and `summarise` raises `TypeError` if handed anything
it did not build.

That is structural rather than stylistic:

- fold standard deviation runs 0.07 to 0.16, which dwarfs every gap between
  models
- one earlier forecaster beat its own fold's floor in 2 of 5 folds while losing
  to persistence on average
- per-seed AUC spread reaches 0.17, so a single fixed seed hides run-to-run
  variance behind one sample of it

`FoldResult` is frozen and carries `label`, `auc`, `floor`, `base_rate_auc`,
`persistence_auc`, `per_seed_aucs`, `n` and the full `report`.

## The floor

```
floor = max(0.5, base_rate_auc, max(persistence_auc, 1 - persistence_auc))
```

Two baselines, and the floor is whichever is higher.

**Base rate** is the majority-class prediction, which scores 0.5 by
construction.

**Persistence** differs by label mode. For `event` it is the rule "it happened
recently, so it will happen again", `days_since_prev <= horizon_days`. For
`rate` it is the trailing count, via `rate_persistence_auc`.

### The floor is oriented

An anti-predictive rule is exactly as exploitable as a predictive one, because
you invert it. So the achievable baseline is `max(auc, 1 - auc)`.

Rate mode always did this. Event mode did not, which silently collapsed the
floor to chance whenever persistence landed below 0.5. That is what made one
n=4 event-mode result look like it cleared a 0.5000 bar when the properly
oriented bar was about 0.58.

`metrics.safe_auc` takes an `oriented` flag for this. **Use it for baselines,
never for the model**, which does not get to choose its sign after seeing the
answer.

## Walk-forward folds

`walk_forward_splits` cuts the record into `cv_folds + 2` chronological blocks.
Training expands one block per fold while validation and test slide forward.

```
fold 1: train B0           val B1  test B2
fold 2: train B0+B1        val B2  test B3
fold 3: train B0+B1+B2     val B3  test B4
```

Training always starts at the beginning of the record and only grows. Test
blocks never overlap, and no fold trains on anything later than what it tests.

Every fold trains a **fresh** model from random initialisation, and a fresh one
per seed. Nothing carries over. Normalisation statistics are also recomputed per
fold from that fold's training block. With three folds and five seeds that is
fifteen independently trained models.

One property to keep in mind when reading results: fold 1 trains on about a
fifth of the record and fold 3 on three fifths, so a fold-to-fold difference
mixes a different test period with a different amount of training data. That is
inherent to expanding-window validation, and it is why the summary reports how
many folds cleared their own floor rather than averaging them.

`--balanced-folds` places boundaries by positive-label mass rather than equal
hour count, so one sustained swarm cannot fill a block. It suits balanced
targets like rate mode. On a rare clustered target it degenerates: measured, it
produced a 0.999-positive test block where AUC is meaningless, so the run warns
when a block falls outside 0.02 to 0.98.

## The embargo

The label at hour H is decided by events in `[H, H + horizon]`. Without a gap,
the last stretch of every block carries labels determined by events in the
*next* block, which is train labels encoding what happens in validation.

```
embargo = seq_hours - 1 + round(horizon_days * 24)
```

At `--seq-hours 24 --horizon-days 14` that is 359 hours, about 15 days. The
`seq_hours - 1` term removes input-window overlap; the horizon term removes
label overlap. Without it, roughly 9% of samples leak. This is the purging and
embargoing of Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7.

scikit-learn only gained a `gap` parameter for this in 0.24, and most code
leaves it at zero.

### Why thinning happens after the cut

`walk_forward_splits` applies the embargo by comparing hour indices. The
single-split path in `train.py` expresses it as a *position* offset into the
index array. Those agree only while every hour is a sample.

So folds are cut on the dense index and each split is thinned afterwards. That
leaves the arithmetic alone, keeps the spatial split's cut aligned with where
the test split starts, and can only widen a realised gap, never narrow one.

## Hourly rows are not observations

At a 14-day horizon a single earthquake turns 336 consecutive rows positive.
Measured on a 150 km region at M4:

| test block | rows reported | distinct events |
|---|---|---|
| fold 1 | 32448 | 10 |
| fold 2 | 32448 | 19 |
| fold 3 | 32448 | 26 |

The standard error on an AUC over ten positives is roughly 0.16, which is larger
than every gap this project reports between a model and its floor. The reported
`n` is not a sample size.

`--eval-stride-hours 168` scores weekly, non-overlapping samples instead.
`--train-stride-hours` does the same for training, which is a separate question
about whether the 336-fold duplication drives the overfitting. Both default to
1, which is how every published number was produced.

**Neither raises the AUC.** They make the count honest and the interval visible,
which has to happen before any other change can be judged.

`print_split_diagnostics` prints each split's distinct event count beside its
sample count, warns below 30 events in a test block with the rough interval, and
reports a block with zero qualifying events as undefined rather than as a low
score. It also warns when the test positive rate differs from training by more
than 1.5x, which usually means a swarm or a quiet period landed in one split.

## Reading a result

In order:

1. **How many folds cleared their own floor.** That is the verdict.
2. **The per-seed spread** within each fold.
3. **Brier skill against persistence**, which can be positive while the AUC
   loses. That combination means the model learned something real but cannot
   rank without a signal it was not given.

A pooled improvement on this data is not evidence of one. `summarise` prints an
explicit warning when the fold spread exceeds the margin over the floor, which
is the condition under which the mean misleads.

## Prospective testing

In earthquake forecasting the accepted standard is prospective testing, as CSEP
runs it: declare the model, then score it on data that did not exist when it was
declared. Retrospective walk-forward is the best available proxy, not a
substitute, because every choice made so far was informed by the whole
catalogue.
