"""How often the record is sampled, and why the default is not the honest one.

An hourly sample and a 14-day horizon mean one earthquake turns 336 consecutive
rows positive. A test block reporting n=32448 can rest on ten events, and an AUC
over ten events carries roughly +/-0.16. That is larger than every gap this
project reports between a model and its floor, so the reported n is not a sample
size and treating it as one is how a noise result gets published.

`--eval-stride-hours 168` scores weekly, non-overlapping samples instead. It
does not raise the AUC. It makes the count honest and the interval visible.

The default stays at 1 because every published number was produced that way, so
the first test here is a regression guard rather than a feature test.

The thinning deliberately happens AFTER the fold boundaries are cut. The embargo
is applied in hour-index units, while the single-split path expresses it as a
position offset, and the two agree only at stride 1. These tests pin that the
gap survives.
"""
import argparse

import numpy as np
import pytest

from forecast.splits import walk_forward_splits
from forecast.train import add_args, run

SEQ_HOURS = 24
HORIZON = 14.0
WEEK = 168


def parse(*extra):
    p = argparse.ArgumentParser()
    add_args(p)
    return p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b", *extra])


def thin(folds, args):
    """The thinning `run` applies, in one place so the tests share its shape."""
    return [(tr[::args.train_stride_hours],
             va[::args.eval_stride_hours],
             te[::args.eval_stride_hours]) for tr, va, te in folds]


@pytest.fixture
def folds():
    """Three walk-forward folds over roughly nineteen years of hours."""
    n = 172321
    valid = np.arange(SEQ_HOURS - 1, n)
    embargo = SEQ_HOURS - 1 + int(round(HORIZON * 24))
    return walk_forward_splits(valid, 3, embargo=embargo), embargo


def test_the_default_keeps_every_hour(folds):
    """The published configuration is one sample per hour, and a stride that
    quietly moved off 1 would rescale every n in the report."""
    dense, _ = folds
    args = parse()
    assert args.train_stride_hours == 1 and args.eval_stride_hours == 1
    for (tr, va, te), (tr2, va2, te2) in zip(dense, thin(dense, args)):
        assert np.array_equal(tr, tr2)
        assert np.array_equal(va, va2)
        assert np.array_equal(te, te2)


def test_the_eval_stride_leaves_training_alone(folds):
    """Scoring on independent samples while training on overlapping ones is the
    standard fix; these are two separate questions and two separate flags."""
    dense, _ = folds
    thinned = thin(dense, parse("--eval-stride-hours", str(WEEK)))
    for (tr, va, te), (tr2, va2, te2) in zip(dense, thinned):
        assert np.array_equal(tr, tr2), "training must be untouched"
        assert len(va2) == pytest.approx(len(va) / WEEK, rel=0.02)
        assert len(te2) == pytest.approx(len(te) / WEEK, rel=0.02)


def test_the_train_stride_leaves_scoring_alone(folds):
    """The other direction, for asking whether the 336-fold duplication in
    training is what drives the overfitting."""
    dense, _ = folds
    thinned = thin(dense, parse("--train-stride-hours", str(WEEK)))
    for (tr, va, te), (tr2, va2, te2) in zip(dense, thinned):
        assert len(tr2) == pytest.approx(len(tr) / WEEK, rel=0.02)
        assert np.array_equal(va, va2)
        assert np.array_equal(te, te2)


def test_weekly_samples_are_a_week_apart(folds):
    """Non-overlapping is the whole point: consecutive samples must not share
    a label window."""
    dense, _ = folds
    _, _, te = thin(dense, parse("--eval-stride-hours", str(WEEK)))[0]
    assert len(np.unique(np.diff(te))) == 1
    assert np.diff(te)[0] == WEEK
    assert WEEK >= SEQ_HOURS, "a stride below the window would still overlap inputs"


def test_the_embargo_survives_thinning(folds):
    """Thinning drops samples inside a block and never moves a boundary, so the
    purge between blocks can only widen. If it narrowed, train labels would
    start encoding what happens in val again."""
    dense, embargo = folds
    for tr, va, te in thin(dense, parse("--eval-stride-hours", str(WEEK),
                                        "--train-stride-hours", str(WEEK))):
        assert va[0] - tr[-1] >= embargo
        assert te[0] - va[-1] >= embargo


def test_a_thin_test_block_says_how_thin(capsys):
    """The count is the point of the diagnostic. A block of 32448 rows driven by
    ten earthquakes has to say so where the row count is printed, not leave it
    to be worked out from the catalogue afterwards."""
    import pandas as pd

    from forecast.splits import print_split_diagnostics
    hours = pd.date_range("2015-01-01", "2015-06-01", freq="h")
    labels = np.zeros(len(hours), dtype=np.float32)
    labels[1700:1900] = 1
    events = pd.to_datetime(["2015-03-16", "2015-03-18"]).to_numpy()
    print_split_diagnostics(hours, labels, np.arange(0, 1000), np.arange(1200, 1500),
                            np.arange(1700, 2000), n_blocks=2, major_times=events)
    out = capsys.readouterr().out
    assert "distinct events" in out
    assert "300 samples,    2 distinct events" in out
    assert "carries roughly +/-0.35" in out


def test_an_empty_test_block_is_undefined_not_bad(capsys):
    """Zero events means the AUC does not exist. Reporting that as a low score
    would be reporting a number that was never computed."""
    import pandas as pd

    from forecast.splits import print_split_diagnostics
    hours = pd.date_range("2015-01-01", "2015-06-01", freq="h")
    labels = np.zeros(len(hours), dtype=np.float32)
    events = pd.to_datetime(["2015-01-05"]).to_numpy()
    print_split_diagnostics(hours, labels, np.arange(0, 1000), np.arange(1200, 1500),
                            np.arange(1700, 2000), n_blocks=2, major_times=events)
    assert "undefined, not low" in capsys.readouterr().out


def test_a_stride_below_one_stops_the_run():
    """0 would empty the splits and -1 would reverse them; neither is a smaller
    run, and numpy would not complain about either."""
    for flag in ("--train-stride-hours", "--eval-stride-hours"):
        with pytest.raises(SystemExit, match="must be at least 1"):
            run(parse(flag, "0"))
        with pytest.raises(SystemExit, match="must be at least 1"):
            run(parse(flag, "-1"))
