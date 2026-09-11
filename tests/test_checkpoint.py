"""What a saved model has to carry to still be the same model tomorrow.

Weights alone are not a checkpoint here. The model reads z-scores computed
from the TRAIN split's mean and sd (`data.py` hands those same stats to val
and test so the later splits' distribution never reaches the model), and it
reads its features in whatever order `--keep-features` listed them. New hours
arriving after the run reconstruct neither: there is no training split to
recompute stats from, and a right-width-wrong-order feature vector loads
without raising and forecasts from the wrong column.

So these tests pin the two silent failures -- stats that came from somewhere
other than the train split, and a feature list that does not match the order
the weights were trained on -- rather than the fact that a file appears.
"""
import argparse

import numpy as np
import pytest
import torch

from forecast.catalog import Region
from forecast.checkpoint import (CHECKPOINT_FORMAT, fold_slug, load_checkpoint,
                                 save_checkpoint)
from forecast.data import CatalogSeqDataset
from forecast.train import add_args, train_one_seed

SEQ_HOURS = 4
N_HOURS = 400


@pytest.fixture
def args(tmp_path):
    """The published defaults, with the training loop cut to one short epoch."""
    p = argparse.ArgumentParser()
    add_args(p)
    a = p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                      "--seq-hours", str(SEQ_HOURS), "--epochs", "1",
                      "--batch-size", "32", "--out-dir", str(tmp_path / "model"),
                      "--keep-features", "log1p_dsp", "mean_mag_30d"])
    return a


@pytest.fixture
def split():
    """A two-feature toy run: features on wildly different scales, so a
    checkpoint carrying the wrong normalization cannot accidentally pass."""
    rng = np.random.default_rng(0)
    cat_features = np.stack([rng.normal(0, 1, N_HOURS),
                             rng.normal(500, 50, N_HOURS)], axis=1).astype(np.float32)
    labels = (rng.random(N_HOURS) < 0.3).astype(np.float32)
    idx = np.arange(SEQ_HOURS - 1, N_HOURS)
    return cat_features, labels, idx[:250], idx[250:320], idx[320:]


def test_the_saved_stats_are_the_train_split_s(args, split):
    """Val and test are scored through the train split's mean and sd, and a
    checkpoint that saved its own would z-score new hours differently than
    every number the run reported."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])

    _, (mu, sd), _ = load_checkpoint(path)
    expected = CatalogSeqDataset(cat_features, labels, SEQ_HOURS, train_idx).stats
    assert np.allclose(mu, expected[0])
    assert np.allclose(sd, expected[1])

    test_only = CatalogSeqDataset(cat_features, labels, SEQ_HOURS, test_idx).stats
    assert not np.allclose(mu, test_only[0]), "these came from the test split"


def test_the_reloaded_model_scores_a_window_identically(args, split):
    """A checkpoint that does not reproduce its own run's scores is a
    different model reported under the run's numbers."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])

    model, (mu, sd), meta = load_checkpoint(path)
    ds = CatalogSeqDataset(cat_features, labels, SEQ_HOURS, test_idx, stats=(mu, sd))
    x = torch.stack([ds[i][0] for i in range(8)])
    with torch.no_grad():
        first, second = model(x), model(x)
    assert torch.equal(first, second), "eval mode should have disabled dropout"
    assert meta["arch"]["seq_hours"] == SEQ_HOURS


def test_the_feature_order_travels_with_the_weights(args, split):
    """`--keep-features` fixes the column order, and a mismatch is the quiet
    kind: the width still matches, so nothing raises."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])
    _, _, meta = load_checkpoint(path)
    assert meta["feature_names"] == ["log1p_dsp", "mean_mag_30d"]


def test_nothing_is_written_without_the_flag(split):
    """Saving is opt-in: a sweep writes one file per fold per seed."""
    p = argparse.ArgumentParser()
    add_args(p)
    a = p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                      "--seq-hours", str(SEQ_HOURS), "--epochs", "1",
                      "--batch-size", "32"])
    assert a.out_dir is None

    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(a, 42, cat_features, labels, train_idx, val_idx,
                                test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])
    assert path is None


def test_a_foreign_format_fails_loudly(args, split, tmp_path):
    """An old file loading into a model that reads a field differently is the
    failure this version number exists to prevent."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])
    blob = torch.load(path, weights_only=True)
    blob["format"] = CHECKPOINT_FORMAT + 1
    torch.save(blob, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="checkpoint format"):
        load_checkpoint(tmp_path / "future.pt")


@pytest.mark.parametrize("label,expected", [
    ("single split", "single_split"),
    ("fold 1/2", "fold_1_of_2"),
    ("fold 10/10", "fold_10_of_10"),
])
def test_the_fold_label_becomes_a_filename(label, expected):
    """Fold labels are printed with a slash in them; filenames cannot hold one."""
    assert fold_slug(label) == expected


def test_a_pooled_run_still_saves_its_normalization(args, split):
    """With extra training regions the training set is a ConcatDataset, which
    carries no stats of its own. Reading them off it saved nothing and crashed
    the run after the first seed had already trained."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    pool = [(Region.from_flags(radius=(37.5, 27.5, 120.0)),
             cat_features * 2.0 + 5.0, labels)]
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"],
                                pool=pool)
    _, (mu, sd), _ = load_checkpoint(path)

    solo = CatalogSeqDataset(cat_features, labels, SEQ_HOURS, train_idx).stats
    assert not np.allclose(mu, solo[0]), \
        "the pool must be part of the saved statistics, not just the study region"
    assert np.all(sd > 0)

    _, _, meta = load_checkpoint(path)
    assert meta["train_regions"] == [[37.5, 27.5]], \
        "the checkpoint has to say which regions it was fit on"


def test_the_checkpoint_records_how_it_was_sampled(args, split):
    """A model fit on weekly, non-overlapping samples is a different fit than
    one fit on every hour, and the weights do not say which."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])
    _, _, meta = load_checkpoint(path)
    assert meta["sampling"] == {"train_stride_hours": 1, "eval_stride_hours": 1}


def test_the_checkpoint_records_what_it_forecasts(args, split):
    """The same weights answer a different question at a different horizon or
    threshold, and nothing in the weights records which one they were fit to."""
    cat_features, labels, train_idx, val_idx, test_idx = split
    _, _, path = train_one_seed(args, 42, cat_features, labels, train_idx,
                                val_idx, test_idx, torch.device("cpu"),
                                fold_label="single split",
                                feature_names=["log1p_dsp", "mean_mag_30d"])
    _, _, meta = load_checkpoint(path)
    assert meta["target"]["horizon_days"] == args.horizon_days
    assert meta["target"]["threshold"] == args.threshold
    assert meta["target"]["label_mode"] == "event"
