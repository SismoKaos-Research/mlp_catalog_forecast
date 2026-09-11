"""The defaults, pinned to the values the published figures were produced with.

This file exists because the port got five of them wrong, and only one showed
up as an error. `--train-frac` and `--val-frac` changed the single-fold split
sizes visibly (12,248 train became 10,498); `--weight-decay` changed nothing
visible at all and silently moved every AUC by ~0.001.

A default is part of the model. The published configuration passes eight flags
and inherits the rest, so a default that drifts is a different model reported
under the same name -- and unlike a renamed flag, nothing raises.
"""
import argparse

import pytest

from forecast.train import add_args

# From `cnn_earthquake/src/sismokaos/forecasting/cnn_lstm_catalog_waveform_fusion.py`,
# read off its `parse_args`. Two are set in its `main()` rather than declared:
# at `--channels catalog` it overrides dropout to 0.2 (a tiny 2-11-input MLP --
# aggressive dropout knocks it off a good basin instead of regularizing it) and
# lr to 3e-4. Since this project IS the catalog-only arm, those are the
# declared defaults here.
ORIGINAL_DEFAULTS = {
    "threshold": 4.5,
    "bg_min_mag": 3.0,
    "horizon_days": 14.0,
    "rate_min_mag": 3.0,
    "seq_hours": 24,
    "cat_hidden": 16,
    "fusion_hidden": 32,
    "epochs": 40,
    "batch_size": 16,
    "weight_decay": 0.1,
    "patience": 8,
    "checkpoint_metric": "auc",
    "ensemble_seeds": "42,43,44",
    "train_frac": 0.70,
    "val_frac": 0.15,
    "cv_folds": 1,
    "region_test_side": "high",
    "dropout": 0.2,          # main() sets this for --channels catalog
    "lr": 3e-4,              # main() sets this for --channels catalog
}


@pytest.fixture
def defaults():
    p = argparse.ArgumentParser()
    add_args(p)
    return vars(p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b"]))


@pytest.mark.parametrize("name,expected", sorted(ORIGINAL_DEFAULTS.items()))
def test_default_matches_the_original(defaults, name, expected):
    assert name in defaults, f"--{name.replace('_', '-')} is missing from this port"
    assert defaults[name] == expected


def test_the_catalogue_is_required():
    """There is no other input; a run without it is not a smaller run."""
    p = argparse.ArgumentParser()
    add_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--catalog-span", "2000-01-01", "2026-08-12"])


def test_the_span_is_required():
    """With no waveform archive, nothing else defines how long the record is."""
    p = argparse.ArgumentParser()
    add_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--catalog-path", "x"])


def test_the_waveform_flags_are_gone(defaults):
    """Their presence would imply an ablation this project cannot run."""
    for gone in ("channels", "data_root", "consolidated", "max_days", "cnn_out",
                 "wave_hidden", "load_catalog_branch", "freeze_catalog",
                 "save_catalog_branch", "detect_window_hours"):
        assert gone not in defaults, f"--{gone.replace('_', '-')} should not exist here"


def test_the_study_region_defaults_to_the_published_one(defaults):
    """Naming no region has to mean the Aegean box every published number was
    produced over -- a default region that drifted would be a different model
    reported under the same name, and nothing would raise."""
    assert defaults["catalog_bbox"] is None
    assert defaults["catalog_radius"] is None
    assert defaults["station_radius"] is None
    assert defaults["station_catalog"] is None


def test_sampling_defaults_to_every_hour(defaults):
    """Every published number was produced one-sample-per-hour. A stride that
    drifted off 1 would silently rescale every n in the report."""
    assert defaults["train_stride_hours"] == 1
    assert defaults["eval_stride_hours"] == 1


def test_the_station_flags_are_gone(defaults):
    """They cut the catalogue down to events near a seismometer, and no
    seismometer is read here -- so they narrowed the M>=4.5 set, already ~10^2
    events, in exchange for nothing. The spatial holdout that does mean
    something for a region-wide branch is --region-split."""
    for gone in ("stations", "max_station_dist_km"):
        assert gone not in defaults, f"--{gone.replace('_', '-')} should not exist here"


def test_detect_is_not_an_offered_label_mode():
    """It needs the waveform branch: its label is a threshold on dsp, which the
    catalogue branch carries, so it would score ~1.0 by construction."""
    p = argparse.ArgumentParser()
    add_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                      "--label-mode", "detect"])
