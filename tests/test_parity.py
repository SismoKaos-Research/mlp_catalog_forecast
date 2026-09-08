"""Does this port still produce what `cnn_earthquake` produced?

The network here is 961 parameters; the substance is the feature builder and the
floor. So parity is checked on those directly, against the parent repo, rather
than inferred from a training run.

Skipped when `cnn_earthquake` is not importable -- it is not a dependency of
this project, and it should not become one. These are the tests you run when
you have both checkouts and are about to change `features.py`.

Set `CNN_EARTHQUAKE_SRC` to point at its `src/` if it is not at the sibling
path this looks for by default.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DEFAULT_SRC = Path(__file__).resolve().parents[2] / "cnn_earthquake" / "src"
SRC = Path(os.environ.get("CNN_EARTHQUAKE_SRC", DEFAULT_SRC))
CATALOG = SRC.parent / "catalogs" / "catalog_current.csv"

pytestmark = pytest.mark.skipif(
    not (SRC / "sismokaos" / "catalog.py").exists() or not CATALOG.exists(),
    reason="cnn_earthquake checkout or its catalogue not found; parity is only "
           "checkable with both present")


@pytest.fixture(scope="module")
def parent():
    """The parent repo's implementations, imported from its checkout."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from sismokaos.baselines import rate_persistence_auc
    from sismokaos.catalog import (label_hours, load_aegean_events,
                                   load_aegean_events_with_location)
    from sismokaos.forecasting.cnn_lstm_catalog_waveform_fusion import \
        build_catalog_features
    return dict(build_catalog_features=build_catalog_features,
                load_aegean_events=load_aegean_events,
                load_aegean_events_with_location=load_aegean_events_with_location,
                label_hours=label_hours,
                rate_persistence_auc=rate_persistence_auc)


@pytest.fixture(scope="module")
def slice_():
    """A fixed three-month slice, small enough to build in seconds."""
    return pd.date_range("2024-01-01", "2024-04-01", freq="h")


def test_all_thirteen_features_are_bit_identical(parent, slice_):
    """The substance of the project, pinned column by column."""
    from forecast.catalog import (load_aegean_events,
                                  load_aegean_events_with_location)
    from forecast.features import FEATURE_NAMES, build_catalog_features

    major = load_aegean_events(str(CATALOG), 4.5)
    bt, bm, bla, blo = load_aegean_events_with_location(str(CATALOG), 3.0)
    dsp = np.zeros(len(slice_))

    mine = build_catalog_features(slice_, major, dsp, bt, bm, 3.0, bla, blo)
    theirs = parent["build_catalog_features"](slice_, major, dsp, bt, bm, 3.0,
                                              bla, blo)
    assert mine.shape == theirs.shape
    for j, name in enumerate(FEATURE_NAMES):
        assert np.array_equal(np.nan_to_num(mine[:, j], nan=-999.0),
                              np.nan_to_num(theirs[:, j], nan=-999.0)), \
            f"feature {name!r} differs from cnn_earthquake's"


def test_the_catalogue_loaders_return_the_same_events(parent, slice_):
    from forecast.catalog import (load_aegean_events,
                                  load_aegean_events_with_location)
    assert np.array_equal(load_aegean_events(str(CATALOG), 4.5),
                          parent["load_aegean_events"](str(CATALOG), 4.5))
    a = load_aegean_events_with_location(str(CATALOG), 3.5)
    b = parent["load_aegean_events_with_location"](str(CATALOG), 3.5)
    for mine, theirs in zip(a, b):
        assert np.array_equal(mine, theirs)


def test_the_labels_are_identical(parent, slice_):
    from forecast.catalog import label_hours, load_aegean_events
    major = load_aegean_events(str(CATALOG), 4.5)
    assert np.array_equal(label_hours(slice_, major, 14.0),
                          parent["label_hours"](slice_, major, 14.0))


def test_the_persistence_floor_is_identical(parent):
    """The bar every result here is reported against."""
    from forecast.evaluate import rate_persistence_auc
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    counts = rng.poisson(3.0, 500)
    assert rate_persistence_auc(y, counts) == parent["rate_persistence_auc"](y, counts)


def test_the_model_initialises_to_the_same_weights(parent):
    """961 parameters, same keys, same values under one seed."""
    import torch

    from forecast.model import CatalogMLPNet
    from forecast.seeding import seed_everything
    sys.path.insert(0, str(SRC))
    from sismokaos.forecasting.cnn_lstm_catalog_waveform_fusion import \
        CatalogWaveformFusionNet

    seed_everything(42)
    theirs = CatalogWaveformFusionNet(catalog_dim=4, cat_hidden=16,
                                      fusion_hidden=32, dropout=0.2,
                                      channels="catalog").state_dict()
    seed_everything(42)
    mine = CatalogMLPNet(catalog_dim=4, cat_hidden=16, fusion_hidden=32,
                         dropout=0.2).state_dict()
    assert list(mine) == list(theirs)
    assert all(torch.equal(mine[k], theirs[k]) for k in mine)
