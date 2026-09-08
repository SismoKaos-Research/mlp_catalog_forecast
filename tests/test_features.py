"""The thirteen catalogue features, and the ones that are expensive or fragile.

The network here is two Linear layers; the substance is entirely in what it is
fed. So these pin the feature builder rather than the model:

**Nothing may look forward.** Every feature is a trailing-window statistic. A
feature that sees past its own hour is the whole task solved by accident, and it
would not be visible in any accuracy number — it would just look like a very
good forecaster.

**The location features are the expensive ones.** The nearest-neighbour
precompute is an O(n_bg x 500) haversine loop in Python, and it fires whenever
coordinates are supplied. A run keeping four non-spatial features must not pay
for two it discards, which is why the trainer passes `bg_lats=None` then.
"""
import numpy as np
import pandas as pd
import pytest

from forecast.features import (ALL_FEATURE_NAMES, CATALOG_DIM, FEATURE_NAMES,
                               RATE_FEATURE_NAMES, build_catalog_features,
                               build_rate_features)


@pytest.fixture
def hours():
    return pd.date_range("2024-03-01", periods=400, freq="h")


@pytest.fixture
def events():
    """Background events through February, then a gap."""
    t = np.array([np.datetime64("2024-02-01") + np.timedelta64(6 * i, "h")
                  for i in range(80)])
    rng = np.random.default_rng(0)
    return t, rng.uniform(3.0, 5.5, len(t)), rng.uniform(36.5, 39.5, len(t)), \
        rng.uniform(25.5, 29.5, len(t))


# --- shape and naming ------------------------------------------------------

def test_the_feature_count_matches_the_names(hours, events):
    bt, bm, bla, blo = events
    dsp = np.full(len(hours), 5.0)
    f = build_catalog_features(hours, bt[:5], dsp, bt, bm, 3.0, bla, blo)
    assert f.shape == (len(hours), CATALOG_DIM)
    assert len(FEATURE_NAMES) == CATALOG_DIM


def test_rate_features_extend_the_name_list_in_order(hours, events):
    bt = events[0]
    r = build_rate_features(hours, bt)
    assert r.shape == (len(hours), len(RATE_FEATURE_NAMES))
    assert ALL_FEATURE_NAMES == FEATURE_NAMES + RATE_FEATURE_NAMES


def test_every_feature_is_finite(hours, events):
    """A NaN column would be silently mean-imputed downstream and train the
    model against a constant."""
    bt, bm, bla, blo = events
    f = build_catalog_features(hours, bt[:5], np.full(len(hours), 5.0),
                               bt, bm, 3.0, bla, blo)
    assert np.isfinite(f).all()


# --- the direction of time -------------------------------------------------

def test_no_feature_looks_past_its_own_hour(hours, events):
    """Truncating the catalogue at hour H must not change any feature at or
    before H. If it does, that feature was reading the future."""
    bt, bm, bla, blo = events
    dsp = np.full(len(hours), 5.0)
    cut = 200
    horizon = hours[cut].to_datetime64()

    full = build_catalog_features(hours, bt[:5], dsp, bt, bm, 3.0)
    past_only = build_catalog_features(hours, bt[:5], dsp,
                                       bt[bt <= horizon], bm[bt <= horizon], 3.0)
    assert np.allclose(full[:cut + 1], past_only[:cut + 1], equal_nan=True), \
        "a feature changed when only FUTURE events were removed"


def test_a_trailing_count_rises_after_events_and_not_before(hours):
    """The sign check the whole feature set depends on."""
    burst = np.array([np.datetime64("2024-03-05") + np.timedelta64(i, "h")
                      for i in range(20)])
    r = build_rate_features(hours, burst)
    i = RATE_FEATURE_NAMES.index("rate_log1p_count_7d")
    before = r[hours < pd.Timestamp("2024-03-05"), i]
    after = r[hours > pd.Timestamp("2024-03-06"), i]
    assert before.max() == 0.0, "a trailing count cannot see a future burst"
    assert after.max() > 0.0


# --- the expensive path ----------------------------------------------------

def test_omitting_coordinates_skips_the_location_features(hours, events):
    """How the trainer avoids an O(n_bg x 500) haversine loop for columns a
    4-feature run then discards."""
    bt, bm, bla, blo = events
    dsp = np.full(len(hours), 5.0)
    without = build_catalog_features(hours, bt[:5], dsp, bt, bm, 3.0)
    withc = build_catalog_features(hours, bt[:5], dsp, bt, bm, 3.0, bla, blo)
    assert without.shape == withc.shape, "the column count must not change"
    nnd = FEATURE_NAMES.index("nnd_log_eta_90d")
    ent = FEATURE_NAMES.index("shannon_entropy_90d")
    assert np.all(without[:, nnd] == 0.0) and np.all(without[:, ent] == 0.0)
    keep = [i for i in range(CATALOG_DIM) if i not in (nnd, ent)]
    assert np.allclose(without[:, keep], withc[:, keep], equal_nan=True), \
        "omitting coordinates must not disturb the non-spatial features"


def test_the_location_features_are_actually_populated_when_given(hours, events):
    """The other half: the skip must be a skip, not a permanent zero."""
    bt, bm, bla, blo = events
    f = build_catalog_features(hours, bt[:5], np.full(len(hours), 5.0),
                               bt, bm, 3.0, bla, blo)
    assert np.any(f[:, FEATURE_NAMES.index("nnd_log_eta_90d")] != 0.0)
    assert np.any(f[:, FEATURE_NAMES.index("shannon_entropy_90d")] != 0.0)


# --- an empty catalogue ----------------------------------------------------

def test_no_background_events_gives_zeros_not_a_crash(hours):
    """The archive's opening hours, before any lookback has filled."""
    empty = np.array([], dtype="datetime64[ns]")
    f = build_catalog_features(hours, empty, np.full(len(hours), np.nan),
                               empty, np.array([]), 3.0)
    assert f.shape == (len(hours), CATALOG_DIM)
    assert np.isfinite(f).all()
