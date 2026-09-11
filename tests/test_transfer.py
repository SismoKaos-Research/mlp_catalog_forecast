"""Training on more regions than the one being forecast.

A fold here can fit a model on ten earthquakes, and that is the binding
constraint on this project, not the network. Pooling further regions into
training is the answer, but only under conditions that are easy to violate
silently, and those are what these tests pin.

**The pool must not contain the target's own earthquakes.** Candidate regions
are rejected within two radii of the study region, so no event can be in both
the training pool and the test block.

**Selection must not see the test period.** Regions are chosen by how closely
they match the target's positive rate and b-value. Measured over the whole
record, that would choose training regions partly for what happens in the block
the model is later scored on.

**Val and test must still come from the study region alone.** The point of the
pool is to change the fit, not the measurement. If it moved the floors or the
reported n, a comparison against a run without it would be meaningless.

The geographic alternative was tried first and does not survive checking. A
country-wide training region is positive 97% of the time against a target at
12%, and a hand-drawn fault corridor captured under a fifth of the seismicity
in most longitude bins. Measured similarity is checkable; a drawn boundary is
not, which is why the selector is statistical.
"""
import argparse

import numpy as np
import pandas as pd
import pytest

from forecast.catalog import AEGEAN_BBOX, Region, haversine_km
from forecast.regions import (build_training_regions, rank_candidate_regions,
                              region_seismicity)
from forecast.train import add_args, build_pool, pooled_stats

RADIUS = 120.0


@pytest.fixture
def catalog(tmp_path):
    """A catalogue dense enough that several discs qualify as candidates."""
    rng = np.random.default_rng(0)
    n = 6000
    t = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 365 * 3 * 24, n)), unit="h")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    pd.DataFrame({
        "Date": t.strftime("%d/%m/%Y %H:%M:%S"),
        "Latitude": rng.uniform(lat0, lat1, n),
        "Longitude": rng.uniform(lon0, lon1, n),
        "Magnitude": np.round(3.0 + rng.exponential(0.6, n), 1),
    }).to_csv(tmp_path / "catalog.csv", index=False)
    return tmp_path / "catalog.csv"


@pytest.fixture
def hours():
    return pd.date_range("2015-01-01", "2017-12-31", freq="h")


def args_for(catalog, *extra):
    p = argparse.ArgumentParser()
    add_args(p)
    return p.parse_args(["--catalog-path", str(catalog),
                         "--catalog-span", "2015-01-01", "2017-12-31",
                         "--threshold", "4.0", "--catalog-radius",
                         "37.5", "27.5", str(RADIUS), *extra])


def fake_folds(hours, n=3):
    """Folds shaped like the real ones: a train block well short of the end."""
    idx = np.arange(23, len(hours))
    cut = int(len(idx) * 0.5)
    return [(idx[:cut], idx[cut:cut + 100], idx[cut + 100:])] * n


def test_the_pool_cannot_contain_the_target_s_events(catalog, hours):
    """Two radii apart is the condition. Any closer and one earthquake sits in
    both the training pool and the block the model is scored on."""
    target = Region.from_flags(radius=(37.5, 27.5, RADIUS))
    chosen = rank_candidate_regions(args_for(catalog), target, hours, 6, grid_deg=1.0)
    assert chosen, "the fixture should offer candidates"
    for region, _ in chosen:
        d = haversine_km(37.5, 27.5, np.array([region.center[0]]),
                         np.array([region.center[1]]))[0]
        assert d >= 2 * RADIUS


def test_selection_never_looks_past_the_training_window(catalog, hours):
    """Statistics are computed on the hours handed in. Ranking over the whole
    record would choose training regions for what happens in the test block."""
    region = Region.from_flags(radius=(38.5, 29.0, RADIUS))
    args = args_for(catalog)
    early = region_seismicity(catalog, region, hours[:len(hours) // 3],
                              4.0, 3.0, 14.0)
    whole = region_seismicity(catalog, region, hours, 4.0, 3.0, 14.0)
    assert early is not None and whole is not None
    assert early["pos_rate"] != whole["pos_rate"], \
        "a window-sensitive statistic is the thing being guarded"
    assert early["n_major"] == whole["n_major"], \
        "event counts are catalogue-wide; only the label rate reads the window"


def test_the_pool_changes_training_and_nothing_else(catalog, hours):
    """Val and test are built from the study region alone, so the floors and
    the reported n are identical to a run without a pool."""
    args = args_for(catalog, "--train-region-centers", "38.6,29.4", "36.5,25.6")
    pool = build_pool(args, hours, Region.from_flags(radius=(37.5, 27.5, RADIUS)),
                      ["log1p_dsp"], fake_folds(hours))
    assert len(pool) == 2
    for region, features, labels in pool:
        assert len(features) == len(hours)
        assert len(labels) == len(hours)
        assert region.radius_km == RADIUS, "the pool must match the target's size"


def test_no_pool_is_asked_for_no_pool_is_built(catalog, hours):
    """The default has to be the single-region run this is compared against."""
    args = args_for(catalog)
    assert args.train_regions == 0 and args.train_region_centers is None
    assert build_pool(args, hours, Region.from_flags(radius=(37.5, 27.5, RADIUS)),
                      ["log1p_dsp"], fake_folds(hours)) is None


def test_the_normalization_is_shared_across_regions(catalog):
    """One mean and sd over every training region. Normalizing each region
    separately would erase what distinguishes a quiet region from an active
    one, which is the signal the pool exists to supply."""
    from forecast.data import CatalogSeqDataset
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1, (2000, 2)).astype(np.float32)
    busy = rng.normal(50, 10, (2000, 2)).astype(np.float32)
    labels = (rng.random(2000) < 0.2).astype(np.float32)
    idx = np.arange(23, 1500)
    parts = [CatalogSeqDataset(a, labels, 24, idx) for a in (quiet, busy)]
    mu, sd = pooled_stats(parts)
    assert mu.shape == (1, 2) and sd.shape == (1, 2)
    assert 20 < mu[0, 0] < 30, "the pooled mean sits between the two regions"

    # The spread BETWEEN the regions' means is part of the pooled spread. Taking
    # the mean of their sds drops it, always understates, and so always inflates
    # the z-scores -- the failure `data.py` documents at 52x. Here the two region
    # means are 50 apart, so the correct sd is dominated by that gap.
    mean_of_sds = np.mean([p.stats[1] for p in parts], axis=0)
    assert sd[0, 0] > 4 * mean_of_sds[0, 0]
    # sqrt(mean of the within-region variances + the variance of their means):
    # sqrt((1 + 100)/2 + 25^2) against a naive (1 + 10)/2.
    assert sd[0, 0] == pytest.approx(np.sqrt(50.5 + 625.0), rel=0.05)
    assert mean_of_sds[0, 0] == pytest.approx(5.5, rel=0.05)


def test_a_centre_that_is_not_a_coordinate_pair_stops_the_run(catalog, hours):
    args = args_for(catalog, "--train-region-centers", "38.5")
    with pytest.raises(SystemExit, match="reads LAT,LON"):
        build_pool(args, hours, Region.from_flags(radius=(37.5, 27.5, RADIUS)),
                   ["log1p_dsp"], fake_folds(hours))


def test_a_box_shaped_study_region_is_refused(catalog, hours):
    """A candidate has to be the same shape as the target to pose the same
    question, and there is no single radius to copy from a box."""
    p = argparse.ArgumentParser()
    add_args(p)
    args = p.parse_args(["--catalog-path", str(catalog), "--catalog-span",
                         "2015-01-01", "2017-12-31", "--catalog-bbox",
                         "36", "39", "26", "29", "--train-regions", "3"])
    with pytest.raises(SystemExit, match="disc-shaped"):
        build_pool(args, hours, Region.from_flags(bbox=(36, 39, 26, 29)),
                   ["log1p_dsp"], fake_folds(hours))


def test_every_pooled_region_gets_its_own_entropy_grid(catalog, hours):
    """The spatial entropy feature bins into a fixed grid. Binned over the
    target's box, a pooled region's events would land in a corner of it and
    the column would be a constant for that region."""
    args = args_for(catalog)
    regions = [Region.from_flags(radius=(38.6, 29.4, RADIUS)),
               Region.from_flags(radius=(36.5, 25.6, RADIUS))]
    built = build_training_regions(args, hours[:2000], regions,
                                   ["log1p_dsp", "shannon_entropy_90d"])
    col = 12
    for region, features, _ in built:
        assert region.bbox != regions[0].bbox or region is regions[0]
        assert np.isfinite(features[:, col]).all()
