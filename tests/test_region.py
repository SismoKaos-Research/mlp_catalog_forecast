"""The study region: what the run is about, and what changes when it narrows.

The region is not a filter applied to a fixed question. Features, labels, the
persistence floor and the fold boundaries are all computed from the events
inside it, so narrowing it asks a different question -- and the M>=4.5 set is
~10^2 events across the whole Aegean, which is few enough that a small region
can leave a run with nothing to score.

Two things here are easy to get quietly wrong. A disc that is really its own
bounding box admits the corners, which at this latitude is a fifth of the area
it should not have. And a spatial entropy feature still binned over the whole
Aegean drops a narrowed region's events into two or three of its hundred cells,
reporting the same near-constant number every hour while looking like a working
feature.
"""
import numpy as np
import pandas as pd
import pytest

from forecast.catalog import (AEGEAN_BBOX, Region, load_aegean_events,
                              load_aegean_events_with_location)
from forecast.features import FEATURE_NAMES, build_catalog_features


@pytest.fixture
def catalog(tmp_path):
    """A catalogue spread across the Aegean box, one event every six hours."""
    rng = np.random.default_rng(0)
    n = 600
    t = pd.to_datetime("2015-01-01") + pd.to_timedelta(np.arange(n) * 6, unit="h")
    lat0, lat1, lon0, lon1 = AEGEAN_BBOX
    df = pd.DataFrame({
        "Date": t.strftime("%d/%m/%Y %H:%M:%S"),
        "Latitude": rng.uniform(lat0 + 0.1, lat1 - 0.1, n),
        "Longitude": rng.uniform(lon0 + 0.1, lon1 - 0.1, n),
        "Magnitude": np.round(3.0 + rng.exponential(0.7, n), 1),
    })
    path = tmp_path / "catalog.csv"
    df.to_csv(path, index=False)
    return path


def test_the_default_region_is_the_published_one(catalog):
    """Every number this project reports was produced over the Aegean box, so
    a run that names no region has to read exactly what it always read."""
    assert Region.from_flags().bbox == AEGEAN_BBOX
    assert Region.from_flags(None, None).center is None

    default = load_aegean_events(catalog, 4.0)
    explicit = load_aegean_events(catalog, 4.0, region=Region(bbox=AEGEAN_BBOX))
    assert np.array_equal(default, explicit)


def test_a_box_narrows_what_the_run_reads(catalog):
    """The flag has to reach the catalogue itself, not just the log line."""
    box = Region.from_flags(bbox=(37.0, 38.0, 26.0, 27.0))
    _, _, lats, lons = load_aegean_events_with_location(catalog, 3.0, region=box)
    assert len(lats), "the fixture should put some events in this box"
    assert lats.min() >= 37.0 and lats.max() <= 38.0
    assert lons.min() >= 26.0 and lons.max() <= 27.0
    assert len(lats) < len(load_aegean_events_with_location(catalog, 3.0)[2])


def test_the_radius_is_a_disc_and_not_its_bounding_box():
    """A box drawn around a 100km disc reaches ~140km into its corners. Reading
    the flag as a box would hand the run a region 27% larger than asked for,
    and nothing downstream would look wrong."""
    disc = Region.from_flags(radius=(37.0, 27.0, 100.0))
    lat0, lat1, lon0, lon1 = disc.bbox

    corner_lat, corner_lon = lat1 - 1e-6, lon1 - 1e-6
    assert lat0 <= corner_lat <= lat1 and lon0 <= corner_lon <= lon1
    inside_box = Region(bbox=disc.bbox).mask(np.array([corner_lat]),
                                             np.array([corner_lon]))
    assert inside_box[0], "the corner is inside the bounding box by construction"
    assert not disc.mask(np.array([corner_lat]), np.array([corner_lon]))[0]

    assert disc.mask(np.array([37.0]), np.array([27.0]))[0], "the centre is inside"
    assert disc.mask(np.array([lat1 - 1e-6]), np.array([27.0]))[0], "due north is inside"


def test_the_radius_reaches_the_catalogue(catalog):
    """Same check through the loader, since the mask is only useful if the
    loader applies it."""
    disc = Region.from_flags(radius=(37.5, 27.5, 120.0))
    _, _, lats, lons = load_aegean_events_with_location(catalog, 3.0, region=disc)
    assert len(lats), "the fixture should put some events in this disc"
    from forecast.catalog import haversine_km
    assert haversine_km(37.5, 27.5, lats, lons).max() <= 120.0


def test_the_entropy_grid_follows_the_region(catalog):
    """Binned over the whole Aegean, a narrow region's events land in a handful
    of the hundred cells and the feature reports the same value every hour --
    a dead column that still looks like a live one."""
    box = Region.from_flags(bbox=(37.0, 37.8, 26.5, 27.3))
    times, mags, lats, lons = load_aegean_events_with_location(catalog, 3.0, region=box)
    hours = pd.date_range("2015-04-01", "2015-08-01", freq="h")
    dsp = np.full(len(hours), 30.0)
    col = FEATURE_NAMES.index("shannon_entropy_90d")

    scoped = build_catalog_features(hours, times, dsp, times, mags, 3.0, lats, lons,
                                    grid_bbox=box.bbox)[:, col]
    aegean = build_catalog_features(hours, times, dsp, times, mags, 3.0, lats, lons)[:, col]
    assert len(np.unique(scoped)) > len(np.unique(aegean))


@pytest.mark.parametrize("bbox,radius,message", [
    ((38.0, 37.0, 26.0, 28.0), None, "inside out"),
    ((37.0, 38.0, 28.0, 26.0), None, "inside out"),
    (None, (37.0, 27.0, 0.0), "must be positive"),
    (None, (37.0, 27.0, -5.0), "must be positive"),
    (None, (137.0, 27.0, 50.0), "not a latitude"),
    ((37.0, 38.0, 26.0, 28.0), (37.0, 27.0, 50.0), "pass one"),
])
def test_a_region_that_cannot_be_read_stops_the_run(bbox, radius, message):
    """These otherwise produce an empty catalogue and a failure much later,
    somewhere that says nothing about the flag that caused it."""
    with pytest.raises(SystemExit, match=message):
        Region.from_flags(bbox, radius)


def test_the_two_region_flags_cannot_both_be_given():
    """They name two different regions, and argparse should say so before the
    catalogue is ever read."""
    import argparse

    from forecast.train import add_args
    p = argparse.ArgumentParser()
    add_args(p)
    with pytest.raises(SystemExit):
        p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                      "--catalog-bbox", "37", "38", "26", "28",
                      "--catalog-radius", "37", "27", "80"])


def test_the_region_is_named_in_one_line():
    """It goes in the run log and in every checkpoint, so it has to read."""
    assert "Aegean default" in Region.from_flags().describe()
    assert Region.from_flags(bbox=(37.0, 38.0, 26.0, 28.0)).describe() == \
        "lat 37..38, lon 26..28"
    assert Region.from_flags(radius=(37.5, 27.5, 120.0)).describe() == \
        "120km around 37.5000, 27.5000"
