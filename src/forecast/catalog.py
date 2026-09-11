"""Loading the earthquake catalogue, and turning it into hourly labels.

Not a runnable script -- imported only.

This is the whole input to the project. There is no waveform archive here: the
2026-08-30 experiment put the waveform and chaotic-feature arms below the
persistence floor, and concluded that "the forecasting signal in this project
comes from the earthquake catalogue, not from the seismogram." What survived
that boundary is what this module reads.

Ported from `cnn_earthquake/src/sismokaos/catalog.py` -- the five functions
catalog_mlp uses, of its twenty. Bodies are unchanged; they produced the
published figures.

The station-distance filter came across with them and is gone: it existed to cut
the catalogue down to events near a seismometer whose waveforms the model read,
and nothing here reads a waveform, so it narrowed the catalogue for a reason that
had stopped existing. What stands in its place is `Region` -- the patch of crust
a run is about, stated by the user and agreed on by everything downstream:
features, labels, the floor and the folds. `stations.py` can name one by station
code, which is a lookup for the centre of a disc and not a return of the filter.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

# lat0, lat1, lon0, lon1
AEGEAN_BBOX = (36.0, 40.0, 25.0, 30.0)

# One degree of latitude, in km. Only ever used to put a box around a disc, so
# the flattening term the geodesic would add is far below the resolution of
# anything downstream (the entropy grid is 10 cells across a 4-degree box).
KM_PER_DEG_LAT = 111.195


def haversine_km(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    r = 6371.0
    la1, lo1 = np.radians(lat0), np.radians(lon0)
    la2, lo2 = np.radians(lats), np.radians(lons)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class Region:
    """The patch of crust a run is about: a box, or a disc around a point.

    The default is the Aegean box every published number was produced over.
    Narrowing it is a real decision rather than a filter, because it changes
    the question: features, labels, the persistence floor and the folds are all
    computed from the events inside, so a run over a 50km disc is forecasting a
    different thing than a run over the Aegean, not the same thing more
    precisely. It is also not free -- the M>=4.5 set is ~10^2 events across the
    whole box -- which is why `train.py` prints what a narrowed region left it.

    `bbox` is carried even for a disc, as the smallest box containing it. The
    spatial Shannon entropy feature bins events into a fixed 10x10 grid, and a
    grid still spanning the whole Aegean would drop a narrowed region's events
    into two or three cells and report the same near-constant entropy every
    hour.
    """

    bbox: tuple
    center: tuple = None
    radius_km: float = None
    station: str = None

    @classmethod
    def from_flags(cls, bbox=None, radius=None):
        """Builds the region named by `--catalog-bbox` / `--catalog-radius`.

        Args:
            bbox: Optional (lat0, lat1, lon0, lon1).
            radius: Optional (lat, lon, km) disc.

        Returns:
            The named `Region`, or the Aegean default if neither was given.

        Raises:
            SystemExit: If the box is inside out or the disc has no size --
                both of which otherwise produce an empty catalogue and a run
                that fails much later, somewhere less informative.
        """
        if bbox and radius:
            raise SystemExit("[ERROR] --catalog-bbox and --catalog-radius name two "
                             "different regions; pass one.")
        if bbox:
            lat0, lat1, lon0, lon1 = (float(v) for v in bbox)
            if lat0 >= lat1 or lon0 >= lon1:
                raise SystemExit(f"[ERROR] --catalog-bbox {lat0} {lat1} {lon0} {lon1} "
                                 f"is inside out; it reads LAT0 LAT1 LON0 LON1 with "
                                 f"LAT0 < LAT1 and LON0 < LON1.")
            return cls(bbox=(lat0, lat1, lon0, lon1))
        if radius:
            lat, lon, km = (float(v) for v in radius)
            if km <= 0:
                raise SystemExit(f"[ERROR] --catalog-radius km must be positive, got {km}.")
            if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
                raise SystemExit(f"[ERROR] --catalog-radius centre {lat} {lon} is not a "
                                 f"latitude and longitude; it reads LAT LON KM.")
            # The enclosing box, for the entropy grid. Longitude degrees shrink
            # with latitude, so the padding is widened by 1/cos(lat) -- clamped
            # because at a pole the disc spans every longitude.
            dlat = km / KM_PER_DEG_LAT
            dlon = km / (KM_PER_DEG_LAT * max(np.cos(np.radians(lat)), 1e-6))
            return cls(bbox=(float(max(lat - dlat, -90.0)), float(min(lat + dlat, 90.0)),
                             float(max(lon - dlon, -180.0)), float(min(lon + dlon, 180.0))),
                       center=(lat, lon), radius_km=km)
        return cls(bbox=AEGEAN_BBOX)

    def mask(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Boolean mask of the events inside this region.

        Args:
            lats: Event latitudes.
            lons: Event longitudes, same order.

        Returns:
            Boolean array, True where the event is inside.
        """
        if self.center is not None:
            # A true great-circle disc, not its bounding box: at 37degN the box
            # corner sits ~40% further out than the radius, so the difference is
            # a fifth of the events, not a rounding detail.
            d = haversine_km(self.center[0], self.center[1], lats, lons)
            return d <= self.radius_km
        lat0, lat1, lon0, lon1 = self.bbox
        return (lats >= lat0) & (lats <= lat1) & (lons >= lon0) & (lons <= lon1)

    def describe(self) -> str:
        """One line naming this region, for the run log and the checkpoint."""
        if self.center is not None:
            where = (f"{self.center[0]:.4f}, {self.center[1]:.4f}"
                     if self.station is None else
                     f"{self.station} ({self.center[0]:.4f}, {self.center[1]:.4f})")
            return f"{self.radius_km:g}km around {where}"
        lat0, lat1, lon0, lon1 = self.bbox
        default = " (Aegean default)" if self.bbox == AEGEAN_BBOX else ""
        return f"lat {lat0:g}..{lat1:g}, lon {lon0:g}..{lon1:g}{default}"


@lru_cache(maxsize=4)
def _read_catalog(catalog_path: str) -> pd.DataFrame:
    """Parses the catalogue once per path, with `dt` already decoded.

    Selecting training regions measures a few hundred candidate discs, and each
    one used to re-read and re-parse the whole CSV. On a 576k-row national
    catalogue that is minutes of work to answer a question about geometry.
    Parsing is pure, so the result is cached and callers filter without
    mutating it.

    Args:
        catalog_path: Catalogue CSV with Date/Latitude/Longitude/Magnitude.

    Returns:
        The parsed catalogue. **Treat as read-only**; it is shared.
    """
    cat = pd.read_csv(catalog_path)
    cat["dt"] = pd.to_datetime(cat["Date"], format="%d/%m/%Y %H:%M:%S",
                               errors="coerce")
    return cat


def load_aegean_events(catalog_path: str, min_magnitude: float = 4.5,
                       region: "Region" = None) -> np.ndarray:
    """Loads catalog events within the study region at or above a magnitude.

    Args:
        catalog_path: Catalogue CSV with Date/Latitude/Longitude/Magnitude.
        min_magnitude: Minimum magnitude to include.
        region: The patch of crust to read. Defaults to the Aegean box.

    Returns:
        Sorted array of event times.
    """
    cat = _read_catalog(str(catalog_path))
    aegean = cat[_inside(cat, region) & (cat.Magnitude >= min_magnitude) & cat.dt.notna()]
    return np.sort(aegean.dt.to_numpy())

def _inside(cat: pd.DataFrame, region: "Region") -> pd.Series:
    """Row mask for the catalogue rows inside `region` (the Aegean box if None)."""
    region = region or Region(bbox=AEGEAN_BBOX)
    return pd.Series(region.mask(cat.Latitude.to_numpy(dtype=np.float64),
                                 cat.Longitude.to_numpy(dtype=np.float64)),
                     index=cat.index)


def load_aegean_events_with_location(catalog_path: str, min_magnitude: float = 3.0,
                                     region: "Region" = None):
    """Loads catalog events (times, magnitudes, AND lat/lon) within the Aegean bbox.

    Companion to `load_aegean_events_with_magnitude`, adding coordinates for
    features that need event location -- nearest-neighbour distance
    (Zaliapin & Ben-Zion) and spatial Shannon entropy, both from Convertito
    et al. 2024 (Sci. Rep. 14:2964).

    Args:
        catalog_path: Path to a catalog CSV with 'Date', 'Latitude',
            'Longitude', 'Magnitude' columns (data_large.csv format).
        min_magnitude: Minimum magnitude to include (completeness
            threshold for the returned "background" catalog).
        region: The patch of crust to read. Defaults to the Aegean box.

    Returns:
        Tuple of (times, magnitudes, lats, lons), all sorted by time, same order.
    """
    cat = _read_catalog(str(catalog_path))
    aegean = cat[_inside(cat, region) & (cat.Magnitude >= min_magnitude)
                 & cat.dt.notna()].sort_values("dt")
    return (aegean.dt.to_numpy(), aegean.Magnitude.to_numpy(dtype=np.float64),
           aegean.Latitude.to_numpy(dtype=np.float64), aegean.Longitude.to_numpy(dtype=np.float64))

def count_events_in_window(hourly_index: pd.DatetimeIndex, times: np.ndarray,
                           window_days: float, forward: bool) -> np.ndarray:
    """Counts events in a trailing or leading window around each hour."""
    t = hourly_index.to_numpy()
    w = np.timedelta64(int(round(window_days * 24)), "h")
    if forward:
        return (np.searchsorted(times, t + w, side="right")
                - np.searchsorted(times, t, side="right")).astype(np.int64)
    return (np.searchsorted(times, t, side="right")
            - np.searchsorted(times, t - w, side="right")).astype(np.int64)

def days_since_prev_major(hourly_index: pd.DatetimeIndex, major_times: np.ndarray) -> np.ndarray:
    """Computes days elapsed since the previous qualifying event, per hour."""
    t = hourly_index.to_numpy()
    out = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        prev = major_times[major_times < ti]
        if len(prev):
            out[i] = (ti - prev[-1]) / np.timedelta64(1, "D")
    return out

def label_hours_rate_change(hourly_index: pd.DatetimeIndex, rate_times: np.ndarray,
                            horizon_days: float, baseline_days: float = None):
    """Labels each hour with whether seismicity RATE will increase ("variant B").

    A different forecasting target from `label_hours`: instead of "does one
    rare M>=threshold event occur in the next horizon" (whose positive class,
    at M>=4.5, is driven by a handful of events per fold -- 4 in fold 1 --
    making the effective sample size far smaller than the hour count
    suggests), this asks "will the next window contain MORE events than the
    trailing window did".

    That is a rate/acceleration forecast, which is what ETAS-family models and
    CSEP evaluation actually target, and it uses a much lower magnitude
    threshold (typically M>=3.0), so the label is driven by ~10^3 events
    instead of ~10^1. It is also the quantity Convertito et al. 2024's
    beta-statistic measures -- but as the target itself rather than as a mask
    on a rare-event label (`label_hours_beta_precursor`), which is what made
    that earlier attempt fail.

    Note the natural baseline here is strongly ANTI-correlated: during an
    aftershock sequence a high trailing rate predicts a DECREASE (Omori
    decay). Score any model against `rate_persistence_auc`, not against 0.5.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        rate_times: Sorted array of event times defining the rate (e.g.
            M>=3.0 events -- a much lower threshold than the label-defining
            `major_times` used by `label_hours`).
        horizon_days: Length of the forward window being forecast.
        baseline_days: Length of the trailing comparison window. Defaults to
            `horizon_days` (a like-for-like comparison, so the label is a
            clean "up or down" with no window-length bias).

    Returns:
        Tuple of (labels, forward_counts, trailing_counts) -- labels is an
        int64 0/1 array (1 = rate increases), the counts are returned so
        callers can build the persistence floor and report diagnostics
        without recomputing them.
    """
    if baseline_days is None:
        baseline_days = horizon_days
    fwd = count_events_in_window(hourly_index, rate_times, horizon_days, forward=True)
    bwd = count_events_in_window(hourly_index, rate_times, baseline_days, forward=False)
    return (fwd > bwd).astype(np.int64), fwd, bwd

def label_hours(hourly_index: pd.DatetimeIndex, major_times: np.ndarray,
                horizon_days: float, feature_hours: float = 1.0) -> np.ndarray:
    """Labels each hour with whether a qualifying event occurs within the horizon.

    **The horizon starts when the features END, not when the hour starts.**
    `hourly_index` holds hour STARTS and the features for hour H are aggregated
    over [H, H+1h], so a horizon opening at H counts an event occurring inside
    the very window the model is shown. That event is visible in the features
    and labelled as future -- the model can read off the answer. It is one hour
    of a 720-hour horizon, so it inflates rather than invents, but it is the
    same window-end mistake `parse_hour_start` and the Zaman_Dk handling above
    were written to avoid.

    `feature_hours=0` restores the old behaviour, for reproducing a figure
    published before this. Every forecasting number in the repo predates it.

    Args:
        hourly_index: Hour-start timestamps, one per sample.
        major_times: Sorted qualifying event times.
        horizon_days: How far ahead to look. Fractional days are honoured.
        feature_hours: Length of the feature window opening at each index, i.e.
            how far past the index the model can already see.

    Returns:
        Int array, 1 where a qualifying event falls in the horizon.
    """
    # timedelta64 with an int day count silently truncated: --horizon-days 0.5
    # became a ZERO-day horizon and every label came out negative. Seconds keep
    # sub-day horizons meaningful.
    horizon = np.timedelta64(int(round(horizon_days * 86400)), "s")
    offset = np.timedelta64(int(round(feature_hours * 3600)), "s")
    t = hourly_index.to_numpy()
    labels = np.zeros(len(t), dtype=np.int64)
    for i, ti in enumerate(t):
        start = ti + offset
        fut = major_times[(major_times > start) & (major_times <= start + horizon)]
        labels[i] = int(len(fut) > 0)
    return labels

def truncate_to_reliable_catalog_end(hour_index: pd.DatetimeIndex, raw: np.ndarray,
                                     major_times: np.ndarray, buffer_days: float = 0):
    """Drops hours past the point where the catalog can no longer reliably inform labels."""
    cutoff = major_times[-1] - np.timedelta64(int(buffer_days * 24), "h")
    n_keep = int((hour_index.to_numpy() <= cutoff).sum())
    if n_keep < len(hour_index):
        print(f"  [!] catalog's last event is {major_times[-1]} -- truncating archive from "
               f"{hour_index[-1]} to {hour_index[n_keep - 1]} ({len(hour_index) - n_keep} hours "
               f"dropped, buffer={buffer_days:.0f}d) to avoid right-censoring the forward-looking "
               f"label near the archive's end.")
    return hour_index[:n_keep], raw[:n_keep]
