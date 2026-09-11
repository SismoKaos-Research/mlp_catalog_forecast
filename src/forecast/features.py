"""The catalogue features, and how each one is derived.

Not a runnable script -- imported only, by `train.py`.

Thirteen features over a trailing window, all from the earthquake catalogue.
Eight describe the magnitude distribution and its timing from a
lower-completeness "background" catalogue (M>=--bg-min-mag) rather than from the
rare M>=threshold events alone, because the rare set gives too few points to
estimate a b-value or an inter-event coefficient of variation from. Two are
spatial, following Convertito et al. 2024 (Sci. Rep. 14:2964): the
Zaliapin-Ben-Zion nearest-neighbour distance and a spatial Shannon entropy.

That paper's coefficient-of-variation feature independently matches this
project's `cv_interevent_90d`, which recursive feature elimination had already
found to be the strongest of the original eleven.

The four `--keep-features` the published configuration uses are `log1p_dsp`,
`mean_mag_30d`, `cv_interevent_90d` and `mag_deficit_90d`.

Bodies are unchanged from `cnn_earthquake`'s
`forecasting/cnn_lstm_catalog_waveform_fusion.py`; they produced the published
figures.
"""
import numpy as np

from forecast.catalog import (AEGEAN_BBOX, count_events_in_window,
                              days_since_prev_major)

CATALOG_DIM = 13
LN10 = np.log(10.0)
FEATURE_NAMES = ["log1p_dsp", "count_7d", "count_30d", "count_90d", "mean_mag_30d",
                 "max_mag_90d", "b_value_90d", "energy_sqrt_30d", "mean_interevent_90d",
                 "cv_interevent_90d", "mag_deficit_90d", "nnd_log_eta_90d",
                 "shannon_entropy_90d"]

NND_FRACTAL_DIM = 1.6  # standard Zaliapin-Ben-Zion literature default
NND_LOOKBACK = 500  # bound the O(n*lookback) nearest-neighbour search for tractability
ENTROPY_GRID_SIZE = 10  # 10x10 spatial cells over the study region for Shannon entropy

# Trailing-rate features, for --label-mode rate. None of the features above encode
# the trailing count of the LOW-magnitude events that define the rate target:
# count_7d/30d/90d count M>=threshold (4.5) events, not M>=rate_min_mag (3.0) ones.
# That left the model trying to beat a persistence floor built from exactly the
# number it was never given -- it lost fold 1 0.6573 vs 0.7991 while scoring a
# POSITIVE Brier skill (+0.129), i.e. learning something real but unable to rank
# without the rate signal. The ratio features are the acceleration term itself
# (short-window rate over long-window rate), which is what the label asks about.
RATE_WINDOWS = (3, 7, 14, 30, 90)
RATE_RATIO_PAIRS = ((3, 14), (7, 30), (14, 90))
RATE_FEATURE_NAMES = ([f"rate_log1p_count_{w}d" for w in RATE_WINDOWS]
                      + [f"rate_logratio_{a}_{b}" for a, b in RATE_RATIO_PAIRS])
ALL_FEATURE_NAMES = FEATURE_NAMES + RATE_FEATURE_NAMES


def build_rate_features(hour_index, rate_times) -> np.ndarray:
    """Backward-looking trailing-rate features for the rate-change target.

    Args:
        hour_index: DatetimeIndex of hour starts.
        rate_times: Sorted array of the event times defining the rate (the
            M>=rate_min_mag set that `label_hours_rate_change` uses).

    Returns:
        float32 array, shape (n_hours, len(RATE_FEATURE_NAMES)). Counts are
        log1p'd (raw counts are heavy-tailed -- median 9, max 297 at 14d) and
        ratios are log'd so acceleration and deceleration are symmetric around 0.
    """
    counts = {w: count_events_in_window(hour_index, rate_times, w, forward=False)
             for w in RATE_WINDOWS}
    cols = [np.log1p(counts[w].astype(np.float64)) for w in RATE_WINDOWS]
    for a, b in RATE_RATIO_PAIRS:
        # per-day rates, so the ratio is a clean acceleration factor rather than a
        # window-length artifact; eps keeps quiet stretches (0 events) finite.
        rate_a = counts[a] / float(a)
        rate_b = counts[b] / float(b)
        cols.append(np.log((rate_a + 1e-3) / (rate_b + 1e-3)))
    return np.stack(cols, axis=1).astype(np.float32)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _nearest_neighbor_log_eta(times, mags, lats, lons, b_value=1.0,
                              df=NND_FRACTAL_DIM, lookback=NND_LOOKBACK):
    """Per-event Zaliapin-Ben-Zion nearest-neighbour distance, log10(eta).

    eta_ij = T_ij * R_ij, rescaled time x rescaled distance to the closest
    preceding candidate "parent" event i:
        T_ij = (t_j - t_i) * 10^(-0.5 b m_i)
        R_ij = (haversine_km(i, j))^df * 10^(-0.5 b m_i)
    Small eta = temporally/spatially close to a prior event relative to its
    magnitude -- the signature of a triggered (foreshock/aftershock-like)
    event; large eta = an independent "background" event. Bounded to the
    `lookback` most recent candidate parents per event (full O(n^2) is
    intractable at ~17k events) -- true nearest neighbors are overwhelmingly
    recent for local clustering, so this is a tractable approximation, not
    the exact full-catalog nearest neighbor.

    Args:
        times: Sorted event times (datetime64).
        mags: Matching magnitudes.
        lats: Matching latitudes.
        lons: Matching longitudes.
        b_value: Gutenberg-Richter b-value for the rescaling.
        df: Fractal dimension of the epicenter distribution.
        lookback: Max number of preceding events considered as candidate parents.

    Returns:
        float64 array, length len(times); NaN for the first event (no candidates).
    """
    n = len(times)
    log_eta = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return log_eta
    t_days = (times - times[0]) / np.timedelta64(1, "D")
    for j in range(1, n):
        lo = max(0, j - lookback)
        dt = t_days[j] - t_days[lo:j]
        dist_km = _haversine_km(lats[lo:j], lons[lo:j], lats[j], lons[j])
        scale = 10.0 ** (-0.5 * b_value * mags[lo:j])
        eta = dt * scale * (np.maximum(dist_km, 1e-6) ** df) * scale
        eta = eta[eta > 0]
        if len(eta):
            log_eta[j] = np.log10(eta.min())
    return log_eta


def build_catalog_features(hour_index, major_times, dsp, bg_times=None, bg_mags=None,
                           bg_min_mag=3.0, bg_lats=None, bg_lons=None,
                           grid_bbox=None) -> np.ndarray:
    """Per-hour backward-looking catalog features -- no leakage.

    Timing features (0-3) come from `major_times` (the M>=threshold set
    labels/persistence use). Magnitude/energy/regularity features (4-10)
    come from a separate, lower-completeness-threshold "background" catalog
    (`bg_times`/`bg_mags`, e.g. M>=3.0) -- the Panakkat-Adeli-style feature
    set used in Nurtas et al. 2025 (IEEE Access, the Central Asia LSTM
    forecasting paper): mean magnitude, sqrt(energy release), Gutenberg-
    Richter b-value, magnitude deficit, mean inter-event time, coefficient
    of variation of inter-event times. `bg_times`/`bg_mags` default to
    `major_times` (with an all-zero-ish magnitude array) if not given, which
    degrades gracefully to just the 4 original timing features plus zeros.

    Features 11-12 (nearest-neighbour log-eta, spatial Shannon entropy) need
    `bg_lats`/`bg_lons`; without them they default to 0.0 for every hour.

    Args:
        hour_index: DatetimeIndex of hour starts.
        major_times: Sorted array of qualifying (M>=threshold) event times.
        dsp: Days since the previous qualifying event, per hour (see
            `days_since_prev_major`); NaN where none exists.
        bg_times: Sorted array of lower-threshold "background" event times
            (see `load_aegean_events_with_magnitude`). Defaults to
            `major_times` if None.
        bg_mags: Matching magnitudes for `bg_times`, same order. Defaults
            to an all-`bg_min_mag` array if None.
        bg_min_mag: Completeness threshold of the background catalog --
            the Gutenberg-Richter reference magnitude for the b-value and
            magnitude-deficit calculations.
        bg_lats: Matching latitudes for `bg_times`, same order (see
            `load_aegean_events_with_location`). Optional.
        bg_lons: Matching longitudes for `bg_times`, same order. Optional.
        grid_bbox: Extent (lat0, lat1, lon0, lon1) the spatial Shannon entropy
            bins over. Defaults to the Aegean box, which is what the published
            figures used. A run restricted to a smaller region must pass that
            region's box: a grid still spanning the whole Aegean would drop all
            of its events into two or three cells and report the same entropy
            every hour.

    Returns:
        float32 array, shape (n_hours, CATALOG_DIM).
    """
    if bg_times is None:
        bg_times = major_times
    if bg_mags is None:
        bg_mags = np.full(len(bg_times), bg_min_mag, dtype=np.float64)

    have_location = bg_lats is not None and bg_lons is not None
    if have_location:
        mean_excess_global = max(bg_mags.mean() - bg_min_mag, 1e-3)
        b_value_global = (1.0 / LN10) / mean_excess_global
        nnd_log_eta = _nearest_neighbor_log_eta(bg_times, bg_mags, bg_lats, bg_lons,
                                                b_value=b_value_global)
        lat0, lat1, lon0, lon1 = grid_bbox or (36.0, 40.0, 25.0, 30.0)  # AEGEAN_BBOX

    t = hour_index.to_numpy()
    feat = np.zeros((len(t), CATALOG_DIM), dtype=np.float32)
    feat[:, 0] = np.log1p(np.nan_to_num(dsp, nan=3650.0))
    for i, ti in enumerate(t):
        past = major_times[major_times < ti]
        feat[i, 1] = np.sum(past > ti - np.timedelta64(7, "D"))
        feat[i, 2] = np.sum(past > ti - np.timedelta64(30, "D"))
        feat[i, 3] = np.sum(past > ti - np.timedelta64(90, "D"))

        mask_30 = (bg_times < ti) & (bg_times > ti - np.timedelta64(30, "D"))
        mask_90 = (bg_times < ti) & (bg_times > ti - np.timedelta64(90, "D"))
        mags_30, mags_90 = bg_mags[mask_30], bg_mags[mask_90]
        times_90 = bg_times[mask_90]
        n90 = len(mags_90)

        feat[i, 4] = mags_30.mean() if len(mags_30) else 0.0
        feat[i, 5] = mags_90.max() if n90 else bg_min_mag
        feat[i, 7] = np.sqrt(np.sum(10.0 ** (1.5 * mags_30))) if len(mags_30) else 0.0

        if n90 >= 5:
            mean_excess = max(mags_90.mean() - bg_min_mag, 1e-3)
            b_value = (1.0 / LN10) / mean_excess
        else:
            b_value = 1.0  # global-average default (standard tectonic seismicity value)
        feat[i, 6] = b_value

        if n90 >= 3:
            intervals = np.diff(np.sort(times_90)) / np.timedelta64(1, "D")
            mean_iv = intervals.mean()
            feat[i, 8] = mean_iv
            feat[i, 9] = (intervals.std() / mean_iv) if mean_iv > 0 else 1.0
        else:
            feat[i, 8] = 90.0  # default: one event per window, i.e. quiet
            feat[i, 9] = 1.0   # default: Poisson-like regularity

        # magnitude deficit: Gutenberg-Richter-extrapolated expected max magnitude
        # (from local b-value + event count) minus what's actually been observed --
        # a proxy for "overdue" stress release.
        expected_max = bg_min_mag + (np.log10(max(n90, 1)) / max(b_value, 1e-3))
        feat[i, 10] = expected_max - feat[i, 5]

        if have_location:
            eta_90 = nnd_log_eta[mask_90]
            eta_90 = eta_90[~np.isnan(eta_90)]
            feat[i, 11] = eta_90.mean() if len(eta_90) else 0.0

            if n90 >= 2:
                lats_90, lons_90 = bg_lats[mask_90], bg_lons[mask_90]
                energies = 10.0 ** (1.5 * mags_90)
                lat_bin = np.clip(((lats_90 - lat0) / (lat1 - lat0) * ENTROPY_GRID_SIZE)
                                  .astype(int), 0, ENTROPY_GRID_SIZE - 1)
                lon_bin = np.clip(((lons_90 - lon0) / (lon1 - lon0) * ENTROPY_GRID_SIZE)
                                  .astype(int), 0, ENTROPY_GRID_SIZE - 1)
                cell_id = lat_bin * ENTROPY_GRID_SIZE + lon_bin
                cell_energy = np.bincount(cell_id, weights=energies,
                                          minlength=ENTROPY_GRID_SIZE ** 2)
                total = cell_energy.sum()
                p = cell_energy[cell_energy > 0] / total if total > 0 else np.array([])
                feat[i, 12] = float(-np.sum(p * np.log(p))) if len(p) else 0.0
    return feat
