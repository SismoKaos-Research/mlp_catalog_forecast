"""Holding out a patch of crust instead of a stretch of time.

Not a runnable script -- imported only, by `train.py`. Off by default
(`--region-split none`); it is kept because it is the honest spatial
generalisation test, and this repo exists to make the evaluation harder to skip.

Ported unchanged from `cnn_earthquake`'s
`forecasting/cnn_lstm_catalog_waveform_fusion.py`, minus the `n_features`
argument the waveform path needed.
"""
import numpy as np

from forecast.catalog import (KM_PER_DEG_LAT, Region, _read_catalog,
                              days_since_prev_major,
                              haversine_km, label_hours, load_aegean_events,
                              load_aegean_events_with_location)
from forecast.features import FEATURE_NAMES, build_catalog_features


def build_region_split(args, hour_index, n, n_features, region=None):
    """Builds a geographic train/test split of the catalog branch.

    Splitting by recording station -- the parent project's spatial test -- is
    meaningless for this branch, and not only because no station data reaches
    it: catalog features and labels are region-wide, and the two stations that
    test used shared 95.9% of their hours, so "train one / test the other"
    would test on the very rows it trained on. The honest analogue is to split
    the *catalog* in space -- train on one half of the study region, test on the
    other -- so the test set is a patch of crust whose events the model has
    never seen. Which region that is comes from `--catalog-bbox` /
    `--catalog-radius`; the boundary is the median of the events actually
    loaded, so the halves follow the region rather than the Aegean default.

    Space alone is not enough. If train covered region A over the whole timeline
    and test covered region B over the whole timeline, a regional swarm at time
    t would raise `count_7d` in A and the label in B simultaneously. The model
    never sees a timestamp, but it does not need one: it would learn "features
    that look like a busy period -> positive", and busy periods are shared
    across the bbox. So the split is space AND time -- region A up to the cut,
    region B after it, with the usual embargo already applied by the caller's
    single-split index arithmetic.

    Args:
        args: The parsed arguments of the run.
        hour_index: DatetimeIndex of hour starts.
        n: Number of hourly rows.
        n_features: Feature width the caller already built, checked against.
        region: The study region to split in half. Defaults to the Aegean box.

    Returns:
        (cat_features, labels, dsp, cut) where rows [0, cut) are built from the
        train-side region and rows [cut, n) from the held-out region.
    """
    axis = args.region_split
    coord_name = "lat" if axis == "lat" else "lon"

    mt, _, mlat, mlon = load_aegean_events_with_location(args.catalog_path,
                                                        args.threshold,
                                                        region=region)
    bt, bm, blat, blon = load_aegean_events_with_location(args.catalog_path,
                                                          args.bg_min_mag,
                                                          region=region)

    boundary = args.region_split_value
    if boundary is None:
        boundary = float(np.median(mlat if axis == "lat" else mlon))

    def side_mask(lats, lons, want_high):
        coord = lats if axis == "lat" else lons
        return coord >= boundary if want_high else coord < boundary

    test_high = args.region_test_side == "high"
    sides = {"train": not test_high, "test": test_high}

    print(f"\n  [region-split] {coord_name} boundary {boundary:.3f}deg, "
          f"test side = {args.region_test_side} "
          f"({'north' if axis == 'lat' and test_high else 'south' if axis == 'lat' else 'east' if test_high else 'west'})")

    built = {}
    for role, want_high in sides.items():
        mm = side_mask(mlat, mlon, want_high)
        bb = side_mask(blat, blon, want_high)
        mt_r, bt_r, bm_r = mt[mm], bt[bb], bm[bb]
        d_r = days_since_prev_major(hour_index, mt_r)
        # Location-derived features (NND/entropy) are intentionally not passed here:
        # inside a half-bbox their neighbour statistics mean something different than
        # the region-wide values every other run used, which would confound the
        # transfer test with a feature-definition change.
        cf_r = build_catalog_features(hour_index, mt_r, d_r, bt_r, bm_r, args.bg_min_mag,
                                      grid_bbox=None if region is None else region.bbox)
        lb_r = label_hours(hour_index, mt_r, args.horizon_days)
        if args.keep_features is not None:
            cf_r = cf_r[:, [FEATURE_NAMES.index(f) for f in args.keep_features]]
        built[role] = (cf_r, lb_r, d_r, mt_r)
        print(f"    {role:5s} side: {len(mt_r)} M>={args.threshold} events, "
              f"{len(bt_r)} M>={args.bg_min_mag} background, "
              f"hourly positive rate {lb_r.mean():.3f}")

    if built["train"][0].shape[1] != n_features:
        raise SystemExit(f"[ERROR] region-split rebuilt {built['train'][0].shape[1]} features, "
                         f"expected {n_features}. --region-split does not support "
                         f"--rate-features or location-derived features.")

    # The label looks horizon_days forward, so that is what a block boundary has
    # to clear on top of the input-window overlap.
    embargo = args.seq_hours - 1 + int(round(args.horizon_days * 24))
    n_valid = n - (args.seq_hours - 1)
    i_val = int(n_valid * (args.train_frac + args.val_frac))
    cut = (args.seq_hours - 1) + i_val + embargo
    if cut >= n:
        raise SystemExit(f"[ERROR] region-split time cut {cut} lands past the archive end "
                         f"({n}). Lower --train-frac/--val-frac or --horizon-days.")

    cf = built["train"][0].copy()
    lb = built["train"][1].copy()
    dd = built["train"][2].copy()
    cf[cut:] = built["test"][0][cut:]
    lb[cut:] = built["test"][1][cut:]
    dd[cut:] = built["test"][2][cut:]

    # Distinct held-out events inside the test block -- the block's effective sample
    # size. Thousands of hourly rows driven by a handful of earthquakes is the
    # recurring trap in this project, so it goes in the log next to the positive rate.
    test_times = built["test"][3]
    lo = hour_index[cut].to_datetime64()
    hi = hour_index[-1].to_datetime64()
    n_teeth = int(np.sum((test_times >= lo) & (test_times <= hi)))
    print(f"    time cut at row {cut} ({hour_index[cut]}), embargo {embargo}h already applied")
    print(f"    test block: {n - cut} rows, positive rate {lb[cut:].mean():.3f}, "
          f"{n_teeth} distinct M>={args.threshold} events (effective n)")
    if n_teeth < 5:
        print(f"    [!] only {n_teeth} distinct events in the held-out block -- its AUC will "
              f"be dominated by a handful of earthquakes. Treat as indicative, not a result.")
    return cf, lb, dd, cut


def catalog_arrays(catalog_path):
    """The catalogue as raw arrays, for scoring many candidate regions.

    Ranking measures a few hundred discs, and going through pandas for each one
    builds a few hundred filtered copies of a 576k-row frame. The geometry
    question only needs four columns, so it is answered in numpy.

    Args:
        catalog_path: Catalogue CSV.

    Returns:
        Tuple of (times, magnitudes, lats, lons), sorted by time.
    """
    cat = _read_catalog(str(catalog_path))
    ok = cat.dt.notna().to_numpy()
    order = np.argsort(cat.dt.to_numpy()[ok], kind="stable")
    return (cat.dt.to_numpy()[ok][order],
            cat.Magnitude.to_numpy(dtype=np.float64)[ok][order],
            cat.Latitude.to_numpy(dtype=np.float64)[ok][order],
            cat.Longitude.to_numpy(dtype=np.float64)[ok][order])


def region_seismicity(catalog_path, region, hour_index, threshold, bg_min_mag,
                      horizon_days, arrays=None):
    """The two properties that decide whether a region can train for another.

    Transfer needs the source and target to pose the same question. Two things
    govern that: how often the label is positive, and how the magnitudes are
    distributed. A region that is positive 97% of the time teaches "always yes"
    to a target that is positive 12% of the time, whatever its features look
    like -- which is exactly what a country-wide training region does here.

    Args:
        catalog_path: Catalogue CSV.
        region: The `Region` to measure.
        hour_index: The hours to measure over. **Pass the training window
            only.** Statistics computed over hours the model will later be
            tested on would select training regions partly for what happens in
            the test block.
        threshold: Magnitude defining a positive label.
        bg_min_mag: Completeness threshold of the background catalogue.
        horizon_days: Label horizon.
        arrays: Optional pre-extracted `catalog_arrays` output, so a sweep over
            many candidates parses and sorts the catalogue once.

    Returns:
        Dict with `n_major`, `n_background`, `pos_rate` and `b_value`, or None
        if the region is too empty to characterise.
    """
    if arrays is None:
        arrays = catalog_arrays(catalog_path)
    times, mags, lats, lons = arrays
    inside = region.mask(lats, lons)
    major = times[inside & (mags >= threshold)]
    bg_sel = inside & (mags >= bg_min_mag)
    bg, bg_mags = times[bg_sel], mags[bg_sel]
    if len(major) < 5 or len(bg) < 50:
        return None
    # Aki maximum-likelihood b, the same estimator `features.py` uses per window.
    b = (1.0 / np.log(10.0)) / max(bg_mags.mean() - bg_min_mag, 1e-3)
    return {"n_major": len(major), "n_background": len(bg), "b_value": float(b),
            "pos_rate": float(label_hours(hour_index, major, horizon_days).mean())}


def rank_candidate_regions(args, eval_region, train_hours, n_wanted, grid_deg=1.2):
    """Ranks same-size regions by how closely they pose the target's question.

    Candidates are discs of the eval region's own radius, on a grid, excluding
    anything within twice that radius of the target so no earthquake can appear
    in both the training pool and the test block. Ranking is by distance in
    positive rate (in log space, since rates span an order of magnitude) plus
    distance in b-value.

    A geographic or fault-zone rule was tried first and does not survive
    checking: seismicity here is not one latitude band, and a hand-drawn
    corridor captured under 20% of events in most longitude bins. Measured
    similarity is checkable against the catalogue; a drawn boundary is not.

    Args:
        args: The parsed arguments of the run.
        eval_region: The target `Region`; must be a disc.
        train_hours: The training window's hours. Statistics come from these
            alone.
        n_wanted: How many regions to return.
        grid_deg: Candidate centre spacing in degrees.

    Returns:
        List of (Region, stats dict), most similar first.

    Raises:
        SystemExit: If the eval region is not a disc, since a candidate has to
            be the same shape as the target to pose the same question.
    """
    if eval_region.center is None:
        raise SystemExit("[ERROR] --train-regions needs a disc-shaped study "
                         "region, so the training regions can be the same shape "
                         "and size. Use --station-radius or --catalog-radius.")
    arrays = catalog_arrays(args.catalog_path)
    target = region_seismicity(args.catalog_path, eval_region, train_hours,
                               args.threshold, args.bg_min_mag, args.horizon_days,
                               arrays=arrays)
    if target is None:
        raise SystemExit("[ERROR] --train-regions: the study region has too few "
                         "events in the training window to match others against.")
    print(f"  [transfer] target: {target['n_major']} M>={args.threshold} events, "
          f"positive rate {target['pos_rate']:.3f}, b-value {target['b_value']:.2f}")

    clat, clon = eval_region.center
    radius = eval_region.radius_km
    span = max(6.0, 2.5 * radius / KM_PER_DEG_LAT)
    scored = []
    for lat in np.arange(clat - span, clat + span + 1e-9, grid_deg):
        for lon in np.arange(clon - 2 * span, clon + 2 * span + 1e-9, grid_deg):
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                continue
            # Two radii apart: the discs cannot share a single earthquake.
            if haversine_km(clat, clon, np.array([lat]), np.array([lon]))[0] < 2 * radius:
                continue
            cand = Region.from_flags(radius=(lat, lon, radius))
            st = region_seismicity(args.catalog_path, cand, train_hours,
                                   args.threshold, args.bg_min_mag,
                                   args.horizon_days, arrays=arrays)
            if st is None:
                continue
            st["score"] = (abs(np.log(st["pos_rate"] / max(target["pos_rate"], 1e-6)))
                           + abs(st["b_value"] - target["b_value"]))
            scored.append((cand, st))
    scored.sort(key=lambda p: p[1]["score"])
    return scored[:n_wanted]


def build_training_regions(args, hour_index, regions, feature_names):
    """Builds one feature and label array per training region.

    Each region is built exactly as the study region is -- same loaders, same
    `--keep-features` subset, and its own `grid_bbox` so the spatial entropy
    feature is binned over that region rather than the target's.

    Args:
        args: The parsed arguments of the run.
        hour_index: DatetimeIndex of hour starts, shared by every region.
        regions: The `Region`s to build.
        feature_names: The active feature names, for the `--keep-features` subset.

    Returns:
        List of (Region, cat_features, labels).
    """
    from forecast.features import ALL_FEATURE_NAMES

    built = []
    for region in regions:
        major = load_aegean_events(args.catalog_path, args.threshold, region=region)
        bg, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
            args.catalog_path, args.bg_min_mag, region=region)
        if not any(f in feature_names for f in ("nnd_log_eta_90d", "shannon_entropy_90d")):
            bg_lats = bg_lons = None
        dsp = days_since_prev_major(hour_index, major)
        cf = build_catalog_features(hour_index, major, dsp, bg, bg_mags,
                                    args.bg_min_mag, bg_lats, bg_lons,
                                    grid_bbox=region.bbox)
        if args.keep_features is not None:
            keep = [ALL_FEATURE_NAMES.index(f) for f in args.keep_features]
            cf = cf[:, [k for k in keep if k < cf.shape[1]]]
        built.append((region, cf, label_hours(hour_index, major, args.horizon_days)))
    return built
