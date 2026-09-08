"""Holding out a patch of crust instead of a stretch of time.

Not a runnable script -- imported only, by `train.py`. Off by default
(`--region-split none`); it is kept because it is the honest spatial
generalisation test, and this repo exists to make the evaluation harder to skip.

Ported unchanged from `cnn_earthquake`'s
`forecasting/cnn_lstm_catalog_waveform_fusion.py`, minus the `n_features`
argument the waveform path needed.
"""
import numpy as np

from forecast.catalog import (days_since_prev_major, label_hours,
                              load_aegean_events_with_location)
from forecast.features import FEATURE_NAMES, build_catalog_features


def build_region_split(args, hour_index, n, n_features):
    """Builds a geographic train/test split of the catalog branch.

    A literal station split is meaningless for this branch: catalog features and
    labels are region-wide, and BODT/DAT share 95.9% of their hours, so
    "train BODT / test DAT" would test on the very rows it trained on. The
    honest analogue is to split the *catalog* in space -- train on one half of
    the AEGEAN bbox, test on the other -- so the test set is a patch of crust
    whose events the model has never seen.

    Space alone is not enough. If train covered region A over the whole timeline
    and test covered region B over the whole timeline, a regional swarm at time
    t would raise `count_7d` in A and the label in B simultaneously. The model
    never sees a timestamp, but it does not need one: it would learn "features
    that look like a busy period -> positive", and busy periods are shared
    across the bbox. So the split is space AND time -- region A up to the cut,
    region B after it, with the usual embargo already applied by the caller's
    single-split index arithmetic.

    Returns:
        (cat_features, labels, dsp, cut) where rows [0, cut) are built from the
        train-side region and rows [cut, n) from the held-out region.
    """
    axis = args.region_split
    coord_name = "lat" if axis == "lat" else "lon"

    mt, _, mlat, mlon = load_aegean_events_with_location(
        args.catalog_path, args.threshold,
        stations=args.stations, max_dist_km=args.max_station_dist_km)
    bt, bm, blat, blon = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag,
        stations=args.stations, max_dist_km=args.max_station_dist_km)

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
        cf_r = build_catalog_features(hour_index, mt_r, d_r, bt_r, bm_r, args.bg_min_mag)
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
