"""catalog_mlp: forecast M>=threshold within a horizon, from the catalogue alone.

    forecast train --catalog-path catalogs/catalog_current.csv \
        --catalog-span 2000-01-01 2026-08-12 \
        --horizon-days 14 --cv-folds 2 --bg-min-mag 3.0 --batch-size 128 \
        --keep-features log1p_dsp mean_mag_30d cv_interevent_90d mag_deficit_90d \
        --ensemble-seeds 42,43,44,45,46

That is the published configuration, and it is the reason this project exists:
of fourteen forecasters tried, this is the one that clears its own fold's floor.

**There is no waveform branch, deliberately.** The original module was
`cnn_lstm_catalog_waveform_fusion.py`, and its waveform arm is what established
the negative half of the result -- neither the raw-waveform CNN nor the
hand-crafted continuous features nor the chaotic features beat persistence, on
the corrected catalogue, across every architecture tried. The 2026-08-30
experiment concluded that "the forecasting signal in this project comes from the
earthquake catalogue, not from the seismogram." That ablation and its figures
stay in `cnn_earthquake`; this repo is the half that survived it.

Consequently `--channels`, `--data-root` and the transfer-learning flags
(`--save/--load/--freeze-catalog-branch`, which existed to seed the fusion
model's catalogue trunk) are gone. `--catalog-span` is no longer optional: it is
how the hourly index is built when no archive defines it.

**Every number goes through `evaluate.fold_result`.** A bare AUC is not a result
on this data -- fold SD runs 0.07-0.16 and models have beaten a pooled number
while losing to persistence on most folds -- so the floor and the per-fold
spread are computed with it, not after it.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from forecast.catalog import (days_since_prev_major, label_hours,
                              label_hours_rate_change,
                              load_aegean_events,
                              load_aegean_events_with_location,
                              truncate_to_reliable_catalog_end)
from forecast.data import CatalogSeqDataset
from forecast.evaluate import fold_result, summarise
from forecast.features import (ALL_FEATURE_NAMES, FEATURE_NAMES,
                               RATE_FEATURE_NAMES, RATE_WINDOWS,
                               build_catalog_features, build_rate_features)
from forecast.metrics import safe_auc
from forecast.model import CatalogMLPNet
from forecast.regions import build_region_split
from forecast.seeding import seed_everything
from forecast.splits import print_split_diagnostics, walk_forward_splits

NAME = "train"
HELP = "train catalog_mlp and score it against its floor"


def add_args(p):
    """The flags catalog_mlp keeps. See the module docstring for what went."""
    p.add_argument("--catalog-path", required=True,
                   help="catalogue CSV with Date/Latitude/Longitude/Magnitude")
    p.add_argument("--catalog-span", nargs=2, metavar=("START", "END"), required=True,
                   help="the hourly index to build, e.g. 2000-01-01 2026-08-12. "
                        "Required: with no waveform archive, nothing else "
                        "defines how long the record is.")
    p.add_argument("--threshold", type=float, default=4.5,
                   help="magnitude defining a positive label")
    p.add_argument("--bg-min-mag", type=float, default=3.0,
                   help="completeness threshold for the BACKGROUND catalogue the "
                        "magnitude-distribution features are estimated from. The "
                        "rare M>=--threshold set is too small to fit a b-value or "
                        "an inter-event CV to.")
    p.add_argument("--horizon-days", type=float, default=14.0)
    p.add_argument("--seq-hours", type=int, default=24)
    p.add_argument("--label-mode", default="event", choices=["event", "rate"],
                   help="event: does an M>=threshold event occur within the "
                        "horizon. rate: will the next window hold MORE events "
                        "than the trailing one -- an acceleration target, driven "
                        "by ~10^3 events instead of ~10^1.")
    p.add_argument("--rate-min-mag", type=float, default=3.0)
    p.add_argument("--rate-baseline-days", type=float, default=None)
    p.add_argument("--rate-features", action="store_true",
                   help="append trailing-rate count/ratio features. Without them "
                        "a rate-mode run is trying to beat a persistence floor "
                        "built from exactly the number it was never given.")
    p.add_argument("--keep-features", nargs="+", default=None, metavar="FEATURE",
                   help=f"restrict to this subset by name from {FEATURE_NAMES}")
    p.add_argument("--stations", nargs="+", default=None)
    p.add_argument("--max-station-dist-km", type=float, default=None)

    g = p.add_argument_group("model")
    g.add_argument("--cat-hidden", type=int, default=16)
    g.add_argument("--fusion-hidden", type=int, default=32)
    g.add_argument("--dropout", type=float, default=0.2,
                   help="0.2 suits this 2-11-input MLP; aggressive dropout can "
                        "knock it off a good basin instead of regularizing it")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=40)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--patience", type=int, default=8)
    g.add_argument("--checkpoint-metric", default="auc", choices=["auc", "loss"])
    g.add_argument("--ensemble-seeds", default="42,43,44")
    g.add_argument("--random-seeds", type=int, default=None,
                   help="draw N seeds instead. Fixed seeds hide run-to-run "
                        "variance behind one sample of it, and per-seed spread "
                        "reaches 0.17 here. Drawn seeds are printed so the run "
                        "can be replayed via --ensemble-seeds.")

    g = p.add_argument_group("evaluation")
    g.add_argument("--cv-folds", type=int, default=1)
    g.add_argument("--train-frac", type=float, default=0.70)
    g.add_argument("--val-frac", type=float, default=0.15)
    g.add_argument("--balanced-folds", action="store_true",
                   help="place block boundaries by positive-label MASS rather "
                        "than equal hour count, so one swarm cannot fill a block")
    g.add_argument("--skip", type=int, nargs="*", default=[])
    g.add_argument("--region-split", default="none",
                   choices=["none", "lat", "lon"],
                   help="hold out half the AEGEAN bbox instead of splitting in "
                        "time -- a patch of crust whose events the model has "
                        "never seen. Forces --cv-folds 1.")
    g.add_argument("--region-split-value", type=float, default=None)
    g.add_argument("--region-test-side", default="high", choices=["low", "high"])
    return p


def train_one_seed(args, seed, cat_features, labels, train_idx, val_idx, test_idx,
                   device):
    """Trains and evaluates one seed's model on one split.

    Returns:
        Tuple of (y_true, y_score) for the test split, from the best epoch's
        weights by `--checkpoint-metric`.
    """
    seed_everything(seed)
    train_ds = CatalogSeqDataset(cat_features, labels, args.seq_hours, train_idx)
    val_ds = CatalogSeqDataset(cat_features, labels, args.seq_hours, val_idx,
                               stats=train_ds.stats)
    test_ds = CatalogSeqDataset(cat_features, labels, args.seq_hours, test_idx,
                                stats=train_ds.stats)

    model = CatalogMLPNet(catalog_dim=cat_features.shape[1],
                          cat_hidden=args.cat_hidden,
                          fusion_hidden=args.fusion_hidden,
                          dropout=args.dropout).to(device)

    def dl(ds, sh):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=2)

    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = labels[train_idx].mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), dtype=torch.float32,
                              device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def score(loader):
        model.eval()
        ys, ss, losses = [], [], []
        with torch.no_grad():
            for cat_seq, y in loader:
                cat_seq, y = cat_seq.to(device), y.to(device)
                logit = model(cat_seq)
                losses.append(criterion(logit, y).item() * y.size(0))
                ss.extend(torch.sigmoid(logit).cpu().tolist())
                ys.extend(y.cpu().tolist())
        return np.array(ys, dtype=np.int64), np.array(ss), sum(losses) / max(len(ys), 1)

    best = float("inf") if args.checkpoint_metric == "loss" else -1.0
    no_improve, best_state = 0, None
    for epoch in range(args.epochs):
        model.train()
        for cat_seq, y in train_loader:
            cat_seq, y = cat_seq.to(device), y.to(device)
            loss = criterion(model(cat_seq), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        yv, sv, val_loss = score(val_loader)
        val_auc = safe_auc(yv, sv)
        print(f"  [seed {seed}] epoch {epoch + 1}/{args.epochs} "
              f"val AUC {val_auc:.4f} val loss {val_loss:.4f}")
        metric = val_loss if args.checkpoint_metric == "loss" else val_auc
        improved = metric < best if args.checkpoint_metric == "loss" else metric > best
        if improved:
            best, no_improve = metric, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, st, _ = score(test_loader)
    print(f"  [seed {seed}] test AUC {safe_auc(yt, st):.4f}")
    return yt, st


def run_fold(fold_label, args, cat_features, labels, dsp, hour_index, train_idx,
             val_idx, test_idx, seeds, device, rate_trailing=None):
    """Trains the seed ensemble on one split and scores it against its floor.

    Returns:
        A `FoldResult`, or None if the split is too thin (fewer than 10 train
        or 5 test windows).
    """
    print(f"\n{'=' * 64}\n{fold_label}\n{'=' * 64}")
    print(f"  splits (chronological): train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)}")
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx):
            print(f"    {name:5s}: positive rate {labels[idx].mean():.3f}")
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores, yt_ref = [], None
    for seed in seeds:
        yt, st = train_one_seed(args, seed, cat_features, labels, train_idx,
                                val_idx, test_idx, device)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)

    return fold_result(
        fold_label, yt_ref, np.mean(per_seed_scores, axis=0), per_seed_scores,
        labels[train_idx], args.label_mode, dsp[test_idx], args.horizon_days,
        rate_trailing_test=None if rate_trailing is None else rate_trailing[test_idx],
        rate_trailing_train=None if rate_trailing is None else rate_trailing[train_idx])


def run(args):
    """Builds the catalogue features and labels, then runs the fold sweep."""
    if args.region_split != "none":
        # One geographic holdout, not a sweep: walk-forward folds would re-slice
        # a time axis whose spatial meaning already changes at the cut.
        if args.cv_folds != 1:
            print(f"  [region-split] forcing --cv-folds 1 (was {args.cv_folds})")
            args.cv_folds = 1
        if args.label_mode != "event" or args.rate_features:
            sys.exit("[ERROR] --region-split supports --label-mode event without "
                     "--rate-features only; the rate target's trailing-count floor "
                     "is defined region-wide and would not match a half-bbox label.")

    import pandas as pd
    print("Building the hourly index and catalogue features...")
    hour_index = pd.date_range(args.catalog_span[0], args.catalog_span[1], freq="h")
    print(f"  [catalog-span] {len(hour_index)} hourly rows, "
          f"{hour_index[0]} -> {hour_index[-1]}")

    major_times = load_aegean_events(args.catalog_path, args.threshold,
                                     stations=args.stations,
                                     max_dist_km=args.max_station_dist_km)
    bg_times, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag, stations=args.stations,
        max_dist_km=args.max_station_dist_km)
    if args.max_station_dist_km:
        print(f"  [station-cap] catalogue restricted to "
              f"<={args.max_station_dist_km:.0f}km from "
              f"{args.stations or 'STATION_COORDS defaults'}")
    # The NND precompute inside build_catalog_features is O(n_bg * NND_LOOKBACK)
    # haversine in a Python loop and fires whenever coordinates are supplied. Skip
    # it when this run's feature subset has no location-derived column -- otherwise
    # a 4-feature run pays the full cost for columns it then discards.
    if args.keep_features is not None and not any(
            f in args.keep_features for f in ("nnd_log_eta_90d", "shannon_entropy_90d")):
        bg_lats = bg_lons = None
    print(f"  {len(major_times)} M>={args.threshold} AEGEAN events, "
          f"{len(bg_times)} M>={args.bg_min_mag} background events")

    # `raw` is a length-1 dummy channel: nothing here reads a waveform, but
    # truncate_to_reliable_catalog_end trims an array alongside the index.
    dummy = np.zeros((len(hour_index), 1), dtype=np.float32)
    hour_index, _ = truncate_to_reliable_catalog_end(hour_index, dummy, major_times,
                                                     buffer_days=args.horizon_days)

    dsp = days_since_prev_major(hour_index, major_times)
    cat_features = build_catalog_features(hour_index, major_times, dsp, bg_times,
                                          bg_mags, args.bg_min_mag, bg_lats, bg_lons)

    # Rate features are appended BEFORE --keep-features subsets, so the flag can
    # select them by name alongside the originals.
    rate_times, active_names = None, FEATURE_NAMES
    if args.label_mode == "rate" or args.rate_features:
        rate_times, _, _, _ = load_aegean_events_with_location(args.catalog_path,
                                                              args.rate_min_mag)
    if args.rate_features:
        cat_features = np.hstack([cat_features,
                                  build_rate_features(hour_index, rate_times)])
        active_names = ALL_FEATURE_NAMES
        print(f"  + {len(RATE_FEATURE_NAMES)} trailing-rate features "
              f"(M>={args.rate_min_mag}, windows {RATE_WINDOWS}d)")

    if args.keep_features is not None:
        unknown = [f for f in args.keep_features if f not in active_names]
        if unknown:
            sys.exit(f"[ERROR] --keep-features got {unknown}. Known: {active_names}"
                     + ("" if args.rate_features else
                        " -- pass --rate-features to enable the rate ones."))
        keep_idx = [active_names.index(f) for f in args.keep_features]
        cat_features = cat_features[:, keep_idx]
        print(f"  restricting to {len(keep_idx)} feature(s): {args.keep_features}")

    rate_trailing = None
    if args.label_mode == "rate":
        labels, _, rate_trailing = label_hours_rate_change(
            hour_index, rate_times, args.horizon_days, args.rate_baseline_days)
    else:
        labels = label_hours(hour_index, major_times, args.horizon_days)
    print(f"  hourly positive rate: {labels.mean():.3f}")

    n = len(hour_index)
    if args.region_split != "none":
        # Rebuilds features and labels per half-bbox from the catalogue itself,
        # so it takes the width to check against rather than the arrays.
        cat_features, labels, dsp, _ = build_region_split(
            args, hour_index, n, cat_features.shape[1])

    valid_end_indices = np.arange(args.seq_hours - 1, n)
    # seq_hours-1 removes *input*-window overlap across a block boundary; the label
    # additionally looks horizon_days forward, so without the extra horizon term the
    # last ~horizon_days of each block carry labels determined by events inside the
    # NEXT block -- i.e. train labels encoding what happens in val, and val labels
    # encoding what happens in test (~9% of samples at horizon=14d, seq=24h). That's
    # the overlapping-label leakage that purging/embargo exists to prevent
    # (Lopez de Prado, Advances in Financial Machine Learning, Ch. 7).
    embargo = args.seq_hours - 1 + int(round(args.horizon_days * 24))

    if args.cv_folds <= 1:
        n_valid = len(valid_end_indices)
        i_train = int(n_valid * args.train_frac)
        i_val = int(n_valid * (args.train_frac + args.val_frac))
        folds = [(valid_end_indices[:i_train],
                  valid_end_indices[i_train + embargo:i_val],
                  valid_end_indices[i_val + embargo:])]
        fold_labels = ["single split"]
    elif args.balanced_folds:
        # Boundaries by equal positive-label MASS rather than equal hour count, so
        # one sustained swarm cannot fill a block almost entirely. Decided from the
        # label series before any model runs and applied uniformly -- not the same
        # as picking whichever fold scores best, which would be test-set selection.
        print("  [balanced-folds] block boundaries placed by positive-label mass, "
              "not equal hour count")
        folds = walk_forward_splits(valid_end_indices, args.cv_folds,
                                    labels=labels[valid_end_indices], embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]
        # Equal positive-MASS boundaries degenerate when the positive class is rare
        # and clustered: at --label-mode event this produced a 0.999-positive test
        # block, where AUC is meaningless. Measured, not hypothetical.
        for k, (_, va, te) in enumerate(folds, 1):
            for split_name, idx in (("val", va), ("test", te)):
                if len(idx) and not 0.02 <= labels[idx].mean() <= 0.98:
                    print(f"  [!] --balanced-folds made fold {k}'s {split_name} block "
                          f"{labels[idx].mean():.3f}-positive -- near-degenerate, so "
                          f"its AUC would be uninformative. Re-run without it (the "
                          f"flag suits balanced targets like --label-mode rate).")
    else:
        folds = walk_forward_splits(valid_end_indices, args.cv_folds, embargo=embargo)
        fold_labels = [f"fold {k + 1}/{args.cv_folds}" for k in range(args.cv_folds)]

    skip = set(args.skip)
    for k in sorted(skip):
        if 1 <= k <= len(folds):
            print(f"\n[skip] {fold_labels[k - 1]} (--skip)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.random_seeds:
        seeds = [int(s) for s in
                 np.random.default_rng().integers(0, 2 ** 31 - 1, size=args.random_seeds)]
        print(f"  [random-seeds] drew {len(seeds)} seeds: "
              f"--ensemble-seeds {','.join(str(s) for s in seeds)}")
    else:
        seeds = [int(s) for s in args.ensemble_seeds.split(",")]

    results = []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(
            zip(fold_labels, folds), 1):
        if k in skip:
            continue
        r = run_fold(fold_label, args, cat_features, labels, dsp, hour_index,
                     train_idx, val_idx, test_idx, seeds, device,
                     rate_trailing=rate_trailing)
        if r is not None:
            results.append(r)

    if args.cv_folds > 1:
        summarise(results, args.cv_folds)
    return 0


def main():
    p = argparse.ArgumentParser(
        prog="forecast train", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
