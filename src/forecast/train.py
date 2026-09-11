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

Consequently `--channels`, `--data-root`, the transfer-learning flags
(`--save/--load/--freeze-catalog-branch`, which existed to seed the fusion
model's catalogue trunk) and the station filter (`--stations`,
`--max-station-dist-km`, which cut the catalogue down to events near a
seismometer this project does not read) are gone. `--catalog-span` is no longer
optional: it is how the hourly index is built when no archive defines it.

`--catalog-bbox`, `--catalog-radius` and `--station-radius` narrow that region:
the run then reads, scores and forecasts only inside a box or a disc, the last
of them centred on a station looked up in `--station-catalog`. The default is
the Aegean box every published number was produced over. `--region-split` still
holds out half of whatever region is in force, since its boundary is the median
of the events actually loaded.

**A fold here can train on ten earthquakes.** That, not the network, is the
binding constraint. `--train-regions N` pools N further regions of the same size
and shape into TRAINING only, chosen by measured similarity in positive rate and
b-value, so the model sees on the order of 10^3 events instead of 10. Val, test
and the floors still come from the study region alone, which is what keeps a
pooled run comparable to one without it. A geographic or fault-zone rule was
tried first and fails: a country-wide region is positive 97% of the time against
a target at 12%, and seismicity here is not one latitude band.

**One earthquake is not 336 observations.** The label at hour H looks
`--horizon-days` forward, so at a 14-day horizon a single qualifying event turns
336 consecutive hourly rows positive. A test block reporting n=32448 has been
measured at ten distinct events, and an AUC over ten events carries roughly
+/-0.16 -- larger than every gap this project reports between a model and its
floor. `--eval-stride-hours 168` scores weekly, non-overlapping samples instead,
and `--train-stride-hours` does the same for training, which is a separate
question about whether that 336-fold duplication is what drives the overfitting.
Both default to 1, which is how every published number was produced. Neither
raises the AUC; they make the count honest and the interval visible.

`--out-dir` writes the trained models: one file per fold per seed, each holding
its weights beside the train split's normalization stats and the feature order
they were fit to, since neither is recoverable from the other two later. That is
not the old `--save-catalog-branch`, which existed to hand a warm trunk to a
fusion model that no longer exists here. See `checkpoint.py`.

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
from torch.utils.data import ConcatDataset, DataLoader

from forecast.catalog import (Region, days_since_prev_major, label_hours,
                              label_hours_rate_change,
                              load_aegean_events,
                              load_aegean_events_with_location,
                              truncate_to_reliable_catalog_end)
from forecast.checkpoint import save_checkpoint, write_manifest
from forecast.data import CatalogSeqDataset
from forecast.evaluate import fold_result, summarise
from forecast.features import (ALL_FEATURE_NAMES, FEATURE_NAMES,
                               RATE_FEATURE_NAMES, RATE_WINDOWS,
                               build_catalog_features, build_rate_features)
from forecast.metrics import safe_auc
from forecast.model import CatalogMLPNet
from forecast.regions import (build_region_split, build_training_regions,
                              rank_candidate_regions, region_seismicity)
from forecast.seeding import seed_everything
from forecast.splits import print_split_diagnostics, walk_forward_splits
from forecast.stations import station_region

NAME = "train"
HELP = "train catalog_mlp and score it against its floor"

# Below this many M>=threshold events, the run is reporting on a handful of
# earthquakes however many hourly rows it has. The whole Aegean holds ~10^2 of
# them, so a narrowed region reaches this quickly.
MIN_MAJOR_EVENTS = 30


def add_args(p):
    """The flags catalog_mlp keeps. See the module docstring for what went."""
    p.add_argument("--catalog-path", required=True,
                   help="catalogue CSV with Date/Latitude/Longitude/Magnitude")
    p.add_argument("--catalog-span", nargs=2, metavar=("START", "END"), required=True,
                   help="the hourly index to build, e.g. 2000-01-01 2026-08-12. "
                        "Required: with no waveform archive, nothing else "
                        "defines how long the record is.")
    region = p.add_mutually_exclusive_group()
    region.add_argument("--catalog-bbox", nargs=4, type=float, default=None,
                        metavar=("LAT0", "LAT1", "LON0", "LON1"),
                        help="forecast only for this box instead of the Aegean "
                             "default (36 40 25 30). Narrowing the region changes "
                             "the question rather than sharpening it: features, "
                             "labels, the floor and the folds are all rebuilt from "
                             "the events inside it.")
    region.add_argument("--catalog-radius", nargs=3, type=float, default=None,
                        metavar=("LAT", "LON", "KM"),
                        help="forecast only within KM of this point -- a true "
                             "great-circle disc, not its bounding box. Same effect "
                             "on the run as --catalog-bbox; the M>=4.5 set is ~10^2 "
                             "events across the whole Aegean, so a small disc can "
                             "leave too few to score.")
    region.add_argument("--station-radius", nargs=2, default=None,
                        metavar=("STATION", "KM"),
                        help="the same disc, centred on a station named "
                             "NETWORK.CODE (e.g. TU.ABT) and looked up in "
                             "--station-catalog. Nothing here reads the station's "
                             "data; it is a way of saying where.")
    p.add_argument("--station-catalog", default=None, metavar="CSV",
                   help="station inventory for --station-radius, with "
                        "Network,Code,Longitude,Latitude columns (Height, Province "
                        "and District are ignored). Read by name, not position: "
                        "longitude precedes latitude in this format.")
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
    g.add_argument("--train-stride-hours", type=int, default=1, metavar="N",
                   help="keep every Nth training sample. At the default 1 every "
                        "hour is a sample, so one event's feature signature is "
                        "repeated across horizon_days*24 rows and weighted that "
                        "many times. 168 makes training samples weekly and "
                        "non-overlapping.")
    g.add_argument("--eval-stride-hours", type=int, default=1, metavar="N",
                   help="keep every Nth val and test sample. At the default 1 a "
                        "single event turns horizon_days*24 consecutive rows "
                        "positive, so a reported n of 32448 can be ten "
                        "earthquakes. 168 scores weekly, non-overlapping "
                        "samples, which is the honest count.")
    g.add_argument("--out-dir", default=None, metavar="DIR",
                   help="save each (fold, seed) model here, with the training "
                        "split's normalization stats and the feature list they "
                        "were computed over -- none of the three is recoverable "
                        "from the others at inference time. Off by default: a "
                        "sweep writes one file per fold per seed.")
    g.add_argument("--random-seeds", type=int, default=None,
                   help="draw N seeds instead. Fixed seeds hide run-to-run "
                        "variance behind one sample of it, and per-seed spread "
                        "reaches 0.17 here. Drawn seeds are printed so the run "
                        "can be replayed via --ensemble-seeds.")

    g = p.add_argument_group("transfer")
    g.add_argument("--train-regions", type=int, default=0, metavar="N",
                   help="pool N further regions of the SAME size and shape as the "
                        "study region into training, chosen by measured similarity "
                        "in positive rate and b-value over the TRAINING window "
                        "only. Val and test still come from the study region "
                        "alone, so the floors and the reported n do not move. "
                        "Fixes the real constraint: a fold here can train on ten "
                        "earthquakes.")
    g.add_argument("--train-region-centers", nargs="+", default=None,
                   metavar="LAT,LON",
                   help="name the training regions explicitly instead of "
                        "selecting them, e.g. 37.7,31.0 40.1,34.0. Takes "
                        "precedence over --train-regions. This is what an "
                        "auto-selected run prints, so the run can be replayed.")
    g.add_argument("--train-region-grid-deg", type=float, default=1.2,
                   metavar="DEG",
                   help="candidate centre spacing for --train-regions")

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


def pooled_stats(datasets):
    """Normalization statistics over every training region at once.

    One shared mean and sd, not one per region. Per-region normalization would
    erase exactly what distinguishes a quiet region from an active one, which
    is the signal the pool exists to supply.

    Args:
        datasets: `CatalogSeqDataset`s covering the training split, each having
            computed its own stats.

    Returns:
        A (mu, sd) pair, shape (1, n_features) each.
    """
    mus = np.concatenate([d.stats[0] for d in datasets], axis=0)
    sds = np.concatenate([d.stats[1] for d in datasets], axis=0)
    # The pooled sd is NOT the mean of the regions' sds: that counts only the
    # spread inside each region and drops the spread BETWEEN their means, so it
    # always understates and the z-scores always come out too large. Measured at
    # 1.01-1.06x on twelve Turkish regions, which is immaterial -- but the same
    # mistake with a 52x understatement is what saturated this model once before
    # (see `data.py`), so the statistic is computed correctly rather than nearly.
    pooled_sd = np.sqrt((sds ** 2).mean(axis=0, keepdims=True)
                        + mus.var(axis=0, keepdims=True))
    return mus.mean(axis=0, keepdims=True), pooled_sd


def train_one_seed(args, seed, cat_features, labels, train_idx, val_idx, test_idx,
                   device, fold_label=None, feature_names=None, region=None,
                   pool=None):
    """Trains and evaluates one seed's model on one split.

    Args:
        args: The parsed arguments of the run.
        seed: The seed to train under.
        cat_features: Per-hour catalogue feature array.
        labels: Per-hour binary labels.
        train_idx: Window end-indices of the train split.
        val_idx: Window end-indices of the val split.
        test_idx: Window end-indices of the test split.
        device: Torch device to train on.
        fold_label: The fold's printed label, used to name a saved checkpoint.
        feature_names: The feature columns, in the order the model reads them.
            Saved with the checkpoint; `--keep-features` makes the order part
            of what a checkpoint means.
        region: The study region, recorded in the checkpoint alongside the
            horizon and threshold as part of what these weights forecast.
        pool: Optional (region, features, labels) triples pooled into TRAINING
            only. Val and test always come from the study region.

    Returns:
        Tuple of (y_true, y_score, checkpoint_path) for the test split, from
        the best epoch's weights by `--checkpoint-metric`. The path is None
        unless `--out-dir` was given.
    """
    seed_everything(seed)
    # The study region is always the first training set, so with no pool this is
    # exactly the single-region path. Extra regions extend training only: val and
    # test are built from the study region below, which is what keeps the floors
    # and the reported n identical to a run without them.
    parts = [CatalogSeqDataset(cat_features, labels, args.seq_hours, train_idx)]
    for _, pool_features, pool_labels in (pool or []):
        parts.append(CatalogSeqDataset(pool_features, pool_labels, args.seq_hours,
                                       train_idx))
    stats = pooled_stats(parts)
    for part in parts:
        part.stats = stats
    train_ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    val_ds = CatalogSeqDataset(cat_features, labels, args.seq_hours, val_idx,
                               stats=stats)
    test_ds = CatalogSeqDataset(cat_features, labels, args.seq_hours, test_idx,
                                stats=stats)

    model = CatalogMLPNet(catalog_dim=cat_features.shape[1],
                          cat_hidden=args.cat_hidden,
                          fusion_hidden=args.fusion_hidden,
                          dropout=args.dropout).to(device)

    def dl(ds, sh):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=2)

    train_loader, val_loader, test_loader = dl(train_ds, True), dl(val_ds, False), dl(test_ds, False)

    pos = float(np.mean([labels[train_idx].mean()]
                        + [pl[train_idx].mean() for _, _, pl in (pool or [])]))
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

    saved = None
    if args.out_dir:
        # The pooled training stats, which is what val and test were scored
        # through and what new hours must be scored through too. Read from the
        # local `stats`, not off the dataset: with a pool the training set is a
        # ConcatDataset and carries no stats of its own.
        saved = save_checkpoint(args.out_dir, fold_label, seed, model,
                                stats, feature_names, args,
                                best if best_state is not None else float("nan"),
                                region=region, pool=pool)
        print(f"  [seed {seed}] saved {saved}")
    return yt, st, saved


def run_fold(fold_label, args, cat_features, labels, dsp, hour_index, train_idx,
             val_idx, test_idx, seeds, device, rate_trailing=None,
             feature_names=None, saved=None, region=None, major_times=None,
             pool=None):
    """Trains the seed ensemble on one split and scores it against its floor.

    Args:
        feature_names: The feature columns, in the order the model reads them,
            recorded in each checkpoint this fold writes.
        saved: Optional list, extended with the path of every checkpoint
            written, so the manifest can list them in the order produced.
        region: The study region, recorded in each checkpoint.
        major_times: Sorted qualifying event times, so the diagnostics can
            report each split's distinct event count beside its sample count.
        pool: Optional training-only regions, pooled into every seed's fit.

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
    print_split_diagnostics(hour_index, labels, train_idx, val_idx, test_idx,
                            major_times=major_times)

    if len(train_idx) < 10 or len(test_idx) < 5:
        print("[ERROR] Not enough hourly data for a meaningful split.")
        return None

    print(f"\nTraining {len(seeds)} seed(s): {seeds}")
    per_seed_scores, yt_ref = [], None
    for seed in seeds:
        yt, st, path = train_one_seed(args, seed, cat_features, labels, train_idx,
                                      val_idx, test_idx, device,
                                      fold_label=fold_label,
                                      feature_names=feature_names, region=region,
                                      pool=pool)
        if yt_ref is None:
            yt_ref = yt
        per_seed_scores.append(st)
        if path is not None and saved is not None:
            saved.append(path)

    return fold_result(
        fold_label, yt_ref, np.mean(per_seed_scores, axis=0), per_seed_scores,
        labels[train_idx], args.label_mode, dsp[test_idx], args.horizon_days,
        rate_trailing_test=None if rate_trailing is None else rate_trailing[test_idx],
        rate_trailing_train=None if rate_trailing is None else rate_trailing[train_idx])


def build_region(args):
    """The study region this run is about, from whichever flag named it.

    Args:
        args: The parsed arguments of the run.

    Returns:
        The named `Region`, or the Aegean default if no flag named one.

    Raises:
        SystemExit: If `--station-radius` was given without the inventory it
            has to look the station up in.
    """
    if args.station_radius:
        if not args.station_catalog:
            raise SystemExit("[ERROR] --station-radius needs --station-catalog to "
                             "look the station up in. Pass --catalog-radius LAT LON "
                             "KM to give the centre directly.")
        station, km = args.station_radius
        try:
            km = float(km)
        except ValueError:
            raise SystemExit(f"[ERROR] --station-radius reads STATION KM; {km!r} is "
                             f"not a radius in km.")
        if km <= 0:
            raise SystemExit(f"[ERROR] --station-radius km must be positive, got {km}.")
        return station_region(args.station_catalog, station, km)
    if args.station_catalog:
        print("  [!] --station-catalog does nothing without --station-radius; "
              "the run is using its default region.")
    return Region.from_flags(args.catalog_bbox, args.catalog_radius)


def build_pool(args, hour_index, region, feature_names, folds):
    """The extra training regions, if any were asked for.

    Args:
        args: The parsed arguments of the run.
        hour_index: DatetimeIndex of hour starts.
        region: The study region. Val and test always come from it alone.
        feature_names: The active feature columns, in model order.
        folds: The cut folds, used for the widest training window so that
            similarity is never measured on hours the model is tested on.

    Returns:
        List of (Region, features, labels), or None if no pool was asked for.

    Raises:
        SystemExit: If a centre cannot be read as LAT,LON.
    """
    if not args.train_region_centers and args.train_regions <= 0:
        return None
    if region.radius_km is None:
        sys.exit("[ERROR] --train-regions needs a disc-shaped study region so the "
                 "training regions can match it. Use --station-radius or "
                 "--catalog-radius.")

    if args.train_region_centers:
        centers = []
        for spec in args.train_region_centers:
            try:
                lat, lon = (float(v) for v in spec.split(","))
            except ValueError:
                sys.exit(f"[ERROR] --train-region-centers reads LAT,LON; got {spec!r}.")
            centers.append(Region.from_flags(radius=(lat, lon, region.radius_km)))
        # Measure them too, so a badly chosen centre shows up in the log rather
        # than only in the result. Same training window as auto-selection uses.
        widest = max((tr for tr, _, _ in folds), key=len)
        train_hours = hour_index[:widest[-1] + 1]
        chosen = [(c, region_seismicity(args.catalog_path, c, train_hours,
                                        args.threshold, args.bg_min_mag,
                                        args.horizon_days)) for c in centers]
    else:
        # The widest training block across folds. Similarity measured past it
        # would pick training regions partly for what happens in a test block.
        widest = max((tr for tr, _, _ in folds), key=len)
        train_hours = hour_index[:widest[-1] + 1]
        print(f"  [transfer] matching over the training window only, "
              f"{train_hours[0].date()}..{train_hours[-1].date()}")
        chosen = rank_candidate_regions(args, region, train_hours,
                                        args.train_regions,
                                        grid_deg=args.train_region_grid_deg)
        if not chosen:
            sys.exit("[ERROR] --train-regions found no candidate region with "
                     "enough events. Widen --train-region-grid-deg or the radius.")
        print("  [transfer] replay this selection with --train-region-centers "
              + " ".join(f"{r.center[0]:.4f},{r.center[1]:.4f}" for r, _ in chosen))

    built = build_training_regions(args, hour_index, [r for r, _ in chosen],
                                   feature_names)
    total = 0
    for (r, st), (_, _, lb) in zip(chosen, built):
        n = st["n_major"] if st else "?"
        total += st["n_major"] if st else 0
        print(f"    {r.center[0]:>7.3f},{r.center[1]:<8.3f} {n:>5} events, "
              f"positive rate {lb.mean():.3f}")
    print(f"  [transfer] {len(built)} training regions pooled"
          + (f", {total} M>={args.threshold} events" if total else ""))
    return built


def run(args):
    """Builds the catalogue features and labels, then runs the fold sweep."""
    for name in ("train_stride_hours", "eval_stride_hours"):
        if getattr(args, name) < 1:
            sys.exit(f"[ERROR] --{name.replace('_', '-')} must be at least 1 "
                     f"(1 keeps every sample), got {getattr(args, name)}.")
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

    # Before anything is read: a region named by a flag that cannot be resolved
    # should say so about the flag, not fail later somewhere less informative.
    region = build_region(args)

    import pandas as pd
    print("Building the hourly index and catalogue features...")
    hour_index = pd.date_range(args.catalog_span[0], args.catalog_span[1], freq="h")
    print(f"  [catalog-span] {len(hour_index)} hourly rows, "
          f"{hour_index[0]} -> {hour_index[-1]}")
    print(f"  [region] {region.describe()}")

    major_times = load_aegean_events(args.catalog_path, args.threshold, region=region)
    bg_times, bg_mags, bg_lats, bg_lons = load_aegean_events_with_location(
        args.catalog_path, args.bg_min_mag, region=region)
    # The NND precompute inside build_catalog_features is O(n_bg * NND_LOOKBACK)
    # haversine in a Python loop and fires whenever coordinates are supplied. Skip
    # it when this run's feature subset has no location-derived column -- otherwise
    # a 4-feature run pays the full cost for columns it then discards.
    if args.keep_features is not None and not any(
            f in args.keep_features for f in ("nnd_log_eta_90d", "shannon_entropy_90d")):
        bg_lats = bg_lons = None
    print(f"  {len(major_times)} M>={args.threshold} events in the region, "
          f"{len(bg_times)} M>={args.bg_min_mag} background events")
    if len(major_times) < MIN_MAJOR_EVENTS:
        # Hourly rows are not the sample size here: thousands of them are driven
        # by a handful of earthquakes, and that is this project's recurring trap.
        # A narrowed region is the easiest way to walk into it, so it is said at
        # the point the narrowing happens rather than left to the fold summary.
        print(f"  [!] only {len(major_times)} M>={args.threshold} events in this "
              f"region -- every AUC below rests on that many earthquakes, whatever "
              f"the row counts say. Treat as indicative, not a result.")

    # `raw` is a length-1 dummy channel: nothing here reads a waveform, but
    # truncate_to_reliable_catalog_end trims an array alongside the index.
    dummy = np.zeros((len(hour_index), 1), dtype=np.float32)
    hour_index, _ = truncate_to_reliable_catalog_end(hour_index, dummy, major_times,
                                                     buffer_days=args.horizon_days)

    dsp = days_since_prev_major(hour_index, major_times)
    cat_features = build_catalog_features(hour_index, major_times, dsp, bg_times,
                                          bg_mags, args.bg_min_mag, bg_lats, bg_lons,
                                          grid_bbox=region.bbox)

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
    # The columns the model actually reads, in order. `--keep-features` makes
    # that order part of what a checkpoint means, so it travels with the file
    # rather than being reconstructed from the flags at load time.
    feature_names = active_names if args.keep_features is None else args.keep_features

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
            args, hour_index, n, cat_features.shape[1], region)

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

    # Thinning happens AFTER the boundaries are cut, never before. The embargo is
    # applied in hour-index units, and the single-split path above expresses it as
    # a POSITION offset into valid_end_indices -- the two agree only while every
    # hour is a sample. Cutting densely and thinning each block afterwards leaves
    # that arithmetic alone, keeps --region-split's `cut` aligned with where the
    # test split starts, and can only widen a realized gap, never narrow one.
    if args.train_stride_hours > 1 or args.eval_stride_hours > 1:
        folds = [(tr[::args.train_stride_hours],
                  va[::args.eval_stride_hours],
                  te[::args.eval_stride_hours]) for tr, va, te in folds]
        print(f"  [stride] train every {args.train_stride_hours}h, val/test every "
              f"{args.eval_stride_hours}h -- samples, not rows: "
              + ", ".join(f"{lbl} {len(tr)}/{len(va)}/{len(te)}"
                          for lbl, (tr, va, te) in zip(fold_labels, folds)))

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

    pool = build_pool(args, hour_index, region, feature_names, folds)

    results, saved = [], []
    for k, (fold_label, (train_idx, val_idx, test_idx)) in enumerate(
            zip(fold_labels, folds), 1):
        if k in skip:
            continue
        r = run_fold(fold_label, args, cat_features, labels, dsp, hour_index,
                     train_idx, val_idx, test_idx, seeds, device,
                     rate_trailing=rate_trailing, feature_names=feature_names,
                     saved=saved, region=region, major_times=major_times,
                     pool=pool)
        if r is not None:
            results.append(r)

    if args.cv_folds > 1:
        summarise(results, args.cv_folds)

    if saved:
        # The manifest carries each fold's floor beside its AUC, because a
        # directory of weights is not a result on this data any more than a
        # bare AUC is.
        manifest = write_manifest(args.out_dir, args, feature_names, seeds,
                                  results, saved, region=region)
        print(f"\nSaved {len(saved)} checkpoint(s) to {args.out_dir}, "
              f"listed in {manifest}")
    return 0


def main():
    p = argparse.ArgumentParser(
        prog="forecast train", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
