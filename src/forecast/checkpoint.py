"""Writing a trained model to disk, and reading one back.

Not a runnable script -- imported only, by `train.py`.

A checkpoint here is one (fold, seed) model, and it carries three things that
are useless apart: the weights, the **training split's** normalization stats,
and the **feature list those stats were computed over**.

The stats are in the file because they are not recoverable at inference time.
`data.py` computes them from the train split alone and hands the same pair to
val and test, precisely so the later splits' distribution never reaches the
model; new hours arriving after the run have no training split to recompute
from, so a checkpoint without its stats can only be fed z-scores it was never
trained on.

The feature list is in the file because `--keep-features` changes the input
width AND the column order, and a mismatch there is the quiet kind. A wrong
width raises on `load_state_dict`; a right width in the wrong order loads
cleanly and forecasts from `mean_mag_30d` as though it were `log1p_dsp`. So
`load_checkpoint` returns the names and `predict`-side code is expected to
check them against the array it built.

One file per seed, not one per ensemble. The published configuration averages
five seeds because per-seed AUC spread reaches 0.17 here, and an average of
weights is not the model that produced the reported number -- the average of
their *scores* is. Keeping the seeds separate is what lets that average be
reproduced rather than approximated.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from forecast.catalog import AEGEAN_BBOX
from forecast.model import CatalogMLPNet

# Bumped when a field's meaning changes, so an old file fails loudly rather
# than loading into a model that reads it differently.
CHECKPOINT_FORMAT = 1


def fold_slug(fold_label: str) -> str:
    """Turns a fold's printed label into a filename component.

    Args:
        fold_label: The label as printed in the run, e.g. `"fold 1/2"` or
            `"single split"`.

    Returns:
        The same label as a lowercase, filesystem-safe slug, e.g.
        `"fold_1_of_2"`, `"single_split"`.
    """
    spelled = fold_label.replace("/", " of ").lower()
    return re.sub(r"[^a-z0-9]+", "_", spelled).strip("_")


def save_checkpoint(out_dir, fold_label, seed, model, stats, feature_names, args,
                    val_metric, region=None, pool=None) -> Path:
    """Writes one seed's best-epoch model, with everything needed to reuse it.

    Args:
        out_dir: Directory to write into; created if absent.
        fold_label: The fold's printed label, used for the filename.
        seed: The seed this model was trained under.
        model: A `CatalogMLPNet` already holding its best-epoch weights.
        stats: The train split's `(cat_mu, cat_sd)`, each shape (1, n_features).
        feature_names: The feature columns, in the order the model reads them.
        args: The parsed arguments of the run.
        val_metric: The best value of `--checkpoint-metric` reached on val.
        region: The study region the model was fit to and forecasts for.
        pool: The extra training regions pooled into this fit, if any.

    Returns:
        Path of the file written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{fold_slug(fold_label)}_seed_{seed}.pt"
    cat_mu, cat_sd = stats
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "state_dict": model.state_dict(),
        # Tensors rather than numpy arrays, so the file loads under
        # `weights_only=True` -- a checkpoint should not need to be trusted
        # code to be read.
        "cat_mu": torch.as_tensor(np.asarray(cat_mu, dtype=np.float32)),
        "cat_sd": torch.as_tensor(np.asarray(cat_sd, dtype=np.float32)),
        "feature_names": list(feature_names),
        "arch": {"catalog_dim": len(feature_names),
                 "cat_hidden": args.cat_hidden,
                 "fusion_hidden": args.fusion_hidden,
                 "dropout": args.dropout,
                 "seq_hours": args.seq_hours},
        # What the model was asked to forecast. A checkpoint scored against a
        # different horizon or threshold is a different question answered by
        # the same weights, and nothing in the weights records which.
        "target": {"label_mode": args.label_mode,
                   "threshold": args.threshold,
                   "horizon_days": args.horizon_days,
                   "bg_min_mag": args.bg_min_mag,
                   "rate_min_mag": args.rate_min_mag,
                   # Where, as well as what. A model fit inside a 100km disc
                   # answers a different question than one fit across the
                   # Aegean, and the weights do not say which.
                   "region": _region_record(region)},
        # How the windows were sampled. A model fit on weekly, non-overlapping
        # samples is a different fit than one fit on every hour, and the weights
        # do not record which. Readers use .get(): files written before this
        # field existed are still valid at this format version.
        "sampling": {"train_stride_hours": args.train_stride_hours,
                     "eval_stride_hours": args.eval_stride_hours},
        # Which regions this model was fit on. It forecasts for the study region
        # either way, but one pretrained on twelve others is not the model the
        # same command produced without them.
        "train_regions": [] if not pool else
                         [[float(r.center[0]), float(r.center[1])] for r, _, _ in pool],
        "fold": fold_label,
        "seed": seed,
        "checkpoint_metric": args.checkpoint_metric,
        "val_metric": float(val_metric),
    }, path)
    return path


def _region_record(region):
    """The study region as plain values, so the file loads `weights_only`."""
    if region is None:
        return {"bbox": list(AEGEAN_BBOX), "center": None, "radius_km": None,
                "station": None,
                "described": "lat 36..40, lon 25..30 (Aegean default)"}
    return {"bbox": [float(v) for v in region.bbox],
            "center": None if region.center is None else [float(v) for v in region.center],
            "radius_km": None if region.radius_km is None else float(region.radius_km),
            "station": region.station,
            "described": region.describe()}


def load_checkpoint(path, map_location="cpu"):
    """Rebuilds a saved model, its normalization stats and its metadata.

    Args:
        path: A file written by `save_checkpoint`.
        map_location: Passed to `torch.load`.

    Returns:
        Tuple of (model in eval mode, `(cat_mu, cat_sd)` as float32 arrays of
        shape (1, n_features), metadata dict carrying at least
        `feature_names`, `arch`, `target`, `fold` and `seed`).

    Raises:
        ValueError: If the file was written by a different checkpoint format.
    """
    blob = torch.load(path, map_location=map_location, weights_only=True)
    if blob.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: checkpoint format {blob.get('format')!r}, "
                         f"this build reads {CHECKPOINT_FORMAT}")
    arch = blob["arch"]
    model = CatalogMLPNet(catalog_dim=arch["catalog_dim"],
                          cat_hidden=arch["cat_hidden"],
                          fusion_hidden=arch["fusion_hidden"],
                          dropout=arch["dropout"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    stats = (blob["cat_mu"].numpy(), blob["cat_sd"].numpy())
    meta = {k: v for k, v in blob.items()
            if k not in ("state_dict", "cat_mu", "cat_sd")}
    return model, stats, meta


def write_manifest(out_dir, args, feature_names, seeds, results, checkpoints,
                   region=None) -> Path:
    """Records what the saved checkpoints are, beside the floors they cleared.

    A directory of `.pt` files is not a result. This project's rule is that an
    AUC means nothing without the floor it had to clear on its own fold, so the
    manifest carries each fold's floor and per-seed spread alongside the files,
    and the command that produced them.

    Args:
        out_dir: Directory the checkpoints were written to.
        args: The parsed arguments of the run.
        region: The study region the models were fit to.
        feature_names: The feature columns the models read, in order.
        seeds: The seeds trained.
        results: The `FoldResult`s produced, in fold order.
        checkpoints: Paths written, in the order they were written.

    Returns:
        Path of the manifest written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps({
        "format": CHECKPOINT_FORMAT,
        "command": " ".join(sys.argv),
        "args": {k: v for k, v in sorted(vars(args).items())},
        "feature_names": list(feature_names),
        "region": _region_record(region),
        "sampling": {"train_stride_hours": args.train_stride_hours,
                     "eval_stride_hours": args.eval_stride_hours},
        "seeds": list(seeds),
        "checkpoints": [p.name for p in checkpoints],
        "folds": [{"fold": r.label,
                   "auc": r.auc,
                   "floor": r.floor,
                   "beats_floor": r.beats_floor,
                   "per_seed_aucs": list(r.per_seed_aucs),
                   "seed_spread": r.seed_spread,
                   "n": r.n} for r in results],
    }, indent=2, default=float) + "\n")
    return path
