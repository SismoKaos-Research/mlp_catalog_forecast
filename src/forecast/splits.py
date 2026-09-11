"""Walk-forward chronological cross-validation and its diagnostics.

Ported unchanged from `cnn_earthquake/src/sismokaos/splits.py`.

Hourly rows are not independent observations: a handful of distinct
earthquakes drive every label in a fold, so the diagnostics print distinct
event counts beside each split rather than row counts alone."""

import numpy as np
import pandas as pd


def print_split_diagnostics(hourly_index: pd.DatetimeIndex, labels: np.ndarray,
                            train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                            n_blocks: int = 10, skew_ratio: float = 1.5,
                            major_times: np.ndarray = None) -> None:
    """Positive rate over time in `n_blocks` equal-width windows.

    Args:
        hourly_index: DatetimeIndex of hour starts.
        labels: Per-hour binary labels.
        train_idx: Window end-indices of the train split.
        val_idx: Window end-indices of the val split.
        test_idx: Window end-indices of the test split.
        n_blocks: Equal-width windows to report the positive rate over.
        skew_ratio: Test/train positive-rate ratio that triggers the warning.
        major_times: Sorted qualifying event times. When given, each split's
            distinct event count is printed beside its sample count -- that is
            the split's effective sample size, and it is the number a result
            actually rests on. At a 14-day horizon one event turns 336
            consecutive hours positive, so a split of 32448 rows can be ten
            earthquakes.
    """
    n = len(hourly_index)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    split_of = np.full(n, "", dtype=object)
    split_of[train_idx] = "train"
    split_of[val_idx] = "val"
    split_of[test_idx] = "test"

    print("\n  positive rate over time (equal-width blocks):")
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        block_splits = split_of[lo:hi]
        present = block_splits[block_splits != ""]
        dominant = pd.Series(present).mode().iloc[0] if len(present) else "-"
        print(f"    {hourly_index[lo].date()} .. {hourly_index[hi - 1].date()}  "
               f"pos rate {labels[lo:hi].mean():.3f}  n={hi - lo:4d}  split~{dominant}")

    if major_times is not None and len(major_times):
        print("\n  effective sample size (distinct events, not rows):")
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            if not len(idx):
                continue
            lo = hourly_index[idx[0]].to_datetime64()
            hi = hourly_index[idx[-1]].to_datetime64()
            k = int(np.sum((major_times >= lo) & (major_times <= hi)))
            print(f"    {name:5s}: {len(idx):>6} samples, {k:>4} distinct events")
            if name == "test" and k < 30:
                # Hanley-McNeil is not worth computing here, but the order of
                # magnitude is: at k positives the AUC's standard error is around
                # 0.5/sqrt(k), which at k=10 is ~0.16 and swamps every gap this
                # project reports between a model and its floor.
                if k == 0:
                    print("    [!] no qualifying events in the test block at all "
                          "-- its AUC is undefined, not low.")
                else:
                    print(f"    [!] a test AUC over {k} events carries roughly "
                          f"+/-{0.5 / np.sqrt(k):.2f}. Read the fold-versus-floor "
                          f"verdict, not the third decimal.")

    rates = {name: labels[idx].mean() for name, idx in
             (("train", train_idx), ("val", val_idx), ("test", test_idx)) if len(idx)}
    if "train" in rates and "test" in rates and rates["train"] > 0:
        ratio = rates["test"] / rates["train"]
        if ratio > skew_ratio or ratio < 1 / skew_ratio:
            print(f"\n  [!] test positive rate ({rates['test']:.3f}) is {ratio:.2f}x train's "
                   f"({rates['train']:.3f}) -- likely a swarm or quiet period concentrated in "
                   "one split rather than the model generalizing. Compare AUC against the "
                   "base-rate/persistence floors below (same skew), not against 0.5.")


def walk_forward_splits(valid_end_indices: np.ndarray, n_folds: int, labels: np.ndarray = None,
                        embargo: int = 0):
    """Expanding-window walk-forward splits."""
    n_blocks = n_folds + 2
    if labels is None:
        edges = np.linspace(0, len(valid_end_indices), n_blocks + 1).astype(int)
    else:
        cum = np.concatenate([[0], np.cumsum(labels)])
        total = cum[-1]
        if total == 0:
            edges = np.linspace(0, len(valid_end_indices), n_blocks + 1).astype(int)
        else:
            targets = np.linspace(0, total, n_blocks + 1)
            edges = np.searchsorted(cum, targets)
            edges[0], edges[-1] = 0, len(valid_end_indices)
            edges = np.maximum.accumulate(edges)
    blocks = [valid_end_indices[edges[i]:edges[i + 1]] for i in range(n_blocks)]
    if embargo > 0:
        for i in range(1, n_blocks):
            if len(blocks[i - 1]) == 0:
                continue
            cutoff = blocks[i - 1][-1] + embargo
            blocks[i] = blocks[i][blocks[i] > cutoff]
    return [(np.concatenate(blocks[:k + 1]), blocks[k + 1], blocks[k + 2])
            for k in range(n_folds)]
