"""Windows of catalogue features, normalized against the training split.

Not a runnable script -- imported only.

One sample per window: `seq_hours` consecutive hours of the feature array, and
the label at the window's last hour. The model reads only that last hour (see
`model.py`), but the window is kept whole because it is what defines a sample's
boundary, and the purge in `train.py` is expressed in those terms.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class CatalogSeqDataset(Dataset):
    """Catalog-feature windows, per-feature normalized with the train split's stats."""

    def __init__(self, cat_features, labels, seq_hours, indices, stats=None):
        """Builds the dataset.

        Args:
            cat_features: Per-hour catalog feature array, shape
                (n_hours, n_features).
            labels: Per-hour binary labels, shape (n_hours,).
            seq_hours: Number of consecutive hours per window.
            indices: Window end-indices.
            stats: Optional (cat_mu, cat_sd) tuple. Val and test MUST be given
                the training split's stats; computing their own would let the
                test split's distribution into the model.
        """
        self.cat_features = cat_features
        self.labels = labels
        self.seq_hours = seq_hours
        self.indices = indices
        if stats is None:
            # Sample windows spread across the WHOLE training split, not the first 50.
            # The archive's opening hours are unrepresentative for any trailing-window
            # feature, whose lookback is still filling up there: measured on the rate
            # features, sd over the first 50 windows understated the true training sd by
            # up to 52x (rate_log1p_count_90d: 0.0157 vs 0.827), so dividing by it
            # produced z-scores up to 156 and saturated the GELU MLP -- val AUC fell from
            # epoch 1 and the "best" checkpoint was the untrained one.
            stat_idx = indices[np.linspace(0, len(indices) - 1,
                                           min(500, len(indices))).astype(int)]
            csub = np.concatenate([cat_features[max(0, i - seq_hours + 1):i + 1]
                                   for i in stat_idx], axis=0)
            stats = (csub.mean(axis=0, keepdims=True),
                     csub.std(axis=0, keepdims=True) + 1e-6)
        self.stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Returns (cat_seq, label) for one window.

        Args:
            idx: Index into `self.indices`.

        Returns:
            Tuple of (float32 tensor shape (seq_hours, n_features), float32
            scalar tensor label).
        """
        end = self.indices[idx]
        start = end - self.seq_hours + 1
        cat_mu, cat_sd = self.stats
        cat_seq = (self.cat_features[start:end + 1] - cat_mu) / cat_sd
        return (torch.from_numpy(cat_seq).float(),
                torch.tensor(self.labels[end], dtype=torch.float32))
