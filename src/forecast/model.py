"""The model: a small MLP on the catalogue feature vector, and its head.

Not a runnable script -- imported only.

**This is deliberately not a sequence model, and that is the finding.** The
original architecture put an LSTM over the 24-hour window and it "got stuck at
chance for the entire catalog-only ablation run." The reason was measured, not
guessed: catalogue features barely move within a window -- within-window std
over overall std came to 0.055 / 0.056 / 0.025 / 0.011 for
log1p(dsp) / count_7d / count_30d / count_90d. An LSTM's whole purpose is
tracking change across a sequence, and fed a near-constant input repeated
twenty-four times there is almost nothing to track.

So the branch reads the window's LAST hour and nothing else. What looks like a
time series is functionally a single point-in-time tabular reading, and a direct
MLP is the better-matched, simpler tool for it.

Ported from `CatalogMLPBranch` and the catalogue-only path of
`CatalogWaveformFusionNet` in `cnn_earthquake`'s
`forecasting/cnn_lstm_catalog_waveform_fusion.py`. The waveform branch is gone;
see the module docstring in `train.py` for why.
"""
import torch.nn as nn

from forecast.features import CATALOG_DIM


class CatalogMLPBranch(nn.Module):
    """Small MLP on the catalog feature vector at the window's LAST hour.

    Catalog features barely change within a window -- empirically ~1-6% of
    their across-dataset variation (measured directly: within-window std /
    overall std for log1p(dsp)/count_7d/count_30d/count_90d was
    0.055/0.056/0.025/0.011). An LSTM's whole point is tracking change
    across a sequence; fed a near-constant-repeated-24-times input, it has
    almost nothing to track and (empirically, in this project) got stuck at
    chance for the entire catalog-only ablation run. A direct MLP on the
    most recent reading is a better-matched, simpler tool for what is
    functionally a single point-in-time tabular reading, not a time series.
    """

    def __init__(self, catalog_dim, hidden=16, dropout=0.4):
        """Initializes the MLP.

        Args:
            catalog_dim: Width of the per-hour catalog feature vector.
            hidden: Hidden width (also the output embedding width).
            dropout: Dropout used between layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(catalog_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.out_dim = hidden

    def forward(self, cat_seq):
        """Embeds the last hour of one batch of catalog-feature sequences.

        Args:
            cat_seq: Shape (batch, seq_hours, catalog_dim).

        Returns:
            Tensor of shape (batch, hidden).
        """
        return self.net(cat_seq[:, -1, :])


class CatalogMLPNet(nn.Module):
    """`CatalogMLPBranch` plus the classification head.

    The head is `CatalogWaveformFusionNet`'s, unchanged -- it fused two branch
    embeddings there and receives one here, so `fused_dim` is just the catalog
    branch's width. Keeping the same layers means a checkpoint trained by the
    original at `--channels catalog` still loads.
    """

    def __init__(self, catalog_dim=CATALOG_DIM, cat_hidden=16, fusion_hidden=32,
                 dropout=0.4):
        """Initializes the branch and head.

        Args:
            catalog_dim: Width of the per-hour catalog feature vector.
            cat_hidden: Catalog MLP hidden size.
            fusion_hidden: Hidden width of the head.
            dropout: Dropout used throughout.
        """
        super().__init__()
        self.cat_branch = CatalogMLPBranch(catalog_dim, hidden=cat_hidden,
                                           dropout=dropout)
        fused_dim = self.cat_branch.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, cat_seq):
        """Forecasts from one batch of catalog-feature windows.

        Args:
            cat_seq: Shape (batch, seq_hours, catalog_dim).

        Returns:
            Tensor of shape (batch,), raw logits.
        """
        return self.head(self.cat_branch(cat_seq)).squeeze(-1)
