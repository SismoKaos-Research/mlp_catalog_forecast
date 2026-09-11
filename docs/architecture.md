# Architecture

`src/forecast/model.py`. Two classes, four linear layers, under 1200 parameters.

## The network

`CatalogMLPNet` is a branch plus a head.

```
CatalogMLPBranch(catalog_dim, hidden=cat_hidden, dropout)
    Linear(catalog_dim, cat_hidden)  GELU  Dropout
    Linear(cat_hidden,  cat_hidden)  GELU  Dropout

head
    LayerNorm(cat_hidden)
    Dropout
    Linear(cat_hidden, fusion_hidden)  GELU  Dropout
    Linear(fusion_hidden, 1)
```

The forward pass takes a batch of shape `(batch, seq_hours, catalog_dim)` and
returns raw logits of shape `(batch,)`. Loss is `BCEWithLogitsLoss` with a
`pos_weight` of `(1 - p) / p`, where `p` is the positive rate of the training
split.

Parameter counts at the default widths, `--cat-hidden 16 --fusion-hidden 32`:

| input features | parameters |
|---|---|
| 4 (the published subset) | 961 |
| 13 (all catalogue features) | 1105 |
| 21 (with `--rate-features`) | 1233 |

## The branch reads one hour, not the window

```python
def forward(self, cat_seq):
    return self.net(cat_seq[:, -1, :])
```

Everything but the window's last hour is discarded. This is deliberate and it
is one of the project's findings rather than a simplification.

The original architecture put an LSTM over the 24-hour window, and it sat at
chance for the entire catalogue-only ablation. The reason was measured, not
inferred. Catalogue features barely move inside a 24-hour window:

| feature | within-window sd / overall sd |
|---|---|
| `log1p_dsp` | 0.055 |
| `count_7d` | 0.056 |
| `count_30d` | 0.025 |
| `count_90d` | 0.011 |

A recurrent layer exists to track change across a sequence. Fed one nearly
constant reading repeated twenty-four times, it has nothing to track. What looks
like a time series is a single point-in-time tabular reading, and a direct MLP
is the better-matched tool.

The window is still carried whole through `CatalogSeqDataset`, because it is
what defines a sample's boundary and the purge in `train.py` is expressed in
those terms. See [evaluation.md](evaluation.md#the-embargo).

## What is deliberately absent

The parent module was `cnn_lstm_catalog_waveform_fusion.py`, and its waveform
arm is the negative half of the result. On the corrected catalogue, neither the
raw-waveform CNN, nor hand-crafted continuous waveform features, nor chaotic
features beat the persistence floor, across every architecture tried. The
conclusion recorded there was that the forecasting signal comes from the
earthquake catalogue and not from the seismogram.

So this repo has no `--channels`, no `--data-root`, and none of the
transfer-learning flags that existed to seed the fusion model's catalogue trunk.
The head still has the shape it had when it fused two branch embeddings; it now
receives one, so `fused_dim` is just the branch width.

That shape is the honest place to cut if capacity ever needs reducing, because
it is carrying structure designed for an input it no longer receives.

## Capacity

Width is a flag. Depth is not, and reducing it means editing this module.

| `--cat-hidden` | `--fusion-hidden` | parameters |
|---|---|---|
| 16 | 32 | 961 |
| 8 | 16 | 289 |
| 4 | 8 | 97 |
| 4 | 4 | 73 |

Whether 961 is too many depends entirely on the training pool. Against a single
150 km region with 90 qualifying events it is about eleven parameters per
earthquake. Against a pooled run of twelve further regions and roughly 2000
events it is well under one. See
[study-regions.md](study-regions.md#training-on-more-regions-than-you-forecast).

`--dropout` defaults to 0.2. The parent project set this explicitly for the
catalogue-only arm: on an input this narrow, aggressive dropout knocks the model
off a good basin rather than regularising it.
