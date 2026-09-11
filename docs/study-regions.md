# Study regions

`src/forecast/catalog.py` (`Region`), `stations.py`, `regions.py`.

## The study region

Every run is about one patch of crust. It defaults to the Aegean box,
`lat 36-40, lon 25-30`, which produced every published number. Three mutually
exclusive flags override it.

```bash
--catalog-bbox 37.0 38.5 26.0 28.0        # a box
--catalog-radius 37.5 27.5 120            # a disc, LAT LON KM
--station-radius KO.BODT 120              # the same disc, centre looked up
```

`Region` is a frozen dataclass carrying `bbox`, and for a disc also `center`,
`radius_km` and an optional `station` name. The bbox is present even for a disc,
as the smallest box containing it, because the spatial entropy feature needs a
grid extent. See [features.md](features.md#the-thirteen).

### The radius is a true disc

`Region.mask` uses a great-circle distance, not the bounding box. At Aegean
latitudes the box corner sits about 40% further out than the radius, so reading
the flag as a box would hand a run a region roughly 27% larger than asked for,
and nothing downstream would look wrong.

### Narrowing changes the question

Features, labels, the persistence floor and the fold boundaries are all rebuilt
from the events inside the region, and the entropy grid is re-cast over it. A
run over a 120 km disc forecasts a different thing than a run over the Aegean,
not the same thing more precisely.

It is also not free. The trainer warns below 30 qualifying events, because that
is the number every AUC rests on however many hourly rows there are.

## Naming a region by station

```
Network,Code,Longitude,Latitude,Height,Province,District
"TU","ABT",31.3208,40.6058,1794,"Bolu","Mudurnu"
"KO","BODT",27.3103,37.0622,120,"Muğla","Bodrum"
```

**Longitude precedes latitude in this format**, the reverse of every other
coordinate pair in the project, so columns are read by name. Read positionally,
a station in Bolu lands in the Persian Gulf and the run continues happily with
an empty catalogue.

Nothing reads the station's data. A station is a lookup for the centre of a
disc, which is why this is not a return of the station-distance filter the port
removed. That filter cut the catalogue to events near a seismometer whose
waveforms the model read, and nothing here reads a waveform.

Behaviour:

- a bare code resolves when only one network carries it, and names the networks
  when several do
- matching is case-insensitive
- a station listed twice within about 1 km resolves to the first row, since real
  inventories re-survey stations and record them again
- a station listed at genuinely different sites is refused rather than resolved
  by row order

## The spatial holdout

`--region-split lat|lon` holds out half the study region instead of a stretch of
time: a patch of crust whose events the model has never seen. It forces
`--cv-folds 1`.

A per-station split would be meaningless here, and not only because no station
data reaches the project. Catalogue features and labels are region-wide, and the
two stations the parent project used shared 95.9% of their hours, so "train one,
test the other" would test on the very rows it trained on.

Space alone is not enough either. If training covered region A across the whole
timeline and test covered region B across the whole timeline, a regional swarm
at time t would raise `count_7d` in A and the label in B simultaneously. The
model never sees a timestamp but does not need one: it would learn that features
resembling a busy period imply positive, and busy periods are shared across the
box. So the split is space **and** time, with the usual embargo applied.

The boundary is the median of the events actually loaded, so it follows whatever
region is in force.

## Training on more regions than you forecast

A fold in a single 150 km region can fit a model on ten earthquakes. That, not
the architecture, is the binding constraint.

```bash
--train-regions 12
```

This pools twelve further regions of the same size and shape into **training
only**. Validation, test, the persistence floor and the reported `n` still come
from the study region alone, which is what keeps a pooled run comparable to one
without it.

### Why the regions are chosen statistically

Two geographic approaches were tried first and neither survives checking.

**One large training region is degenerate.** Measured at M4 with a 14-day
horizon on the Turkish catalogue:

| training region | events | positive rate |
|---|---|---|
| a 150 km disc | 90 | 0.122 |
| a North Anatolian corridor | 413 | 0.377 |
| Turkey-wide box | 4039 | **0.971** |

A national label is positive 97% of the time, so training on it teaches "always
yes" to a target that is positive 12% of the time.

**A hand-drawn fault corridor is the wrong shape.** Taking 15571 events above
M3.5 and asking where seismicity actually sits per longitude bin, a
`lat 39.5-41.5` corridor captures between 4% and 48% of events, under 20% in
most bins. Turkish seismicity is not one latitude band.

So selection is by measured similarity in positive rate and Aki b-value:

```
score = abs(log(pos_rate / target_pos_rate)) + abs(b - target_b)
```

Every number in it is recomputable from the catalogue, which a drawn boundary is
not. On the Turkish catalogue the top twelve carry about 1470 qualifying events
against the target's 90, with positive rates from 0.07 to 0.21 around a target
of 0.11.

Worth knowing, because it contradicts the intuition: the closest match to a
Marmara target sits in southwest Anatolia, on a different fault system.
Behavioural similarity does not track tectonic proximity here.

### Two guards

**Candidates within two radii of the target are excluded**, so no earthquake can
be in both the training pool and the test block.

**Similarity is measured over the training window only.** Ranking over the whole
record would choose training regions partly for what happens in the block the
model is later scored on. `build_pool` takes the widest training block across
folds and passes only those hours.

The selection is printed in `--train-region-centers` form so a run replays
exactly, the same way `--random-seeds` prints its draw.

### Cost

Selecting twelve regions from a 576k-event catalogue takes about a minute.
Building their features is the slow part, roughly 17 seconds per region for
164k hours, because the feature builder loops over every hour of every region.
Restricting `--keep-features` to columns that exclude `nnd_log_eta_90d` and
`shannon_entropy_90d` skips the nearest-neighbour precompute entirely and is the
single largest saving.

### What it cannot fix

Pooling enlarges training. The test block still holds however many earthquakes
it held before, so per-fold uncertainty is unchanged. Judge a pooled run by
whether the fold-versus-floor verdict moves consistently with the per-seed
spread narrowing, not by whether the mean AUC rises.
