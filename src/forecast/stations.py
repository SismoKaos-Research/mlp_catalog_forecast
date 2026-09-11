"""Naming the study region by a station instead of by coordinates.

Not a runnable script -- imported only, by `train.py`.

**This is not the station filter that was removed, and the difference matters.**
That one masked the catalogue down to events near a seismometer because the
model read that seismometer's waveforms, and it survived into a project that
reads no waveforms at all -- so it narrowed the catalogue for a reason that no
longer existed. Nothing here reads a station's data. A station is used purely as
a convenient way to say *where*: `--station-radius TU.ABT 150` is
`--catalog-radius 40.6058 31.3208 150` with the coordinates looked up for you,
and the resulting region behaves exactly like any other.

The inventory format is the one station files come in:

    Network,Code,Longitude,Latitude,Height,Province,District
    "TU","ABT",31.3208,40.6058,1794,"Bolu","Mudurnu"

Note the column order: **longitude precedes latitude**, the reverse of how every
coordinate pair in this project is written. Reading them positionally is how a
station in Bolu ends up placed in the Indian Ocean, so they are read by name.
"""
import numpy as np
import pandas as pd

from forecast.catalog import Region

REQUIRED_COLUMNS = ("network", "code", "latitude", "longitude")

# Two rows for the same station further apart than this are two different
# places, not one place written twice. Loose enough for a re-survey: real
# inventories carry TU.ERZM at two points 320m apart, which is one station
# measured twice and is immaterial against a radius in the tens of km. Tight
# enough that a station actually moved to a new site is still refused.
SAME_SITE_DEG = 0.01


def load_station_inventory(path) -> pd.DataFrame:
    """Reads a station inventory into `station`, `lat`, `lon` columns.

    Args:
        path: CSV with Network, Code, Latitude and Longitude columns. Extra
            columns (Height, Province, District) are ignored.

    Returns:
        DataFrame with `station` ("NETWORK.CODE", upper case), `lat`, `lon`.

    Raises:
        SystemExit: If a required column is missing.
    """
    raw = pd.read_csv(path)
    by_lower = {c.strip().lower(): c for c in raw.columns}
    missing = [c for c in REQUIRED_COLUMNS if c not in by_lower]
    if missing:
        raise SystemExit(
            f"[ERROR] {path}: station inventory is missing "
            f"{', '.join(m.capitalize() for m in missing)}. It reads "
            f"Network,Code,Longitude,Latitude (Height/Province/District optional), "
            f"and found {list(raw.columns)}.")

    net = raw[by_lower["network"]].astype(str).str.strip().str.strip('"').str.upper()
    code = raw[by_lower["code"]].astype(str).str.strip().str.strip('"').str.upper()
    return pd.DataFrame({
        "station": net + "." + code,
        "code": code,
        "lat": pd.to_numeric(raw[by_lower["latitude"]], errors="coerce"),
        "lon": pd.to_numeric(raw[by_lower["longitude"]], errors="coerce"),
    })


def station_region(inventory_path, station_id, radius_km) -> Region:
    """Builds the disc of `radius_km` around a named station.

    Args:
        inventory_path: The station inventory CSV.
        station_id: "NETWORK.CODE", e.g. "TU.ABT". A bare code is accepted
            when it is unambiguous in the inventory.
        radius_km: Disc radius in km.

    Returns:
        A `Region` disc centred on that station, carrying its name.

    Raises:
        SystemExit: If the station is not in the inventory, if a bare code
            matches several networks, or if the inventory places one station
            in two different spots.
    """
    inv = load_station_inventory(inventory_path)
    wanted = str(station_id).strip().strip('"').upper()
    # A bare code is matched on its own, so "ABT" works when only one network
    # carries it -- and says which networks do when several carry it.
    hits = inv[inv.station == wanted] if "." in wanted else inv[inv.code == wanted]

    if not len(hits):
        raise SystemExit(
            f"[ERROR] --station-radius: {station_id!r} is not in {inventory_path} "
            f"({len(inv)} stations). They are named NETWORK.CODE, e.g. "
            f"{inv.station.iloc[0] if len(inv) else 'TU.ABT'}.")
    sites = hits[["lat", "lon"]].dropna().drop_duplicates()
    if not len(sites):
        raise SystemExit(f"[ERROR] --station-radius: {station_id!r} has no usable "
                         f"coordinates in {inventory_path}.")
    if "." not in wanted and hits.station.nunique() > 1:
        raise SystemExit(f"[ERROR] --station-radius: the code {station_id!r} is in "
                         f"{hits.station.nunique()} networks "
                         f"({', '.join(sorted(hits.station.unique()))}). Name one.")
    if len(sites) > 1 and (np.ptp(sites.lat.to_numpy()) > SAME_SITE_DEG
                           or np.ptp(sites.lon.to_numpy()) > SAME_SITE_DEG):
        raise SystemExit(
            f"[ERROR] --station-radius: {station_id!r} is at "
            f"{len(sites)} different places in {inventory_path} "
            f"{[(round(float(a), 4), round(float(b), 4)) for a, b in sites.to_numpy()]}. "
            f"Pass --catalog-radius with the one you mean.")

    lat, lon = float(sites.lat.iloc[0]), float(sites.lon.iloc[0])
    region = Region.from_flags(radius=(lat, lon, radius_km))
    return Region(bbox=region.bbox, center=region.center,
                  radius_km=region.radius_km,
                  station=hits.station.iloc[0])
