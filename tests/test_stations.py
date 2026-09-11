"""Naming the study region by station code.

A station here is a lookup for the centre of a disc, and nothing more: no
station data is read, and the region it produces is the same object
`--catalog-radius` produces. What these tests pin is the lookup, because the
inventory format invites two silent mistakes.

The first is the column order -- **longitude precedes latitude**, the reverse of
every other coordinate pair in this project -- so a positional read puts a
station in Bolu somewhere in the Indian Ocean and the run continues happily with
an empty catalogue. The second is a station that the inventory lists twice at
different coordinates, where picking the first row silently decides which of two
regions the whole run was about.
"""
import numpy as np
import pytest

from forecast.catalog import haversine_km
from forecast.stations import load_station_inventory, station_region

INVENTORY = """Network,Code,Longitude,Latitude,Height,Province,District
"TU","ABT",31.3208,40.6058,1794,"Bolu","Mudurnu"
"TU","AKAS",29.6052,36.2326,1165,"Antalya","Kaş"
"TU","AKKT",37.0135,40.7844,1575,"Ordu","Akkuş"
"TU","ATAB",38.2949,37.4696,551,"Şanlıurfa","Bozova"
"KO","BODT",27.3103,37.0622,120,"Muğla","Bodrum"
"""


@pytest.fixture
def inventory(tmp_path):
    path = tmp_path / "stations.csv"
    path.write_text(INVENTORY, encoding="utf-8")
    return path


def test_longitude_precedes_latitude_in_this_format(inventory):
    """ABT is in Bolu, at 40.6N 31.3E. Read positionally it lands at 31.3N
    40.6E, which is the Persian Gulf -- and an empty catalogue, not an error."""
    inv = load_station_inventory(inventory)
    abt = inv[inv.station == "TU.ABT"].iloc[0]
    assert abt.lat == pytest.approx(40.6058)
    assert abt.lon == pytest.approx(31.3208)


def test_the_station_becomes_a_disc_around_its_coordinates(inventory):
    """The region it produces is the one --catalog-radius would produce."""
    region = station_region(inventory, "KO.BODT", 150.0)
    assert region.center == pytest.approx((37.0622, 27.3103))
    assert region.radius_km == 150.0
    assert region.station == "KO.BODT"

    edge = haversine_km(37.0622, 27.3103,
                        np.array([37.0622]), np.array([27.3103 + 3.0]))
    assert edge[0] > 150.0, "this fixture point should sit outside the disc"
    assert not region.mask(np.array([37.0622]), np.array([27.3103 + 3.0]))[0]
    assert region.mask(np.array([37.0622]), np.array([27.3103]))[0]


def test_the_station_is_named_in_the_region_s_description(inventory):
    """It goes in the run log and in every checkpoint; a bare pair of
    coordinates there would not say which station was meant."""
    assert station_region(inventory, "KO.BODT", 150.0).describe() == \
        "150km around KO.BODT (37.0622, 27.3103)"


@pytest.mark.parametrize("given", ["KO.BODT", "ko.bodt", "BODT", "bodt"])
def test_the_code_is_matched_case_insensitively_and_bare(inventory, given):
    """Station codes are written upper case, and a bare code is unambiguous
    here, so both should resolve rather than making the user guess."""
    assert station_region(inventory, given, 100.0).station == "KO.BODT"


def test_a_bare_code_in_two_networks_is_refused(inventory, tmp_path):
    """Picking one silently would decide which patch of crust the whole run
    was about."""
    path = tmp_path / "dup.csv"
    path.write_text(INVENTORY + '"XX","BODT",30.0,38.0,10,"a","b"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="Name one"):
        station_region(path, "BODT", 100.0)


def test_a_re_surveyed_station_still_resolves(inventory, tmp_path):
    """Real inventories carry a station at slightly different coordinates when
    it has been measured twice. TU.ERZM appears 320m apart in the AFAD file,
    which is one station, not two regions."""
    path = tmp_path / "resurveyed.csv"
    path.write_text(INVENTORY + '"KO","BODT",27.3139,37.0630,120,"a","b"\n',
                    encoding="utf-8")
    assert station_region(path, "KO.BODT", 150.0).center[0] == pytest.approx(37.0622)


def test_one_station_at_two_places_is_refused(inventory, tmp_path):
    """Inventories carry a station's history. Two rows 100km apart are two
    regions, and the first row is not an answer to which one was meant."""
    path = tmp_path / "moved.csv"
    path.write_text(INVENTORY + '"KO","BODT",28.9,38.2,10,"a","b"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="different places"):
        station_region(path, "KO.BODT", 100.0)


def test_a_station_that_is_not_there_says_so(inventory):
    with pytest.raises(SystemExit, match="not in"):
        station_region(inventory, "KO.NOPE", 100.0)


def test_a_file_that_is_not_an_inventory_says_which_columns_it_wanted(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="missing Network, Code, Latitude, Longitude"):
        station_region(path, "KO.BODT", 100.0)


def test_the_radius_flag_needs_an_inventory():
    """Without one there is nothing to look the station up in, and the fix is
    a different flag rather than a different value."""
    import argparse

    from forecast.train import add_args, build_region
    p = argparse.ArgumentParser()
    add_args(p)
    args = p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                         "--station-radius", "KO.BODT", "150"])
    with pytest.raises(SystemExit, match="needs --station-catalog"):
        build_region(args)


def test_an_unusable_radius_is_caught_before_the_catalogue_is_read(inventory):
    import argparse

    from forecast.train import add_args, build_region
    p = argparse.ArgumentParser()
    add_args(p)
    for km, message in (("abc", "not a radius"), ("0", "must be positive")):
        args = p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                             "--station-catalog", str(inventory),
                             "--station-radius", "KO.BODT", km])
        with pytest.raises(SystemExit, match=message):
            build_region(args)


def test_no_region_flag_still_means_the_aegean(inventory):
    """An inventory on its own names no region, and must not quietly become one."""
    import argparse

    from forecast.catalog import AEGEAN_BBOX
    from forecast.train import add_args, build_region
    p = argparse.ArgumentParser()
    add_args(p)
    args = p.parse_args(["--catalog-path", "x", "--catalog-span", "a", "b",
                         "--station-catalog", str(inventory)])
    assert build_region(args).bbox == AEGEAN_BBOX
