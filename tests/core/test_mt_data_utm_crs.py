"""Tests for UTM CRS propagation in MTData and MTStations."""

import mtpy_data
import numpy as np
import pandas as pd
from pyproj import CRS

from mtpy.core import MTData
from mtpy.core.mt import MT
from mtpy.core.mt_stations import MTStations


def _loaded_mts(n: int = 2) -> list[MT]:
    mts = []
    for fn in sorted(mtpy_data.PROFILE_LIST)[:n]:
        mt_obj = MT(fn)
        mt_obj.read()
        mts.append(mt_obj)
    return mts


def _en(attrs):
    return float(attrs["easting"]), float(attrs["northing"])


def test_add_station_adopts_root_utm_crs():
    mts = _loaded_mts(1)
    mts[0].utm_crs = 32611
    data = MTData()
    data.utm_crs = 32613
    path = data.add_station(mts[0])

    attrs = data.get_station(path).attrs
    assert CRS.from_user_input(attrs["utm_crs"]).to_epsg() == 32613

    ref = MT()
    ref.latitude = attrs["latitude"]
    ref.longitude = attrs["longitude"]
    ref.utm_crs = 32613
    assert np.allclose(_en(attrs), (ref.east, ref.north))


def test_add_stations_bulk_adopts_root_utm_crs():
    data = MTData()
    data.utm_crs = 32613
    paths = data.add_stations(_loaded_mts(2))
    for path in paths:
        attrs = data.get_station(path).attrs
        assert CRS.from_user_input(attrs["utm_crs"]).to_epsg() == 32613


def test_changing_utm_crs_updates_easting_northing():
    data = MTData()
    data.utm_crs = 32611
    path = data.add_station(_loaded_mts(1)[0])
    before = _en(data.get_station(path).attrs)

    data.utm_crs = 32612
    attrs = data.get_station(path).attrs
    after = _en(attrs)

    assert CRS.from_user_input(attrs["utm_crs"]).to_epsg() == 32612
    assert not np.allclose(before, after)


def test_same_utm_crs_does_not_change_coordinates():
    data = MTData()
    data.utm_crs = 32611
    path = data.add_station(_loaded_mts(1)[0])
    data.get_station(path).attrs["easting"] = 123.0

    data.utm_crs = 32611
    assert data.get_station(path).attrs["easting"] == 123.0


def test_utm_change_updates_index():
    data = MTData(use_index=True)
    data.utm_crs = 32611
    path = data.add_station(_loaded_mts(1)[0])
    data.utm_crs = 32612

    attrs = data.get_station(path).attrs
    row = data._index.station_record(path)
    assert str(row.utm_epsg) == "32612"
    assert np.isclose(row.east, float(attrs["easting"]))


def test_reprojection_failure_keeps_station_consistent():
    data = MTData()
    path = data.add_station(_loaded_mts(1)[0])
    attrs = data.get_station(path).attrs
    attrs["latitude"] = "not-a-number"
    old_crs, old_east = attrs.get("utm_crs"), attrs.get("easting")

    data.utm_crs = 32613
    assert attrs.get("utm_crs") == old_crs
    assert attrs.get("easting") == old_east


def _station_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "survey": ["s", "s"],
            "station": ["a", "b"],
            "latitude": [40.0, 40.5],
            "longitude": [-117.0, -116.5],
            "elevation": [1000.0, 1100.0],
            "datum_epsg": ["4326", "4326"],
            "east": [0.0, 0.0],
            "north": [0.0, 0.0],
            "utm_epsg": ["32611", "32611"],
            "model_east": [0.0, 0.0],
            "model_north": [0.0, 0.0],
            "model_elevation": [0.0, 0.0],
            "profile_offset": [0.0, 0.0],
        }
    )


def test_mt_stations_utm_crs_change_reprojects():
    stations = MTStations(32611, station_locations=_station_frame())
    before = stations.station_locations[["east", "north"]].to_numpy().copy()

    stations.utm_crs = 32612
    df = stations.station_locations
    assert (df["utm_epsg"] == "32612").all()
    assert not np.allclose(before, df[["east", "north"]].to_numpy())

    from pyproj import Transformer

    tf = Transformer.from_crs(4326, 32612, always_xy=True)
    e, n = tf.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    assert np.allclose(df["east"], e)
    assert np.allclose(df["north"], n)


def test_mt_stations_sync_reprojects_mismatched_station():
    data = MTData()
    data.utm_crs = 32611
    path = data.add_station(_loaded_mts(1)[0])
    stations = data.to_mt_stations()
    stations.utm_crs = 32612
    data._sync_station_locations_from_mt_stations(stations)

    attrs = data.get_station(path).attrs
    assert CRS.from_user_input(attrs["utm_crs"]).to_epsg() == 32612
    row = stations.station_locations.iloc[0]
    assert np.isclose(float(attrs["easting"]), row["east"])
