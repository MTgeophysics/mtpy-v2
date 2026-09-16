"""Tests for public MTData station replacement and update methods."""

import mtpy_data
import numpy as np

from mtpy.core import MTData
from mtpy.core.mt import MT


def _loaded_mt() -> MT:
    """Load one real MT response for station update tests."""
    mt_obj = MT(sorted(mtpy_data.PROFILE_LIST)[0])
    mt_obj.read()
    return mt_obj


def test_set_station_replaces_data_and_preserves_metadata():
    """Set a replacement MT object without changing station metadata."""
    tree = MTData()
    station_path = tree.add_station(_loaded_mt())
    original_attrs = dict(tree.get_station(station_path).attrs)
    replacement = tree.get_station(station_path, as_mt=True)
    replacement._transfer_function["transfer_function"].values *= 2

    tree.set_station(station_path, replacement)

    updated = tree.get_station(station_path, as_mt=True)
    assert np.allclose(
        updated._transfer_function["transfer_function"].values,
        replacement._transfer_function["transfer_function"].values,
    )
    assert dict(tree.get_station(station_path).attrs) == original_attrs


def test_set_station_preserves_indexed_station_without_periods():
    """Retain an indexed station record when its replacement has no periods."""
    tree = MTData(use_index=True)
    station_path = tree.add_station(_loaded_mt())
    replacement = MT()

    tree.set_station(station_path, replacement)

    assert tree.n_stations == 1
    assert tree.query_station_paths() == [station_path]


def test_update_station_saves_inplace_transform():
    """Persist a transform that modifies its station object in place."""
    tree = MTData()
    station_path = tree.add_station(_loaded_mt())
    original_tf = (
        tree.get_station(station_path, as_mt=True)
        ._transfer_function["transfer_function"]
        .values.copy()
    )

    def double_impedance(station: MT) -> None:
        station._transfer_function["transfer_function"].values *= 2

    tree.update_station(station_path, double_impedance)

    updated = tree.get_station(station_path, as_mt=True)
    assert np.allclose(
        updated._transfer_function["transfer_function"].values,
        original_tf * 2,
    )


def test_update_station_saves_replacement_transform():
    """Persist a transform that returns a replacement station object."""
    tree = MTData()
    station_path = tree.add_station(_loaded_mt())
    original_tf = (
        tree.get_station(station_path, as_mt=True)
        ._transfer_function["transfer_function"]
        .values.copy()
    )

    def replacement_with_scaled_impedance(station: MT) -> MT:
        station._transfer_function["transfer_function"].values *= 3
        return station

    tree.update_station(station_path, replacement_with_scaled_impedance)

    updated = tree.get_station(station_path, as_mt=True)
    assert np.allclose(
        updated._transfer_function["transfer_function"].values,
        original_tf * 3,
    )
