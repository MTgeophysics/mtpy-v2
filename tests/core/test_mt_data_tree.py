# -*- coding: utf-8 -*-
"""Pytest suite for MTDataTree scaffold behavior."""


import mtpy_data
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mtpy.core import MTDataTree
from mtpy.core.mt import MT
from mtpy.core.mt_dataframe import MTDataFrame
from mtpy.core.mt_stations import MTStations


@pytest.fixture(scope="session")
def profile_edi_files():
    """Session-scoped subset of real profile EDI files from mtpy_data."""
    return sorted(mtpy_data.PROFILE_LIST)[:2]


@pytest.fixture(scope="session")
def profile_edi_file(profile_edi_files):
    """Single real profile EDI file path."""
    return profile_edi_files[0]


@pytest.fixture(scope="session")
def loaded_profile_mt_objects(profile_edi_files):
    """Session-scoped real MT objects loaded from mtpy_data profile EDIs."""
    mt_objects = []
    for fn in profile_edi_files:
        mt_obj = MT(fn)
        mt_obj.read()
        mt_objects.append(mt_obj)
    return mt_objects


@pytest.fixture(scope="session")
def loaded_profile_mt(loaded_profile_mt_objects):
    """Single real MT object loaded from mtpy_data profile EDI."""
    return loaded_profile_mt_objects[0]


class TestMTDataTreeInitialization:
    def test_init_creates_root_and_surveys_node(self):
        tree = MTDataTree()

        assert tree.tree.name == MTDataTree.ROOT_NAME
        assert MTDataTree.SURVEYS_NODE in tree.tree.children
        assert tree.tree.attrs["schema_name"] == "mtpy.mt_data_tree"
        assert tree.tree.attrs["schema_version"] == "0.1.0"

    def test_init_applies_custom_attrs(self):
        tree = MTDataTree(coordinate_reference_frame="ned", impedance_units="mt")

        assert tree.tree.attrs["coordinate_reference_frame"] == "ned"
        assert tree.tree.attrs["impedance_units"] == "mt"
        assert tree.attrs is tree.tree.attrs


class TestMTDataTreeAddStation:
    @pytest.mark.parametrize(
        "survey, station, expected_path",
        [
            ("survey_a", "station_01", "surveys/survey_a/stations/station_01"),
            ("survey_b", "station_02", "surveys/survey_b/stations/station_02"),
        ],
    )
    def test_add_station_path_and_attrs(self, survey, station, expected_path):
        mt = MT()
        mt.survey = survey
        mt.station = station

        tree = MTDataTree()
        station_path = tree.add_station(mt)

        assert station_path == expected_path

        ds = tree.get_station(station_path)
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["survey"] == survey.replace("/", "_")
        assert ds.attrs["station"] == station.replace("/", "_")
        assert "survey_metadata" in ds.attrs
        assert "station_metadata" in ds.attrs

    def test_clean_name_replaces_path_separators(self):
        assert MTDataTree._clean_name("survey/a", "default") == "survey_a"
        assert MTDataTree._clean_name("station/01", "unknown_station") == "station_01"

    def test_add_station_uses_metadata_id_fallback(self):
        mt = MT()
        mt.survey = None
        mt.station = None
        mt.survey_metadata.id = "survey_from_metadata"
        mt.station_metadata.id = "station_from_metadata"

        tree = MTDataTree()
        station_path = tree.add_station(mt)

        assert (
            station_path
            == "surveys/survey_from_metadata/stations/station_from_metadata"
        )

    def test_add_station_default_name_fallbacks(self):
        mt = MT()
        mt.survey = ""
        mt.station = ""
        mt.survey_metadata.id = ""
        mt.station_metadata.id = ""

        tree = MTDataTree()
        station_path = tree.add_station(mt)

        assert station_path == "surveys/default/stations/unknown_station"

    def test_add_station_none_raises(self):
        tree = MTDataTree()
        with pytest.raises(TypeError, match="mt_obj cannot be None"):
            tree.add_station(None)

    def test_add_station_from_filename(self, profile_edi_file, loaded_profile_mt):
        tree = MTDataTree()
        station_path = tree.add_station(str(profile_edi_file))

        expected_path = (
            f"surveys/{MTDataTree._clean_name(loaded_profile_mt.survey, 'default')}"
            f"/stations/{MTDataTree._clean_name(loaded_profile_mt.station, 'unknown_station')}"
        )
        assert station_path == expected_path

    def test_add_station_from_path(self, profile_edi_file, loaded_profile_mt):
        tree = MTDataTree()
        station_path = tree.add_station(profile_edi_file)

        expected_path = (
            f"surveys/{MTDataTree._clean_name(loaded_profile_mt.survey, 'default')}"
            f"/stations/{MTDataTree._clean_name(loaded_profile_mt.station, 'unknown_station')}"
        )
        assert station_path == expected_path

    def test_add_station_list_mixed_inputs(
        self, basic_mt, profile_edi_file, loaded_profile_mt
    ):
        tree = MTDataTree()
        out_paths = tree.add_station([basic_mt, profile_edi_file])

        assert isinstance(out_paths, list)
        assert len(out_paths) == 2
        assert out_paths[0] == "surveys/big/stations/test_01"
        assert out_paths[1] == (
            f"surveys/{MTDataTree._clean_name(loaded_profile_mt.survey, 'default')}"
            f"/stations/{MTDataTree._clean_name(loaded_profile_mt.station, 'unknown_station')}"
        )

    def test_add_station_invalid_type_raises(self):
        tree = MTDataTree()
        with pytest.raises(TypeError, match="mt_obj must be an MT instance"):
            tree.add_station(42)

    def test_add_station_includes_location_attrs(self):
        mt = MT()
        mt.survey = "loc_survey"
        mt.station = "loc_station"
        mt.latitude = 40.25
        mt.longitude = -118.75
        mt.elevation = 1234.5
        mt.east = 500000.0
        mt.north = 4450000.0

        tree = MTDataTree()
        station_path = tree.add_station(mt)
        ds = tree.get_station(station_path)

        assert ds.attrs["latitude"] == 40.25
        assert ds.attrs["longitude"] == -118.75
        assert ds.attrs["elevation"] == 1234.5
        assert ds.attrs["easting"] == 500000.0
        assert ds.attrs["northing"] == 4450000.0
        assert "datum_crs" in ds.attrs
        assert "utm_crs" in ds.attrs

    def test_add_station_real_data_includes_location_attrs(self, loaded_profile_mt):
        tree = MTDataTree()
        station_path = tree.add_station(loaded_profile_mt)
        ds = tree.get_station(station_path)

        assert ds.attrs["latitude"] == loaded_profile_mt.latitude
        assert ds.attrs["longitude"] == loaded_profile_mt.longitude
        assert ds.attrs["elevation"] == loaded_profile_mt.elevation

    def test_add_station_overwrite_false_raises(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        with pytest.raises(KeyError, match="Station path already exists"):
            tree.add_station(basic_mt, overwrite=False)

        # Sanity check existing node remains accessible
        ds = tree.get_station(station_path)
        assert isinstance(ds, xr.Dataset)

    def test_add_station_summary_metadata_storage(self):
        mt = MT()
        mt.survey = "summary_survey"
        mt.station = "summary_station"
        mt.survey_metadata.id = "summary_survey_id"
        mt.station_metadata.id = "summary_station_id"

        tree = MTDataTree(metadata_storage="summary")
        station_path = tree.add_station(mt)
        ds = tree.get_station(station_path)

        assert ds.attrs["survey_metadata"] == {"id": "summary_survey_id"}
        assert ds.attrs["station_metadata"] == {"id": "summary_station_id"}
        assert ds.attrs["survey_metadata_ref"] is None
        assert ds.attrs["station_metadata_ref"] is None

    def test_add_station_cache_metadata_storage(self):
        mt = MT()
        mt.survey = "cache_survey"
        mt.station = "cache_station"
        mt.survey_metadata.id = "cache_survey_id"
        mt.station_metadata.id = "cache_station_id"

        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(mt)
        ds = tree.get_station(station_path)

        assert ds.attrs["survey_metadata"] == {"id": "cache_survey_id"}
        assert ds.attrs["station_metadata"] == {"id": "cache_station_id"}
        assert ds.attrs["survey_metadata_ref"] == station_path
        assert ds.attrs["station_metadata_ref"] == station_path
        assert tree.metadata_cache["survey"][station_path] is mt.survey_metadata
        assert tree.metadata_cache["station"][station_path] is mt.station_metadata

    def test_get_station_as_mt_hydrates_cache_metadata(self):
        mt = MT()
        mt.survey = "cache_hydrate_survey"
        mt.station = "cache_hydrate_station"
        mt.survey_metadata.id = "cache_hydrate_survey_id"
        mt.station_metadata.id = "cache_hydrate_station_id"

        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(mt)

        out = tree.get_station(station_path, as_mt=True)

        assert out.survey_metadata.id == "cache_hydrate_survey_id"
        assert out.station_metadata.id == "cache_hydrate_station_id"


class TestMTDataTreeNodeOperations:
    def test_get_station_default_returns_dataset(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        out = tree.get_station(station_path)
        assert isinstance(out, xr.Dataset)

    def test_get_station_as_mt_returns_mt_object(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        out = tree.get_station(station_path, as_mt=True)
        assert isinstance(out, MT)
        assert out.survey == basic_mt.survey
        assert out.station == basic_mt.station
        assert isinstance(out._transfer_function, xr.Dataset)

    def test_remove_station(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        assert tree._path_exists(station_path)
        tree.remove_station(station_path)
        assert not tree._path_exists(station_path)

    def test_keys_returns_top_level_children(self, basic_mt):
        tree = MTDataTree()
        tree.add_station(basic_mt)

        keys = tree.keys()
        assert MTDataTree.SURVEYS_NODE in keys

    def test_to_mt_stations_returns_station_locations(self, loaded_profile_mt_objects):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        stations = tree.to_mt_stations()

        assert isinstance(stations, MTStations)
        assert len(stations.mt_list) == len(loaded_profile_mt_objects)
        assert stations.station_locations is not None
        assert len(stations.station_locations) == len(loaded_profile_mt_objects)

    def test_to_mt_stations_does_not_require_dataset_to_mt(
        self, loaded_profile_mt_objects, monkeypatch
    ):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        def _fail(_station_ds):
            raise AssertionError(
                "to_mt_stations should not reconstruct full MT objects"
            )

        monkeypatch.setattr(MTDataTree, "_dataset_to_mt", staticmethod(_fail))

        stations = tree.to_mt_stations()

        assert isinstance(stations, MTStations)
        assert len(stations.station_locations) == len(loaded_profile_mt_objects)

    def test_mt_stations_property_returns_mtstations(self, loaded_profile_mt_objects):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        assert isinstance(tree.mt_stations, MTStations)

    def test_get_station_as_mt_restores_location_attrs(self, loaded_profile_mt):
        tree = MTDataTree()
        station_path = tree.add_station(loaded_profile_mt)

        out = tree.get_station(station_path, as_mt=True)

        assert out.latitude == loaded_profile_mt.latitude
        assert out.longitude == loaded_profile_mt.longitude
        assert out.elevation == loaded_profile_mt.elevation


class TestMTDataTreePeriods:
    def test_get_periods_empty_tree(self):
        tree = MTDataTree()
        periods = tree.get_periods()

        assert isinstance(periods, np.ndarray)
        assert periods.size == 0

    def test_get_periods_unique_sorted(self):
        mt1 = MT()
        mt1.survey = "s1"
        mt1.station = "st01"
        mt1._transfer_function = xr.Dataset(coords={"period": [10.0, 1.0, 3.0]})

        mt2 = MT()
        mt2.survey = "s1"
        mt2.station = "st02"
        mt2._transfer_function = xr.Dataset(coords={"period": [3.0, 5.0, 20.0]})

        tree = MTDataTree()
        tree.add_station([mt1, mt2])

        periods = tree.get_periods()
        assert np.array_equal(periods, np.array([1.0, 3.0, 5.0, 10.0, 20.0]))

    def test_get_periods_real_data(self, loaded_profile_mt_objects):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        expected = np.unique(
            np.concatenate(
                [np.asarray(mt.period, dtype=float) for mt in loaded_profile_mt_objects]
            )
        )
        expected.sort()

        periods = tree.get_periods()
        assert np.array_equal(periods, expected)


class TestMTDataTreeSpatialFiltering:
    def test_station_locations_returns_dataframe_without_mt_conversion(
        self, loaded_profile_mt_objects, monkeypatch
    ):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        def _fail(_station_ds):
            raise AssertionError(
                "station_locations should not reconstruct full MT objects"
            )

        monkeypatch.setattr(MTDataTree, "_dataset_to_mt", staticmethod(_fail))

        station_df = tree.station_locations

        assert isinstance(station_df, pd.DataFrame)
        assert len(station_df) == len(loaded_profile_mt_objects)
        assert set(station_df.station) == {
            mt.station for mt in loaded_profile_mt_objects
        }

    def test_apply_bounding_box_returns_subset_tree(self):
        mt1 = MT()
        mt1.survey = "s1"
        mt1.station = "inside"
        mt1.latitude = 40.0
        mt1.longitude = -120.0

        mt2 = MT()
        mt2.survey = "s1"
        mt2.station = "outside"
        mt2.latitude = 50.0
        mt2.longitude = -100.0

        tree = MTDataTree()
        tree.add_station([mt1, mt2])

        subset = tree.apply_bounding_box(-121.0, -119.0, 39.0, 41.0)

        assert isinstance(subset, MTDataTree)
        assert subset._path_exists("surveys/s1/stations/inside")
        assert not subset._path_exists("surveys/s1/stations/outside")

    def test_apply_bounding_box_real_data(self, loaded_profile_mt_objects):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        lon_values = [mt.longitude for mt in loaded_profile_mt_objects]
        lat_values = [mt.latitude for mt in loaded_profile_mt_objects]
        subset = tree.apply_bounding_box(
            min(lon_values), max(lon_values), min(lat_values), max(lat_values)
        )

        assert len(subset.mt_stations.station_locations) == len(
            loaded_profile_mt_objects
        )

    def test_apply_bounding_box_does_not_require_dataset_to_mt(
        self, loaded_profile_mt_objects, monkeypatch
    ):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        def _fail(_station_ds):
            raise AssertionError(
                "apply_bounding_box should not reconstruct full MT objects"
            )

        monkeypatch.setattr(MTDataTree, "_dataset_to_mt", staticmethod(_fail))

        lon_values = [mt.longitude for mt in loaded_profile_mt_objects]
        lat_values = [mt.latitude for mt in loaded_profile_mt_objects]
        subset = tree.apply_bounding_box(
            min(lon_values), max(lon_values), min(lat_values), max(lat_values)
        )

        assert len(subset.station_locations) == len(loaded_profile_mt_objects)


class TestMTDataTreeDataFrames:
    def test_to_dataframe_returns_concatenated_station_data(
        self, loaded_profile_mt_objects
    ):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        df = tree.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert set(df.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }
        assert np.array_equal(np.sort(df.period.unique()), tree.get_periods())

    def test_to_mt_dataframe_returns_mt_dataframe(self, loaded_profile_mt_objects):
        tree = MTDataTree()
        tree.add_station(loaded_profile_mt_objects)

        mt_df = tree.to_mt_dataframe()

        assert isinstance(mt_df, MTDataFrame)
        assert set(mt_df.dataframe.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }

    def test_from_dataframe_populates_tree(self, loaded_profile_mt_objects):
        source_tree = MTDataTree()
        source_tree.add_station(loaded_profile_mt_objects)
        df = source_tree.to_dataframe()

        tree = MTDataTree()
        tree.from_dataframe(df)

        assert len(tree._iter_station_paths()) == len(loaded_profile_mt_objects)
        assert np.array_equal(tree.get_periods(), source_tree.get_periods())

    def test_from_mt_dataframe_populates_tree(self, loaded_profile_mt_objects):
        source_tree = MTDataTree()
        source_tree.add_station(loaded_profile_mt_objects)
        mt_df = source_tree.to_mt_dataframe()

        tree = MTDataTree()
        tree.from_mt_dataframe(mt_df)

        assert len(tree._iter_station_paths()) == len(loaded_profile_mt_objects)
        assert np.array_equal(tree.get_periods(), source_tree.get_periods())

    def test_to_dataframe_empty_tree_returns_empty_dataframe(self):
        tree = MTDataTree()

        df = tree.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.empty
