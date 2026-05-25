# -*- coding: utf-8 -*-
"""Pytest suite for MTData scaffold behavior."""

import mtpy_data
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from mtpy.core import MTData
from mtpy.core.mt import MT
from mtpy.core.mt_dataframe import MTDataFrame
from mtpy.core.mt_stations import MTStations
from mtpy.modeling.modem import Data
from mtpy.modeling.occam2d import Occam2DData


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


class TestMTDataInitialization:
    def test_init_creates_root_and_surveys_node(self):
        tree = MTData()

        assert tree.tree.name == MTData.ROOT_NAME
        assert MTData.SURVEYS_NODE in tree.tree.children
        assert tree.tree.attrs["schema_name"] == "mtpy.mt_data_tree"
        assert tree.tree.attrs["schema_version"] == "0.1.0"
        assert tree.metadata_storage == "cache"

    def test_init_applies_custom_attrs(self):
        tree = MTData(coordinate_reference_frame="ned", impedance_units="mt")

        assert tree.tree.attrs["coordinate_reference_frame"] == "NED"
        assert tree.tree.attrs["impedance_units"] == "mt"
        assert tree.attrs is tree.tree.attrs

    def test_coordinate_reference_frame_set_propagates_to_stations(
        self, loaded_profile_mt_objects
    ):
        tree = MTData()
        station_paths = tree.add_stations(loaded_profile_mt_objects)

        tree.coordinate_reference_frame = "enu"

        assert tree.coordinate_reference_frame == "ENU"
        assert tree.tree.attrs["coordinate_reference_frame"] == "ENU"
        for station_path in station_paths:
            assert (
                tree.get_station(station_path).attrs["coordinate_reference_frame"]
                == "ENU"
            )

    def test_impedance_units_set_propagates_to_stations(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        tree.impedance_units = "ohm"

        assert tree.impedance_units == "ohm"
        assert tree.tree.attrs["impedance_units"] == "ohm"
        assert tree.get_station(station_path).attrs["impedance_units"] == "ohm"

    def test_coordinate_reference_frame_invalid_raises(self):
        tree = MTData()

        with pytest.raises(ValueError, match="is not understood as a reference frame"):
            tree.coordinate_reference_frame = "bad_frame"

    def test_impedance_units_invalid_raises(self):
        tree = MTData()

        with pytest.raises(ValueError, match="is not an acceptable unit"):
            tree.impedance_units = "bad_unit"


class TestMTDataStringRepresentation:
    def test_str_empty_tree_contains_summary_sections(self):
        tree = MTData()

        text = str(tree)

        assert "MTData Summary" in text
        assert "stations: 0" in text
        assert "surveys: 0" in text
        assert "survey names:" in text
        assert "station paths:" in text
        assert "<none>" in text

    def test_str_populated_tree_lists_station_paths(self):
        mt_1 = MT()
        mt_1.survey = "survey_a"
        mt_1.station = "station_01"

        mt_2 = MT()
        mt_2.survey = "survey_b"
        mt_2.station = "station_02"

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        text = str(tree)

        assert "stations: 2" in text
        assert "surveys: 2" in text
        assert "surveys/survey_a/stations/station_01" in text
        assert "surveys/survey_b/stations/station_02" in text

    def test_repr_includes_core_counts_and_modes(self):
        mt = MT()
        mt.survey = "survey_a"
        mt.station = "station_01"

        tree = MTData(metadata_storage="summary", dataset_copy_mode="deep")
        tree.add_station(mt)

        text = repr(tree)

        assert text.startswith("MTData(")
        assert "stations=1" in text
        assert "surveys=1" in text
        assert "metadata_storage='summary'" in text
        assert "dataset_copy_mode='deep'" in text


class TestMTDataAddDunder:
    def test_add_returns_new_tree_with_union_of_station_paths(self):
        mt_left = MT()
        mt_left.survey = "survey_a"
        mt_left.station = "station_01"

        mt_right = MT()
        mt_right.survey = "survey_b"
        mt_right.station = "station_02"

        left = MTData()
        right = MTData()
        left_path = left.add_station(mt_left)
        right_path = right.add_station(mt_right)

        merged = left + right

        assert isinstance(merged, MTData)
        assert merged is not left
        assert merged is not right
        assert set(merged._iter_station_paths()) == {left_path, right_path}
        assert set(left._iter_station_paths()) == {left_path}
        assert set(right._iter_station_paths()) == {right_path}

    def test_add_overwrites_duplicate_paths_and_logs_warning(self):
        mt_old = MT()
        mt_old.survey = "survey_a"
        mt_old.station = "station_01"
        mt_old.latitude = 10.0

        mt_new = MT()
        mt_new.survey = "survey_a"
        mt_new.station = "station_01"
        mt_new.latitude = 20.0

        left = MTData()
        right = MTData()
        station_path = left.add_station(mt_old)
        right.add_station(mt_new)

        warning_messages = []

        sink_id = logger.add(
            lambda message: warning_messages.append(str(message)), level="WARNING"
        )

        merged = left + right

        logger.remove(sink_id)

        assert merged.get_station(station_path).attrs["latitude"] == 20.0
        assert left.get_station(station_path).attrs["latitude"] == 10.0
        assert any(
            "Overwriting existing station path during MTData merge" in msg
            for msg in warning_messages
        )


class TestMTDataAddStation:
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

        tree = MTData()
        station_path = tree.add_station(mt)

        assert station_path == expected_path

        ds = tree.get_station(station_path)
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["survey"] == survey.replace("/", "_")
        assert ds.attrs["station"] == station.replace("/", "_")
        assert "survey_metadata" in ds.attrs
        assert "station_metadata" in ds.attrs

    def test_clean_name_replaces_path_separators(self):
        assert MTData._clean_name("survey/a", "default") == "survey_a"
        assert MTData._clean_name("station/01", "unknown_station") == "station_01"

    def test_add_station_uses_metadata_id_fallback(self):
        mt = MT()
        mt.survey = None
        mt.station = None
        mt.survey_metadata.id = "survey_from_metadata"
        mt.station_metadata.id = "station_from_metadata"

        tree = MTData()
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

        tree = MTData()
        station_path = tree.add_station(mt)

        assert station_path == "surveys/default/stations/unknown_station"

    def test_add_station_none_raises(self):
        tree = MTData()
        with pytest.raises(TypeError, match="mt_obj cannot be None"):
            tree.add_station(None)

    def test_add_station_from_filename(self, profile_edi_file, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(str(profile_edi_file))

        expected_path = (
            f"surveys/{MTData._clean_name(loaded_profile_mt.survey, 'default')}"
            f"/stations/{MTData._clean_name(loaded_profile_mt.station, 'unknown_station')}"
        )
        assert station_path == expected_path

    def test_add_station_from_path(self, profile_edi_file, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(profile_edi_file)

        expected_path = (
            f"surveys/{MTData._clean_name(loaded_profile_mt.survey, 'default')}"
            f"/stations/{MTData._clean_name(loaded_profile_mt.station, 'unknown_station')}"
        )
        assert station_path == expected_path

    def test_add_station_list_mixed_inputs(
        self, basic_mt, profile_edi_file, loaded_profile_mt
    ):
        tree = MTData()
        out_paths = tree.add_station([basic_mt, profile_edi_file])

        assert isinstance(out_paths, list)
        assert len(out_paths) == 2
        assert out_paths[0] == "surveys/big/stations/test_01"
        assert out_paths[1] == (
            f"surveys/{MTData._clean_name(loaded_profile_mt.survey, 'default')}"
            f"/stations/{MTData._clean_name(loaded_profile_mt.station, 'unknown_station')}"
        )

    def test_add_station_invalid_type_raises(self):
        tree = MTData()
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

        tree = MTData()
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
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        ds = tree.get_station(station_path)

        assert ds.attrs["latitude"] == loaded_profile_mt.latitude
        assert ds.attrs["longitude"] == loaded_profile_mt.longitude
        assert ds.attrs["elevation"] == loaded_profile_mt.elevation

    def test_add_station_overwrite_false_raises(self, basic_mt):
        tree = MTData()
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

        tree = MTData(metadata_storage="summary")
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

        tree = MTData(metadata_storage="cache")
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

        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(mt)

        out = tree.get_station(station_path, as_mt=True)

        assert out.survey_metadata.id == "cache_hydrate_survey_id"
        assert out.station_metadata.id == "cache_hydrate_station_id"

    def test_add_station_invalid_dataset_copy_mode_raises(self, basic_mt):
        tree = MTData()
        with pytest.raises(ValueError, match="dataset_copy_mode must be one of"):
            tree.add_station(basic_mt, dataset_copy_mode="bad_mode")

    def test_add_stations_bulk_returns_paths(self):
        mt_1 = MT()
        mt_1.survey = "bulk_survey"
        mt_1.station = "bulk_01"

        mt_2 = MT()
        mt_2.survey = "bulk_survey"
        mt_2.station = "bulk_02"

        tree = MTData()
        out_paths = tree.add_stations([mt_1, mt_2])

        assert out_paths == [
            "surveys/bulk_survey/stations/bulk_01",
            "surveys/bulk_survey/stations/bulk_02",
        ]

    def test_add_stations_precomputed_attrs(self):
        mt_1 = MT()
        mt_1.survey = "bulk_attr"
        mt_1.station = "bulk_attr_01"

        mt_2 = MT()
        mt_2.survey = "bulk_attr"
        mt_2.station = "bulk_attr_02"

        precomputed = [
            {"latitude": 1.25, "longitude": -2.5},
            {"latitude": 3.5, "longitude": -4.75},
        ]

        tree = MTData()
        out_paths = tree.add_stations([mt_1, mt_2], precomputed_attrs=precomputed)

        ds_1 = tree.get_station(out_paths[0])
        ds_2 = tree.get_station(out_paths[1])

        assert ds_1.attrs["latitude"] == 1.25
        assert ds_1.attrs["longitude"] == -2.5
        assert ds_2.attrs["latitude"] == 3.5
        assert ds_2.attrs["longitude"] == -4.75

    def test_add_station_overwrite_false_does_not_mutate_cache(self):
        mt_existing = MT()
        mt_existing.survey = "cache_overwrite"
        mt_existing.station = "same_station"

        mt_new = MT()
        mt_new.survey = "cache_overwrite"
        mt_new.station = "same_station"

        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(mt_existing)

        with pytest.raises(KeyError, match="Station path already exists"):
            tree.add_station(mt_new, overwrite=False)

        assert (
            tree.metadata_cache["survey"][station_path] is mt_existing.survey_metadata
        )
        assert (
            tree.metadata_cache["station"][station_path] is mt_existing.station_metadata
        )

    def test_add_stations_failure_does_not_mutate_cache(self):
        mt_existing = MT()
        mt_existing.survey = "cache_bulk"
        mt_existing.station = "same_station"

        mt_new = MT()
        mt_new.survey = "cache_bulk"
        mt_new.station = "same_station"

        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(mt_existing)

        with pytest.raises(KeyError, match="Station path already exists"):
            tree.add_stations([mt_new], overwrite=False)

        assert (
            tree.metadata_cache["survey"][station_path] is mt_existing.survey_metadata
        )
        assert (
            tree.metadata_cache["station"][station_path] is mt_existing.station_metadata
        )


class TestMTDataNodeOperations:
    def test_get_station_default_returns_dataset(self, basic_mt):
        tree = MTData()
        station_path = tree.add_station(basic_mt)

        out = tree.get_station(station_path)
        assert isinstance(out, xr.Dataset)

    def test_get_station_as_mt_returns_mt_object(self, basic_mt):
        tree = MTData()
        station_path = tree.add_station(basic_mt)

        out = tree.get_station(station_path, as_mt=True)
        assert isinstance(out, MT)

    def test_get_station_accepts_short_path(self, basic_mt):
        tree = MTData()
        station_path = tree.add_station(basic_mt)
        short_path = f"{basic_mt.survey}/{basic_mt.station}"

        out = tree.get_station(short_path)

        assert isinstance(out, xr.Dataset)
        assert out.identical(tree.get_station(station_path))

    def test_remove_station(self, basic_mt):
        tree = MTData()
        station_path = tree.add_station(basic_mt)

        assert tree._path_exists(station_path)
        tree.remove_station(station_path)
        assert not tree._path_exists(station_path)

    def test_remove_station_accepts_short_path(self, basic_mt):
        tree = MTData()
        station_path = tree.add_station(basic_mt)
        short_path = f"{basic_mt.survey}/{basic_mt.station}"

        tree.remove_station(short_path)

        assert not tree._path_exists(station_path)

    def test_get_subset_accepts_short_path(self, basic_mt):
        tree = MTData()
        station_path = tree.add_station(basic_mt)
        short_path = f"{basic_mt.survey}/{basic_mt.station}"

        subset = tree.get_subset([short_path])

        assert subset._path_exists(station_path)

    def test_remove_station_clears_cached_metadata(self):
        mt = MT()
        mt.survey = "remove_cache"
        mt.station = "remove_station"
        mt.survey_metadata.id = "remove_survey_id"
        mt.station_metadata.id = "remove_station_id"

        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(mt)

        tree.remove_station(station_path)

        assert station_path not in tree.metadata_cache["survey"]
        assert station_path not in tree.metadata_cache["station"]

    def test_keys_returns_top_level_children(self, basic_mt):
        tree = MTData()
        tree.add_station(basic_mt)

        keys = tree.keys()
        assert MTData.SURVEYS_NODE in keys

    def test_to_mt_stations_returns_station_locations(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        stations = tree.to_mt_stations()

        assert isinstance(stations, MTStations)
        assert len(stations) == len(loaded_profile_mt_objects)
        assert stations.station_locations is not None
        assert len(stations.station_locations) == len(loaded_profile_mt_objects)

    def test_to_mt_stations_does_not_require_dataset_to_mt(
        self, loaded_profile_mt_objects, monkeypatch
    ):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        def _fail(_station_ds):
            raise AssertionError(
                "to_mt_stations should not reconstruct full MT objects"
            )

        monkeypatch.setattr(MTData, "_dataset_to_mt", staticmethod(_fail))

        stations = tree.to_mt_stations()

        assert isinstance(stations, MTStations)
        assert len(stations.station_locations) == len(loaded_profile_mt_objects)

    def test_mt_stations_property_returns_mtstations(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        assert isinstance(tree.mt_stations, MTStations)

    def test_compute_relative_locations_wrapper_round_trip(self):
        mt_1 = MT()
        mt_1.survey = "wrap"
        mt_1.station = "s01"
        mt_1.latitude = 40.0
        mt_1.longitude = -120.0
        mt_1.east = 500000.0
        mt_1.north = 4400000.0
        mt_1.utm_epsg = 32611

        mt_2 = MT()
        mt_2.survey = "wrap"
        mt_2.station = "s02"
        mt_2.latitude = 40.1
        mt_2.longitude = -120.1
        mt_2.east = 500400.0
        mt_2.north = 4400600.0
        mt_2.utm_epsg = 32611

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        expected = tree.to_mt_stations()
        expected.compute_relative_locations()
        expected_df = expected.station_locations.set_index(["survey", "station"])

        tree.compute_relative_locations()
        actual_df = tree.station_locations.set_index(["survey", "station"])

        for column in ["model_east", "model_north", "model_elevation"]:
            assert np.allclose(
                actual_df[column].to_numpy(dtype=float),
                expected_df[column].to_numpy(dtype=float),
            )

    def test_rotate_stations_wrapper_round_trip(self):
        mt_1 = MT()
        mt_1.survey = "rotate_wrap"
        mt_1.station = "s01"
        mt_1.latitude = 40.0
        mt_1.longitude = -120.0
        mt_1.east = 500000.0
        mt_1.north = 4400000.0
        mt_1.utm_epsg = 32611

        mt_2 = MT()
        mt_2.survey = "rotate_wrap"
        mt_2.station = "s02"
        mt_2.latitude = 40.05
        mt_2.longitude = -120.05
        mt_2.east = 500700.0
        mt_2.north = 4400400.0
        mt_2.utm_epsg = 32611

        tree = MTData()
        tree.add_stations([mt_1, mt_2])
        tree.compute_relative_locations()

        expected = tree.to_mt_stations()
        expected.rotate_stations(30.0)
        expected_df = expected.station_locations.set_index(["survey", "station"])

        tree.rotate_stations(30.0)
        actual_df = tree.station_locations.set_index(["survey", "station"])

        for column in ["model_east", "model_north"]:
            assert np.allclose(
                actual_df[column].to_numpy(dtype=float),
                expected_df[column].to_numpy(dtype=float),
            )
        assert np.isclose(tree.data_rotation_angle, 30.0)

    def test_center_stations_wrapper_round_trip(self):
        mt_1 = MT()
        mt_1.survey = "center_wrap"
        mt_1.station = "s01"
        mt_1.latitude = 40.0
        mt_1.longitude = -120.0
        mt_1.east = 500000.0
        mt_1.north = 4400000.0
        mt_1.utm_epsg = 32611

        mt_2 = MT()
        mt_2.survey = "center_wrap"
        mt_2.station = "s02"
        mt_2.latitude = 40.1
        mt_2.longitude = -120.1
        mt_2.east = 501000.0
        mt_2.north = 4401000.0
        mt_2.utm_epsg = 32611

        class _Model:
            grid_east = np.array([-2000.0, 0.0, 1000.0, 2000.0])
            grid_north = np.array([-2000.0, 0.0, 1000.0, 2000.0])

        tree = MTData()
        tree.add_stations([mt_1, mt_2])
        tree.compute_relative_locations()

        expected = tree.to_mt_stations()
        expected.center_stations(_Model())
        expected_df = expected.station_locations.set_index(["survey", "station"])

        tree.center_stations(_Model())
        actual_df = tree.station_locations.set_index(["survey", "station"])

        for column in ["model_east", "model_north"]:
            assert np.allclose(
                actual_df[column].to_numpy(dtype=float),
                expected_df[column].to_numpy(dtype=float),
            )

    def test_project_stations_on_topography_wrapper_round_trip(self):
        mt_1 = MT()
        mt_1.survey = "topo_wrap"
        mt_1.station = "s01"
        mt_1.latitude = 40.0
        mt_1.longitude = -120.0
        mt_1.east = 500000.0
        mt_1.north = 4400000.0
        mt_1.model_east = 500.0
        mt_1.model_north = 500.0
        mt_1.utm_epsg = 32611

        mt_2 = MT()
        mt_2.survey = "topo_wrap"
        mt_2.station = "s02"
        mt_2.latitude = 40.1
        mt_2.longitude = -120.1
        mt_2.east = 500800.0
        mt_2.north = 4400900.0
        mt_2.model_east = 1500.0
        mt_2.model_north = 1500.0
        mt_2.utm_epsg = 32611

        class _TopoModel:
            grid_east = np.array([0.0, 1000.0, 2000.0])
            grid_north = np.array([0.0, 1000.0, 2000.0])
            grid_z = np.array([100.0, 50.0, 0.0])
            res_model = np.array(
                [
                    [[1e12, 100.0, 100.0], [1e12, 100.0, 100.0]],
                    [[1e12, 100.0, 100.0], [1e12, 100.0, 100.0]],
                ]
            )

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        expected = tree.to_mt_stations()
        expected.project_stations_on_topography(_TopoModel())
        expected_df = expected.station_locations.set_index(["survey", "station"])

        tree.project_stations_on_topography(_TopoModel())
        actual_df = tree.station_locations.set_index(["survey", "station"])

        assert np.allclose(
            actual_df["model_elevation"].to_numpy(dtype=float),
            expected_df["model_elevation"].to_numpy(dtype=float),
        )
        assert np.isclose(tree._center_elev, 100.0)

    def test_to_geopd_wrapper_delegates_to_mt_stations(self, monkeypatch):
        tree = MTData()
        expected = object()

        class _FakeStations:
            def to_geopd(self):
                return expected

        monkeypatch.setattr(tree, "to_mt_stations", lambda: _FakeStations())

        out = tree.to_geopd()

        assert out is expected

    def test_to_csv_wrapper_delegates_to_mt_stations(self, monkeypatch, tmp_path):
        tree = MTData()
        captured = {}

        class _FakeStations:
            def to_csv(self, csv_fn, geometry=False):
                captured["csv_fn"] = csv_fn
                captured["geometry"] = geometry

        monkeypatch.setattr(tree, "to_mt_stations", lambda: _FakeStations())

        out = tree.to_csv(tmp_path / "stations.csv", geometry=True)

        assert out is None
        assert captured["csv_fn"] == tmp_path / "stations.csv"
        assert captured["geometry"] is True

    def test_to_shp_wrapper_delegates_to_mt_stations(self, monkeypatch, tmp_path):
        tree = MTData()
        expected = tmp_path / "stations.shp"
        captured = {}

        class _FakeStations:
            def to_shp(self, shp_fn):
                captured["shp_fn"] = shp_fn
                return expected

        monkeypatch.setattr(tree, "to_mt_stations", lambda: _FakeStations())

        out = tree.to_shp(tmp_path / "stations.shp")

        assert out == expected
        assert captured["shp_fn"] == tmp_path / "stations.shp"

    def test_to_vtk_wrapper_delegates_to_mt_stations(self, monkeypatch, tmp_path):
        tree = MTData()
        expected = tmp_path / "stations.vtu"
        captured = {}

        class _FakeStations:
            def to_vtk(self, **kwargs):
                captured.update(kwargs)
                return expected

        monkeypatch.setattr(tree, "to_mt_stations", lambda: _FakeStations())

        out = tree.to_vtk(
            vtk_fn=tmp_path / "stations.vtk",
            vtk_save_path=tmp_path,
            vtk_fn_basename="wrap_stations",
            geographic=True,
            shift_east=1.0,
            shift_north=2.0,
            shift_elev=3.0,
            units="m",
            coordinate_system="enz-",
        )

        assert out == expected
        assert captured["vtk_fn"] == tmp_path / "stations.vtk"
        assert captured["vtk_save_path"] == tmp_path
        assert captured["vtk_fn_basename"] == "wrap_stations"
        assert captured["geographic"] is True
        assert np.isclose(captured["shift_east"], 1.0)
        assert np.isclose(captured["shift_north"], 2.0)
        assert np.isclose(captured["shift_elev"], 3.0)
        assert captured["units"] == "m"
        assert captured["coordinate_system"] == "enz-"

    def test_generate_profile_wrapper_delegates_to_mt_stations(self, monkeypatch):
        tree = MTData()
        expected = (1.0, 2.0, 3.0, 4.0, {"s01": 0.0})
        captured = {}

        class _FakeStations:
            def generate_profile(self, units="deg"):
                captured["units"] = units
                return expected

        monkeypatch.setattr(tree, "to_mt_stations", lambda: _FakeStations())

        out = tree.generate_profile(units="m")

        assert out == expected
        assert captured["units"] == "m"

    def test_generate_profile_from_strike_wrapper_delegates_to_mt_stations(
        self, monkeypatch
    ):
        tree = MTData()
        expected = (0.0, 1.0, 2.0, 3.0, {"s01": 100.0})
        captured = {}

        class _FakeStations:
            def generate_profile_from_strike(self, strike, units="deg"):
                captured["strike"] = strike
                captured["units"] = units
                return expected

        monkeypatch.setattr(tree, "to_mt_stations", lambda: _FakeStations())

        out = tree.generate_profile_from_strike(35.0, units="m")

        assert out == expected
        assert np.isclose(captured["strike"], 35.0)
        assert captured["units"] == "m"

    def test_get_station_as_mt_restores_location_attrs(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        out = tree.get_station(station_path, as_mt=True)

        assert out.latitude == loaded_profile_mt.latitude
        assert out.longitude == loaded_profile_mt.longitude
        assert out.elevation == loaded_profile_mt.elevation

    def test_get_station_as_mt_restores_profile_offset(self, loaded_profile_mt):
        mt_obj = loaded_profile_mt.copy()
        mt_obj.profile_offset = 123.456

        tree = MTData()
        station_path = tree.add_station(mt_obj)

        out = tree.get_station(station_path, as_mt=True)

        assert np.isclose(out.profile_offset, 123.456)

    def test_survey_ids_empty_tree(self):
        tree = MTData()

        assert tree.survey_ids == []

    def test_survey_ids_returns_unique_surveys(self):
        mt_1 = MT()
        mt_1.survey = "survey_a"
        mt_1.station = "station_01"

        mt_2 = MT()
        mt_2.survey = "survey_a"
        mt_2.station = "station_02"

        mt_3 = MT()
        mt_3.survey = "survey_b"
        mt_3.station = "station_03"

        tree = MTData()
        tree.add_stations([mt_1, mt_2, mt_3])

        assert set(tree.survey_ids) == {"survey_a", "survey_b"}

    def test_short_station_paths_returns_survey_station_form(self):
        mt_1 = MT()
        mt_1.survey = "survey_a"
        mt_1.station = "station_01"

        mt_2 = MT()
        mt_2.survey = "survey_b"
        mt_2.station = "station_02"

        tree = MTData()
        tree.add_stations([mt_2, mt_1])

        assert tree.short_station_paths == [
            "survey_a/station_01",
            "survey_b/station_02",
        ]

    def test_get_survey_returns_filtered_tree(self):
        mt_1 = MT()
        mt_1.survey = "survey_a"
        mt_1.station = "station_01"

        mt_2 = MT()
        mt_2.survey = "survey_a"
        mt_2.station = "station_02"

        mt_3 = MT()
        mt_3.survey = "survey_b"
        mt_3.station = "station_03"

        tree = MTData()
        tree.add_stations([mt_1, mt_2, mt_3])

        survey_tree = tree.get_survey("survey_a")

        assert isinstance(survey_tree, MTData)
        assert set(survey_tree.survey_ids) == {"survey_a"}
        assert set(survey_tree._iter_station_paths()) == {
            "surveys/survey_a/stations/station_01",
            "surveys/survey_a/stations/station_02",
        }

    def test_get_survey_missing_id_returns_empty_tree(self):
        mt = MT()
        mt.survey = "survey_a"
        mt.station = "station_01"

        tree = MTData()
        tree.add_station(mt)

        survey_tree = tree.get_survey("not_present")

        assert isinstance(survey_tree, MTData)
        assert survey_tree._iter_station_paths() == []
        assert survey_tree.survey_ids == []

    def test_to_geo_df_station_locations(self):
        geopandas = pytest.importorskip("geopandas")

        mt_1 = MT()
        mt_1.survey = "geo"
        mt_1.station = "s01"
        mt_1.latitude = 40.0
        mt_1.longitude = -120.0
        mt_1.east = 100.0
        mt_1.north = 200.0
        mt_1.utm_epsg = 32611

        mt_2 = MT()
        mt_2.survey = "geo"
        mt_2.station = "s02"
        mt_2.latitude = 41.0
        mt_2.longitude = -121.0
        mt_2.east = 130.0
        mt_2.north = 240.0
        mt_2.utm_epsg = 32611

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        gdf = tree.to_geo_df()

        assert isinstance(gdf, geopandas.GeoDataFrame)
        assert len(gdf) == 2
        assert "geometry" in gdf.columns
        assert set(gdf.station.tolist()) == {"s01", "s02"}

    def test_to_geo_df_model_locations_uses_model_coordinates(self):
        pytest.importorskip("geopandas")

        mt = MT()
        mt.survey = "geo_model"
        mt.station = "m01"
        mt.latitude = 40.0
        mt.longitude = -120.0
        mt.model_east = 12.5
        mt.model_north = 34.5

        tree = MTData()
        tree.add_station(mt)

        gdf = tree.to_geo_df(model_locations=True)

        assert np.isclose(float(gdf.geometry.x.iloc[0]), 12.5)
        assert np.isclose(float(gdf.geometry.y.iloc[0]), 34.5)

    def test_to_geo_df_invalid_data_type_raises(self):
        tree = MTData()

        with pytest.raises(ValueError, match="unsupported"):
            tree.to_geo_df(data_type="bad_type")

    def test_to_shp_pt_tipper_estimates_sizes_and_forwards_defaults(
        self, basic_mt, monkeypatch, tmp_path
    ):
        captured = {}

        class _FakeShapefileCreator:
            def __init__(self, mt_df, output_crs, save_dir):
                captured["mt_df"] = mt_df
                captured["output_crs"] = output_crs
                captured["save_dir"] = save_dir
                self.utm = False
                self.ellipse_size = None
                self.arrow_size = None

            def estimate_ellipse_size(self):
                captured["ellipse_estimated"] = True
                return 12.5

            def estimate_arrow_size(self):
                captured["arrow_estimated"] = True
                return 7.25

            def make_shp_files(
                self, pt=True, tipper=True, periods=None, period_tol=None
            ):
                captured["pt"] = pt
                captured["tipper"] = tipper
                captured["periods"] = periods
                captured["period_tol"] = period_tol
                captured["utm"] = self.utm
                captured["ellipse_size"] = self.ellipse_size
                captured["arrow_size"] = self.arrow_size
                return {"pt": ["pt.shp"], "tipper": ["tipper.shp"]}

        monkeypatch.setattr(
            "mtpy.gis.shapefile_creator.ShapefileCreator", _FakeShapefileCreator
        )

        tree = MTData()
        tree.add_station(basic_mt)

        out = tree.to_shp_pt_tipper(save_dir=tmp_path, output_crs=4326, utm=True)

        assert isinstance(captured["mt_df"], MTDataFrame)
        assert captured["output_crs"] == 4326
        assert captured["save_dir"] == tmp_path
        assert captured["ellipse_estimated"] is True
        assert captured["arrow_estimated"] is True
        assert captured["pt"] is True
        assert captured["tipper"] is True
        assert captured["periods"] is None
        assert captured["period_tol"] is None
        assert captured["utm"] is True
        assert np.isclose(captured["ellipse_size"], 12.5)
        assert np.isclose(captured["arrow_size"], 7.25)
        assert out == {"pt": ["pt.shp"], "tipper": ["tipper.shp"]}

    def test_to_shp_pt_tipper_uses_explicit_sizes_and_options(
        self, basic_mt, monkeypatch, tmp_path
    ):
        captured = {}

        class _FakeShapefileCreator:
            def __init__(self, _mt_df, _output_crs, save_dir):
                captured["save_dir"] = save_dir
                self.utm = False
                self.ellipse_size = None
                self.arrow_size = None

            def estimate_ellipse_size(self):
                raise AssertionError("estimate_ellipse_size should not be called")

            def estimate_arrow_size(self):
                raise AssertionError("estimate_arrow_size should not be called")

            def make_shp_files(
                self, pt=True, tipper=True, periods=None, period_tol=None
            ):
                captured["pt"] = pt
                captured["tipper"] = tipper
                captured["periods"] = periods
                captured["period_tol"] = period_tol
                captured["utm"] = self.utm
                captured["ellipse_size"] = self.ellipse_size
                captured["arrow_size"] = self.arrow_size
                return {"pt": [], "tipper": []}

        monkeypatch.setattr(
            "mtpy.gis.shapefile_creator.ShapefileCreator", _FakeShapefileCreator
        )

        tree = MTData()
        tree.add_station(basic_mt)
        periods = np.array([1.0, 10.0])

        out = tree.to_shp_pt_tipper(
            save_dir=tmp_path,
            output_crs="EPSG:4326",
            utm=False,
            pt=False,
            tipper=True,
            periods=periods,
            period_tol=0.05,
            ellipse_size=9.0,
            arrow_size=3.5,
        )

        assert captured["save_dir"] == tmp_path
        assert captured["pt"] is False
        assert captured["tipper"] is True
        assert np.array_equal(captured["periods"], periods)
        assert np.isclose(captured["period_tol"], 0.05)
        assert captured["utm"] is False
        assert np.isclose(captured["ellipse_size"], 9.0)
        assert np.isclose(captured["arrow_size"], 3.5)
        assert out == {"pt": [], "tipper": []}

    def test_get_nearby_stations_meters(self):
        mt_1 = MT()
        mt_1.survey = "near"
        mt_1.station = "s01"
        mt_1.utm_epsg = 32611
        mt_1.east = 1.0
        mt_1.north = 1.0

        mt_2 = MT()
        mt_2.survey = "near"
        mt_2.station = "s02"
        mt_2.utm_epsg = 32611
        mt_2.east = 4.0
        mt_2.north = 5.0

        mt_3 = MT()
        mt_3.survey = "near"
        mt_3.station = "s03"
        mt_3.utm_epsg = 32611
        mt_3.east = 100.0
        mt_3.north = 100.0

        tree = MTData()
        tree.add_stations([mt_1, mt_2, mt_3])

        out = tree.get_nearby_stations("near.s01", radius=6.0, radius_units="m")

        assert out == ["near.s02"]

    def test_get_nearby_stations_requires_utm_for_meters(self):
        mt_1 = MT()
        mt_1.survey = "no_utm"
        mt_1.station = "s01"
        mt_1.east = 0.0
        mt_1.north = 0.0

        mt_2 = MT()
        mt_2.survey = "no_utm"
        mt_2.station = "s02"
        mt_2.east = 1.0
        mt_2.north = 1.0

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        with pytest.raises(ValueError, match="UTM CRS"):
            tree.get_nearby_stations("no_utm.s01", radius=10.0, radius_units="m")

    def test_estimate_spatial_static_shift_no_nearby_returns_identity(
        self, monkeypatch
    ):
        tree = MTData()

        monkeypatch.setattr(tree, "get_nearby_stations", lambda *_args, **_kwargs: [])

        sx, sy = tree.estimate_spatial_static_shift(
            station_key="survey.station",
            radius=1000.0,
            period_min=1.0,
            period_max=10.0,
        )

        assert sx == 1.0
        assert sy == 1.0

    def test_estimate_spatial_static_shift_computes_and_applies_tolerance(
        self, monkeypatch
    ):
        tree = MTData()

        class _FakeLocalSite:
            def __init__(self):
                self.period = np.array([1.0, 5.0, 10.0])

                class _Z:
                    res_xy = np.array([10.0, 10.0, 10.0])
                    res_yx = np.array([10.0, 10.0, 10.0])

                self.Z = _Z()

            def interpolate(self, _periods):
                return self

        class _FakeSubset:
            def interpolate(self, _periods):
                return None

            def to_dataframe(self):
                return pd.DataFrame(
                    {
                        "res_xy": [20.0, 20.0, 20.0],
                        "res_yx": [10.5, 10.5, 10.5],
                    }
                )

        monkeypatch.setattr(
            tree,
            "get_nearby_stations",
            lambda *_args, **_kwargs: ["near.s01"],
        )
        monkeypatch.setattr(tree, "_resolve_station_path", lambda key: key)
        monkeypatch.setattr(tree, "get_subset", lambda _paths: _FakeSubset())
        monkeypatch.setattr(
            tree,
            "get_station",
            lambda _key, as_mt=False: _FakeLocalSite() if as_mt else None,
        )

        sx, sy = tree.estimate_spatial_static_shift(
            station_key="local.s00",
            radius=1000.0,
            period_min=1.0,
            period_max=10.0,
            shift_tolerance=0.1,
        )

        assert np.isclose(sx, 2.0)
        assert np.isclose(sy, 1.0)

    def test_get_profile_returns_subset_with_profile_offsets(self):
        mt_1 = MT()
        mt_1.survey = "prof"
        mt_1.station = "s01"
        mt_1.utm_epsg = 32611
        mt_1.east = 1.0
        mt_1.north = 1.0

        mt_2 = MT()
        mt_2.survey = "prof"
        mt_2.station = "s02"
        mt_2.utm_epsg = 32611
        mt_2.east = 20.0
        mt_2.north = 1.0

        mt_3 = MT()
        mt_3.survey = "prof"
        mt_3.station = "s03"
        mt_3.utm_epsg = 32611
        mt_3.east = 1.0
        mt_3.north = 60.0

        tree = MTData()
        tree.add_stations([mt_1, mt_2, mt_3])

        profile_tree = tree.get_profile(
            x1=1.0,
            y1=1.0,
            x2=200.0,
            y2=1.0,
            radius=5.0,
        )

        assert isinstance(profile_tree, MTData)
        assert set(profile_tree._iter_station_paths()) == {
            "surveys/prof/stations/s01",
            "surveys/prof/stations/s02",
        }
        offset_1 = profile_tree.get_station("surveys/prof/stations/s01").attrs[
            "profile_offset"
        ]
        offset_2 = profile_tree.get_station("surveys/prof/stations/s02").attrs[
            "profile_offset"
        ]
        assert np.isclose(offset_1, 0.0)
        assert offset_2 > offset_1

    def test_get_profile_no_matches_returns_empty_tree(self):
        mt = MT()
        mt.survey = "prof_empty"
        mt.station = "s01"
        mt.latitude = 40.0
        mt.longitude = -120.0
        mt.east = 100.0
        mt.north = 100.0
        mt.utm_epsg = 32611

        tree = MTData()
        tree.add_station(mt)

        profile_tree = tree.get_profile(
            x1=0.0,
            y1=0.0,
            x2=200.0,
            y2=0.0,
            radius=5.0,
        )

        assert isinstance(profile_tree, MTData)
        assert profile_tree.n_stations == 0

    def test_compute_model_errors_updates_transfer_function_model_error(
        self, loaded_profile_mt
    ):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt.copy())

        before = tree.get_station(station_path)[
            "transfer_function_model_error"
        ].values.copy()

        tree.compute_model_errors(
            z_error_value=11,
            z_error_type="absolute",
            z_floor=False,
            t_error_value=0.11,
            t_error_type="absolute",
            t_floor=False,
        )

        after = tree.get_station(station_path)["transfer_function_model_error"].values

        assert not np.allclose(before, after, equal_nan=True)

    def test_compute_model_errors_matches_mt_compute_methods(self, loaded_profile_mt):
        mt_obj = loaded_profile_mt.copy()

        tree = MTData()
        station_path = tree.add_station(mt_obj.copy())

        z_kwargs = {
            "error_value": 9,
            "error_type": "absolute",
            "floor": False,
        }
        t_kwargs = {
            "error_value": 0.08,
            "error_type": "absolute",
            "floor": False,
        }

        mt_obj.compute_model_z_errors(**z_kwargs)
        mt_obj.compute_model_t_errors(**t_kwargs)

        tree.compute_model_errors(
            z_error_value=z_kwargs["error_value"],
            z_error_type=z_kwargs["error_type"],
            z_floor=z_kwargs["floor"],
            t_error_value=t_kwargs["error_value"],
            t_error_type=t_kwargs["error_type"],
            t_floor=t_kwargs["floor"],
        )

        expected = mt_obj._transfer_function["transfer_function_model_error"].values
        actual = tree.get_station(station_path)["transfer_function_model_error"].values

        assert np.allclose(actual, expected, equal_nan=True)

    def test_estimate_starting_rho_plots_mean_and_median(self, loaded_profile_mt):
        plt = pytest.importorskip("matplotlib.pyplot")

        tree = MTData()
        tree.add_stations([loaded_profile_mt.copy(), loaded_profile_mt.copy()])

        class _FakeAxes:
            def __init__(self):
                self.legend_labels = None
                self.xlim = None

            def loglog(self, x, y, **kwargs):
                class _Line:
                    pass

                return (_Line(),)

            def set_xlabel(self, *_args, **_kwargs):
                return None

            def set_ylabel(self, *_args, **_kwargs):
                return None

            def legend(self, _handles, labels, loc="upper left"):
                self.legend_labels = labels
                return None

            def grid(self, **_kwargs):
                return None

            def set_xlim(self, value):
                self.xlim = value
                return None

        class _FakeFigure:
            def __init__(self, axes):
                self._axes = axes

            def add_subplot(self, *_args, **_kwargs):
                return self._axes

        fake_axes = _FakeAxes()
        show_called = {"value": False}

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(plt, "figure", lambda: _FakeFigure(fake_axes))
        monkeypatch.setattr(
            plt,
            "show",
            lambda: show_called.__setitem__("value", True),
        )

        try:
            tree.estimate_starting_rho()
        finally:
            monkeypatch.undo()

        assert show_called["value"]
        assert fake_axes.legend_labels is not None
        assert fake_axes.legend_labels[0].startswith("Mean =")
        assert fake_axes.legend_labels[1].startswith("Median =")
        assert fake_axes.xlim is not None

    def test_n_stations_counts_nodes(self):
        tree = MTData()
        assert tree.n_stations == 0

        mt_1 = MT()
        mt_1.survey = "survey_a"
        mt_1.station = "station_01"

        mt_2 = MT()
        mt_2.survey = "survey_b"
        mt_2.station = "station_02"

        tree.add_stations([mt_1, mt_2])

        assert tree.n_stations == 2

    def test_add_tf_aliases_add_station(self):
        mt = MT()
        mt.survey = "tf_survey"
        mt.station = "tf_station"

        tree = MTData()
        station_path = tree.add_tf(mt)

        assert station_path == "surveys/tf_survey/stations/tf_station"
        assert tree.n_stations == 1
        assert isinstance(tree.get_station(station_path), xr.Dataset)


class TestMTDataCopy:
    def test_copy_returns_independent_tree(self, loaded_profile_mt):
        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)

        copied = tree.copy()

        copied_ds = copied.get_station(station_path)
        original_ds = tree.get_station(station_path)

        copied_ds.attrs["survey"] = "changed_survey"

        copied_values = copied_ds["transfer_function"].values
        non_zero = np.argwhere(np.abs(copied_values) > 0)
        assert non_zero.size > 0
        idx = tuple(non_zero[0])
        copied_ds["transfer_function"].values[idx] = copied_ds[
            "transfer_function"
        ].values[idx] + (1.0 + 1.0j)

        assert original_ds.attrs["survey"] != "changed_survey"
        assert not np.isclose(
            original_ds["transfer_function"].values[idx],
            copied_ds["transfer_function"].values[idx],
        )

    def test_copy_deepcopies_metadata_cache(self, loaded_profile_mt):
        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)

        copied = tree.copy()

        copied.metadata_cache["survey"][station_path].id = "copied_survey"
        copied.metadata_cache["station"][station_path].id = "copied_station"

        assert tree.metadata_cache["survey"][station_path].id != "copied_survey"
        assert tree.metadata_cache["station"][station_path].id != "copied_station"

    def test_clone_empty_preserves_attrs_without_stations(self, basic_mt):
        tree = MTData(
            metadata_storage="cache",
            dataset_copy_mode="shallow",
            coordinate_reference_frame="ned",
            impedance_units="mt",
        )
        tree.add_station(basic_mt)

        empty = tree.clone_empty()

        assert isinstance(empty, MTData)
        assert empty.metadata_storage == tree.metadata_storage
        assert empty.dataset_copy_mode == tree.dataset_copy_mode
        assert empty.coordinate_reference_frame == tree.coordinate_reference_frame
        assert empty.impedance_units == tree.impedance_units
        assert empty._iter_station_paths() == []
        assert empty.n_stations == 0


class TestMTDataPeriods:
    def test_get_periods_empty_tree(self):
        tree = MTData()
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

        tree = MTData()
        tree.add_station([mt1, mt2])

        periods = tree.get_periods()
        assert np.array_equal(periods, np.array([1.0, 3.0, 5.0, 10.0, 20.0]))

    def test_get_periods_real_data(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        expected = np.unique(
            np.concatenate(
                [np.asarray(mt.period, dtype=float) for mt in loaded_profile_mt_objects]
            )
        )
        expected.sort()

        periods = tree.get_periods()
        assert np.array_equal(periods, expected)


class TestMTDataInterpolation:
    def test_interpolate_matches_single_mt_interpolate(self, loaded_profile_mt):
        mt_obj = loaded_profile_mt.copy()
        source_periods = np.asarray(mt_obj.period, dtype=float)
        target_periods = source_periods[[6, 15, 25, 35]]

        mt_interp = mt_obj.interpolate(
            target_periods, f_type="period", bounds_error=True
        )

        tree = MTData()
        station_path = tree.add_station(mt_obj)
        tree_interp = tree.interpolate(
            target_periods,
            f_type="period",
            bounds_error=True,
            inplace=False,
        )
        tree_ds = tree_interp.get_station(station_path)
        mt_ds = mt_interp._transfer_function

        assert np.array_equal(tree_ds.period.values, mt_ds.period.values)
        for var_name in [
            "transfer_function",
            "transfer_function_error",
            "transfer_function_model_error",
        ]:
            assert var_name in tree_ds
            assert var_name in mt_ds
            assert np.allclose(
                tree_ds[var_name].values,
                mt_ds[var_name].values,
                equal_nan=True,
            )

    def test_interpolate_returns_tree_without_mt_conversion(
        self, loaded_profile_mt, monkeypatch
    ):
        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)
        original_periods = tree.get_station(station_path).period.values.copy()
        target_periods = original_periods[[5, 10, 20]]

        def _fail(_station_ds):
            raise AssertionError("interpolate should not reconstruct full MT objects")

        monkeypatch.setattr(MTData, "_dataset_to_mt", staticmethod(_fail))

        out = tree.interpolate(target_periods, inplace=False)
        out_ds = out.get_station(station_path)

        assert isinstance(out, MTData)
        assert np.array_equal(out_ds.period.values, target_periods)
        assert np.array_equal(
            tree.get_station(station_path).period.values, original_periods
        )
        assert (
            out.metadata_cache["survey"][station_path]
            is loaded_profile_mt.survey_metadata
        )
        assert (
            out.metadata_cache["station"][station_path]
            is loaded_profile_mt.station_metadata
        )

    def test_interpolate_bounds_error_filters_to_station_range(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        station_periods = tree.get_station(station_path).period.values
        target_periods = np.array(
            [
                station_periods.min() / 10.0,
                station_periods[12],
                station_periods.max() * 10.0,
            ]
        )

        out = tree.interpolate(target_periods, inplace=False, bounds_error=True)

        assert np.array_equal(
            out.get_station(station_path).period.values, np.array([station_periods[12]])
        )

    def test_interpolate_inplace_updates_station_dataset(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        target_periods = tree.get_station(station_path).period.values[[3, 7, 11]]

        result = tree.interpolate(target_periods, inplace=True)

        assert result is None
        assert np.array_equal(
            tree.get_station(station_path).period.values, target_periods
        )

    def test_interpolate_lazy_defers_materialization(self, loaded_profile_mt):
        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)
        original_periods = tree.get_station(station_path).period.values.copy()
        target_periods = original_periods[[4, 9, 13]]

        lazy_tree = tree.interpolate_lazy(target_periods, inplace=False)

        assert isinstance(lazy_tree, MTData)
        assert lazy_tree.is_lazy
        assert lazy_tree.lazy_station_count == 1
        assert np.array_equal(
            lazy_tree.tree[station_path].ds.period.values,
            original_periods,
        )

        lazy_tree.compute()

        assert not lazy_tree.is_lazy
        assert lazy_tree.lazy_station_count == 0
        assert np.array_equal(
            lazy_tree.get_station(station_path).period.values, target_periods
        )

    def test_interpolate_lazy_matches_eager_after_compute(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        target_periods = tree.get_station(station_path).period.values[[5, 10, 15]]

        eager_tree = tree.interpolate(target_periods, inplace=False)
        lazy_tree = tree.interpolate_lazy(target_periods, inplace=False)
        lazy_tree.compute()

        eager_ds = eager_tree.get_station(station_path)
        lazy_ds = lazy_tree.get_station(station_path)

        assert np.array_equal(lazy_ds.period.values, eager_ds.period.values)
        for var_name in [
            "transfer_function",
            "transfer_function_error",
            "transfer_function_model_error",
        ]:
            assert np.allclose(
                lazy_ds[var_name].values,
                eager_ds[var_name].values,
                equal_nan=True,
            )

    def test_interpolate_dask_matches_eager(self, loaded_profile_mt):
        pytest.importorskip("dask")

        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        target_periods = tree.get_station(station_path).period.values[[6, 11, 19]]

        eager_tree = tree.interpolate(target_periods, inplace=False)
        dask_tree = tree.interpolate_dask(
            target_periods,
            chunks={"period": 16},
            compute=True,
            inplace=False,
        )

        eager_ds = eager_tree.get_station(station_path)
        dask_ds = dask_tree.get_station(station_path)
        assert np.array_equal(dask_ds.period.values, eager_ds.period.values)
        for var_name in [
            "transfer_function",
            "transfer_function_error",
            "transfer_function_model_error",
        ]:
            assert np.allclose(
                dask_ds[var_name].values,
                eager_ds[var_name].values,
                equal_nan=True,
            )


class TestMTDataRotation:
    def test_rotate_matches_single_mt_rotate(self, loaded_profile_mt):
        mt_obj = loaded_profile_mt.copy()
        rotation_angle = 13.0

        mt_rot = mt_obj.rotate(rotation_angle, inplace=False)

        tree = MTData()
        station_path = tree.add_station(mt_obj)
        tree_rot = tree.rotate(rotation_angle, inplace=False)

        tree_ds = tree_rot.get_station(station_path)
        mt_ds = mt_rot._transfer_function

        for var_name in [
            "transfer_function",
            "transfer_function_error",
            "transfer_function_model_error",
        ]:
            assert var_name in tree_ds
            assert var_name in mt_ds
            assert np.allclose(
                tree_ds[var_name].values,
                mt_ds[var_name].values,
                equal_nan=True,
            )

    def test_rotate_returns_tree_without_mt_conversion(
        self, loaded_profile_mt, monkeypatch
    ):
        tree = MTData(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)
        original_ds = tree.get_station(station_path).copy(deep=True)
        rotation_angle = 22.5

        def _fail(_station_ds):
            raise AssertionError("rotate should not reconstruct full MT objects")

        monkeypatch.setattr(MTData, "_dataset_to_mt", staticmethod(_fail))

        out = tree.rotate(rotation_angle, inplace=False)
        out_ds = out.get_station(station_path)

        assert isinstance(out, MTData)
        assert not np.allclose(
            out_ds["transfer_function"].values,
            original_ds["transfer_function"].values,
            equal_nan=True,
        )
        assert np.allclose(
            tree.get_station(station_path)["transfer_function"].values,
            original_ds["transfer_function"].values,
            equal_nan=True,
        )
        assert (
            out.metadata_cache["survey"][station_path]
            is loaded_profile_mt.survey_metadata
        )
        assert (
            out.metadata_cache["station"][station_path]
            is loaded_profile_mt.station_metadata
        )

    def test_rotate_inplace_updates_station_dataset(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        before = tree.get_station(station_path).copy(deep=True)

        result = tree.rotate(17.5, inplace=True)

        assert result is None
        after = tree.get_station(station_path)
        assert np.array_equal(after.period.values, before.period.values)
        assert not np.allclose(
            after["transfer_function"].values,
            before["transfer_function"].values,
            equal_nan=True,
        )

    def test_rotate_dask_matches_eager(self, loaded_profile_mt):
        pytest.importorskip("dask")

        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        rotation_angle = 14.0

        eager_tree = tree.rotate(rotation_angle, inplace=False)
        dask_tree = tree.rotate_dask(
            rotation_angle,
            chunks={"period": 16},
            compute=True,
            inplace=False,
        )

        eager_ds = eager_tree.get_station(station_path)
        dask_ds = dask_tree.get_station(station_path)

        for var_name in [
            "transfer_function",
            "transfer_function_error",
            "transfer_function_model_error",
        ]:
            assert np.allclose(
                dask_ds[var_name].values,
                eager_ds[var_name].values,
                equal_nan=True,
            )


class TestMTDataDaskMethods:
    def test_as_dask_and_chunk_plan(self, loaded_profile_mt):
        pytest.importorskip("dask")

        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        dask_tree = tree.as_dask(chunks={"period": 16}, inplace=False)

        assert dask_tree.is_dask_backed([station_path])
        plan = dask_tree.chunk_plan([station_path])
        assert plan[station_path]["transfer_function"] is not None

    def test_rechunk_updates_chunk_layout(self, loaded_profile_mt):
        pytest.importorskip("dask")

        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        tree = tree.as_dask(chunks={"period": 20}, inplace=False)

        tree.rechunk(chunks={"period": 10}, inplace=True)

        plan = tree.chunk_plan([station_path])
        tf_chunks = plan[station_path]["transfer_function"]
        assert tf_chunks is not None
        assert max(tf_chunks[0]) <= 10

    def test_map_stations_lazy_then_compute(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        lazy_tree = tree.map_stations(
            lambda ds: ds.isel(period=slice(0, 5)),
            lazy=True,
            inplace=False,
        )

        assert lazy_tree.is_lazy
        lazy_tree.compute()
        assert not lazy_tree.is_lazy
        assert lazy_tree.get_station(station_path).sizes["period"] == 5

    def test_map_stations_eager_delegates_to_tree_accessor(
        self, loaded_profile_mt, monkeypatch
    ):
        import mtpy.core.mt_data_accessor as mt_data_accessor

        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        called = {"value": False}

        original_map = mt_data_accessor.MTDataTreeAccessor.map_stations

        def _wrapped(self, transform, station_paths=None, inplace=False):
            called["value"] = True
            return original_map(
                self,
                transform,
                station_paths=station_paths,
                inplace=inplace,
            )

        monkeypatch.setattr(
            mt_data_accessor.MTDataTreeAccessor,
            "map_stations",
            _wrapped,
        )

        out = tree.map_stations(
            lambda ds: ds.isel(period=slice(0, 4)),
            lazy=False,
            inplace=False,
        )

        assert called["value"] is True
        assert out.get_station(station_path).sizes["period"] == 4


class TestMTDataSpatialFiltering:
    def test_station_locations_returns_dataframe_without_mt_conversion(
        self, loaded_profile_mt_objects, monkeypatch
    ):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        def _fail(_station_ds):
            raise AssertionError(
                "station_locations should not reconstruct full MT objects"
            )

        monkeypatch.setattr(MTData, "_dataset_to_mt", staticmethod(_fail))

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

        tree = MTData()
        tree.add_station([mt1, mt2])

        subset = tree.apply_bounding_box(-121.0, -119.0, 39.0, 41.0)

        assert isinstance(subset, MTData)
        assert subset._path_exists("surveys/s1/stations/inside")
        assert not subset._path_exists("surveys/s1/stations/outside")

    def test_apply_bounding_box_real_data(self, loaded_profile_mt_objects):
        tree = MTData()
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
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        def _fail(_station_ds):
            raise AssertionError(
                "apply_bounding_box should not reconstruct full MT objects"
            )

        monkeypatch.setattr(MTData, "_dataset_to_mt", staticmethod(_fail))

        lon_values = [mt.longitude for mt in loaded_profile_mt_objects]
        lat_values = [mt.latitude for mt in loaded_profile_mt_objects]
        subset = tree.apply_bounding_box(
            min(lon_values), max(lon_values), min(lat_values), max(lat_values)
        )

        assert len(subset.station_locations) == len(loaded_profile_mt_objects)


class TestMTDataDataFrames:
    def test_to_dataframe_returns_concatenated_station_data(
        self, loaded_profile_mt_objects
    ):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        df = tree.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert set(df.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }
        assert np.array_equal(np.sort(df.period.unique()), tree.get_periods())

    def test_to_mt_dataframe_returns_mt_dataframe(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        mt_df = tree.to_mt_dataframe()

        assert isinstance(mt_df, MTDataFrame)
        assert set(mt_df.dataframe.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }

    def test_from_dataframe_populates_tree(self, loaded_profile_mt_objects):
        source_tree = MTData()
        source_tree.add_station(loaded_profile_mt_objects)
        df = source_tree.to_dataframe()

        tree = MTData()
        tree.from_dataframe(df)

        assert len(tree._iter_station_paths()) == len(loaded_profile_mt_objects)
        assert np.array_equal(tree.get_periods(), source_tree.get_periods())

    def test_from_mt_dataframe_populates_tree(self, loaded_profile_mt_objects):
        source_tree = MTData()
        source_tree.add_station(loaded_profile_mt_objects)
        mt_df = source_tree.to_mt_dataframe()

        tree = MTData()
        tree.from_mt_dataframe(mt_df)

        assert len(tree._iter_station_paths()) == len(loaded_profile_mt_objects)
        assert np.array_equal(tree.get_periods(), source_tree.get_periods())

    def test_to_dataframe_empty_tree_returns_empty_dataframe(self):
        tree = MTData()

        df = tree.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestMTDataModEM:
    def test_to_modem_returns_data_object(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        modem_data = tree.to_modem()

        assert isinstance(modem_data, Data)
        assert np.isclose(modem_data.center_point.latitude, tree.center_point.latitude)
        assert np.isclose(
            modem_data.center_point.longitude,
            tree.center_point.longitude,
        )
        assert set(modem_data.dataframe.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }
        assert (
            modem_data.z_model_error.error_parameters
            == tree.z_model_error.error_parameters
        )
        assert (
            modem_data.t_model_error.error_parameters
            == tree.t_model_error.error_parameters
        )

    def test_to_modem_writes_file(self, loaded_profile_mt_objects, tmp_path):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)
        out_fn = tmp_path / "tree_modem.dat"

        modem_data = tree.to_modem(data_filename=out_fn)

        assert isinstance(modem_data, Data)
        assert out_fn.exists()

    def test_from_modem_populates_tree_and_metadata(
        self, loaded_profile_mt_objects, tmp_path
    ):
        source_tree = MTData()
        source_tree.add_station(loaded_profile_mt_objects)
        source_tree.model_parameters = {"inv_mode": "2", "formatting": "1"}
        out_fn = tmp_path / "roundtrip_modem.dat"
        modem_data = source_tree.to_modem(data_filename=out_fn)

        tree = MTData()
        tree.from_modem(out_fn, survey="tree_modem")

        assert tree.survey_ids == ["tree_modem"]
        assert tree.n_stations == len(loaded_profile_mt_objects)
        assert tree.data_rotation_angle == 0
        assert np.isclose(tree.center_point.latitude, modem_data.center_point.latitude)
        assert np.isclose(
            tree.center_point.longitude,
            modem_data.center_point.longitude,
        )
        assert "inv_mode" in tree.model_parameters
        assert "formatting" in tree.model_parameters


class TestMTDataOccam2D:
    def test_to_occam2d_returns_occam2d_data_object(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        occam_data = tree.to_occam2d()

        assert isinstance(occam_data, Occam2DData)
        assert set(occam_data.dataframe.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }

    def test_to_occam2d_writes_file(self, loaded_profile_mt_objects, tmp_path):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)
        out_fn = tmp_path / "tree_occam.dat"

        occam_data = tree.to_occam2d(data_filename=out_fn)

        assert isinstance(occam_data, Occam2DData)
        assert out_fn.exists()

    def test_from_occam2d_populates_tree(self, loaded_profile_mt_objects, tmp_path):
        source_tree = MTData()
        source_tree.add_station(loaded_profile_mt_objects)
        out_fn = tmp_path / "roundtrip_occam.dat"
        source_tree.to_occam2d(data_filename=out_fn)

        tree = MTData()
        tree.from_occam2d(out_fn, survey="data")

        assert tree.survey_ids == ["data"]
        assert tree.n_stations == len(loaded_profile_mt_objects)
        assert "profile_origin" in tree.model_parameters
        assert "profile_angle" in tree.model_parameters
        assert "model_mode" in tree.model_parameters


class TestMTDataSimPEG:
    def test_to_simpeg_2d_returns_object(self, loaded_profile_mt_objects):
        from mtpy.modeling.simpeg.data_2d import Simpeg2DData

        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        simpeg_data = tree.to_simpeg_2d()

        assert isinstance(simpeg_data, Simpeg2DData)
        assert set(simpeg_data.dataframe.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }

    def test_to_simpeg_2d_kwargs_applied(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        simpeg_data = tree.to_simpeg_2d(include_elevation=False, invert_tm=False)

        assert simpeg_data.include_elevation is False
        assert simpeg_data.invert_tm is False

    def test_to_simpeg_3d_returns_object(self, loaded_profile_mt_objects):
        from mtpy.modeling.simpeg.data_3d import Simpeg3DData

        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        simpeg_data = tree.to_simpeg_3d()

        assert isinstance(simpeg_data, Simpeg3DData)
        assert set(simpeg_data.dataframe.station.unique()) == {
            mt.station for mt in loaded_profile_mt_objects
        }

    def test_to_simpeg_3d_kwargs_applied(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        simpeg_data = tree.to_simpeg_3d(include_elevation=True, invert_z_yy=False)

        assert simpeg_data.include_elevation is True
        assert simpeg_data.invert_z_yy is False


class TestMTDataAddWhiteNoise:
    def test_add_white_noise_inplace_changes_data(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        before = tree.to_dataframe()
        before_z = before[["z_xx", "z_xy", "z_yx", "z_yy"]].values.copy()

        tree.add_white_noise(0.05)

        after = tree.to_dataframe()
        after_z = after[["z_xx", "z_xy", "z_yx", "z_yy"]].values

        assert not np.allclose(before_z, after_z, equal_nan=True)

    def test_add_white_noise_inplace_returns_none(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        result = tree.add_white_noise(5)

        assert result is None

    def test_add_white_noise_not_inplace_returns_new_tree(
        self, loaded_profile_mt_objects
    ):
        tree = MTData()
        tree.add_station(loaded_profile_mt_objects)

        before_z = tree.to_dataframe()[["z_xx", "z_xy", "z_yx", "z_yy"]].values.copy()

        new_tree = tree.add_white_noise(0.05, inplace=False)

        assert isinstance(new_tree, MTData)
        assert new_tree.n_stations == tree.n_stations
        after_z = new_tree.to_dataframe()[["z_xx", "z_xy", "z_yx", "z_yy"]].values
        assert not np.allclose(before_z, after_z, equal_nan=True)
        # original tree must be unchanged
        orig_z = tree.to_dataframe()[["z_xx", "z_xy", "z_yx", "z_yy"]].values
        assert np.allclose(before_z, orig_z, equal_nan=True)


class TestMTDataPlottingCompatibility:
    def test_resolve_plot_station_key_with_station_id_and_survey(
        self, loaded_profile_mt
    ):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        resolved = tree._resolve_plot_station_key(
            station_id=loaded_profile_mt.station,
            survey_id=loaded_profile_mt.survey,
        )

        assert resolved == station_path

    def test_resolve_plot_station_key_ambiguous_without_survey(self):
        mt_1 = MT()
        mt_1.survey = "survey_01"
        mt_1.station = "same_station"

        mt_2 = MT()
        mt_2.survey = "survey_02"
        mt_2.station = "same_station"

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        with pytest.raises(ValueError, match="Provide survey_id"):
            tree._resolve_plot_station_key(station_id="same_station")

    def test_plot_phase_tensor_uses_mt_object_accessor(
        self, loaded_profile_mt, monkeypatch
    ):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        called = {}

        class _FakeMT:
            def plot_phase_tensor(self, **kwargs):
                called["kwargs"] = kwargs
                return "phase_tensor_plot"

        def _fake_get_station(station_key, as_mt=False):
            called["station_key"] = station_key
            called["as_mt"] = as_mt
            return _FakeMT()

        monkeypatch.setattr(tree, "get_station", _fake_get_station)

        out = tree.plot_phase_tensor(
            station_key=station_path, color="k", backend="matplotlib"
        )

        assert out == "phase_tensor_plot"
        assert called["station_key"] == station_path
        assert called["as_mt"] is True
        assert called["kwargs"] == {"color": "k"}

    def test_plot_phase_tensor_pseudosection_uses_mt_data_argument(
        self, loaded_profile_mt, monkeypatch
    ):
        import mtpy.core.mt_data as mt_data_module

        tree = MTData()
        tree.add_station(loaded_profile_mt)
        other = MTData()

        captured = {}

        def _fake_plotter(*, mt_data, **kwargs):
            captured["mt_data"] = mt_data
            captured["kwargs"] = kwargs
            return "pseudo"

        monkeypatch.setattr(
            mt_data_module, "PlotPhaseTensorPseudoSection", _fake_plotter
        )

        out = tree.plot_phase_tensor_pseudosection(
            mt_data=other, foo=1, backend="matplotlib"
        )

        assert out == "pseudo"
        assert captured["mt_data"] is other
        assert captured["kwargs"] == {"foo": 1}

    def test_plot_stations_applies_bounding_box(self, loaded_profile_mt, monkeypatch):
        import mtpy.core.mt_data as mt_data_module

        tree = MTData()
        tree.add_station(loaded_profile_mt)

        captured = {}

        class _Subset:
            def to_geo_df(self, model_locations=False, data_type="station_locations"):
                captured["model_locations"] = model_locations
                captured["data_type"] = data_type
                return "gdf"

        def _fake_apply_bounding_box(lon_min, lon_max, lat_min, lat_max):
            captured["bbox"] = (lon_min, lon_max, lat_min, lat_max)
            return _Subset()

        def _fake_plot_stations(gdf, **kwargs):
            captured["gdf"] = gdf
            captured["kwargs"] = kwargs
            return "stations_plot"

        monkeypatch.setattr(tree, "apply_bounding_box", _fake_apply_bounding_box)
        monkeypatch.setattr(mt_data_module, "PlotStations", _fake_plot_stations)

        out = tree.plot_stations(
            map_epsg=3857, bounding_box=(0, 1, 2, 3), backend="matplotlib"
        )

        assert out == "stations_plot"
        assert captured["bbox"] == (0, 1, 2, 3)
        assert captured["gdf"] == "gdf"
        assert captured["kwargs"]["map_epsg"] == 3857

    def test_plot_tipper_map_preserves_explicit_kwargs(
        self, loaded_profile_mt, monkeypatch
    ):
        import mtpy.core.mt_data as mt_data_module

        tree = MTData()
        tree.add_station(loaded_profile_mt)

        captured = {}

        def _fake_plotter(*, mt_data, **kwargs):
            captured["mt_data"] = mt_data
            captured["kwargs"] = kwargs
            return "tipper"

        monkeypatch.setattr(mt_data_module, "PlotPhaseTensorMaps", _fake_plotter)

        out = tree.plot_tipper_map(
            plot_pt=True, plot_tipper="abc", backend="matplotlib"
        )

        assert out == "tipper"
        assert captured["mt_data"] is tree
        assert captured["kwargs"]["plot_pt"] is True
        assert captured["kwargs"]["plot_tipper"] == "abc"

    def test_plot_resistivity_phase_wrappers_pass_mt_data(
        self, loaded_profile_mt, monkeypatch
    ):
        import mtpy.core.mt_data as mt_data_module

        tree = MTData()
        tree.add_station(loaded_profile_mt)

        captured = {}

        def _fake_maps(*, mt_data, **kwargs):
            captured["maps"] = (mt_data, kwargs)
            return "maps"

        def _fake_ps(*, mt_data, **kwargs):
            captured["ps"] = (mt_data, kwargs)
            return "ps"

        monkeypatch.setattr(mt_data_module, "PlotResPhaseMaps", _fake_maps)
        monkeypatch.setattr(mt_data_module, "PlotResPhasePseudoSection", _fake_ps)

        out_maps = tree.plot_resistivity_phase_maps(a=1, backend="matplotlib")
        out_ps = tree.plot_resistivity_phase_pseudosections(b=2, backend="matplotlib")

        assert out_maps == "maps"
        assert out_ps == "ps"
        assert captured["maps"][0] is tree
        assert captured["maps"][1] == {"a": 1}
        assert captured["ps"][0] is tree
        assert captured["ps"][1] == {"b": 2}

    def test_plot_penetration_depth_map_passes_mt_data(
        self, loaded_profile_mt, monkeypatch
    ):
        import mtpy.core.mt_data as mt_data_module

        tree = MTData()
        tree.add_station(loaded_profile_mt)

        captured = {}

        def _fake_plotter(*, mt_data, **kwargs):
            captured["mt_data"] = mt_data
            captured["kwargs"] = kwargs
            return "depth_map"

        monkeypatch.setattr(mt_data_module, "PlotPenetrationDepthMap", _fake_plotter)

        out = tree.plot_penetration_depth_map(plot_period=10, backend="matplotlib")

        assert out == "depth_map"
        assert captured["mt_data"] is tree
        assert captured["kwargs"] == {"plot_period": 10}

    def test_plot_residual_phase_tensor_maps_validates_surveys(
        self, loaded_profile_mt, monkeypatch
    ):
        import mtpy.core.mt_data as mt_data_module

        mt_1 = MT()
        mt_1.survey = "survey_a"
        mt_1.station = "st01"

        mt_2 = MT()
        mt_2.survey = "survey_b"
        mt_2.station = "st02"

        tree = MTData()
        tree.add_stations([mt_1, mt_2])

        captured = {}

        def _fake_plotter(mt_data_01, mt_data_02, **kwargs):
            captured["s1"] = mt_data_01
            captured["s2"] = mt_data_02
            captured["kwargs"] = kwargs
            return "residual"

        monkeypatch.setattr(mt_data_module, "PlotResidualPTMaps", _fake_plotter)

        out = tree.plot_residual_phase_tensor_maps("survey_a", "survey_b", c=3)

        assert out == "residual"
        assert captured["s1"].n_stations == 1
        assert captured["s2"].n_stations == 1
        assert captured["kwargs"] == {"c": 3}

        with pytest.raises(KeyError, match="Survey not found"):
            tree.plot_residual_phase_tensor_maps("survey_a", "missing")


class TestMTDataTreeAccessor:
    def test_station_paths_and_short_paths(self, loaded_profile_mt_objects):
        tree = MTData()
        station_paths = tree.add_stations(loaded_profile_mt_objects)

        assert sorted(tree.tree.mt.station_paths) == sorted(station_paths)
        assert len(tree.tree.mt.short_station_paths) == len(station_paths)
        assert sorted(tree.tree.mt.survey_names) == sorted(tree.survey_ids)

    def test_get_station_dataset_from_short_key(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)
        survey = station_path.split("/")[1]
        station = station_path.split("/")[3]

        ds_short = tree.tree.mt.get_station_dataset(f"{survey}/{station}")
        ds_full = tree.get_station(station_path)
        assert ds_short.identical(ds_full)

    def test_set_station_dataset_non_inplace_returns_new_tree(self, loaded_profile_mt):
        tree = MTData()
        station_path = tree.add_station(loaded_profile_mt)

        source_ds = tree.tree.mt.get_station_dataset(station_path, copy=True, deep=True)
        source_ds.attrs["custom_accessor_flag"] = "new"

        out_tree = tree.tree.mt.set_station_dataset(
            station_path,
            source_ds,
            inplace=False,
        )

        assert out_tree is not None
        assert "custom_accessor_flag" not in tree.get_station(station_path).attrs
        assert out_tree[station_path].ds.attrs["custom_accessor_flag"] == "new"

    def test_map_stations_applies_transform(self, loaded_profile_mt_objects):
        tree = MTData()
        tree.add_stations(loaded_profile_mt_objects)

        def _trim(ds):
            return ds.isel(period=slice(0, 3))

        mapped_tree = tree.tree.mt.map_stations(_trim, inplace=False)
        assert mapped_tree is not None
        for station_path in tree.tree.mt.station_paths:
            assert mapped_tree[station_path].ds.sizes["period"] == 3
            assert tree.get_station(station_path).sizes["period"] > 3

    def test_select_stations_returns_subset(self, loaded_profile_mt_objects):
        tree = MTData()
        station_paths = tree.add_stations(loaded_profile_mt_objects)
        selected = station_paths[:1]

        subset_tree = tree.tree.mt.select_stations(selected)

        assert sorted(subset_tree.mt.station_paths) == sorted(selected)
        assert subset_tree.attrs["schema_name"] == tree.tree.attrs["schema_name"]
