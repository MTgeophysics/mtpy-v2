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
        assert tree.metadata_storage == "cache"

    def test_init_applies_custom_attrs(self):
        tree = MTDataTree(coordinate_reference_frame="ned", impedance_units="mt")

        assert tree.tree.attrs["coordinate_reference_frame"] == "NED"
        assert tree.tree.attrs["impedance_units"] == "mt"
        assert tree.attrs is tree.tree.attrs

    def test_coordinate_reference_frame_set_propagates_to_stations(
        self, loaded_profile_mt_objects
    ):
        tree = MTDataTree()
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
        tree = MTDataTree()
        station_path = tree.add_station(loaded_profile_mt)

        tree.impedance_units = "ohm"

        assert tree.impedance_units == "ohm"
        assert tree.tree.attrs["impedance_units"] == "ohm"
        assert tree.get_station(station_path).attrs["impedance_units"] == "ohm"

    def test_coordinate_reference_frame_invalid_raises(self):
        tree = MTDataTree()

        with pytest.raises(ValueError, match="is not understood as a reference frame"):
            tree.coordinate_reference_frame = "bad_frame"

    def test_impedance_units_invalid_raises(self):
        tree = MTDataTree()

        with pytest.raises(ValueError, match="is not an acceptable unit"):
            tree.impedance_units = "bad_unit"


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

    def test_add_station_invalid_dataset_copy_mode_raises(self, basic_mt):
        tree = MTDataTree()
        with pytest.raises(ValueError, match="dataset_copy_mode must be one of"):
            tree.add_station(basic_mt, dataset_copy_mode="bad_mode")

    def test_add_stations_bulk_returns_paths(self):
        mt_1 = MT()
        mt_1.survey = "bulk_survey"
        mt_1.station = "bulk_01"

        mt_2 = MT()
        mt_2.survey = "bulk_survey"
        mt_2.station = "bulk_02"

        tree = MTDataTree()
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

        tree = MTDataTree()
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

        tree = MTDataTree(metadata_storage="cache")
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

        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(mt_existing)

        with pytest.raises(KeyError, match="Station path already exists"):
            tree.add_stations([mt_new], overwrite=False)

        assert (
            tree.metadata_cache["survey"][station_path] is mt_existing.survey_metadata
        )
        assert (
            tree.metadata_cache["station"][station_path] is mt_existing.station_metadata
        )


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

    def test_remove_station(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        assert tree._path_exists(station_path)
        tree.remove_station(station_path)
        assert not tree._path_exists(station_path)

    def test_remove_station_clears_cached_metadata(self):
        mt = MT()
        mt.survey = "remove_cache"
        mt.station = "remove_station"
        mt.survey_metadata.id = "remove_survey_id"
        mt.station_metadata.id = "remove_station_id"

        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(mt)

        tree.remove_station(station_path)

        assert station_path not in tree.metadata_cache["survey"]
        assert station_path not in tree.metadata_cache["station"]

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
        assert len(stations) == len(loaded_profile_mt_objects)
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


class TestMTDataTreeCopy:
    def test_copy_returns_independent_tree(self, loaded_profile_mt):
        tree = MTDataTree(metadata_storage="cache")
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
        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)

        copied = tree.copy()

        copied.metadata_cache["survey"][station_path].id = "copied_survey"
        copied.metadata_cache["station"][station_path].id = "copied_station"

        assert tree.metadata_cache["survey"][station_path].id != "copied_survey"
        assert tree.metadata_cache["station"][station_path].id != "copied_station"


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


class TestMTDataTreeInterpolation:
    def test_interpolate_matches_single_mt_interpolate(self, loaded_profile_mt):
        mt_obj = loaded_profile_mt.copy()
        source_periods = np.asarray(mt_obj.period, dtype=float)
        target_periods = source_periods[[6, 15, 25, 35]]

        mt_interp = mt_obj.interpolate(
            target_periods, f_type="period", bounds_error=True
        )

        tree = MTDataTree()
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
        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)
        original_periods = tree.get_station(station_path).period.values.copy()
        target_periods = original_periods[[5, 10, 20]]

        def _fail(_station_ds):
            raise AssertionError("interpolate should not reconstruct full MT objects")

        monkeypatch.setattr(MTDataTree, "_dataset_to_mt", staticmethod(_fail))

        out = tree.interpolate(target_periods, inplace=False)
        out_ds = out.get_station(station_path)

        assert isinstance(out, MTDataTree)
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
        tree = MTDataTree()
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
        tree = MTDataTree()
        station_path = tree.add_station(loaded_profile_mt)
        target_periods = tree.get_station(station_path).period.values[[3, 7, 11]]

        result = tree.interpolate(target_periods, inplace=True)

        assert result is None
        assert np.array_equal(
            tree.get_station(station_path).period.values, target_periods
        )

    def test_interpolate_lazy_defers_materialization(self, loaded_profile_mt):
        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)
        original_periods = tree.get_station(station_path).period.values.copy()
        target_periods = original_periods[[4, 9, 13]]

        lazy_tree = tree.interpolate_lazy(target_periods, inplace=False)

        assert isinstance(lazy_tree, MTDataTree)
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
        tree = MTDataTree()
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

        tree = MTDataTree()
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


class TestMTDataTreeRotation:
    def test_rotate_matches_single_mt_rotate(self, loaded_profile_mt):
        mt_obj = loaded_profile_mt.copy()
        rotation_angle = 13.0

        mt_rot = mt_obj.rotate(rotation_angle, inplace=False)

        tree = MTDataTree()
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
        tree = MTDataTree(metadata_storage="cache")
        station_path = tree.add_station(loaded_profile_mt)
        original_ds = tree.get_station(station_path).copy(deep=True)
        rotation_angle = 22.5

        def _fail(_station_ds):
            raise AssertionError("rotate should not reconstruct full MT objects")

        monkeypatch.setattr(MTDataTree, "_dataset_to_mt", staticmethod(_fail))

        out = tree.rotate(rotation_angle, inplace=False)
        out_ds = out.get_station(station_path)

        assert isinstance(out, MTDataTree)
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
        tree = MTDataTree()
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

        tree = MTDataTree()
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


class TestMTDataTreeDaskMethods:
    def test_as_dask_and_chunk_plan(self, loaded_profile_mt):
        pytest.importorskip("dask")

        tree = MTDataTree()
        station_path = tree.add_station(loaded_profile_mt)

        dask_tree = tree.as_dask(chunks={"period": 16}, inplace=False)

        assert dask_tree.is_dask_backed([station_path])
        plan = dask_tree.chunk_plan([station_path])
        assert plan[station_path]["transfer_function"] is not None

    def test_rechunk_updates_chunk_layout(self, loaded_profile_mt):
        pytest.importorskip("dask")

        tree = MTDataTree()
        station_path = tree.add_station(loaded_profile_mt)
        tree = tree.as_dask(chunks={"period": 20}, inplace=False)

        tree.rechunk(chunks={"period": 10}, inplace=True)

        plan = tree.chunk_plan([station_path])
        tf_chunks = plan[station_path]["transfer_function"]
        assert tf_chunks is not None
        assert max(tf_chunks[0]) <= 10

    def test_map_stations_lazy_then_compute(self, loaded_profile_mt):
        tree = MTDataTree()
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
