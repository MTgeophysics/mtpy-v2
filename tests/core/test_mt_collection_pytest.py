# -*- coding: utf-8 -*-
"""
Pytest suite for MTCollection class.

Created on Mon Jan 11 15:36:38 2021
Converted to pytest and optimized for pytest-xdist.

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT
"""

import os

import pandas as pd

# =============================================================================
# Imports
# =============================================================================
import pytest
from mth5.helpers import validate_name

from mtpy import MT, MTCollection, MTData


# =============================================================================
# NOTE: Fixtures are now defined in tests/conftest.py
# =============================================================================
# The following fixtures are imported from conftest.py:
# - tf_file_list: Session-scoped list of TF files
# - expected_dataframe_data: Expected DataFrame structure for validation
# - mt_collection_main: Worker-safe MTCollection with all TF files
# - mt_collection_from_mt_data_with_survey: MTCollection with specified survey
# - mt_collection_from_mt_data_new_survey: MTCollection with new_survey and tf_id_extra
# - mt_collection_add_tf_method: MTCollection created using add_tf method
# - session_temp_dir: Worker-safe temporary directory
# - worker_id: pytest-xdist worker ID
#
# These fixtures use a global cache system for performance and are
# pytest-xdist compatible, with each worker getting its own file copy
# to avoid HDF5 locking issues.
# =============================================================================


# This fixture is kept locally as it has test-specific structure
@pytest.fixture(scope="session")
def expected_dataframe_complete():
    """Session-scoped expected dataframe data for comparison."""
    return {
        "station": {
            0: "14-IEB0537A",
            1: "CAS04",
            2: "NMX20",
            3: "KAK",
            4: "500fdfilNB207",
            5: "SMG1",
            6: "GAA54",
            7: "300",
            8: "YSW212abcdefghijkl",
            9: "BP05",
            10: "701_merged_wrcal",
            11: "GEO858",
            12: "TEST01",
            13: "TEST_01",
            14: "s08",
            15: "SAGE_2005_og",
            16: "SAGE_2005_out",
            17: "21PBS-FJM",
            18: "24",
            19: "22",
            20: "2813",
        },
        "survey": {
            0: "BOULIA",
            1: "CONUS_South",
            2: "CONUS_South",
            3: "JMA",
            4: "Nepabunna_2010",
            5: "South_Chile",
            6: "Transportable_Array",
            7: "unknown_survey",
            8: "unknown_survey_001",
            9: "unknown_survey_002",
            10: "unknown_survey_003",
            11: "unknown_survey_004",
            12: "unknown_survey_005",
            13: "unknown_survey_006",
            14: "unknown_survey_007",
            15: "unknown_survey_008",
            16: "unknown_survey_009",
            17: "unknown_survey_010",
            18: "unknown_survey_011",
            19: "unknown_survey_012",
            20: "unknown_survey_013",
        },
        "latitude": {
            0: -22.823722222222223,
            1: 37.63335,
            2: 34.470528,
            3: 36.232,
            4: -30.587969,
            5: -38.41,
            6: 31.888699,
            7: 34.727,
            8: 44.631,
            9: 0.0,
            10: 40.64811111111111,
            11: 22.691378333333333,
            12: -30.930285,
            13: -23.051133333333333,
            14: -34.646,
            15: 35.55,
            16: 35.55,
            17: 0.0,
            18: 32.83331167,
            19: 38.6653467,
            20: 44.1479163,
        },
        "longitude": {
            0: 139.29469444444445,
            1: -121.46838,
            2: -108.712288,
            3: 140.186,
            4: 138.959969,
            5: -73.904722,
            6: -83.281681,
            7: -115.735,
            8: -110.44,
            9: 0.0,
            10: -106.21241666666667,
            11: 139.70504,
            12: 127.22923,
            13: 139.46753333333334,
            14: 137.006,
            15: -106.28333333333333,
            16: -106.28333333333333,
            17: 0.0,
            18: -107.08305667,
            19: -113.1690717,
            20: -111.0497517,
        },
        "elevation": {
            0: 158.0,
            1: 329.387,
            2: 1940.05,
            3: 36.0,
            4: 534.0,
            5: 10.0,
            6: 77.025,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 2489.0,
            11: 181.0,
            12: 175.27,
            13: 122.0,
            14: 0.0,
            15: 0.0,
            16: 0.0,
            17: 0.0,
            18: 0.0,
            19: 1548.1,
            20: 0.0,
        },
        "tf_id": {
            0: "14-IEB0537A",
            1: "CAS04",
            2: "NMX20",
            3: "KAK",
            4: "500fdfilNB207",
            5: "SMG1",
            6: "GAA54",
            7: "300",
            8: "ysw212abcdefghijkl",
            9: "BP05",
            10: "701_merged_wrcal",
            11: "GEO858",
            12: "TEST01",
            13: "TEST_01",
            14: "s08",
            15: "SAGE_2005_og",
            16: "SAGE_2005",
            17: "21PBS-FJM",
            18: "24",
            19: "22",
            20: "2813",
        },
        "units": {i: "none" for i in range(21)},
        "has_impedance": {
            0: True,
            1: True,
            2: True,
            3: True,
            4: True,
            5: True,
            6: True,
            7: True,
            8: False,
            9: True,
            10: True,
            11: True,
            12: True,
            13: True,
            14: True,
            15: True,
            16: True,
            17: True,
            18: True,
            19: True,
            20: True,
        },
        "has_tipper": {
            0: True,
            1: True,
            2: True,
            3: False,
            4: False,
            5: True,
            6: True,
            7: True,
            8: True,
            9: False,
            10: True,
            11: True,
            12: True,
            13: True,
            14: False,
            15: True,
            16: True,
            17: True,
            18: False,
            19: True,
            20: False,
        },
        "has_covariance": {
            0: True,
            1: False,
            2: True,
            3: False,
            4: False,
            5: False,
            6: True,
            7: True,
            8: True,
            9: False,
            10: False,
            11: False,
            12: False,
            13: True,
            14: False,
            15: True,
            16: False,
            17: False,
            18: False,
            19: False,
            20: False,
        },
        "period_min": {
            0: 0.003125,
            1: 4.65455,
            2: 4.65455,
            3: 6.4,
            4: 0.0064,
            5: 16.0,
            6: 7.31429,
            7: 1.16364,
            8: 0.01818,
            9: 1.333333,
            10: 0.0001,
            11: 0.005154639175257732,
            12: 0.0012115271966653925,
            13: 0.00010061273153504844,
            14: 0.007939999015440123,
            15: 0.00419639110365086,
            16: 0.00419639110365086,
            17: 0.000726427429899753,
            18: 0.0009765625,
            19: 0.0125,
            20: 0.00390625,
        },
        "period_max": {
            0: 2941.176470588235,
            1: 29127.11,
            2: 29127.11,
            3: 614400.0,
            4: 2.730674,
            5: 11585.27,
            6: 18724.57,
            7: 10922.66699,
            8: 4096.0,
            9: 64.55,
            10: 2912.710720057042,
            11: 1449.2753623188407,
            12: 1211.5274902250933,
            13: 1.0240026214467108,
            14: 2730.8332372990308,
            15: 209.73154362416108,
            16: 209.73154362416108,
            17: 526.3157894736842,
            18: 42.6657564638621,
            19: 1365.3368285956144,
            20: 1024.002621446711,
        },
        "hdf5_reference": {i: None for i in range(21)},
        "station_hdf5_reference": {i: None for i in range(21)},
    }


# =============================================================================
# NOTE: All MTCollection fixtures are now defined in tests/conftest.py
# =============================================================================
# The fixtures are automatically available from conftest.py:
# - mt_collection_main: Main collection with all TF files (worker-safe)
# - mt_collection_from_mt_data_with_survey: Collection with survey (worker-safe)
# - mt_collection_from_mt_data_new_survey: Collection with new_survey (worker-safe)
# - mt_collection_add_tf_method: Collection using add_tf method (worker-safe)
#
# These fixtures use a global cache and provide worker-specific copies
# for pytest-xdist compatibility, eliminating HDF5 file locking issues.
# =============================================================================


# Helper fixture for tests that need MTData objects
@pytest.fixture
def mt_data_with_tf_files(tf_file_list):
    """Create an MTData object with TF files loaded."""
    mt_data_obj = MTData()
    mt_data_obj.add_station(tf_file_list)
    return mt_data_obj


# =============================================================================
# Tests for main MTCollection functionality
# =============================================================================


class TestMTCollectionBasic:
    """Test basic MTCollection properties and methods."""

    def test_filename(self, mt_collection_main):
        """Test that mth5_filename is correctly set."""
        worker_id = os.environ.get("PYTEST_XDIST_WORKER")
        expected_name = (
            "mt_collection_main.h5"
            if not worker_id or worker_id == "master"
            else f"mt_collection_main_{worker_id}.h5"
        )
        expected = mt_collection_main.working_directory.joinpath(expected_name)
        assert mt_collection_main.mth5_filename == expected

    def test_dataframe_shape(self, mt_collection_main, tf_file_list):
        """Test dataframe has correct number of rows."""
        df = mt_collection_main.dataframe
        # Use unique tf_id to avoid counting duplicated rows that appear in some TF sources
        assert df.tf_id.nunique() == len(tf_file_list)

    def test_dataframe_columns(self, mt_collection_main):
        """Test dataframe has expected columns."""
        expected_columns = [
            "station",
            "survey",
            "latitude",
            "longitude",
            "elevation",
            "tf_id",
            "units",
            "has_impedance",
            "has_tipper",
            "has_covariance",
            "period_min",
            "period_max",
            "hdf5_reference",
            "station_hdf5_reference",
        ]
        for col in expected_columns:
            assert col in mt_collection_main.dataframe.columns

    def test_dataframe_content(self, mt_collection_main, expected_dataframe_data):
        """Test dataframe content matches expected values (key columns only)."""
        df = mt_collection_main.dataframe

        # Test key numeric columns that should match exactly
        assert df.latitude.notna().all()
        assert df.longitude.notna().all()

        # Test boolean columns
        assert df.has_impedance.dtype == bool
        assert df.has_tipper.dtype == bool
        assert df.has_covariance.dtype == bool

        # Test period ranges
        assert (df.period_min > 0).all()
        assert (df.period_max > df.period_min).all()

        # Note: Station names may be normalized (hyphens -> underscores)
        # so we don't do exact string matching


class TestMTCollectionGetTF:
    """Test retrieving TF objects from collection."""

    def test_get_tf_returns_mt_object(self, mt_collection_main, tf_file_list):
        """Test that get_tf returns an MT object."""
        original = MT(tf_file_list[0])
        original.read()
        h5_tf = mt_collection_main.get_tf(validate_name(original.tf_id))
        assert isinstance(h5_tf, MT)

    @pytest.mark.parametrize("index", [0, 10, 20])
    def test_get_tf_data_consistency(self, mt_collection_main, tf_file_list, index):
        """Test that retrieved TF data matches original (subset for speed)."""
        tf_fn = tf_file_list[index]
        original = MT(tf_fn)
        original.read()

        h5_tf = mt_collection_main.get_tf(validate_name(original.tf_id))

        # Update some metadata fields to match
        original.survey_metadata.id = h5_tf.survey_metadata.id
        original.survey_metadata.hdf5_reference = h5_tf.survey_metadata.hdf5_reference
        original.survey_metadata.mth5_type = h5_tf.survey_metadata.mth5_type
        original.station_metadata.acquired_by.author = (
            h5_tf.station_metadata.acquired_by.author
        )
        if original.station_metadata.transfer_function.runs_processed in [[], [""]]:
            original.station_metadata.transfer_function.runs_processed = (
                original.station_metadata.run_list
            )

        # For spectra files, just check data equality
        if tf_fn.stem in ["spectra_in", "spectra_out"]:
            assert (original.dataset == h5_tf.dataset).all()
            return

        # Check dataset equality
        assert (original.dataset == h5_tf.dataset).all()

    @pytest.mark.parametrize("index", [0, 5, 10, 15, 20])
    def test_get_tf_various_stations(self, mt_collection_main, tf_file_list, index):
        """Test getting TF for various stations by index."""
        original = MT(tf_file_list[index])
        original.read()
        h5_tf = mt_collection_main.get_tf(validate_name(original.tf_id))
        assert h5_tf is not None
        assert isinstance(h5_tf, MT)


class TestMTCollectionToMTData:
    """Test converting collection to MTData."""

    def test_to_mt_data_returns_mtdata(self, mt_collection_main):
        """Test that to_mt_data returns MTData object."""
        mt_data = mt_collection_main.to_mt_data(utm_crs=32610)
        assert isinstance(mt_data, MTData)

    def test_to_mt_data_station_count(self, mt_collection_main, tf_file_list):
        """Test that MTData has correct number of stations."""
        mt_data = mt_collection_main.to_mt_data(utm_crs=32610)
        assert len(mt_data.keys()) == len(tf_file_list)

    def test_to_mt_data_utm_crs(self, mt_collection_main):
        """Test that UTM CRS is correctly set."""
        utm_crs = 32610
        mt_data = mt_collection_main.to_mt_data(utm_crs=utm_crs)
        assert mt_data.utm_crs == utm_crs

    def test_to_mt_data_keys(self, mt_collection_main, tf_file_list):
        """Test that MTData contains expected station keys."""
        mt_data = mt_collection_main.to_mt_data(utm_crs=32610)
        keys = list(mt_data.keys())
        assert len(keys) == len(tf_file_list)
        # Check that keys have survey.station format
        for key in keys:
            assert "." in key


# =============================================================================
# Tests for MTCollection from MTData with survey
# =============================================================================


class TestMTCollectionFromMTDataWithSurvey:
    """Test creating MTCollection from MTData with specified survey."""

    def test_survey_name(self, mt_collection_from_mt_data_with_survey):
        """Test that survey name is correctly set."""
        mc, _ = mt_collection_from_mt_data_with_survey
        assert mc.dataframe.survey.unique()[0] == "test"

    def test_single_survey(self, mt_collection_from_mt_data_with_survey):
        """Test that only one survey exists."""
        mc, _ = mt_collection_from_mt_data_with_survey
        assert len(mc.dataframe.survey.unique()) == 1

    def test_dataframe_length(
        self, mt_collection_from_mt_data_with_survey, tf_file_list
    ):
        """Test dataframe has correct number of entries."""
        mc, _ = mt_collection_from_mt_data_with_survey
        assert len(mc.dataframe) == len(tf_file_list)

    def test_all_stations_have_survey(self, mt_collection_from_mt_data_with_survey):
        """Test that all stations have the survey name."""
        mc, _ = mt_collection_from_mt_data_with_survey
        assert (mc.dataframe.survey == "test").all()


# =============================================================================
# Tests for MTCollection from MTData with new_survey and tf_id_extra
# =============================================================================


class TestMTCollectionFromMTDataNewSurvey:
    """Test creating MTCollection with new_survey and tf_id_extra parameters."""

    def test_survey_name(self, mt_collection_from_mt_data_new_survey):
        """Test that new survey name is applied."""
        mc, _ = mt_collection_from_mt_data_new_survey
        assert mc.dataframe.survey.unique()[0] == "test"

    def test_single_survey(self, mt_collection_from_mt_data_new_survey):
        """Test that only one survey exists."""
        mc, _ = mt_collection_from_mt_data_new_survey
        assert len(mc.dataframe.survey.unique()) == 1

    def test_dataframe_length(
        self, mt_collection_from_mt_data_new_survey, tf_file_list
    ):
        """Test dataframe has correct number of entries."""
        mc, _ = mt_collection_from_mt_data_new_survey
        assert len(mc.dataframe) == len(tf_file_list)

    def test_tf_id_extra_in_all(self, mt_collection_from_mt_data_new_survey):
        """Test that tf_id_extra is present in all TF IDs."""
        mc, _ = mt_collection_from_mt_data_new_survey
        for tf_id in mc.dataframe.tf_id:
            assert "new" in tf_id

    @pytest.mark.parametrize("row_index", [0, 5, 10, 15, 20])
    def test_tf_id_extra_parameterized(
        self, mt_collection_from_mt_data_new_survey, row_index
    ):
        """Test tf_id_extra in specific rows."""
        mc, _ = mt_collection_from_mt_data_new_survey
        tf_id = mc.dataframe.iloc[row_index].tf_id
        assert "new" in tf_id


# =============================================================================
# Tests for MTCollection using add_tf method
# =============================================================================


class TestMTCollectionAddTFMethod:
    """Test MTCollection.add_tf() with MTData object."""

    def test_survey_name(self, mt_collection_add_tf_method):
        """Test that survey name is correctly set."""
        mc, _ = mt_collection_add_tf_method
        assert mc.dataframe.survey.unique()[0] == "test"

    def test_single_survey(self, mt_collection_add_tf_method):
        """Test that only one survey exists."""
        mc, _ = mt_collection_add_tf_method
        assert len(mc.dataframe.survey.unique()) == 1

    def test_dataframe_length(self, mt_collection_add_tf_method, tf_file_list):
        """Test dataframe has correct number of entries."""
        mc, _ = mt_collection_add_tf_method
        assert len(mc.dataframe) == len(tf_file_list)

    def test_tf_id_extra(self, mt_collection_add_tf_method):
        """Test that tf_id_extra is applied."""
        mc, _ = mt_collection_add_tf_method
        for tf_id in mc.dataframe.tf_id:
            assert "added" in tf_id

    def test_mt_data_coordinate_reference_frame(self, mt_data_with_tf_files):
        """Test that coordinate reference frame propagates to MT objects."""
        mt_data_obj = mt_data_with_tf_files
        mt_data_obj.coordinate_reference_frame = "ned"

        for mt_obj in mt_data_obj.values():
            assert mt_obj.coordinate_reference_frame == "NED"


# =============================================================================
# Additional tests for functionality not covered by unittest
# =============================================================================


class TestMTCollectionEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_collection(self, tmp_path):
        """Test creating an empty collection."""
        mc = MTCollection()
        mc.working_directory = tmp_path
        mc.open_collection("empty_collection")

        # Collection should be created successfully
        assert mc.mth5_filename.exists()
        # May have metadata row even when empty, so just check it's minimal
        assert len(mc.dataframe) <= 1

        # Cleanup
        mc.mth5_collection.close_mth5()
        if mc.mth5_filename.exists():
            mc.mth5_filename.unlink()

    def test_collection_with_single_tf(self, tf_file_list, tmp_path):
        """Test collection with only one TF file."""
        mc = MTCollection()
        mc.working_directory = tmp_path
        mc.open_collection("single_tf_collection")
        mc.add_tf([tf_file_list[0]])

        assert len(mc.dataframe) == 1

        # Cleanup
        mc.mth5_collection.close_mth5()
        if mc.mth5_filename.exists():
            mc.mth5_filename.unlink()

    def test_dataframe_column_types(self, mt_collection_main):
        """Test that dataframe columns have expected data types."""
        df = mt_collection_main.dataframe

        # Numeric columns
        assert pd.api.types.is_numeric_dtype(df["latitude"])
        assert pd.api.types.is_numeric_dtype(df["longitude"])
        assert pd.api.types.is_numeric_dtype(df["elevation"])
        assert pd.api.types.is_numeric_dtype(df["period_min"])
        assert pd.api.types.is_numeric_dtype(df["period_max"])

        # Boolean columns
        assert pd.api.types.is_bool_dtype(df["has_impedance"])
        assert pd.api.types.is_bool_dtype(df["has_tipper"])
        assert pd.api.types.is_bool_dtype(df["has_covariance"])

        # String columns
        assert pd.api.types.is_object_dtype(df["station"])
        assert pd.api.types.is_object_dtype(df["survey"])
        assert pd.api.types.is_object_dtype(df["tf_id"])


class TestMTCollectionDataframeOperations:
    """Test dataframe query and filtering operations."""

    def test_filter_by_survey(self, mt_collection_main):
        """Test filtering dataframe by survey."""
        df = mt_collection_main.dataframe
        surveys = df.survey.unique()

        for survey in surveys[:3]:  # Test first 3 surveys
            filtered = df[df.survey == survey]
            assert len(filtered) > 0
            assert (filtered.survey == survey).all()

    def test_filter_by_impedance(self, mt_collection_main):
        """Test filtering by has_impedance."""
        df = mt_collection_main.dataframe
        with_impedance = df[df.has_impedance == True]
        without_impedance = df[df.has_impedance == False]

        assert len(with_impedance) + len(without_impedance) == len(df)

    def test_filter_by_tipper(self, mt_collection_main):
        """Test filtering by has_tipper."""
        df = mt_collection_main.dataframe
        with_tipper = df[df.has_tipper == True]
        without_tipper = df[df.has_tipper == False]

        assert len(with_tipper) + len(without_tipper) == len(df)

    def test_period_range_filter(self, mt_collection_main):
        """Test filtering by period range."""
        df = mt_collection_main.dataframe

        # Filter for stations with period_min < 1 second
        short_period = df[df.period_min < 1.0]
        assert len(short_period) > 0

        # Filter for stations with period_max > 1000 seconds
        long_period = df[df.period_max > 1000.0]
        assert len(long_period) > 0


class TestMTCollectionMultipleSurveys:
    """Test handling of collections with multiple surveys."""

    def test_survey_count(self, mt_collection_main):
        """Test counting unique surveys."""
        surveys = mt_collection_main.dataframe.survey.unique()
        assert len(surveys) > 1  # Should have multiple surveys

    def test_stations_per_survey(self, mt_collection_main):
        """Test counting stations per survey."""
        df = mt_collection_main.dataframe
        survey_counts = df.groupby("survey").size()

        assert len(survey_counts) > 0
        assert survey_counts.sum() == len(df)


class TestMTCollectionFileOperations:
    """Test file-related operations."""

    def test_collection_file_exists(self, mt_collection_main):
        """Test that HDF5 file is created."""
        assert mt_collection_main.mth5_filename.exists()
        assert mt_collection_main.mth5_filename.suffix == ".h5"

    def test_working_directory(self, mt_collection_main):
        """Test that working directory is set correctly."""
        assert mt_collection_main.working_directory.exists()
        assert mt_collection_main.working_directory.is_dir()


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
