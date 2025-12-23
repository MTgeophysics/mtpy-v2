# -*- coding: utf-8 -*-
"""
Pytest configuration and global fixtures for mtpy-v2 tests.

This module provides shared fixtures that can be used across all test modules
to improve test efficiency and reduce redundant setup operations.

Created on December 22, 2025
"""

# =============================================================================
# Imports
# =============================================================================
import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import mt_metadata
import numpy as np
import pandas as pd
import pytest
from loguru import logger
from mt_metadata import TF_EDI_CGG
from mth5.helpers import close_open_files

from mtpy import MT, MTCollection, MTData
from mtpy.core.transfer_function import MT_TO_OHM_FACTOR, Z


# =============================================================================
# Global cache management for MTCollection tests
# =============================================================================

# Global cache directory - persists across test sessions
_DEFAULT_CACHE_DIR = Path.home() / "mtpy_v2_global_test_cache"

# Allow override via environment variable
GLOBAL_CACHE_DIR = Path(os.getenv("MTPY_TEST_CACHE_DIR", _DEFAULT_CACHE_DIR))

# Track cached files for cleanup
_CACHED_COLLECTION_FILES = {}


def get_tf_file_list():
    """Get list of all TF files from mt_metadata package."""
    return [
        value for key, value in mt_metadata.__dict__.items() if key.startswith("TF")
    ]


def ensure_global_mt_collection_cache():
    """
    Ensure global MTCollection cache exists.

    This creates a single cache file that all workers and test sessions share.
    Returns the path to the global cache file.
    """
    cache_name = "test_collection_global.h5"
    global_cache_path = GLOBAL_CACHE_DIR / cache_name

    if global_cache_path.exists():
        logger.debug(f"Global MTCollection cache exists: {global_cache_path}")
        return global_cache_path

    # Create cache directory if needed
    GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Creating global MTCollection cache...")

    # Create in temporary directory first
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / cache_name

        # Create the collection
        mc = MTCollection()
        mc.open_collection(temp_path)

        # Add all TF files
        fn_list = get_tf_file_list()
        mc.add_tf(fn_list)

        # Close the collection
        mc.mth5_collection.close_mth5()

        # Copy to global cache
        try:
            shutil.copy2(temp_path, global_cache_path)
            logger.info(f"Global MTCollection cache created: {global_cache_path}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create global cache: {e}")
            # Return temp path as fallback, though it will be deleted
            return temp_path

    return global_cache_path


def create_session_mt_collection(worker_id="master"):
    """
    Create a worker-specific copy of the global MTCollection cache.

    This function copies the global cache to a worker-specific file.
    Each worker gets its own copy to avoid conflicts.
    """
    # Ensure global cache exists (only created once across all workers)
    global_cache_path = ensure_global_mt_collection_cache()

    # Create unique worker-specific file
    session_dir = tempfile.mkdtemp(prefix=f"mtpy_worker_{worker_id}_")
    unique_id = str(uuid.uuid4())[:8]
    session_file = Path(session_dir) / f"test_collection_{worker_id}_{unique_id}.h5"

    # Copy from global cache (much faster than creating from scratch)
    shutil.copy2(global_cache_path, session_file)

    # Track for cleanup
    _CACHED_COLLECTION_FILES[f"{worker_id}_{unique_id}"] = session_file

    logger.debug(f"Created worker copy for {worker_id}: {session_file}")
    return session_file


def cleanup_collection_files():
    """Clean up session files and temporary directories."""
    close_open_files()
    for cache_key, file_path in _CACHED_COLLECTION_FILES.items():
        try:
            if file_path.exists():
                file_path.unlink()
                # Try to remove parent directory if empty
                try:
                    file_path.parent.rmdir()
                except OSError:
                    pass  # Directory not empty or other error
        except (OSError, PermissionError):
            logger.warning(f"Failed to cleanup {cache_key} file: {file_path}")


atexit.register(cleanup_collection_files)


# =============================================================================
# Session-level fixtures (created once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def tf_file_list():
    """Provide list of all TF files for testing."""
    return get_tf_file_list()


@pytest.fixture(scope="session")
def loaded_mt_objects(tf_file_list):
    """
    Session-scoped fixture providing pre-loaded MT objects.

    This caches MT object creation and reading to avoid repeated I/O.
    Returns a dictionary mapping file path to MT object.
    """
    mt_cache = {}
    for tf_fn in tf_file_list:
        mt_obj = MT(tf_fn)
        mt_obj.read()
        mt_cache[tf_fn] = mt_obj
    return mt_cache


@pytest.fixture(scope="session")
def expected_dataframe():
    """
    Provide expected dataframe structure for MTCollection tests.

    This is the ground truth dataframe used to validate MTCollection behavior.
    """
    return pd.DataFrame(
        {
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
            "units": {
                0: "none",
                1: "none",
                2: "none",
                3: "none",
                4: "none",
                5: "none",
                6: "none",
                7: "none",
                8: "none",
                9: "none",
                10: "none",
                11: "none",
                12: "none",
                13: "none",
                14: "none",
                15: "none",
                16: "none",
                17: "none",
                18: "none",
                19: "none",
                20: "none",
            },
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
            "hdf5_reference": {
                0: None,
                1: None,
                2: None,
                3: None,
                4: None,
                5: None,
                6: None,
                7: None,
                8: None,
                9: None,
                10: None,
                11: None,
                12: None,
                13: None,
                14: None,
                15: None,
                16: None,
                17: None,
                18: None,
                19: None,
                20: None,
            },
            "station_hdf5_reference": {
                0: None,
                1: None,
                2: None,
                3: None,
                4: None,
                5: None,
                6: None,
                7: None,
                8: None,
                9: None,
                10: None,
                11: None,
                12: None,
                13: None,
                14: None,
                15: None,
                16: None,
                17: None,
                18: None,
                19: None,
                20: None,
            },
        }
    )


@pytest.fixture(scope="session")
def global_mt_collection(worker_id):
    """
    Session-scoped fixture providing a cached MTCollection file.

    This collection uses a global cache shared across all workers and sessions.
    Each worker gets its own copy to ensure thread-safety.
    """
    file_path = create_session_mt_collection(worker_id=worker_id)
    yield file_path
    # Cleanup handled by atexit


@pytest.fixture(scope="session")
def sample_edi_file():
    """Provide path to sample EDI file for testing."""
    return TF_EDI_CGG


@pytest.fixture(scope="session")
def sample_impedance_array():
    """
    Provide a sample impedance tensor array for testing.

    Returns a 1x2x2 complex impedance array.
    """
    return np.array([[0.1 - 0.1j, 10 + 10j], [-10 - 10j, -0.1 + 0.1j]]).reshape(
        (1, 2, 2)
    )


@pytest.fixture(scope="session")
def sample_impedance_error_array():
    """Provide a sample impedance error array for testing."""
    return np.array([[0.1, 0.05], [0.05, 0.1]]).reshape((1, 2, 2))


@pytest.fixture(scope="session")
def sample_tipper_array():
    """Provide a sample tipper array for testing."""
    return np.array([[[0.25 - 0.2j, 0.25 + 0.2j]]])


@pytest.fixture(scope="session")
def sample_tipper_error_array():
    """Provide a sample tipper error array for testing."""
    return np.array([[[0.02, 0.03]]])


@pytest.fixture(scope="session")
def expected_resistivity():
    """Provide expected resistivity values for sample impedance."""
    return np.array([[[4.0e-03, 4.0e01], [4.0e01, 4.0e-03]]])


@pytest.fixture(scope="session")
def expected_resistivity_error():
    """Provide expected resistivity error values."""
    return np.array([[[0.00565685, 0.28284271], [0.28284271, 0.00565685]]])


@pytest.fixture(scope="session")
def expected_phase():
    """Provide expected phase values for sample impedance."""
    return np.array([[[-45.0, 45.0], [-135.0, 135.0]]])


@pytest.fixture(scope="session")
def expected_phase_error():
    """Provide expected phase error values."""
    return np.array([[[35.26438968, 0.20257033], [0.20257033, 35.26438968]]])


@pytest.fixture(scope="session")
def expected_phase_tensor():
    """Provide expected phase tensor values."""
    return np.array([[[1.00020002, -0.020002], [-0.020002, 1.00020002]]])


@pytest.fixture(scope="session")
def expected_phase_tensor_error():
    """Provide expected phase tensor error values."""
    return np.array([[[0.01040308, 0.02020604], [0.02020604, 0.01040308]]])


@pytest.fixture(scope="session")
def expected_phase_tensor_azimuth():
    """Provide expected phase tensor azimuth."""
    return np.array([315.0])


@pytest.fixture(scope="session")
def expected_phase_tensor_azimuth_error():
    """Provide expected phase tensor azimuth error."""
    return np.array([3.30832308])


@pytest.fixture(scope="session")
def expected_phase_tensor_skew():
    """Provide expected phase tensor skew."""
    return np.array([0])


@pytest.fixture(scope="session")
def expected_phase_tensor_skew_error():
    """Provide expected phase tensor skew error."""
    return np.array([0.40923428])


# =============================================================================
# Function-level fixtures (created for each test function)
# =============================================================================


@pytest.fixture
def fresh_mt_collection(global_mt_collection, worker_id):
    """
    Function-scoped fixture providing a fresh copy of MTCollection file.

    Use this fixture when your test needs to modify the MTCollection.
    Each test gets its own isolated copy.
    """
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    fresh_file = Path(temp_dir) / f"test_collection_fresh_{worker_id}_{unique_id}.h5"

    shutil.copy2(global_mt_collection, fresh_file)

    mc = MTCollection()
    mc.open_collection(fresh_file)

    yield mc

    # Cleanup
    try:
        mc.mth5_collection.close_mth5()
    except:
        pass
    close_open_files()
    try:
        fresh_file.unlink()
        fresh_file.parent.rmdir()
    except (OSError, PermissionError):
        pass


@pytest.fixture
def mt_collection_from_mt_data(tf_file_list, worker_id):
    """
    Function-scoped fixture providing MTCollection created from MTData.

    This creates a fresh collection for each test.
    """
    temp_dir = tempfile.mkdtemp()
    unique_id = str(uuid.uuid4())[:8]
    collection_file = (
        Path(temp_dir) / f"test_collection_mtdata_{worker_id}_{unique_id}.h5"
    )

    mt_data_obj = MTData()
    mt_data_obj.add_station(tf_file_list, survey="test")

    mc = MTCollection()
    mc.open_collection(collection_file)
    mc.from_mt_data(mt_data_obj)

    yield mc, mt_data_obj

    # Cleanup
    try:
        mc.mth5_collection.close_mth5()
    except:
        pass
    close_open_files()
    try:
        collection_file.unlink()
        collection_file.parent.rmdir()
    except (OSError, PermissionError):
        pass


@pytest.fixture
def basic_mt():
    """
    Create a basic MT object with minimal setup.

    Returns a new MT instance for each test to ensure test isolation.
    """
    mt = MT()
    mt.station = "test_01"
    mt.survey = "big"
    mt.latitude = 10
    mt.longitude = 20
    return mt


@pytest.fixture
def mt_with_impedance(sample_impedance_array, sample_impedance_error_array):
    """
    Create an MT object with impedance data.

    Returns an MT instance with impedance and error data set.
    """
    mt = MT()
    mt.station = "mt001"
    mt.impedance = sample_impedance_array
    mt.impedance_error = sample_impedance_error_array
    mt.impedance_model_error = sample_impedance_error_array
    return mt


@pytest.fixture
def mt_with_impedance_ohm(sample_impedance_array, sample_impedance_error_array):
    """
    Create an MT object with impedance data in ohm units.

    Returns an MT instance with impedance in ohm units.
    """
    z_ohm = sample_impedance_array / MT_TO_OHM_FACTOR
    z_err_ohm = sample_impedance_error_array / MT_TO_OHM_FACTOR

    z_object = Z(
        z=z_ohm,
        z_error=z_err_ohm,
        z_model_error=z_err_ohm,
        units="ohm",
    )

    mt = MT()
    mt.station = "mt001"
    mt.Z = z_object
    return mt


@pytest.fixture
def mt_with_tipper(sample_tipper_array, sample_tipper_error_array):
    """
    Create an MT object with tipper data.

    Returns an MT instance with tipper and error data set.
    """
    mt = MT()
    mt.tipper = sample_tipper_array
    mt.tipper_error = sample_tipper_error_array
    return mt


@pytest.fixture
def mt_from_edi(sample_edi_file):
    """
    Create an MT object from EDI file.

    Returns an MT instance loaded from the sample EDI file.
    """
    mt = MT(sample_edi_file)
    mt.read()
    return mt


# =============================================================================
# Pytest-xdist worker-safe fixtures
# =============================================================================


@pytest.fixture(scope="session")
def worker_id(request):
    """
    Get the current pytest-xdist worker ID.

    Returns:
        str: Worker ID (e.g., 'gw0', 'gw1', 'master') or 'master' if not using xdist
    """
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


# =============================================================================
# Pytest configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to optimize for pytest-xdist.

    This ensures tests are properly isolated and can run in parallel.
    """
    # Add markers based on test names
    for item in items:
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)


def pytest_sessionstart(session):
    """
    Called before the test session starts.

    Pre-populate the global cache to ensure best performance.
    This creates the cache once, and all workers will copy from it.
    """
    logger.info("MTpy-v2 Test Session Starting - Pre-populating global cache...")

    # Pre-create the global MTCollection cache
    # This happens once for the main process, all workers will copy from it
    try:
        ensure_global_mt_collection_cache()
        logger.info("Global MTCollection cache ready")
    except Exception as e:
        logger.warning(f"Failed to pre-populate global cache: {e}")
        logger.warning("Tests will still work but may be slower on first run")


def pytest_sessionfinish(session, exitstatus):
    """
    Called after the test session finishes.

    Perform final cleanup.
    """
    logger.info("MTpy-v2 Test Session Ending - Cleaning up...")
    cleanup_collection_files()
    close_open_files()
    logger.info("MTpy-v2 Test Session cleanup completed")
