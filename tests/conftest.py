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
import os


# Disable HDF5 file locking early to avoid Windows lock errors in cache builds
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core.transfer_function import MT_TO_OHM_FACTOR, Z


# =============================================================================
# Session-level fixtures (created once per test session)
# =============================================================================


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


# =============================================================================
# MTCollection caching infrastructure for pytest-xdist
# =============================================================================

import os
import shutil
import tempfile
import time
from pathlib import Path

from loguru import logger
from mth5.helpers import close_open_files

from mtpy.core.mt_collection import MTCollection
from mtpy.core.mt_data import MTData


# Avoid HDF5 file locking on Windows to reduce cache creation failures
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Global cache directory - persists across test sessions
_DEFAULT_CACHE_DIR = Path.home() / "mtpy_v2_test_cache"
GLOBAL_CACHE_DIR = Path(os.getenv("MTPY_TEST_CACHE_DIR", _DEFAULT_CACHE_DIR))


def get_tf_file_list():
    """Get list of transfer function files from mt_metadata."""
    import mt_metadata

    # Get TF files from mt_metadata module attributes
    raw_tf_list = [
        value for key, value in mt_metadata.__dict__.items() if key.startswith("TF")
    ]

    # Deduplicate by normalized path to prevent repeated additions
    unique_paths = []
    seen = set()
    for path in raw_tf_list:
        norm = Path(path).resolve()
        if norm not in seen:
            seen.add(norm)
            unique_paths.append(path)

    return sorted(unique_paths)


def create_cached_mt_collection(collection_name, setup_func, force_recreate=False):
    """
    Create or retrieve a cached MTCollection file.

    This creates a global cache that persists across test sessions,
    dramatically improving test performance.

    Args:
        collection_name: Name for the collection (used in cache filename)
        setup_func: Function that creates and configures the MTCollection
        force_recreate: Force recreation even if cache exists

    Returns:
        Path to the cached HDF5 file
    """
    cache_filename = f"{collection_name}_cache.h5"
    cache_path = GLOBAL_CACHE_DIR / cache_filename

    if cache_path.exists() and not force_recreate:
        logger.debug(f"Using cached MTCollection: {cache_path}")
        return cache_path

    # Close any open HDF5 files to avoid locking issues
    close_open_files()

    # Create cache directory if needed
    GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating cached MTCollection: {collection_name}...")

    # Create in temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"mtpy_cache_create_{collection_name}_")
    temp_dir_path = Path(temp_dir)
    temp_file = None

    try:
        # Call the setup function to create the collection
        temp_file = setup_func(temp_dir_path, collection_name)

        # Make sure files are closed before copying
        close_open_files()

        # Copy to global cache
        shutil.copy2(temp_file, cache_path)
        logger.info(f"Cached MTCollection created: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to create cache: {e}")

        # If cache already exists but was locked, return it instead of failing
        if cache_path.exists():
            logger.warning(
                f"Using existing cached file despite creation failure: {cache_path}"
            )
            return cache_path

        # If temp_file was created, return it as fallback
        if temp_file and temp_file.exists():
            return temp_file

        # Otherwise raise the error
        raise
    finally:
        # Clean up temp directory
        try:
            if temp_dir_path.exists():
                shutil.rmtree(temp_dir)
        except (OSError, PermissionError):
            pass

    return cache_path


# Setup functions for different MTCollection configurations


def setup_main_collection(working_dir, collection_name):
    """Create main MTCollection with all transfer functions."""
    tf_list = get_tf_file_list()

    mc = MTCollection()
    mc.working_directory = working_dir
    mc.open_collection(collection_name)
    mc.add_tf(tf_list)
    mc.close_collection()

    # Close the underlying HDF5 file explicitly
    try:
        mc.mth5_collection.close_mth5()
    except Exception:
        pass

    return working_dir / f"{collection_name}.h5"


def setup_collection_from_mt_data_with_survey(working_dir, collection_name):
    """Create MTCollection from MTData with specified survey."""
    tf_list = get_tf_file_list()

    mt_data_obj = MTData()
    mt_data_obj.add_station(tf_list, survey="test")

    mc = MTCollection()
    mc.working_directory = working_dir
    mc.open_collection(collection_name)
    mc.from_mt_data(mt_data_obj)
    mc.close_collection()

    return working_dir / f"{collection_name}.h5"


def setup_collection_from_mt_data_new_survey(working_dir, collection_name):
    """Create MTCollection from MTData with new_survey and tf_id_extra."""
    tf_list = get_tf_file_list()

    mt_data_obj = MTData()
    mt_data_obj.add_station(tf_list)

    mc = MTCollection()
    mc.working_directory = working_dir
    mc.open_collection(collection_name)
    mc.from_mt_data(mt_data_obj, new_survey="test", tf_id_extra="new")
    mc.close_collection()

    return working_dir / f"{collection_name}.h5"


def setup_collection_add_tf_method(working_dir, collection_name):
    """Create MTCollection using add_tf with MTData object."""
    tf_list = get_tf_file_list()

    mt_data_obj = MTData()
    mt_data_obj.add_station(tf_list, survey="test")

    mc = MTCollection()
    mc.working_directory = working_dir
    mc.open_collection(collection_name)
    # add_tf with MTData object - survey is already set in MTData
    mc.add_tf(mt_data_obj, tf_id_extra="added")
    mc.close_collection()

    return working_dir / f"{collection_name}.h5"


# Pytest-xdist worker-safe utilities


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


def get_worker_safe_filename(base_filename: str, worker_id: str) -> str:
    """
    Generate a worker-safe filename by inserting the worker ID before the extension.

    Args:
        base_filename: Original filename (e.g., "test.h5")
        worker_id: Worker ID from pytest-xdist (e.g., "gw0", "master")

    Returns:
        str: Worker-safe filename (e.g., "test_gw0.h5", "test_master.h5")
    """
    path = Path(base_filename)
    stem = path.stem
    suffix = path.suffix
    return f"{stem}_{worker_id}{suffix}"


def create_worker_copy(cache_path: Path, worker_id: str, target_dir: Path) -> Path:
    """
    Create a worker-specific copy of a cached file.

    This ensures each pytest-xdist worker has its own file, avoiding HDF5 locking issues.

    Args:
        cache_path: Path to the cached source file
        worker_id: Worker ID for unique naming
        target_dir: Directory to place the copy

    Returns:
        Path to the worker-specific copy
    """
    worker_filename = get_worker_safe_filename(cache_path.name, worker_id)
    worker_path = target_dir / worker_filename

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, worker_path)

    logger.debug(f"Created worker copy: {worker_path}")
    return worker_path


def get_worker_collection_file(
    collection_name: str,
    setup_func,
    worker_id: str,
    session_temp_dir: Path,
):
    """Return a worker-local MTCollection file using cached source when available.

    Prefers the controller-built cache (fast copy). Falls back to building a
    worker-local file if the cache is missing.
    """

    cache_path = GLOBAL_CACHE_DIR / f"{collection_name}_cache.h5"
    lock_path = GLOBAL_CACHE_DIR / f"{collection_name}.lock"

    # Fast path: cache already present
    if cache_path.exists():
        try:
            return create_worker_copy(cache_path, worker_id, session_temp_dir)
        except Exception as e:
            logger.warning(
                f"Failed to copy cache {cache_path} for worker {worker_id}: {e}"
            )

    # Slow path: coordinate a single cache build across workers using a lock file
    GLOBAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    acquired_lock = False
    cache_ready = False
    start_time = time.time()
    wait_seconds = 300  # generous to allow first builder to finish

    # Poll for existing cache or try to become the builder
    while time.time() - start_time < wait_seconds:
        if cache_path.exists():
            cache_ready = True
            break

        try:
            # os.O_EXCL ensures only one worker creates the lock
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired_lock = True
            break
        except FileExistsError:
            time.sleep(1.0)

    if acquired_lock:
        try:
            close_open_files()
            created = create_cached_mt_collection(collection_name, setup_func)
            cache_ready = created.exists()
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    if cache_ready and cache_path.exists():
        try:
            return create_worker_copy(cache_path, worker_id, session_temp_dir)
        except Exception as e:
            logger.warning(
                f"Failed to copy cache {cache_path} for worker {worker_id} after build: {e}"
            )

    # If a stale lock blocked us and no cache is ready, clear it before fallback
    if not cache_ready and not acquired_lock and lock_path.exists():
        try:
            lock_path.unlink()
        except Exception:
            pass

    # Cache unavailable or copy failed — build a worker-local file
    close_open_files()
    safe_name = f"{collection_name}_{worker_id}"
    return setup_func(session_temp_dir, safe_name)


# Session-scoped fixtures for MTCollection testing


@pytest.fixture(scope="session")
def session_temp_dir(worker_id):
    """Create a session-scoped temporary directory that's worker-safe."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"mtpy_test_{worker_id}_"))
    yield temp_dir

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except (OSError, PermissionError):
        pass


@pytest.fixture(scope="session")
def tf_file_list():
    """Session-scoped fixture providing list of transfer function files."""
    return get_tf_file_list()


@pytest.fixture(scope="session")
def mt_collection_main_cache(worker_id, session_temp_dir):
    """
    Session-scoped fixture providing a worker-safe copy of the main MTCollection.

    Creates the collection once per worker session directly in the temp directory,
    avoiding global cache to prevent HDF5 file locking issues on Windows.
    """
    worker_file = get_worker_collection_file(
        "mt_collection_main", setup_main_collection, worker_id, session_temp_dir
    )

    yield worker_file


@pytest.fixture(scope="session")
def mt_collection_with_survey_cache(worker_id, session_temp_dir):
    """Session-scoped fixture for MTCollection created from MTData with survey."""
    worker_file = get_worker_collection_file(
        "mt_collection_with_survey",
        setup_collection_from_mt_data_with_survey,
        worker_id,
        session_temp_dir,
    )

    yield worker_file


@pytest.fixture(scope="session")
def mt_collection_new_survey_cache(worker_id, session_temp_dir):
    """Session-scoped fixture for MTCollection with new_survey and tf_id_extra."""
    worker_file = get_worker_collection_file(
        "mt_collection_new_survey",
        setup_collection_from_mt_data_new_survey,
        worker_id,
        session_temp_dir,
    )

    yield worker_file


@pytest.fixture(scope="session")
def mt_collection_add_tf_cache(worker_id, session_temp_dir):
    """Session-scoped fixture for MTCollection created using add_tf method."""
    worker_file = get_worker_collection_file(
        "mt_collection_add_tf",
        setup_collection_add_tf_method,
        worker_id,
        session_temp_dir,
    )

    yield worker_file


# Convenience fixtures that open MTCollection objects


@pytest.fixture
def mt_collection_main(mt_collection_main_cache):
    """
    Function-scoped fixture providing an opened MTCollection.

    Opens in append mode so mth5 can write estimate metadata while keeping
    the underlying session file worker-safe.
    """
    mc = MTCollection()
    mc.working_directory = mt_collection_main_cache.parent
    mc.open_collection(mt_collection_main_cache.stem, mode="a")

    # Drop any duplicate transfer functions that can appear from multi-attachment
    # XML examples so downstream tests see one entry per TF file.
    if mc.dataframe is not None:
        mc.working_dataframe = mc.dataframe.drop_duplicates(subset="tf_id")

    yield mc

    try:
        mc.close_collection()
    except Exception:
        pass


@pytest.fixture
def mt_collection_from_mt_data_with_survey(mt_collection_with_survey_cache):
    """Function-scoped fixture for MTCollection with survey."""
    mc = MTCollection()
    mc.working_directory = mt_collection_with_survey_cache.parent
    mc.open_collection(mt_collection_with_survey_cache.stem, mode="a")

    if mc.dataframe is not None:
        mc.working_dataframe = mc.dataframe.drop_duplicates(subset="tf_id")

    yield mc, None

    try:
        mc.close_collection()
    except Exception:
        pass


@pytest.fixture
def mt_collection_from_mt_data_new_survey(mt_collection_new_survey_cache):
    """Function-scoped fixture for MTCollection with new_survey."""
    mc = MTCollection()
    mc.working_directory = mt_collection_new_survey_cache.parent
    mc.open_collection(mt_collection_new_survey_cache.stem, mode="a")

    if mc.dataframe is not None:
        mc.working_dataframe = mc.dataframe.drop_duplicates(subset="tf_id")

    yield mc, None

    try:
        mc.close_collection()
    except Exception:
        pass


@pytest.fixture
def mt_collection_add_tf_method(mt_collection_add_tf_cache):
    """Function-scoped fixture for MTCollection created with add_tf."""
    mc = MTCollection()
    mc.working_directory = mt_collection_add_tf_cache.parent
    mc.open_collection(mt_collection_add_tf_cache.stem, mode="a")

    if mc.dataframe is not None:
        mc.working_dataframe = mc.dataframe.drop_duplicates(subset="tf_id")

    yield mc, None

    try:
        mc.close_collection()
    except Exception:
        pass


@pytest.fixture(scope="session")
def expected_dataframe_data():
    """Expected data for DataFrame validation tests."""
    return {
        "num_rows": 21,
        "num_columns": 23,
        "has_latitude": True,
        "has_longitude": True,
        "has_impedance": True,
        "has_tipper": True,
    }


def pytest_sessionstart(session):
    """Called before the test session starts."""
    logger.info("MTpy-v2 Test Session Starting - Initializing cache...")

    # When running under pytest-xdist workers, skip global cache creation to
    # prevent multiple processes from writing/locking the same file.
    if hasattr(session.config, "workerinput"):
        logger.info("Skipping cache pre-population on xdist worker")
        return

    # Pre-create the most commonly used cache files
    try:
        logger.info(
            "Pre-populating MTCollection cache (this may take a few minutes on first run)..."
        )
        create_cached_mt_collection("mt_collection_main", setup_main_collection)
        create_cached_mt_collection(
            "mt_collection_with_survey", setup_collection_from_mt_data_with_survey
        )
        create_cached_mt_collection(
            "mt_collection_new_survey", setup_collection_from_mt_data_new_survey
        )
        create_cached_mt_collection(
            "mt_collection_add_tf", setup_collection_add_tf_method
        )
        logger.info("Cache pre-population completed successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-populate cache: {e}")
        logger.warning("Tests will still work but may be slower on first run")


def pytest_sessionfinish(session, exitstatus):
    """Called after the test session finishes."""
    logger.info("MTpy-v2 Test Session Ending")
    logger.info(f"Global cache location: {GLOBAL_CACHE_DIR}")
    logger.info("Note: Global cache is preserved for faster subsequent test runs")
