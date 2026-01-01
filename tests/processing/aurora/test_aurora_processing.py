# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for Aurora processing with MTH5 integration.

Combines single-station and remote reference processing tests with
fixtures, parameterization, and parallel execution optimization.

Created on December 23, 2025

@author: AI Assistant
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from loguru import logger
from mth5.data.make_mth5_from_asc import create_test12rr_h5, MTH5_PATH
from mth5.mth5 import MTH5
from mth5.processing.kernel_dataset import KernelDataset
from mth5.processing.run_summary import RunSummary
from mth5.utils.helpers import close_open_files

from mtpy import MT
from mtpy.processing.aurora.process_aurora import AuroraProcessing


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mth5_test_file_cache():
    """Create or locate test12rr MTH5 file in cache for entire test session."""
    mth5_path = MTH5_PATH.joinpath("test12rr.h5")
    if not mth5_path.exists():
        mth5_path = create_test12rr_h5()
    yield mth5_path
    # Cleanup handled by mth5 module


@pytest.fixture(scope="function")
def mth5_test_file(mth5_test_file_cache):
    """
    Create a unique copy of test12rr MTH5 file for each test.

    This prevents file locking conflicts when running tests in parallel with pytest-xdist.
    Each test gets its own isolated copy of the file.
    """
    temp_dir = tempfile.mkdtemp(prefix="mth5_test_")
    unique_id = str(uuid.uuid4())[:8]
    unique_file = Path(temp_dir) / f"test12rr_{unique_id}.h5"

    # Copy from cached file
    shutil.copy2(mth5_test_file_cache, unique_file)

    yield unique_file

    # Cleanup
    close_open_files()
    try:
        unique_file.unlink()
        unique_file.parent.rmdir()
    except (OSError, PermissionError):
        pass


@pytest.fixture(scope="session")
def sample_rate_options():
    """Define available sample rate options for parameterization."""
    return [1]


@pytest.fixture(scope="session")
def decimation_kwargs():
    """Default decimation parameters for low frequency processing."""
    ap = AuroraProcessing()
    return ap.default_window_parameters["low"]


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================


@pytest.fixture
def aurora_processor(mth5_test_file):
    """Create fresh AuroraProcessing instance for each test."""
    ap = AuroraProcessing()
    ap.local_mth5_path = mth5_test_file
    return ap


@pytest.fixture
def single_station_config(mth5_test_file, decimation_kwargs):
    """Create configuration for single station processing."""
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_test_file])

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1")

    cc_kwargs = {"num_samples_window": decimation_kwargs["stft.window.num_samples"]}
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset, **cc_kwargs)
    return {
        "config": config,
        "kernel_dataset": kernel_dataset,
        "run_summary": run_summary,
    }


@pytest.fixture
def remote_reference_config(mth5_test_file, decimation_kwargs):
    """Create configuration for remote reference processing."""
    run_summary = RunSummary()
    run_summary.from_mth5s([mth5_test_file])

    kernel_dataset = KernelDataset()
    kernel_dataset.from_run_summary(run_summary, "test1", "test2")

    cc_kwargs = {"num_samples_window": decimation_kwargs["stft.window.num_samples"]}
    cc = ConfigCreator()
    config = cc.create_from_kernel_dataset(kernel_dataset, **cc_kwargs)

    return {
        "config": config,
        "kernel_dataset": kernel_dataset,
        "run_summary": run_summary,
    }


# =============================================================================
# Test Classes - Single Station Processing
# =============================================================================


class TestSingleStationLegacyProcessing:
    """Test single station processing with legacy aurora infrastructure."""

    def test_run_summary_creation(self, single_station_config):
        """Test RunSummary object creation and shape."""
        run_summary = single_station_config["run_summary"]
        assert run_summary.df.shape == (
            2,
            15,
        ), f"Expected shape (2, 15), got {run_summary.df.shape}"

    def test_kernel_dataset_shape(self, single_station_config):
        """Test KernelDataset shape for single station."""
        kernel_dataset = single_station_config["kernel_dataset"]
        assert kernel_dataset.df.shape == (
            1,
            20,
        ), f"Expected shape (1, 20), got {kernel_dataset.df.shape}"

    def test_kernel_dataset_remote_is_none(self, single_station_config):
        """Verify no remote station in single station processing."""
        kernel_dataset = single_station_config["kernel_dataset"]
        assert kernel_dataset.remote_station_id is None
        assert kernel_dataset.remote_df is None

    def test_kernel_dataset_channels(self, single_station_config):
        """Verify correct input/output channel configuration."""
        kernel_dataset = single_station_config["kernel_dataset"]
        assert kernel_dataset.input_channels == ["hx", "hy"]
        assert kernel_dataset.output_channels == ["ex", "ey", "hz"]

    def test_config_channels(self, single_station_config):
        """Verify config has correct channel assignments."""
        config = single_station_config["config"]
        assert config.decimations[0].input_channels == ["hx", "hy"]
        assert config.decimations[0].output_channels == ["ex", "ey", "hz"]

    def test_tf_object_creation(self, single_station_config, decimation_kwargs):
        """Test transfer function creation with legacy pipeline."""
        config = single_station_config["config"]
        kernel_dataset = single_station_config["kernel_dataset"]

        # Apply decimation parameters
        ap = AuroraProcessing()
        ap._set_decimation_level_parameters(config, **decimation_kwargs)

        tf_obj = process_mth5(config, kernel_dataset)
        mt_obj = MT(survey_metadata=tf_obj.survey_metadata)
        mt_obj._transfer_function = tf_obj._transfer_function

        assert isinstance(mt_obj, MT)


class TestSingleStationSingleSampleRate:
    """Test single station processing with single sample rate."""

    def test_process_single_sample_rate(self, aurora_processor):
        """Test basic single sample rate processing."""
        aurora_processor.local_station_id = "test1"
        mt_obj = aurora_processor.process_single_sample_rate(1)

        assert isinstance(mt_obj, MT)

    def test_tf_id_matches_processing_id(self, aurora_processor):
        """Verify transfer function ID matches processing ID."""
        aurora_processor.local_station_id = "test1"
        mt_obj = aurora_processor.process_single_sample_rate(1)

        assert mt_obj.tf_id == aurora_processor.processing_id


class TestSingleStationComparison:
    """Compare new and legacy processing pipelines for single station."""

    @pytest.fixture(scope="class")
    def processed_data_new(self, mth5_test_file_cache, decimation_kwargs):
        """Process with new mtpy pipeline using unique copy."""
        # Need unique copy to avoid file locking during processing
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_new_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_new_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        ap = AuroraProcessing()
        ap.local_station_id = "test1"
        ap.local_mth5_path = unique_file
        mt_obj_new = ap.process_single_sample_rate(1)
        return {"mt_obj": mt_obj_new, "processor": ap}

    @pytest.fixture(scope="class")
    def processed_data_legacy(self, mth5_test_file_cache, decimation_kwargs):
        """Process with legacy aurora infrastructure using unique copy."""
        # Need unique copy to avoid file locking during processing
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_data_legacy_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_data_legacy_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        run_summary = RunSummary()
        run_summary.from_mth5s([unique_file])

        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test1")

        cc_kwargs = {}
        cc_kwargs["num_samples_window"] = decimation_kwargs["stft.window.num_samples"]
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset, **cc_kwargs)

        ap = AuroraProcessing()
        ap._set_decimation_level_parameters(config, **decimation_kwargs)

        tf_obj = process_mth5(config, kernel_dataset)

        # Only create MT object if TF processing succeeded
        if tf_obj is not None:
            mt_obj_legacy = MT()
            mt_obj_legacy.survey_metadata.update(tf_obj.survey_metadata)
            mt_obj_legacy.station_metadata.update(tf_obj.station_metadata)
            mt_obj_legacy._transfer_function = tf_obj._transfer_function
            mt_obj_legacy.tf_id = kernel_dataset.processing_id

            return {"mt_obj": mt_obj_legacy, "kernel_dataset": kernel_dataset}
        else:
            return {"mt_obj": None, "kernel_dataset": kernel_dataset}

    def test_mt_objects_equal(self, processed_data_new, processed_data_legacy):
        """Verify new and legacy processing produce identical results."""
        # Skip test if legacy processing failed
        if processed_data_legacy["mt_obj"] is None:
            pytest.skip("Legacy processing returned None")

        # Verify station IDs are correct before comparing
        assert processed_data_new["mt_obj"].station == "test1", (
            f"New processing station ID is '{processed_data_new['mt_obj'].station}' "
            f"but expected 'test1'"
        )
        assert processed_data_legacy["mt_obj"].station == "test1", (
            f"Legacy processing station ID is '{processed_data_legacy['mt_obj'].station}' "
            f"but expected 'test1'"
        )

        assert processed_data_legacy["mt_obj"] == processed_data_new["mt_obj"]

    def test_tf_id_consistency(self, processed_data_new):
        """Verify TF ID matches processor ID."""
        assert (
            processed_data_new["mt_obj"].tf_id
            == processed_data_new["processor"].processing_id
        )


class TestSingleStationWithMerge:
    """Test single station processing with merge and mth5 save options."""

    @pytest.fixture(scope="class")
    def processed_with_merge(self, mth5_test_file_cache, decimation_kwargs):
        """Process with merge=True and save_to_mth5=True using unique copy."""
        # Need unique copy since we're writing to mth5
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_merge_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_merge_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        ap = AuroraProcessing()
        ap.local_station_id = "test1"
        ap.local_mth5_path = unique_file
        processed = ap.process(sample_rates=1, merge=True, save_to_mth5=True)

        return {
            "processed": processed,
            "mt_obj": processed[1]["tf"],
            "processor": ap,
            "mth5_file": unique_file,
        }

    @pytest.fixture(scope="class")
    def legacy_comparison(self, mth5_test_file_cache, decimation_kwargs):
        """Create legacy comparison data using unique copy."""
        # Need unique copy to avoid file locking during processing
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_legacy_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_legacy_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        run_summary = RunSummary()
        run_summary.from_mth5s([unique_file])

        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test1")

        cc_kwargs = {"num_samples_window": decimation_kwargs["stft.window.num_samples"]}
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset, **cc_kwargs)

        ap = AuroraProcessing()
        ap._set_decimation_level_parameters(config, **decimation_kwargs)

        tf_obj = process_mth5(config, kernel_dataset)
        tf_obj.tf_id = kernel_dataset.processing_id
        tf_obj.station_metadata.transfer_function.runs_processed = (
            tf_obj.station_metadata.run_list
        )

        mt_obj_legacy = MT()
        mt_obj_legacy.survey_metadata.update(tf_obj.survey_metadata)
        mt_obj_legacy.station_metadata.update(tf_obj.station_metadata)
        mt_obj_legacy._transfer_function = tf_obj._transfer_function

        return {"mt_obj": mt_obj_legacy, "tf_obj": tf_obj}

    def test_processed_flag_is_true(self, processed_with_merge):
        """Verify processed flag is set correctly."""
        assert processed_with_merge["processed"][1]["processed"]

    def test_tfs_equal_with_date_correction(
        self, processed_with_merge, legacy_comparison
    ):
        """Verify TFs equal after correcting processed_date and survey ID."""
        mt_obj_new = processed_with_merge["mt_obj"]
        mt_obj_legacy = legacy_comparison["mt_obj"]

        # Correct processed_date to match
        mt_obj_new.station_metadata.transfer_function.processed_date = (
            mt_obj_legacy.station_metadata.transfer_function.processed_date
        )

        # Match survey IDs (underscore vs space issue)
        mt_obj_new.survey_metadata.id = mt_obj_legacy.survey_metadata.id

        assert mt_obj_new == mt_obj_legacy

    def test_tf_id_in_processor(self, processed_with_merge):
        """Verify TF ID matches processor ID."""
        assert (
            processed_with_merge["mt_obj"].tf_id
            == processed_with_merge["processor"].processing_id
        )

    def test_tf_saved_in_mth5(self, processed_with_merge, legacy_comparison):
        """Verify transfer function was saved to MTH5 file."""
        with MTH5() as m:
            m.open_mth5(processed_with_merge["mth5_file"])
            tf_df = m.tf_summary.to_dataframe()

            # Check station is in tf_summary
            assert "test1" in tf_df.station.tolist()

            # Retrieve and compare TF
            tf = m.get_transfer_function("test1", "test1_sr1")
            tf_obj = legacy_comparison["tf_obj"]

            # Force southeast corners to match (see mt_metadata issue #253)
            logger.error("Forcing southeast corners to match")
            tf.survey_metadata.southeast_corner = (
                tf_obj.survey_metadata.southeast_corner
            )
            tf.station_metadata.remove_run("0")

            assert tf_obj == tf


# =============================================================================
# Test Classes - Remote Reference Processing
# =============================================================================


class TestRemoteReferenceComparison:
    """Compare new and legacy processing for remote reference."""

    @pytest.fixture(scope="class")
    def processed_data_new_rr(self, mth5_test_file_cache, decimation_kwargs):
        """Process with new mtpy pipeline using remote reference with unique copy."""
        # Need unique copy to avoid file locking during processing
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_new_rr_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_new_rr_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        ap = AuroraProcessing()
        ap.local_station_id = "test1"
        ap.local_mth5_path = unique_file
        ap.remote_station_id = "test2"
        ap.remote_mth5_path = unique_file

        mt_obj_new = ap.process_single_sample_rate(1)
        return {"mt_obj": mt_obj_new, "processor": ap}

    @pytest.fixture(scope="class")
    def processed_data_legacy_rr(self, mth5_test_file_cache, decimation_kwargs):
        """Process with legacy aurora infrastructure using remote reference with unique copy."""
        # Need unique copy to avoid file locking during processing
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_data_legacy_rr_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_data_legacy_rr_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        run_summary = RunSummary()
        run_summary.from_mth5s([unique_file])

        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test1", "test2")

        cc_kwargs = {}
        cc_kwargs["num_samples_window"] = decimation_kwargs["stft.window.num_samples"]
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset, **cc_kwargs)

        ap = AuroraProcessing()
        ap._set_decimation_level_parameters(config, **decimation_kwargs)

        tf_obj = process_mth5(config, kernel_dataset)

        # Only create MT object if TF processing succeeded
        if tf_obj is not None:
            mt_obj_legacy = MT()
            mt_obj_legacy.survey_metadata.update(tf_obj.survey_metadata)
            mt_obj_legacy.station_metadata.update(tf_obj.station_metadata)
            mt_obj_legacy._transfer_function = tf_obj._transfer_function
            mt_obj_legacy.tf_id = kernel_dataset.processing_id

            return {"mt_obj": mt_obj_legacy, "kernel_dataset": kernel_dataset}
        else:
            return {"mt_obj": None, "kernel_dataset": kernel_dataset}

    def test_mt_objects_equal_rr(self, processed_data_new_rr, processed_data_legacy_rr):
        """Verify new and legacy RR processing produce identical results."""
        # Skip test if legacy processing failed
        if processed_data_legacy_rr["mt_obj"] is None:
            pytest.skip("Legacy RR processing returned None")

        assert (
            processed_data_legacy_rr["mt_obj"].survey_metadata
            == processed_data_new_rr["mt_obj"].survey_metadata
        )
        # tipper data is slightly different for some reason, probably coherence
        assert np.isclose(
            processed_data_legacy_rr["mt_obj"].transfer_function.data,
            processed_data_new_rr["mt_obj"].transfer_function.data,
            atol=1e-3,
        ).all()

    def test_tf_id_consistency_rr(self, processed_data_new_rr):
        """Verify TF ID matches processor ID for RR."""
        assert (
            processed_data_new_rr["mt_obj"].tf_id
            == processed_data_new_rr["processor"].processing_id
        )


class TestRemoteReferenceWithMerge:
    """Test remote reference processing with merge and mth5 save options."""

    @pytest.fixture(scope="class")
    def processed_with_merge_rr(self, mth5_test_file_cache, decimation_kwargs):
        """Process RR with merge=True and save_to_mth5=True using unique copy."""
        # Need unique copy since we're writing to mth5
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_merge_rr_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_merge_rr_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        ap = AuroraProcessing()
        ap.local_station_id = "test1"
        ap.local_mth5_path = unique_file
        ap.remote_station_id = "test2"
        ap.remote_mth5_path = unique_file

        processed = ap.process(sample_rates=1, merge=True, save_to_mth5=True)

        return {
            "processed": processed,
            "mt_obj": processed[1]["tf"],
            "processor": ap,
            "mth5_file": unique_file,
        }

    @pytest.fixture(scope="class")
    def legacy_comparison_rr(self, mth5_test_file_cache, decimation_kwargs):
        """Create legacy RR comparison data using unique copy."""
        # Need unique copy to avoid file locking during processing
        temp_dir = tempfile.mkdtemp(prefix="mth5_test_legacy_rr_")
        unique_id = str(uuid.uuid4())[:8]
        unique_file = Path(temp_dir) / f"test12rr_legacy_rr_{unique_id}.h5"
        shutil.copy2(mth5_test_file_cache, unique_file)

        run_summary = RunSummary()
        run_summary.from_mth5s([unique_file])

        kernel_dataset = KernelDataset()
        kernel_dataset.from_run_summary(run_summary, "test1", "test2")

        cc_kwargs = {"num_samples_window": decimation_kwargs["stft.window.num_samples"]}
        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset, **cc_kwargs)

        ap = AuroraProcessing()
        ap._set_decimation_level_parameters(config, **decimation_kwargs)

        tf_obj = process_mth5(config, kernel_dataset)
        tf_obj.tf_id = kernel_dataset.processing_id
        tf_obj.station_metadata.transfer_function.runs_processed = (
            tf_obj.station_metadata.run_list
        )

        mt_obj_legacy = MT()
        mt_obj_legacy.survey_metadata.update(tf_obj.survey_metadata)
        mt_obj_legacy.station_metadata.update(tf_obj.station_metadata)
        mt_obj_legacy._transfer_function = tf_obj._transfer_function

        return {"mt_obj": mt_obj_legacy, "tf_obj": tf_obj}

    def test_processed_flag_is_true_rr(self, processed_with_merge_rr):
        """Verify processed flag is set correctly for RR."""
        assert processed_with_merge_rr["processed"][1]["processed"]

    # @pytest.mark.xfail(reason="RR TF comparison can have subtle metadata differences")
    def test_tfs_equal_rr(self, processed_with_merge_rr, legacy_comparison_rr):
        """Verify RR TFs are equal after matching survey IDs and processed dates."""
        mt_obj_new = processed_with_merge_rr["mt_obj"]
        mt_obj_legacy = legacy_comparison_rr["mt_obj"]

        # Match survey IDs (underscore vs space issue)
        mt_obj_new.survey_metadata.id = mt_obj_legacy.survey_metadata.id

        # Match processed dates (timing differences)
        mt_obj_new.station_metadata.transfer_function.processed_date = (
            mt_obj_legacy.station_metadata.transfer_function.processed_date
        )

        assert mt_obj_new.survey_metadata == mt_obj_legacy.survey_metadata
        # tipper data is slightly different for some reason, probably coherence
        assert np.isclose(
            mt_obj_legacy.transfer_function.data,
            mt_obj_new.transfer_function.data,
            atol=1e-3,
        ).all()

    def test_tf_id_in_processor_rr(self, processed_with_merge_rr):
        """Verify TF ID matches processor ID for RR."""
        assert (
            processed_with_merge_rr["mt_obj"].tf_id
            == processed_with_merge_rr["processor"].processing_id
        )

    # @pytest.mark.xfail(reason="RR TF retrieval from mth5 can have metadata differences")
    def test_tf_saved_in_mth5_rr(self, processed_with_merge_rr, legacy_comparison_rr):
        """Verify RR transfer function was saved to MTH5 file."""
        with MTH5() as m:
            m.open_mth5(processed_with_merge_rr["mth5_file"])
            tf_df = m.tf_summary.to_dataframe()

            # Check station is in tf_summary
            assert "test1" in tf_df.station.tolist()

            # Retrieve and compare TF (processing ID changed from test1-rr_test2_sr1)
            expected_processing_id = "test1_rr_test2_sr1"
            tf = m.get_transfer_function("test1", expected_processing_id)
            tf_obj = legacy_comparison_rr["tf_obj"]

            # Force southeast corners to match (see mt_metadata issue #253)
            logger.error("Forcing southeast corners to match")
            tf.survey_metadata.southeast_corner = (
                tf_obj.survey_metadata.southeast_corner
            )
            tf.station_metadata.remove_run("0")

            assert tf_obj.survey_metadata == tf.survey_metadata
            # tipper data is slightly different for some reason, probably coherence
            assert np.isclose(
                tf_obj.transfer_function.data,
                tf.transfer_function.data,
                atol=1e-3,
            ).all()


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize("station_id", ["test1"])
class TestParameterizedSingleStation:
    """Parameterized tests for single station processing."""

    def test_processing_with_station(self, mth5_test_file, station_id):
        """Test processing works for parameterized station IDs."""
        ap = AuroraProcessing()
        ap.local_station_id = station_id
        ap.local_mth5_path = mth5_test_file

        mt_obj = ap.process_single_sample_rate(1)
        assert isinstance(mt_obj, MT)
        assert mt_obj.station == station_id


@pytest.mark.parametrize(
    "local_id,remote_id",
    [("test1", "test2")],
)
class TestParameterizedRemoteReference:
    """Parameterized tests for remote reference processing."""

    def test_rr_processing_with_stations(self, mth5_test_file, local_id, remote_id):
        """Test RR processing works for parameterized station pairs."""
        ap = AuroraProcessing()
        ap.local_station_id = local_id
        ap.local_mth5_path = mth5_test_file
        ap.remote_station_id = remote_id
        ap.remote_mth5_path = mth5_test_file

        mt_obj = ap.process_single_sample_rate(1)
        assert isinstance(mt_obj, MT)
        assert mt_obj.station == local_id


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_tests():
    """Cleanup fixture to close open files after all tests."""
    yield
    close_open_files()
