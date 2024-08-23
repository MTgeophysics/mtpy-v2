# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:09:12 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.data.make_mth5_from_asc import MTH5_PATH, create_test12rr_h5
from mth5.utils.helpers import close_open_files
from mth5.mth5 import MTH5

from mtpy.processing.run_summary import RunSummary
from mtpy.processing.kernel_dataset import KernelDataset
from mtpy import MT

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from mtpy.processing.aurora.process_aurora import AuroraProcessing


# =============================================================================
class TestProcessSingleStationCompare(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()

        # process with mtpy
        self.ap = AuroraProcessing()
        self.ap.local_station_id = "test1"
        self.ap.local_mth5_path = self.mth5_path
        self.processed = self.ap.process(
            sample_rates=1, merge=True, save_to_mth5=True
        )
        self.mt_obj_new = self.processed[1]["tf"]

        # process with aurora infrastructure
        self.run_summary = RunSummary()
        self.run_summary.from_mth5s([self.mth5_path])
        self.kernel_dataset = KernelDataset()
        self.kernel_dataset.from_run_summary(self.run_summary, "test1")
        cc = ConfigCreator()
        self.config = cc.create_from_kernel_dataset(self.kernel_dataset)
        ## need to set same config parameters
        self.ap._set_decimation_level_parameters(
            self.config,
            **self.ap.default_window_parameters["low"],
        )

        self.tf_obj = process_mth5(self.config, self.kernel_dataset)
        self.tf_obj.tf_id = self.kernel_dataset.processing_id
        self.tf_obj.station_metadata.transfer_function.runs_processed = (
            self.tf_obj.station_metadata.run_list
        )
        self.mt_obj_legacy = MT(survey_metadata=self.tf_obj.survey_metadata)
        self.mt_obj_legacy._transfer_function = self.tf_obj._transfer_function

        self.mt_obj_legacy.survey_metadata.id = (
            self.mt_obj_new.survey_metadata.id
        )

    def test_tfs_equal(self):
        self.assertEqual(self.mt_obj_new, self.mt_obj_legacy)

    def test_tf_id(self):
        self.assertEqual(self.mt_obj_new.tf_id, self.ap.processing_id)

    def test_tf_in_mth5(self):
        with MTH5() as m:
            m.open_mth5(self.mth5_path)
            tf_df = m.tf_summary.to_dataframe()
            with self.subTest("station is in tf_summary"):
                self.assertIn("test1", tf_df.station.tolist())
            with self.subTest("tf's are equal"):
                tf = m.get_transfer_function("test1", "test1_sr1")
                self.assertEqual(self.tf_obj, tf)

    def test_processed_dict(self):
        self.assertTrue(self.processed[1]["processed"])

    @classmethod
    def tearDownClass(self):
        close_open_files()


class TestProcessRRCompare(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()

        # process with mtpy
        self.ap = AuroraProcessing()
        self.ap.local_station_id = "test1"
        self.ap.local_mth5_path = self.mth5_path
        self.ap.remote_station_id = "test2"
        self.ap.remote_mth5_path = self.mth5_path
        self.processed = self.ap.process(
            sample_rates=1, merge=True, save_to_mth5=True
        )
        self.mt_obj_new = self.processed[1]["tf"]

        # process with aurora infrastructure
        self.run_summary = RunSummary()
        self.run_summary.from_mth5s([self.mth5_path])
        self.kernel_dataset = KernelDataset()
        self.kernel_dataset.from_run_summary(
            self.run_summary, "test1", "test2"
        )
        cc = ConfigCreator()
        self.config = cc.create_from_kernel_dataset(self.kernel_dataset)
        ## need to set same config parameters
        self.ap._set_decimation_level_parameters(
            self.config,
            **self.ap.default_window_parameters["low"],
        )

        self.tf_obj = process_mth5(self.config, self.kernel_dataset)
        self.tf_obj.tf_id = self.kernel_dataset.processing_id
        self.tf_obj.station_metadata.transfer_function.runs_processed = (
            self.tf_obj.station_metadata.run_list
        )
        self.mt_obj_legacy = MT(survey_metadata=self.tf_obj.survey_metadata)
        self.mt_obj_legacy._transfer_function = self.tf_obj._transfer_function

        self.mt_obj_legacy.survey_metadata.id = (
            self.mt_obj_new.survey_metadata.id
        )

    def test_tfs_equal(self):
        self.assertEqual(self.mt_obj_new, self.mt_obj_legacy)

    def test_tf_id(self):
        self.assertEqual(self.mt_obj_new.tf_id, self.ap.processing_id)

    def test_tf_in_mth5(self):
        with MTH5() as m:
            m.open_mth5(self.mth5_path)
            tf_df = m.tf_summary.to_dataframe()
            with self.subTest("station is in tf_summary"):
                self.assertIn("test1", tf_df.station.tolist())
            with self.subTest("tf's are equal"):
                tf = m.get_transfer_function("test1", "test1-rr_test2_sr1")
                self.assertEqual(self.tf_obj, tf)

    def test_processed_dict(self):
        self.assertTrue(self.processed[1]["processed"])

    @classmethod
    def tearDownClass(self):
        close_open_files()


# =============================================================================
# run
# =============================================================================
if __name__ in "__main__":
    unittest.main()
