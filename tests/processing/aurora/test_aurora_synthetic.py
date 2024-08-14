# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:53:28 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

import unittest

from mth5.data.make_mth5_from_asc import MTH5_PATH, create_test12rr_h5
from mth5.utils.helpers import close_open_files

from mtpy.processing.run_summary import RunSummary
from mtpy.processing.kernel_dataset import KernelDataset
from mtpy import MT

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from mtpy.processing.aurora.process_aurora import AuroraProcessing


# =============================================================================
class TestProcessingSingleStationLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()

        self.run_summary = RunSummary()
        self.run_summary.from_mth5s([self.mth5_path])
        self.kernel_dataset = KernelDataset()
        self.kernel_dataset.from_run_summary(self.run_summary, "test1")
        cc = ConfigCreator()
        self.config = cc.create_from_kernel_dataset(self.kernel_dataset)

        self.tf_obj = process_mth5(self.config, self.kernel_dataset)
        self.mt_obj = MT(survey_metadata=self.tf_obj.survey_metadata)
        self.mt_obj._transfer_function = self.tf_obj._transfer_function

    def test_tf_obj(self):
        self.assertIsInstance(self.mt_obj, MT)

    def test_run_summary(self):
        """
        Most of the testing for run_summary is in test, this is just a quick
        check

        """

        with self.subTest("shape"):
            self.assertEqual(self.run_summary.df.shape, (2, 15))

    def test_kernel_dataset(self):
        with self.subTest("shape"):
            self.assertEqual(self.kernel_dataset.df.shape, (1, 20))
        with self.subTest("remote is None"):
            self.assertEqual(None, self.kernel_dataset.remote_station_id)
        with self.subTest("remote_df is None"):
            self.assertEqual(None, self.kernel_dataset.remote_df)
        with self.subTest("input_channels"):
            self.assertListEqual(
                self.kernel_dataset.input_channels, ["hx", "hy"]
            )
        with self.subTest("output_channels"):
            self.assertListEqual(
                self.kernel_dataset.output_channels, ["ex", "ey", "hz"]
            )

    def test_config(self):
        with self.subTest("input_channels"):
            self.assertListEqual(
                self.config.decimations[0].input_channels, ["hx", "hy"]
            )
        with self.subTest("output_channels"):
            self.assertListEqual(
                self.config.decimations[0].output_channels, ["ex", "ey", "hz"]
            )

    @classmethod
    def tearDownClass(self):
        close_open_files()


class TestAuroraProcessingSingleStationSingleSampleRate(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()

        self.ap = AuroraProcessing()
        self.ap.local_station_id = "test1"
        self.ap.local_mth5_path = self.mth5_path

        self.mt_obj = self.ap.process_single_sample_rate(1)

    def test_tf_obj(self):
        self.assertIsInstance(self.mt_obj, MT)

    def test_tf_id(self):
        self.assertEqual(self.mt_obj.tf_id, self.ap.processing_id)

    @classmethod
    def tearDownClass(self):
        close_open_files()


class TestProcessingSingleStationCompare(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()

        self.run_summary = RunSummary()
        self.run_summary.from_mth5s([self.mth5_path])
        self.kernel_dataset = KernelDataset()
        self.kernel_dataset.from_run_summary(self.run_summary, "test1")
        cc = ConfigCreator()
        self.config = cc.create_from_kernel_dataset(self.kernel_dataset)

        self.ap = AuroraProcessing()
        self.ap.local_station_id = "test1"
        self.ap.local_mth5_path = self.mth5_path

        ## need to set same config parameters
        self.ap._set_decimation_level_parameters(
            self.config,
            **self.ap.default_window_parameters["low"],
        )

        self.tf_obj = process_mth5(self.config, self.kernel_dataset)
        self.mt_obj_legacy = MT(survey_metadata=self.tf_obj.survey_metadata)
        self.mt_obj_legacy._transfer_function = self.tf_obj._transfer_function
        self.mt_obj_legacy.tf_id = self.kernel_dataset.processing_id

        self.mt_obj_new = self.ap.process_single_sample_rate(1)

    def test_mt_objs_equal(self):
        self.assertEqual(self.mt_obj_legacy, self.mt_obj_new)

    def test_tf_id(self):
        self.assertEqual(self.mt_obj_new.tf_id, self.ap.processing_id)

    @classmethod
    def tearDownClass(self):
        close_open_files()


class TestProcessingRRStationCompare(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()

        self.run_summary = RunSummary()
        self.run_summary.from_mth5s([self.mth5_path])
        self.kernel_dataset = KernelDataset()
        self.kernel_dataset.from_run_summary(
            self.run_summary, "test1", "test2"
        )
        cc = ConfigCreator()
        self.config = cc.create_from_kernel_dataset(self.kernel_dataset)

        self.ap = AuroraProcessing()
        self.ap.local_station_id = "test1"
        self.ap.local_mth5_path = self.mth5_path
        self.ap.remote_station_id = "test2"
        self.ap.remote_mth5_path = self.mth5_path

        ## need to set same config parameters
        self.ap._set_decimation_level_parameters(
            self.config,
            **self.ap.default_window_parameters["low"],
        )

        self.tf_obj = process_mth5(self.config, self.kernel_dataset)
        self.mt_obj_legacy = MT(survey_metadata=self.tf_obj.survey_metadata)
        self.mt_obj_legacy._transfer_function = self.tf_obj._transfer_function
        self.mt_obj_legacy.tf_id = self.kernel_dataset.processing_id

        self.mt_obj_new = self.ap.process_single_sample_rate(1)

    def test_mt_objs_equal(self):
        self.assertEqual(self.mt_obj_legacy, self.mt_obj_new)

    def test_tf_id(self):
        self.assertEqual(self.mt_obj_new.tf_id, self.ap.processing_id)

    @classmethod
    def tearDownClass(self):
        close_open_files()


# =============================================================================
# run
# =============================================================================
if __name__ in "__main__":
    unittest.main()
