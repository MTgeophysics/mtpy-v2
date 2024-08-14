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


# =============================================================================
class TestProcessingSingleStation(unittest.TestCase):
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

    @classmethod
    def tearDownClass(self):
        close_open_files()


# =============================================================================
# run
# =============================================================================
if __name__ in "__main__":
    unittest.main()