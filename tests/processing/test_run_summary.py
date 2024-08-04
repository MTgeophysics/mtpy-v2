# import logging
# =============================================================================
# Imports
# =============================================================================
import unittest
import pandas as pd

from mtpy.processing.run_summary import RunSummary
from mtpy.processing import RUN_SUMMARY_COLUMNS
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.paths import DATA_PATH

# =============================================================================


class TestRunSummary(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.mth5_path = DATA_PATH.joinpath("synthetic", "mth5", "test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()
        self.rs = RunSummary()
        self.rs.from_mth5s(
            [
                self.mth5_path,
            ]
        )

        self.maxDiff = None

    def test_df_columns(self):
        self.assertListEqual(
            sorted(RUN_SUMMARY_COLUMNS), sorted(self.rs.df.columns)
        )

    def test_set_df_fail_bad_type(self):
        def set_df(value):
            self.rs.df = value

        self.assertRaises(TypeError, set_df, 10)

    def test_set_df_fail_bad_df(self):
        def set_df(value):
            self.rs.df = value

        self.assertRaises(ValueError, set_df, pd.DataFrame({"test": [0]}))

    def test_df_shape(self):
        self.assertEqual((2, 13), self.rs.df.shape)

    def test_clone(self):
        rs_clone = self.rs.clone()
        self.assertEqual(True, (self.rs.df == rs_clone.df).all().all())

    def test_mini_summary(self):
        mini_df = self.rs.mini_summary
        self.assertListEqual(
            sorted(self.rs._mini_summary_columns), sorted(mini_df.columns)
        )

    def test_drop_no_data_rows(self):
        rs_clone = self.rs.clone()
        rs_clone.df.loc[0, "has_data"] = False

        rs_clone._warn_no_data_runs()
        rs_clone.drop_no_data_rows()
        self.assertEqual((1, 13), rs_clone.df.shape)

    def test_set_sample_rate(self):
        new_rs = self.rs.set_sample_rate(1)
        self.assertEqual(True, (self.rs.df == new_rs.df).all().all())

    def test_set_sample_rate_faile(self):
        self.assertRaises(ValueError, self.rs.set_sample_rate, 10)

    # @classmethod
    # def tearDownClass(self):
    #     self.mth5_path.unlink()


# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    unittest.main()
