# import logging
# =============================================================================
# Imports
# =============================================================================
import unittest

from mtpy.processing.run_summary import RunSummary
from mtpy.processing import RUN_SUMMARY_COLUMNS
from aurora.test_utils.synthetic.make_mth5_from_asc import create_test12rr_h5
from aurora.test_utils.synthetic.paths import DATA_PATH

# =============================================================================


class TestRunSummary(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.mth5_path = DATA_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()
        self.rs = RunSummary()
        self.rs.from_mth5s(
            [
                self.mth5_path,
            ]
        )

    def test_df_columns(self):
        self.assertListEqual(
            sorted(RUN_SUMMARY_COLUMNS), sorted(self.rs.df.columns)
        )

    def test_df_shape(self):
        self.assertEqual((2, 13), self.rs.df.shape)

    def test_clone(self):
        rs_clone = self.rs.clone()
        self.assertEqual(rs_clone.df, self.rs.df)

    def test_mini_summary(self):
        mini_df = self.rs.mini_summary
        self.assertListEqual(
            sorted(self.rs._mini_summary_columns), sorted(mini_df.columns)
        )

    @classmethod
    def tearDownClass(self):
        self.mth5_path.unlink()


# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    unittest.main()
