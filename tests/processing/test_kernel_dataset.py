# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import unittest

from mth5.data.make_mth5_from_asc import MTH5_PATH, create_test12rr_h5

from mtpy.processing.run_summary import RunSummary

from mtpy.processing.kernel_dataset import intervals_overlap
from mtpy.processing.kernel_dataset import overlap
from mtpy.processing.kernel_dataset import KernelDataset

# =============================================================================


class TestKernelDataset(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(self):
        self.mth5_path = MTH5_PATH.joinpath("test12rr.h5")
        if not self.mth5_path.exists():
            self.mth5_path = create_test12rr_h5()
        self.run_summary = RunSummary()
        self.run_summary.from_mth5s(
            [
                self.mth5_path,
            ]
        )

    def setUp(self):
        self.run_summary = self.run_summary.clone()
        self.kd = KernelDataset()
        self.kd.from_run_summary(self.run_summary, "test1", "test2")

    def test_exception_from_empty_run_summary(self):
        # make the run summary df empty
        self.run_summary.df.valid = False
        self.run_summary.drop_invalid_rows()
        with self.assertRaises(ValueError):  # as context:
            self.kd.from_run_summary(self.run_summary, "test1", "test2")

    def test_clone_dataframe(self):
        cloned_df = self.kd.clone_dataframe()

        # fc column is None so this wont be true
        self.assertFalse((cloned_df == self.kd.df).all().all())

        cloned_df["fc"] = False
        self.kd.df["fc"] = False
        assert (cloned_df == self.kd.df).all().all()

    def test_clone(self):
        clone = self.kd.clone()

        # fc column is None so this wont be true
        self.assertFalse((clone.df == self.kd.df).all().all())

        clone.df["fc"] = False
        self.kd.df["fc"] = False
        assert (clone.df == self.kd.df).all().all()
        # add more checks


class TestOverlapFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Pick some time intervals and test that the overlap logic is correct

        Returns
        -------

        """
        # A day long interal
        self.ti1_start = pd.Timestamp(1980, 1, 1, 12, 30, 0)
        self.ti1_end = pd.Timestamp(1980, 1, 2, 12, 30, 0)
        self.shift_1_hours = 5
        self.shift_2_hours = 25
        # hours

        # shift the interval forward, leave it overlapping
        self.ti2_start = self.ti1_start + pd.Timedelta(hours=self.shift_1_hours)
        self.ti2_end = self.ti1_end + pd.Timedelta(hours=self.shift_1_hours)

        # shift the interval forward, non-verlapping
        self.ti3_start = self.ti1_start + pd.Timedelta(hours=self.shift_2_hours)
        self.ti3_end = self.ti1_end + pd.Timedelta(hours=self.shift_2_hours)

    def test_overlaps_boolean(self):
        self.assertTrue(
            intervals_overlap(
                self.ti1_start, self.ti1_end, self.ti2_start, self.ti2_end
            )
        )

        self.assertFalse(
            intervals_overlap(
                self.ti1_start, self.ti1_end, self.ti3_start, self.ti3_end
            )
        )

    def test_overlap_returns_interval(self):
        """
        TODO: there are four cases being handled ---
        add a subtest for each of the four
        Returns
        -------

        """
        # This test corresponds to the second line in the if/elif logic.
        tmp = overlap(
            self.ti1_start, self.ti1_end, self.ti2_start, self.ti2_end
        )
        self.assertTrue(
            tmp[0] == self.ti1_start + pd.Timedelta(hours=self.shift_1_hours)
        )
        self.assertTrue(tmp[1] == self.ti1_end)

        # TODO To test first line, we need t1 to completely enclose t2


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
