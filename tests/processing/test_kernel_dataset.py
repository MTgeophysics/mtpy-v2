# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import unittest

from mth5.data.make_mth5_from_asc import MTH5_PATH, create_test12rr_h5

from mtpy.processing import KERNEL_DATASET_COLUMNS
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
        self.run_summary.from_mth5s([self.mth5_path])

    def setUp(self):
        self.run_summary = self.run_summary.clone()
        self.kd = KernelDataset()
        self.kd.from_run_summary(self.run_summary, "test1", "test2")

    def test_df_columns(self):
        self.assertListEqual(
            sorted(KERNEL_DATASET_COLUMNS), sorted(self.kd.df.columns)
        )

    def test_set_df_fail_bad_type(self):
        def set_df(value):
            self.kd.df = value

        self.assertRaises(TypeError, set_df, 10)

    def test_set_df_fail_bad_df(self):
        def set_df(value):
            self.kd.df = value

        self.assertRaises(ValueError, set_df, pd.DataFrame({"test": [0]}))

    def test_df_shape(self):
        self.assertEqual((2, 15), self.kd.df.shape)

    def test_exception_from_empty_run_summary(self):
        self.run_summary.df.loc[:, "has_data"] = False
        self.run_summary.drop_no_data_rows()
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

    def test_mini_summary(self):
        mini_df = self.kd.mini_summary
        self.assertListEqual(
            sorted(self.kd._mini_summary_columns), sorted(mini_df.columns)
        )

    # @classmethod
    # def tearDownClass(self):
    #     self.mth5_path.unlink()


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


class TestKernelDatasetMethods(unittest.TestCase):
    def setUp(self):
        self.local = "mt01"
        self.remote = "mt02"
        self.rs_df = pd.DataFrame(
            {
                "channel_scale_factors": [1, 1, 1, 1],
                "duration": [0, 0, 0, 0],
                "end": [
                    "2020-01-01T23:59:59",
                    "2020-01-02T02:59:59",
                    "2020-01-01T22:00:00",
                    "2020-01-02T02:00:00.5",
                ],
                "has_data": [True, True, True, True],
                "input_channels": ["hx, hy"] * 4,
                "mth5_path": ["path"] * 4,
                "n_samples": [86400, 86400, 79200, 7200],
                "output_channels": ["hz, ex, ey"] * 4,
                "run": ["01", "02", "03", "04"],
                "sample_rate": [1, 1, 1, 1],
                "start": [
                    "2020-01-01T00:00:00",
                    "2020-01-02T00:00:00",
                    "2020-01-01T00:00:00",
                    "2020-01-02T00:00:00.5",
                ],
                "station": [self.local, self.local, self.remote, self.remote],
                "survey": ["test"] * 4,
            }
        )
        self.run_summary = RunSummary(df=self.rs_df)
        self.kd = KernelDataset()
        self.kd.from_run_summary(self.run_summary, self.local, self.remote)

    def test_from_run_summary(self):

        with self.subTest("local_station_id"):
            self.assertEqual(self.kd.local_station_id, self.local)
        with self.subTest("remote_station_id"):
            self.assertEqual(self.kd.remote_station_id, self.remote)
        with self.subTest("has remote column"):
            self.assertIn("remote", self.kd.df.columns)
        with self.subTest("has fc column"):
            self.assertIn("fc", self.kd.df.columns)

    def test_num_sample_rates(self):
        self.assertEqual(self.kd.num_sample_rates, 1)

    def test_sample_rate(self):
        self.assertEqual(self.kd.sample_rate, 1)


class TestKernelDatasetMethodsFail(unittest.TestCase):
    def setUp(self):
        self.local = "mt01"
        self.remote = "mt02"
        self.rs_df = pd.DataFrame(
            {
                "channel_scale_factors": [1, 1, 1, 1],
                "duration": [0, 0, 0, 0],
                "end": [
                    "2020-01-01T23:59:59",
                    "2020-01-01T02:59:59",
                    "2020-02-02T22:00:00",
                    "2020-02-02T02:00:00.5",
                ],
                "has_data": [True, True, True, True],
                "input_channels": ["hx, hy"] * 4,
                "mth5_path": ["path"] * 4,
                "n_samples": [86400, 86400, 79200, 7200],
                "output_channels": ["hz, ex, ey"] * 4,
                "run": ["01", "02", "03", "04"],
                "sample_rate": [1, 4, 1, 4],
                "start": [
                    "2020-01-01T00:00:00",
                    "2020-01-01T00:00:00",
                    "2020-02-02T00:00:00",
                    "2020-02-02T00:00:00.5",
                ],
                "station": [self.local, self.local, self.remote, self.remote],
                "survey": ["test"] * 4,
            }
        )
        self.run_summary = RunSummary(df=self.rs_df)
        self.kd = KernelDataset()

    def test_from_run_summary(self):
        self.assertRaises(
            ValueError,
            self.kd.from_run_summary,
            self.run_summary,
            self.local,
            self.remote,
        )

    def test_num_sample_rates(self):
        self.kd.from_run_summary(self.run_summary, self.local)
        self.assertEqual(self.kd.num_sample_rates, 2)

    def test_sample_rate_fail(self):
        self.kd.from_run_summary(self.run_summary, self.local)

        def get_sample_rate():
            return self.kd.sample_rate

        self.assertRaises(NotImplementedError, get_sample_rate)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
