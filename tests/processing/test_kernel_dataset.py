# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import pandas as pd
import unittest

from mth5.data.make_mth5_from_asc import MTH5_PATH, create_test12rr_h5

from mtpy.processing import KERNEL_DATASET_COLUMNS, KernelDataset, RunSummary
from mtpy.processing.kernel_dataset import (
    intervals_overlap,
    overlap,
)

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
        self.setup_run_summary = self.run_summary.clone()
        self.kd = KernelDataset()
        self.kd.from_run_summary(self.setup_run_summary, "test1", "test2")

    def test_df(self):
        self.assertTrue(self.kd._has_df())

    def test_df_has_local_station_id(self):
        self.assertTrue(self.kd._df_has_local_station_id)

    def test_df_has_remote_station_id(self):
        self.assertTrue(self.kd._df_has_remote_station_id)

    def test_set_datetime_columns(self):
        new_df = self.kd._set_datetime_columns(self.kd.df)
        with self.subTest("start"):
            self.assertEqual(new_df.start.dtype.type, pd.Timestamp)
        with self.subTest("end"):
            self.assertEqual(new_df.end.dtype.type, pd.Timestamp)

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
        self.assertEqual((2, 20), self.kd.df.shape)

    def test_exception_from_empty_run_summary(self):
        rs = self.run_summary.clone()
        rs.df.loc[:, "has_data"] = False
        rs.drop_no_data_rows()
        with self.assertRaises(ValueError):
            self.kd.from_run_summary(rs, "test1", "test2")

    def test_clone_dataframe(self):
        cloned_df = self.kd.clone_dataframe()

        # need to change na to a value
        cloned_df.fillna(0, inplace=True)
        self.kd.df.fillna(0, inplace=True)

        # fc column is None so this wont be true
        self.assertTrue((cloned_df == self.kd.df).all().all())

    def test_clone(self):
        clone = self.kd.clone()

        # need to change na to a value
        clone.df.fillna(0, inplace=True)
        self.kd.df.fillna(0, inplace=True)

        self.assertTrue((clone.df == self.kd.df).all().all())

    def test_mini_summary(self):
        mini_df = self.kd.mini_summary
        self.assertListEqual(
            sorted(self.kd._mini_summary_columns), sorted(mini_df.columns)
        )

    def test_local_station_id(self):
        self.assertEqual("test1", self.kd.local_station_id)

    def test_set_local_station_id_fail(self):
        self.assertRaises(
            NameError, self.kd.__setattr__, "local_station_id", "test3"
        )

    def test_set_remote_station_id_fail(self):
        self.assertRaises(
            NameError, self.kd.__setattr__, "remote_station_id", "test3"
        )

    def test_str(self):
        mini_df = self.kd.mini_summary
        self.assertEqual(str(mini_df.head()), str(self.kd))

    def test_input_channels(self):
        self.assertListEqual(["hx", "hy"], self.kd.input_channels)

    def test_output_channels(self):
        self.assertListEqual(["ex", "ey", "hz"], self.kd.output_channels)

    def test_local_df(self):
        with self.subTest("shape"):
            self.assertEqual(self.kd.local_df.shape, (1, 20))
        with self.subTest("local station only length"):
            self.assertEqual(len(self.kd.local_df.station.unique()), 1)
        with self.subTest("local station only"):
            self.assertListEqual(
                list(self.kd.local_df.station.unique()), ["test1"]
            )

    def test_remote_df(self):
        with self.subTest("shape"):
            self.assertEqual(self.kd.remote_df.shape, (1, 20))
        with self.subTest("remote station only length"):
            self.assertEqual(len(self.kd.remote_df.station.unique()), 1)
        with self.subTest("remote station only"):
            self.assertListEqual(
                list(self.kd.remote_df.station.unique()), ["test2"]
            )

    def test_processing_id(self):
        self.assertEqual(self.kd.processing_id, "test1-rr_test2_sr1")

    def test_local_survey_id(self):
        self.assertEqual("EMTF Synthetic", self.kd.local_survey_id)

    def test_set_run_times(self):
        times = {
            "001": {
                "start": "1980-01-01T01:00:00+00:00",
                "end": "1980-01-01T08:00:00+00:00",
            }
        }

        self.kd.set_run_times(times)
        with self.subTest("local duration"):
            self.assertEqual(self.kd.df.iloc[0].duration, 25200)
        with self.subTest("remote duration"):
            self.assertEqual(self.kd.df.iloc[1].duration, 25200)

    def test_set_run_times_bad_input(self):
        self.assertRaises(TypeError, self.kd.set_run_times, 10)

    def test_set_run_times_bad_dict(self):
        self.assertRaises(TypeError, self.kd.set_run_times, {"001": 10})

    def test_set_run_times_bad_dict_keys(self):
        self.assertRaises(KeyError, self.kd.set_run_times, {"001": {"a": 1}})

    def test_sample_rate(self):
        self.assertEqual(1, self.kd.sample_rate)

    def test_num_sample_rates(self):
        self.assertEqual(1, self.kd.num_sample_rates)

    @classmethod
    def tearDownClass(self):
        self.mth5_path.unlink()


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
                "mth5_path": ["path"] * 2 + ["remote_path"] * 2,
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
                "run_hdf5_reference": [None] * 4,
                "station_hdf5_reference": [None] * 4,
            }
        )
        self.run_summary = RunSummary(df=self.rs_df)
        self.kd = KernelDataset()
        self.kd.from_run_summary(self.run_summary, self.local, self.remote)

    def test_from_run_summary_local_station_id(self):
        self.assertEqual(self.kd.local_station_id, self.local)

    def test_from_run_summary_remote_station_id(self):
        self.assertEqual(self.kd.remote_station_id, self.remote)

    def test_from_run_summary_has_duration(self):
        self.assertFalse((self.kd.df.duration == 0).all())

    def test_from_run_summary_has_all_columns(self):
        self.assertListEqual(
            sorted(self.kd.df.columns), sorted(KERNEL_DATASET_COLUMNS)
        )

    def test_from_run_summary_local_mth5_path(self):
        self.assertEqual(self.kd.local_mth5_path, Path("path"))

    def test_from_run_summary_remote_mth5_path(self):
        self.assertEqual(self.kd.remote_mth5_path, Path("remote_path"))

    def test_from_run_summary_local_mth5_path_False(self):
        self.assertFalse(self.kd.has_local_mth5())

    def test_from_run_summary_remote_mth5_path_False(self):
        self.assertFalse(self.kd.has_remote_mth5())

    def test_num_sample_rates(self):
        self.assertEqual(self.kd.num_sample_rates, 1)

    def test_sample_rate(self):
        self.assertEqual(self.kd.sample_rate, 1)

    def test_drop_runs_shorter_than(self):
        self.kd.drop_runs_shorter_than(8000)
        self.assertEqual((2, 20), self.kd.df.shape)

    def test_survey_id(self):
        self.assertEqual(self.kd.local_survey_id, "test")

    def test_update_duration_column_not_inplace(self):
        new_df = self.kd._update_duration_column(inplace=False)

        self.assertTrue((new_df.duration == self.kd.df.duration).all())

    def test_set_local_station_id_fail(self):
        def set_station(value):
            self.kd.local_station_id = value

        self.assertRaises(NameError, set_station, "mt03")

    def test_set_remote_station_id_fail(self):
        def set_station(value):
            self.kd.remote_station_id = value

        self.assertRaises(NameError, set_station, "mt03")


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
                "run_hdf5_reference": [None] * 4,
                "station_hdf5_reference": [None] * 4,
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
