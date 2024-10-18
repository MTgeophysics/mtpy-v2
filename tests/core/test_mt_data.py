# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:40:56 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

from mtpy import MT, MTData

# =============================================================================


class TestMTData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.utm_epsg = 3216
        self.datum_epsg = 4236
        self.mt_list_01 = [
            MT(
                survey="a",
                station=f"mt{ii:02}",
                latitude=40 + ii,
                longitude=-118,
            )
            for ii in range(4)
        ]
        self.mt_list_02 = [
            MT(
                survey="b",
                station=f"mt{ii:02}",
                latitude=45 + ii,
                longitude=-118,
            )
            for ii in range(4)
        ]

        self.md = MTData(
            mt_list=self.mt_list_01 + self.mt_list_02, utm_epsg=self.utm_epsg
        )

    def test_validate_item_fail(self):
        self.assertRaises(TypeError, self.md._validate_item, 10)

    def test_eq(self):
        md = MTData(
            mt_list=self.mt_list_01 + self.mt_list_02, utm_epsg=self.utm_epsg
        )

        self.assertEqual(self.md, md)

    def test_neq(self):
        md = MTData(mt_list=self.mt_list_01, utm_epsg=self.utm_epsg)

        self.assertNotEqual(self.md, md)

    def test_deep_copy(self):
        md = self.md.copy()
        self.assertEqual(self.md, md)

    def test_utm_epsg(self):
        self.assertEqual(self.md.utm_epsg, self.utm_epsg)

    def test_clone_empty(self):
        md_empty = self.md.clone_empty()

        for attr in self.md._copy_attrs:
            with self.subTest(attr):
                self.assertEqual(
                    getattr(self.md, attr), getattr(md_empty, attr)
                )

    def test_initialization_utm_epsg_no_mt_list(self):
        md = MTData(utm_epsg=self.utm_epsg)
        self.assertEqual(md.utm_epsg, self.utm_epsg)

    def test_coordinate_reference_frame(self):
        self.assertEqual("NED", self.md.coordinate_reference_frame)

    def test_coordinate_reference_frame_set(self):
        md = MTData(mt_list=self.mt_list_01, coordinate_reference_frame="enu")

        with self.subTest("mtdata"):
            self.assertEqual("ENU", md.coordinate_reference_frame)

        for mt_obj in md.values():
            with self.subTest(mt_obj.station):
                self.assertEqual("ENU", mt_obj.coordinate_reference_frame)

    def test_initialization_datum_epsg_no_mt_list(self):
        md = MTData(datum_epsg=self.datum_epsg)
        self.assertEqual(md.datum_epsg, self.datum_epsg)

    def test_survey_ids(self):
        self.assertListEqual(["a", "b"], sorted(self.md.survey_ids))

    def test_get_survey(self):
        a = self.md.get_survey("a")

        with self.subTest("length"):
            self.assertEqual(4, len(a))

        for attr in self.md._copy_attrs:
            with self.subTest(attr):
                self.assertEqual(getattr(self.md, attr), getattr(a, attr))

    def test_rotate_inplace(self):
        md = self.md.copy()
        md.rotate(30)

        with self.subTest("MTData rotation angle"):
            self.assertEqual(md.data_rotation_angle, 30)

        with self.subTest("MT rotation angle"):
            self.assertEqual(md["a.mt01"].rotation_angle, 30)

    def test_rotate_not_inplace(self):
        md_rot = self.md.rotate(30, inplace=False)

        with self.subTest("MTData rotation angle"):
            self.assertEqual(md_rot.data_rotation_angle, 30)

        with self.subTest("MT rotation angle"):
            self.assertEqual(md_rot["a.mt01"].rotation_angle, 30)

    # def test_get_station_from_id(self):
    #     a = self.md.get_station("mt01")
    #     self.assertEqual(a.station, "mt01")

    def test_get_station_from_key(self):
        a = self.md.get_station(station_key="a.mt01")
        self.assertEqual(a.station, "mt01")

    def test_get_subset_from_ids_fail(self):
        station_list = ["mt01", "mt02"]
        self.assertRaises(KeyError, self.md.get_subset, station_list)

    def test_get_subset_from_keys(self):
        station_keys = ["a.mt01", "b.mt02"]
        md = self.md.get_subset(station_keys)
        with self.subTest("keys"):
            self.assertListEqual(station_keys, list(md.keys()))

        for attr in self.md._copy_attrs:
            with self.subTest(attr):
                self.assertEqual(getattr(self.md, attr), getattr(md, attr))

    def test_n_station(self):
        self.assertEqual(8, self.md.n_stations)


class TestMTDataMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.utm_epsg = 3216
        self.datum_epsg = 4236
        self.mt_list_01 = [
            MT(
                survey="a",
                station=f"mt{ii:02}",
                latitude=40 + ii,
                longitude=-118,
            )
            for ii in range(4)
        ]

    def setUp(self):
        self.md = MTData()

    def test_add_station(self):
        self.md.add_station(self.mt_list_01[0])

        self.assertListEqual(["a.mt00"], list(self.md.keys()))

    def test_remove_station(self):
        self.md.add_station(self.mt_list_01)
        self.md.remove_station("mt00", "a")

        self.assertNotIn("a.mt00", list(self.md.keys()))

    def test_get_station_key(self):
        self.md.add_station(self.mt_list_01)

        with self.subTest("no survey"):
            self.assertEqual("a.mt01", self.md._get_station_key("mt01", None))
        with self.subTest("with survey"):
            self.assertEqual("a.mt01", self.md._get_station_key("mt01", "a"))
        with self.subTest("fail"):
            self.assertRaises(KeyError, self.md._get_station_key, None, "a")


# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    unittest.main()
