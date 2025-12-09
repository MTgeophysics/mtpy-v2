# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 10:59:50 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

import numpy as np
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core.mt_dataframe import MTDataFrame


# =============================================================================


class TestMTDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m1 = MT(TF_EDI_CGG)
        self.m1.read()
        self.m1.utm_epsg = 32752
        self.m1.model_east = 200
        self.m1.model_north = 1000
        self.m1.model_elevation = 20

        self.sdf = self.m1.to_dataframe()
        self.sdf.working_station = "TEST01"

    def test_column_names(self):
        self.assertListEqual(
            self.sdf._column_names, [col[0] for col in self.sdf._dtype_list]
        )

    def test_pt_attrs(self):
        self.assertListEqual(
            self.sdf._pt_attrs,
            [col for col in self.sdf._column_names if col.startswith("pt")],
        )

    def test_tipper_attrs(self):
        self.assertListEqual(
            self.sdf._tipper_attrs,
            [col for col in self.sdf._column_names if col.startswith("t_")],
        )

    def test_station(self):
        self.assertEqual(self.sdf.station, "TEST01")

    def test_period(self):
        self.assertEqual(self.sdf.period.size, 73)

    def test_latitude(self):
        self.assertEqual(self.sdf.latitude, -30.930285)

    def test_longitude(self):
        self.assertEqual(self.sdf.longitude, 127.22923)

    def test_elevation(self):
        self.assertEqual(self.sdf.elevation, 175.27)

    def test_east(self):
        self.assertAlmostEqual(self.sdf.east, 330815.90793634474)

    def test_north(self):
        self.assertAlmostEqual(self.sdf.north, 6576780.151722098)

    def test_utm_epsg(self):
        self.assertAlmostEqual(self.sdf.utm_epsg, 32752)

    def test_model_east(self):
        self.assertEqual(self.sdf.model_east, self.m1.model_east)

    def test_model_north(self):
        self.assertEqual(self.sdf.model_north, self.m1.model_north)

    def test_model_elevation(self):
        self.assertEqual(self.sdf.model_elevation, self.m1.model_elevation)

    def test_to_z_object(self):
        new_z = self.sdf.to_z_object()
        self.assertTrue(self.m1.Z == new_z)

    def test_to_t_object(self):
        new_t = self.sdf.to_t_object()
        self.assertTrue(self.m1.Tipper == new_t)

    def test_from_z_object(self):
        new_df = MTDataFrame(n_entries=self.sdf.size)
        new_df.from_z_object(self.m1.Z)
        new_z = new_df.to_z_object()
        self.assertTrue(self.m1.Z == new_z)

    def test_from_t_object(self):
        new_df = MTDataFrame(n_entries=self.sdf.size)
        new_df.from_t_object(self.m1.Tipper)
        new_t = new_df.to_t_object()
        self.assertTrue(self.m1.Tipper == new_t)

    def test_pt_df(self):
        pt_df = self.sdf.phase_tensor

        comp_dict = {
            "pt_xx": (0, 0),
            "pt_xy": (0, 1),
            "pt_yx": (1, 0),
            "pt_yy": (1, 1),
        }

        for comp, index in comp_dict.items():
            with self.subTest(comp):
                self.assertTrue(
                    np.all(pt_df[comp] == self.m1.pt.pt[:, index[0], index[1]])
                )
        for comp in [
            "azimuth",
            "skew",
            "phimin",
            "phimax",
            "det",
            "ellipticity",
        ]:
            with self.subTest(comp):
                self.assertTrue(
                    np.all(pt_df[f"pt_{comp}"] == getattr(self.m1.pt, comp))
                )

    def test_tip_df(self):
        tip_df = self.sdf.tipper

        comp_dict = {
            "t_zx": (0, 0),
            "t_zy": (0, 1),
        }

        for comp, index in comp_dict.items():
            with self.subTest(comp):
                self.assertTrue(
                    np.all(tip_df[comp] == self.m1.Tipper.tipper[:, index[0], index[1]])
                )
        for comp in ["angle_real", "angle_imag", "mag_real", "mag_imag"]:
            with self.subTest(comp):
                self.assertTrue(
                    np.all(tip_df[f"t_{comp}"] == getattr(self.m1.Tipper, comp))
                )


class TestMTDataFrameValidation(unittest.TestCase):
    def setUp(self):
        self.sdf = MTDataFrame()

    def test_bad_input_fail(self):
        self.assertRaises(TypeError, self.sdf._validate_data, 10)

    def test_from_dict(self):
        df = MTDataFrame(
            {
                "station": "a",
                "period": [0, 1],
                "latitude": 10,
                "longitude": 20,
                "elevation": 30,
            }
        )
        with self.subTest("size"):
            self.assertTrue(df.size == 2)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
