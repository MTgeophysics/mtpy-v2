# -*- coding: utf-8 -*-
"""
tests update to v2
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np
from loguru import logger

from mtpy.utils import gis_tools

# =============================================================================


class TestGisTools(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.lat_hhmmss = "-34:17:57.99"
        self.lat_str = "-34.299442"
        self.lat_fail = "-34:29.9442"
        self.lat_d = -34.299442

        self.lon_hhmmss = "149:12:03.71"
        self.lon_str = "149.2010301"
        self.lon_fail = "149:12.0371"
        self.lon_d = 149.2010301

        self.elev_d = 1254.1
        self.elev_fail = "1200m"

        self.zone = "55H"
        self.zone_number = 55
        self.is_northern = False
        self.utm_letter = "H"
        self.zone_epsg = 32755
        self.easting = 702562.690286
        self.northing = 6202448.52785
        self.atol = 0.3  # tolerance of error
        self.from_epsg = 4326
        self.to_epsg = 28355

    def test_assert_minutes(self):
        self.assertEqual(10, gis_tools.assert_minutes(10))

    def test_assert_seconds(self):
        self.assertEqual(10, gis_tools.assert_seconds(10))

    def test_convert_position_str2float(self):
        with self.subTest("time deg"):
            self.assertAlmostEqual(
                gis_tools.convert_position_str2float(self.lat_hhmmss),
                self.lat_d,
                places=5,
            )
        with self.subTest("decimal deg"):
            self.assertAlmostEqual(
                gis_tools.convert_position_str2float(self.lat_str),
                self.lat_d,
                places=5,
            )

    def test_convert_position_str2float_fail(self):
        self.assertRaises(
            ValueError, gis_tools.convert_position_str2float, self.lat_fail
        )

    def test_assert_lat_value_none(self):
        self.assertEqual(0, gis_tools.assert_lat_value(None))

    def test_assert_lat_value_pass(self):
        self.assertEqual(10.0, gis_tools.assert_lat_value(10))

    def test_assert_lat_value_type_error(self):
        self.assertRaises(TypeError, gis_tools.assert_lat_value, {})

    def test_assert_lat_value_value_error(self):
        self.assertRaises(ValueError, gis_tools.assert_lat_value, 100)

    def test_assert_lon_value_none(self):
        self.assertEqual(0, gis_tools.assert_lon_value(None))

    def test_assert_lon_value_pass(self):
        self.assertEqual(10.0, gis_tools.assert_lon_value(10))

    def test_assert_lon_value_type_error(self):
        self.assertRaises(TypeError, gis_tools.assert_lon_value, {})

    def test_assert_lon_value_value_error(self):
        self.assertRaises(ValueError, gis_tools.assert_lon_value, 200)

    def test_assert_elevation_value_pass(self):
        self.assertEqual(10, gis_tools.assert_elevation_value(10))

    def test_assert_elevation_value_pass_fall(self):
        self.assertEqual(0, gis_tools.assert_elevation_value({}))

    def test_convert_position_float2str(self):
        self.assertEqual(
            self.lat_hhmmss, gis_tools.convert_position_float2str(self.lat_d)
        )

    def test_convert_position_float2str_fail(self):
        self.assertRaises(TypeError, gis_tools.convert_position_float2str, {})

    def test_validate_input_values(self):
        in_array = np.array([1, 20, 45, 67.2342])
        self.assertTrue(
            (
                in_array
                == gis_tools.validate_input_values(in_array.tolist(), "lat")
            ).all()
        )

    def test_validate_input_values_list_of_strings(self):
        in_array = np.array([1, 20, 45, 67.2342])
        self.assertTrue(
            (
                in_array
                == gis_tools.validate_input_values(
                    in_array.astype(str).tolist(), "lat"
                )
            ).all()
        )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
