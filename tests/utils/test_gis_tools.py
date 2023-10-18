# -*- coding: utf-8 -*-
"""
tests update to v2
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np

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

        self.easting = 702562.1378419038
        self.easting_36 = 702562.690285791
        self.northing = 6202447.112735093
        self.northing_36 = 6202448.527851131
        self.datum_epsg = 4326
        self.utm_epsg = 28355

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

    def test_convert_position_str2float_None(self):
        self.assertEqual(0, gis_tools.convert_position_str2float(None))

    def test_assert_lat_value_none(self):
        self.assertEqual(0, gis_tools.assert_lat_value(None))

    def test_assert_lat_value_pass(self):
        self.assertEqual(10.0, gis_tools.assert_lat_value(10))

    def test_assert_lat_value_str(self):
        self.assertEqual(10.0, gis_tools.assert_lat_value("10"))

    def test_assert_lat_value_type_error(self):
        self.assertRaises(TypeError, gis_tools.assert_lat_value, {})

    def test_assert_lat_value_value_error(self):
        self.assertRaises(ValueError, gis_tools.assert_lat_value, 100)

    def test_assert_lon_value_none(self):
        self.assertEqual(0, gis_tools.assert_lon_value(None))

    def test_assert_lon_value_pass(self):
        self.assertEqual(10.0, gis_tools.assert_lon_value(10))

    def test_assert_lon_value_str(self):
        self.assertEqual(10.0, gis_tools.assert_lon_value("10"))

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

    def test_validate_input_values_lat(self):
        in_array = np.array([1, 20, 45, 67.2342])
        self.assertTrue(
            (
                in_array
                == gis_tools.validate_input_values(in_array.tolist(), "lat")
            ).all()
        )

    def test_validate_input_values_latitude(self):
        in_array = np.array([1, 20, 45, 67.2342])
        self.assertTrue(
            (
                in_array
                == gis_tools.validate_input_values(
                    in_array.tolist(), "latitude"
                )
            ).all()
        )

    def test_validate_input_values_lon(self):
        in_array = np.array([1, 20, 45, 67.2342])
        self.assertTrue(
            (
                in_array
                == gis_tools.validate_input_values(in_array.tolist(), "lon")
            ).all()
        )

    def test_validate_input_values_longitude(self):
        in_array = np.array([1, 20, 45, 67.2342])
        self.assertTrue(
            (
                in_array
                == gis_tools.validate_input_values(
                    in_array.tolist(), "longitude"
                )
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

    def test_validate_input_values_fail(self):
        in_array = np.array([1, 20, 45, 167.2342])
        self.assertRaises(
            gis_tools.GISError,
            gis_tools.validate_input_values,
            in_array,
            "lat",
        )

    def test_project_point_fail_old_epsg(self):
        self.assertRaises(
            ValueError, gis_tools.project_point, 1, 2, None, None
        )

    def test_project_point_fail_new_epsg(self):
        self.assertRaises(
            ValueError, gis_tools.project_point, 1, 2, 4326, None
        )

    def test_project_point_fail_x_zeros(self):
        self.assertRaises(
            ValueError, gis_tools.project_point, 0, 10, 4326, 32611
        )

    def test_project_point_fail_y_zeros(self):
        self.assertRaises(
            ValueError, gis_tools.project_point, 10, 0, 4326, 32611
        )

    def test_project_point(self):
        point = gis_tools.project_point(
            self.lon_d, self.lat_d, self.datum_epsg, self.utm_epsg
        )

        if round(point[0], 3) == round(self.easting, 3):
            with self.subTest("easting"):
                self.assertAlmostEqual(point[0], self.easting, 5)
            with self.subTest("northing"):
                self.assertAlmostEqual(point[1], self.northing, 5)

        else:
            with self.subTest("easting"):
                self.assertAlmostEqual(point[0], self.easting_36, 5)
            with self.subTest("northing"):
                self.assertAlmostEqual(point[1], self.northing_36, 5)

    def test_project_point_ll2utm_point(self):
        point = gis_tools.project_point_ll2utm(
            self.lat_d,
            self.lon_d,
            datum="WGS84",
            epsg=self.utm_epsg,
        )

        with self.subTest("type"):
            self.assertIsInstance(point, tuple)

        if round(point[0], 3) == round(self.easting, 3):
            with self.subTest("easting"):
                self.assertAlmostEqual(self.easting, point[0], 5)
            with self.subTest("northing"):
                self.assertAlmostEqual(self.northing, point[1], 5)
        else:
            with self.subTest("easting"):
                self.assertAlmostEqual(self.easting_36, point[0], 5)
            with self.subTest("northing"):
                self.assertAlmostEqual(self.northing_36, point[1], 5)
        with self.subTest("zone"):
            self.assertEqual("None", point[2])

    def test_project_point_ll2utm_arrays(self):
        points = gis_tools.project_point_ll2utm(
            np.repeat(self.lat_d, 5),
            np.repeat(self.lon_d, 5),
            datum="WGS84",
            epsg=self.utm_epsg,
        )

        with self.subTest("type"):
            self.assertIsInstance(points, np.recarray)

        if np.isclose(points.easting[0], self.easting):
            with self.subTest("easting"):
                self.assertTrue(
                    np.isclose(
                        np.repeat(self.easting, 5), points.easting
                    ).all()
                )

            with self.subTest("northing"):
                self.assertTrue(
                    np.isclose(
                        np.repeat(self.northing, 5), points.northing
                    ).all()
                )
        else:
            with self.subTest("easting"):
                self.assertTrue(
                    np.isclose(
                        np.repeat(self.easting_36, 5), points.easting
                    ).all()
                )

            with self.subTest("northing"):
                self.assertTrue(
                    np.isclose(
                        np.repeat(self.northing_36, 5), points.northing
                    ).all()
                )
        with self.subTest("elevation"):
            self.assertTrue(np.isclose(np.repeat(0, 5), points.elev).all())
        with self.subTest("zone"):
            self.assertTrue((np.repeat("None", 5) == points.utm_zone).all())

    def test_project_point_utm2ll_point(self):
        points = gis_tools.project_point_utm2ll(
            self.easting,
            self.northing,
            self.utm_epsg,
            datum_epsg=self.datum_epsg,
        )

        with self.subTest("type"):
            self.assertIsInstance(points, tuple)

        with self.subTest("lat"):
            self.assertAlmostEqual(self.lat_d, points[0], 3)
        with self.subTest("northing"):
            self.assertAlmostEqual(self.lon_d, points[1], 3)

    def test_project_point_utm2ll_arrays(self):
        points = gis_tools.project_point_utm2ll(
            np.repeat(self.easting, 5),
            np.repeat(self.northing, 5),
            self.utm_epsg,
            datum_epsg=self.datum_epsg,
        )

        with self.subTest("type"):
            self.assertIsInstance(points, np.recarray)

        with self.subTest("lat"):
            self.assertTrue(
                np.isclose(np.repeat(self.lat_d, 5), points.latitude).all()
            )
        with self.subTest("lon"):
            self.assertTrue(
                np.isclose(np.repeat(self.lon_d, 5), points.longitude).all()
            )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
