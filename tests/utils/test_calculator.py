# -*- coding: utf-8 -*-
"""

"""
# =============================================================================
# imports
# =============================================================================
import unittest
import numpy as np

from mtpy.utils import calculator

# =============================================================================


class TestCentrePoint(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.x = np.linspace(-100, 100, 20)
        self.y = np.linspace(-200, 200, 20)

    def test_center_point(self):
        cp = calculator.centre_point(self.x, self.y)

        self.assertTupleEqual(cp, (0, 0))


class TestRoundsf(unittest.TestCase):
    def test_round_1(self):
        self.assertEqual(100, calculator.roundsf(102, 1))

    def test_round_2(self):
        self.assertEqual(100, calculator.roundsf(102, 2))

    def test_round_3(self):
        self.assertEqual(102, calculator.roundsf(102, 3))


class TestErrorCalculation(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.z = np.array(
            [
                [
                    [-6.4 - 1.320e01j, 45.2 + 9.920e01j],
                    [-42.5 - 1.014e02j, 10.4 + 1.930e01j],
                ],
                [
                    [-1.0 - 2.100e00j, 8.7 + 1.590e01j],
                    [-7.5 - 1.520e01j, 1.7 + 3.100e00j],
                ],
                [
                    [-1.3 - 1.000e-01j, 6.3 + 2.000e00j],
                    [-6.3 - 1.600e00j, 1.6 + 4.000e-01j],
                ],
            ]
        )
        self.z_err = np.array(
            [
                [[1.5, 10.9], [11.0, 2.2]],
                [[0.2, 1.8], [1.7, 0.4]],
                [[0.1, 0.7], [0.6, 2.0]],
            ]
        )

        # relative error in resistivity is 2 * relative error in z
        self.res_rel_err_test = 2.0 * self.z_err / np.abs(self.z)

        self.phase_err_test = np.rad2deg(
            np.arctan(self.z_err / np.abs(self.z))
        )
        self.phase_err_test[self.res_rel_err_test > 1.0] = 90.0

        # test providing an array
        self.res_rel_err, self.phase_err = calculator.z_error2r_phi_error(
            self.z.real, self.z.imag, self.z_err
        )

    def test_resistivity(self):
        self.assertTrue(
            np.isclose(self.res_rel_err, self.res_rel_err_test).all()
        )

    def test_phase(self):
        self.assertTrue(np.isclose(self.phase_err, self.phase_err_test).all())


class TestGetIndex(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.freq = np.array([100.0, 10.0, 1.0])

    def test_nearest_index(self):

        for index, freq in zip([1, 2, 0], [8, 1.2, 1000]):
            self.assertEqual(calculator.nearest_index(freq, self.freq), index)


class TestMakeLogIncreasing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z1_layer = 80
        self.target_depth = 400e3
        self.n_layers = 120
        self.increment_factor = 0.999

    def test_make_log_increasing_array(self):

        array1 = calculator.make_log_increasing_array(
            self.z1_layer,
            self.target_depth,
            self.n_layers,
            increment_factor=self.increment_factor,
        )

        self.assertTrue(
            np.abs(array1.sum() / self.target_depth - 1.0)
            < 1.0 - self.increment_factor
        )


class TestGetPeriod(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.array1 = np.array(
            [
                1.77827941e-02,
                3.16227766e-02,
                5.62341325e-02,
                1.00000000e-01,
                1.77827941e-01,
                3.16227766e-01,
                5.62341325e-01,
                1.00000000e00,
                1.77827941e00,
                3.16227766e00,
                5.62341325e00,
                1.00000000e01,
                1.77827941e01,
                3.16227766e01,
                5.62341325e01,
                1.00000000e02,
                1.77827941e02,
                3.16227766e02,
                5.62341325e02,
            ]
        )
        self.array2 = np.array(
            [
                3.16227766e-02,
                5.62341325e-02,
                1.00000000e-01,
                1.77827941e-01,
                3.16227766e-01,
                5.62341325e-01,
                1.00000000e00,
                1.77827941e00,
                3.16227766e00,
                5.62341325e00,
                1.00000000e01,
                1.77827941e01,
                3.16227766e01,
                5.62341325e01,
                1.00000000e02,
                1.77827941e02,
                3.16227766e02,
            ]
        )

        self.array3 = np.array(
            [
                0.1,
                0.21544347,
                0.46415888,
                1.0,
                2.15443469,
                4.64158883,
                10.0,
                21.5443469,
                46.41588834,
                100.0,
            ]
        )

    def test_get_period_list_include(self):

        test_array = calculator.get_period_list(
            0.02, 400, 4, include_outside_range=True
        )

        self.assertTrue(np.isclose(test_array, self.array1).all())

    def test_get_period_list_not_include(self):
        test_array = calculator.get_period_list(
            0.02, 400, 4, include_outside_range=False
        )
        self.assertTrue(np.isclose(test_array, self.array2).all())

    def test_get_period_list_exact_include(self):
        # test with ends of input range on an exact decade
        test_array = calculator.get_period_list(
            0.1, 100, 3, include_outside_range=True
        )
        self.assertTrue(np.isclose(test_array, self.array3).all())

    def test_get_period_list_exact_not_include(self):
        test_array = calculator.get_period_list(
            0.1, 100, 3, include_outside_range=False
        )
        self.assertTrue(np.isclose(test_array, self.array3).all())


class TestRotation(unittest.TestCase):
    def setUp(self):
        # this is a 45 degree vector
        self.a = np.array(np.array([[np.sqrt(2), 1], [1, np.sqrt(2)]]))
        self.azimuth = self.compute_azimuth(self.a)

    def compute_azimuth(self, array):
        return np.degrees(
            0.5
            * np.arctan2(array[0, 1] + array[1, 0], array[0, 0] - array[1, 1])
        )

    def compute_strike(self, array):
        return np.degrees(
            0.25
            * np.arctan2(
                (array[0, 0] - array[1, 1])
                * np.conj(array[0, 1] + array[1, 0])
                + np.conj(array[0, 0] - array[1, 1])
                * (array[0, 1] + array[1, 0]),
                (
                    abs(array[0, 0] - array[1, 1]) ** 2
                    - abs(array[0, 1] - array[1, 0]) ** 2
                ),
            )
        )

    def test_get_rotation_matrix_0_clockwise(self):
        expected_rot = np.array([[1.0, 0.0], [-0.0, 1.0]])

        self.assertTrue(
            np.allclose(expected_rot, calculator.get_rotation_matrix(0))
        )

    def test_get_rotation_matrix_0_counterclockwise(self):
        expected_rot = np.array([[1.0, -0.0], [0.0, 1.0]])

        self.assertTrue(
            np.allclose(
                expected_rot,
                calculator.get_rotation_matrix(0, clockwise=False),
            )
        )

    def test_rotate_30_plus(self):
        angle = 30
        ar, ar_err = calculator.rotate_matrix_with_errors(self.a, angle)

        # we are rotating clockwise so in the reference frame towards zero,
        # therefore we need to subtract
        with self.subTest("azimuth"):
            self.assertAlmostEqual(
                self.azimuth - angle, self.compute_azimuth(ar)
            )

        r = calculator.get_rotation_matrix(angle, clockwise=True)
        b = r @ self.a @ r.T
        with self.subTest("matrix"):
            self.assertTrue(np.allclose(ar, b))

    def test_rotate_30_minus(self):
        angle = -30
        ar, ar_err = calculator.rotate_matrix_with_errors(self.a, angle)

        # we are rotating clockwise so in the reference frame away from zero,
        # therefore we need to add
        with self.subTest("azimuth"):
            self.assertAlmostEqual(
                self.azimuth - angle, self.compute_azimuth(ar)
            )

        r = calculator.get_rotation_matrix(angle, clockwise=True)
        b = r @ self.a @ r.T
        with self.subTest("matrix"):
            self.assertTrue(np.allclose(ar, b))


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
