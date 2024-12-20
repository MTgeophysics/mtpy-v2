# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:04:38 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
import numpy as np

from mtpy.core.transfer_function import MT_TO_OHM_FACTOR
from mtpy.core.transfer_function.z import Z


# =============================================================================


class TestZInitialize(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z = Z()

    def test_n_periods(self):
        self.assertEqual(0, self.z.n_periods)

    def test_is_empty(self):
        self.assertTrue(self.z._is_empty())

    def test_has_tf(self):
        self.assertFalse(self.z._has_tf())

    def test_has_tf_error(self):
        self.assertFalse(self.z._has_tf_error())

    def test_has_tf_model_error(self):
        self.assertFalse(self.z._has_tf_model_error())

    def test_z(self):
        self.assertEqual(None, self.z.z)

    def test_z_error(self):
        self.assertEqual(None, self.z.z_error)

    def test_z_model_error(self):
        self.assertEqual(None, self.z.z_model_error)

    def test_resistivity(self):
        self.assertEqual(None, self.z.resistivity)

    def test_phase(self):
        self.assertEqual(None, self.z.phase)

    def test_resistivity_error(self):
        self.assertEqual(None, self.z.resistivity_error)

    def test_phase_error(self):
        self.assertEqual(None, self.z.phase_error)

    def test_resistivity_model_error(self):
        self.assertEqual(None, self.z.resistivity_model_error)

    def test_phase_model_error(self):
        self.assertEqual(None, self.z.phase_model_error)

    def test_units(self):
        self.assertEqual("mt", self.z.units)


class TestZSetResPhase(unittest.TestCase):
    """
    set the error to 1%
    """

    @classmethod
    def setUpClass(self):
        self.z = Z()
        self.resistivity = np.array([[[5.0, 100.0], [100.0, 5.0]]])
        self.phase = np.array([[[90.0, 45.0], [-135.0, -90.0]]])
        self.resistivity_error = np.array([[0.1, 2], [2, 0.1]])
        self.phase_error = np.array([[0.573, 0.573], [0.573, 0.573]])
        self.resistivity_model_error = np.array([[0.1, 2], [2, 0.1]])
        self.phase_model_error = np.array([[0.573, 0.573], [0.573, 0.573]])
        self.z.set_resistivity_phase(
            self.resistivity,
            self.phase,
            np.array([1]),
            res_error=self.resistivity_error,
            phase_error=self.phase_error,
            res_model_error=self.resistivity_model_error,
            phase_model_error=self.phase_model_error,
        )

    def test_is_empty(self):
        self.assertFalse(self.z._is_empty())

    def test_has_tf(self):
        self.assertTrue(self.z._has_tf())

    def test_has_tf_error(self):
        self.assertTrue(self.z._has_tf_error())

    def test_has_tf_model_error(self):
        self.assertTrue(self.z._has_tf_model_error())

    def test_resistivity(self):
        self.assertTrue(np.isclose(self.z.resistivity, self.resistivity).all())

    def test_phase(self):
        self.assertTrue(np.isclose(self.z.phase, self.phase).all())

    def test_resistivity_error(self):
        self.assertTrue(
            np.isclose(
                self.resistivity_error, self.z.resistivity_error, atol=0.001
            ).all()
        )

    def test_phase_error(self):
        self.assertTrue(
            np.isclose(
                self.z.phase_error,
                self.phase_error,
            ).all()
        )

    def test_resistivity_model_error(self):
        self.assertTrue(
            np.isclose(
                self.resistivity_model_error,
                self.z.resistivity_model_error,
                atol=0.001,
            ).all()
        )

    def test_phase_model_error(self):
        self.assertTrue(
            np.isclose(
                self.z.phase_model_error,
                self.phase_model_error,
            ).all()
        )

    def test_rotation_plus_5_ned(self):
        zr = self.z.rotate(5, coordinate_reference_frame="ned")
        with self.subTest("PT Strike"):
            self.assertAlmostEqual(zr.phase_tensor.azimuth[0], 40)
        with self.subTest("Invariant Strike"):
            self.assertAlmostEqual(zr.invariants.strike[0], 40)

    def test_rotation_minus_5_ned(self):
        zr = self.z.rotate(-5, coordinate_reference_frame="ned")
        with self.subTest("PT Strike"):
            self.assertAlmostEqual(zr.phase_tensor.azimuth[0], 50)
        with self.subTest("Invariant Strike"):
            self.assertAlmostEqual(zr.invariants.strike[0], 50)

    def test_rotation_plus_5_enu(self):
        zr = self.z.rotate(5, coordinate_reference_frame="enu")
        with self.subTest("PT Strike"):
            self.assertAlmostEqual(zr.phase_tensor.azimuth[0], 50)
        with self.subTest("Invariant Strike"):
            self.assertAlmostEqual(zr.invariants.strike[0], 50)

    def test_rotation_minus_5_enu(self):
        zr = self.z.rotate(-5, coordinate_reference_frame="enu")
        with self.subTest("PT Strike"):
            self.assertAlmostEqual(zr.phase_tensor.azimuth[0], 40)
        with self.subTest("Invariant Strike"):
            self.assertAlmostEqual(zr.invariants.strike[0], 40)

    def test_rotation_angle(self):
        zr = self.z.rotate(40)
        self.assertTrue(np.all(zr.rotation_angle == np.array([40])))


class TestRemoveStaticShift(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z = Z()
        self.z.z = np.array(
            [[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]]
        )

        self.ss_x = 0.5
        self.ss_y = 1.5

        self.ss_z = np.array(
            [
                [
                    [0.14142136 - 0.14142136j, 14.14213562 + 14.14213562j],
                    [-8.16496581 - 8.16496581j, -0.08164966 + 0.08164966j],
                ]
            ]
        )

        self.new_z = self.z.remove_ss(self.ss_x, self.ss_y, inplace=False)

    def test_remove_ss(self):
        self.assertTrue(np.allclose(self.new_z.z, self.ss_z))

    def test_ss_factors(self):
        self.assertTrue(
            np.allclose(
                (self.z.z / self.new_z.z) ** 2,
                np.array(
                    [[[0.5 + 0.0j, 0.5 + 0.0j], [1.5 - 0.0j, 1.5 - 0.0j]]]
                ),
            )
        )

    def test_set_factor_fail_single(self):
        self.assertRaises(ValueError, self.z.remove_ss, "k")

    def test_set_factor_fail_too_many(self):
        self.assertRaises(ValueError, self.z.remove_ss, [1, 2, 3])


class TestRemoveDistortion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        z = np.array(
            [[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]]
        )
        self.z = Z()
        self.z.z = z

        self.distortion = np.array(np.real([[0.5, 0.1], [0.2, 0.6]]))

        self.dz = Z()
        self.dz.z = np.array(np.dot(self.distortion, z)).reshape((1, 2, 2))

    def test_remove_distortion(self):
        new_z = self.dz.remove_distortion(self.distortion)

        self.assertTrue(np.allclose(new_z.z, self.z.z))

    def test_fail_bad_input_shape_too_many(self):
        self.assertRaises(
            ValueError, self.z.remove_distortion, np.random.rand(4, 3, 3, 3)
        )

    def test_fail_bad_input_shape_not_z_shape(self):
        self.assertRaises(
            ValueError, self.z.remove_distortion, np.random.rand(4, 3, 3)
        )

    def test_fail_bad_input_singular_matrix(self):
        self.assertRaises(
            ValueError, self.z.remove_distortion, np.array([[0, 1], [0, 0]])
        )


class TestInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z = Z(
            z=np.array(
                [
                    [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                    [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
                ]
            )
        )

    def test_normalizing_real(self):
        self.assertAlmostEqual(
            51.753349, self.z.invariants.normalizing_real[0], 5
        )

    def test_normalizing_imag(self):
        self.assertAlmostEqual(
            115.504607, self.z.invariants.normalizing_imag[0], 5
        )

    def test_anisotropy_real(self):
        self.assertAlmostEqual(
            0.190142, self.z.invariants.anisotropic_real[0], 5
        )

    def test_anisotropy_imag(self):
        self.assertAlmostEqual(
            0.158448, self.z.invariants.anisotropic_imag[0], 5
        )

    def test_electric_twist(self):
        self.assertAlmostEqual(0.071840, self.z.invariants.electric_twist[0], 5)

    def test_phase_distortion(self):
        self.assertAlmostEqual(
            -0.015665, self.z.invariants.phase_distortion[0], 5
        )

    def test_dimensionality(self):
        self.assertAlmostEqual(
            0.0503231, self.z.invariants.dimensionality[0], 5
        )

    def test_structure_3d(self):
        self.assertAlmostEqual(-0.180011, self.z.invariants.structure_3d[0], 5)

    def test_strike(self):
        self.assertAlmostEqual(17.266664, self.z.invariants.strike[0], 5)

    def test_strike_error(self):
        self.assertAlmostEqual(5.185204, self.z.invariants.strike_error[0], 5)

    def test_estimate_dimensionality(self):
        self.assertTrue(
            np.all(self.z.estimate_dimensionality() == np.array([1]))
        )

    def test_rotate_plus_5(self):
        new_z = self.z.rotate(5)
        self.assertAlmostEqual(new_z.invariants.strike[0], 12.266664, 5)

    def test_rotate_minus_5(self):
        new_z = self.z.rotate(-5)
        self.assertAlmostEqual(new_z.invariants.strike[0], 22.266664, 5)


class TestZAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z = Z(
            z=np.array(
                [
                    [
                        [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                        [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
                    ],
                    [
                        [-1.420305 - 1.02897j, 603.44306 + 814.4988j],
                        [-10.96444 - 21.4191j, 1.95081 + 1.52367j],
                    ],
                    [
                        [-70.420305 - 111.02897j, 3.44306 + 214.4988j],
                        [-19.96444 - 56.4191j, 81.95081 + 314.52367j],
                    ],
                ]
            ),
            frequency=np.array([10, 1, 0.1]),
        )

    def test_estimate_dimensionality(self):
        self.assertTrue(
            np.all(self.z.estimate_dimensionality() == np.array([1, 2, 3]))
        )

    def test_estimate_distortion(self):
        d, d_err = self.z.estimate_distortion()
        with self.subTest("distortion"):
            self.assertTrue(
                np.isclose(
                    d,
                    # np.array(
                    #     [
                    #         [0.99707684, -1.01783035],
                    #         [0.04933412, 1.08934035],
                    #     ]
                    # ),
                    # rotation with R A R^-1
                    np.array(
                        [[0.99586079, 1.17387806], [0.09359908, 1.1030191]]
                    ),
                ).all()
            )
        with self.subTest("distortion error"):
            self.assertTrue(
                np.isclose(
                    d_err,
                    np.array(
                        [[0.57735027, 0.57735027], [0.57735027, 0.57735027]]
                    ),
                ).all()
            )

    def test_remove_distortion(self):
        new_z = self.z.remove_distortion()

        self.assertTrue(
            np.isclose(
                new_z.z,
                # np.array(
                #     [
                #         [
                #             [
                #                 -51.86565263 - 118.68192661j,
                #                 61.93545384 + 129.03863553j,
                #             ],
                #             [
                #                 -43.51779826 - 101.49631509j,
                #                 8.1657482 + 13.91453345j,
                #             ],
                #         ],
                #         [
                #             [
                #                 -11.18221849 - 20.17117021j,
                #                 580.21646901 + 782.15493955j,
                #             ],
                #             [
                #                 -9.55878948 - 18.74893648j,
                #                 -24.48606379 - 34.02357534j,
                #             ],
                #         ],
                #         [
                #             [
                #                 -85.38777896 - 156.96774979j,
                #                 76.70274771 + 487.33602186j,
                #             ],
                #             [
                #                 -14.46004384 - 44.68321988j,
                #                 71.75603776 + 266.65805243j,
                #             ],
                #         ],
                #     ]
                # ),
                # for R A R^-1
                np.array(
                    [
                        [
                            [
                                51.05044024 + 121.47184822j,
                                45.43885369 + 102.19531416j,
                            ],
                            [
                                -49.62988816 - 115.85361851j,
                                6.9788229 + 10.84140998j,
                            ],
                        ],
                        [
                            [
                                11.43488341 + 24.28579241j,
                                670.98211541 + 906.97720461j,
                            ],
                            [
                                -10.91072189 - 21.47943583j,
                                -55.16903678 - 75.5821594j,
                            ],
                        ],
                        [
                            [
                                -54.86571355 - 56.88761188j,
                                -93.46996499 - 134.14819458j,
                            ],
                            [
                                -13.44406414 - 46.32238172j,
                                82.22841591 + 296.53141905j,
                            ],
                        ],
                    ]
                ),
            ).all()
        )

    def test_depth_of_investigation(self):
        doi = self.z.estimate_depth_of_investigation()

        with self.subTest("depth determinant"):
            self.assertTrue(
                np.all(
                    np.isclose(
                        np.array(
                            [1987.75038069, 24854.87498141, 283705.23967805]
                        ),
                        doi["depth_det"],
                    )
                )
            )

        with self.subTest("depth xy"):
            self.assertTrue(
                np.all(
                    np.isclose(
                        np.array(
                            [2011.03691158, 161332.55006745, 341429.42016186]
                        ),
                        doi["depth_xy"],
                    )
                )
            )

        with self.subTest("depth yx"):
            self.assertTrue(
                np.all(
                    np.isclose(
                        np.array(
                            [2016.30231674, 3829.64228158, 95249.86168927]
                        ),
                        doi["depth_yx"],
                    )
                )
            )
        with self.subTest("depth min"):
            self.assertTrue(
                np.all(
                    np.isclose(
                        np.array(
                            [1987.75038069, 3829.64228158, 95249.86168927]
                        ),
                        doi["depth_min"],
                    )
                )
            )
        with self.subTest("depth max"):
            self.assertTrue(
                np.all(
                    np.isclose(
                        np.array(
                            [2016.30231674, 161332.55006745, 341429.42016186]
                        ),
                        doi["depth_max"],
                    )
                )
            )

        with self.subTest("period"):
            self.assertTrue(np.all(np.isclose(doi["period"], self.z.period)))


class TestUnits(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z = np.array(
            [
                [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
            ]
        )
        self.z_in_ohms = self.z / MT_TO_OHM_FACTOR

    def test_initialize_with_units_ohm(self):
        z_obj = Z(z=self.z_in_ohms, units="ohm")
        with self.subTest("data in mt units"):
            self.assertTrue(
                np.allclose(self.z, z_obj._dataset.transfer_function.values)
            )
        with self.subTest("data in mt units"):
            self.assertTrue(np.allclose(self.z_in_ohms, z_obj.z))
        with self.subTest("units"):
            self.assertEqual("ohm", z_obj.units)

    def test_initialize_with_units_mt(self):
        z_obj = Z(z=self.z, units="mt")
        with self.subTest("data in mt units"):
            self.assertTrue(
                np.allclose(self.z, z_obj._dataset.transfer_function.values)
            )
        with self.subTest("data in mt units"):
            self.assertTrue(np.allclose(self.z, z_obj.z))
        with self.subTest("units"):
            self.assertEqual("mt", z_obj.units)

    def test_units_change_ohm_to_mt(self):
        z_obj = Z(z=self.z_in_ohms, units="ohm")
        z_obj.units = "mt"
        with self.subTest("data in mt units"):
            self.assertTrue(
                np.allclose(self.z, z_obj._dataset.transfer_function.values)
            )
        with self.subTest("data in mt units"):
            self.assertTrue(np.allclose(self.z, z_obj.z))
        with self.subTest("units"):
            self.assertEqual("mt", z_obj.units)

    def test_units_change_mt_to_ohm(self):
        z_obj = Z(z=self.z, units="mt")
        z_obj.units = "ohm"
        with self.subTest("data in mt units"):
            self.assertTrue(
                np.allclose(self.z, z_obj._dataset.transfer_function.values)
            )
        with self.subTest("data in ohm units"):
            self.assertTrue(np.allclose(self.z_in_ohms, z_obj.z))
        with self.subTest("units"):
            self.assertEqual("ohm", z_obj.units)

    def test_set_unit_fail(self):
        z_obj = Z(z=self.z)

        def set_units(unit):
            z_obj.units = unit

        with self.subTest("bad type"):
            self.assertRaises(TypeError, set_units, 4)
        with self.subTest("bad choice"):
            self.assertRaises(ValueError, set_units, "ants")

    def test_phase_tensor_equal(self):
        z_ohm = Z(z=self.z_in_ohms, units="ohm")
        z_mt = Z(z=self.z, units="mt")

        self.assertTrue(
            np.allclose(z_ohm.phase_tensor.pt, z_mt.phase_tensor.pt)
        )

    def test_resistivity_phase_equal(self):
        z_ohm = Z(z=self.z_in_ohms, units="ohm")
        z_mt = Z(z=self.z, units="mt")

        with self.subTest("resistivity"):
            self.assertTrue(np.allclose(z_ohm.resistivity, z_mt.resistivity))
        with self.subTest("phase"):
            self.assertTrue(np.allclose(z_ohm.phase, z_mt.phase))


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
