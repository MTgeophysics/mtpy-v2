# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:39:46 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
import unittest.mock
from mtpy import MT
from mtpy.imaging import plot_mt_response

from mt_metadata import TF_EDI_CGG


# =============================================================================


class TestPlotMTResponse(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        m1 = MT(TF_EDI_CGG)
        m1.read()

        self.z_object = m1.Z.copy()
        self.t_object = m1.Tipper.copy()
        self.pt_object = m1.pt.copy()
        self.station = m1.station

        self.plot_object = plot_mt_response.PlotMTResponse(
            z_object=self.z_object,
            t_object=self.t_object,
            pt_obj=self.pt_object,
            station=self.station,
            show_plot=False,
        )

    def test_z_object(self):
        self.assertEqual(self.z_object, self.plot_object.Z)

    def test_t_object(self):
        self.assertEqual(self.t_object, self.plot_object.Tipper)

    def test_pt_object(self):
        self.assertEqual(self.pt_object, self.plot_object.pt)

    def test_period(self):
        self.assertEqual(
            True, (self.z_object.period == self.plot_object.period).all()
        )

    def test_set_model_error_to_true(self):
        self.plot_object.plot_model_error = True
        with self.subTest("set to true"):
            self.assertEqual(self.plot_object.plot_model_error, True)
        with self.subTest("error string"):
            self.assertEqual(self.plot_object._error_str, "model_error")

    def test_set_model_error_to_false(self):
        self.plot_object.plot_model_error = False
        with self.subTest("set to true"):
            self.assertEqual(self.plot_object.plot_model_error, False)
        with self.subTest("error string"):
            self.assertEqual(self.plot_object._error_str, "error")

    def test_set_rotation_angle(self):
        self.plot_object.rotation_angle = 45
        with self.subTest("rotation_angle"):
            self.assertEqual(self.plot_object.rotation_angle, 45)

        with self.subTest("Z rotation_angle"):
            self.assertEqual(self.plot_object.Z.rotation_angle.mean(), 45)
        with self.subTest("T rotation_angle"):
            self.assertEqual(self.plot_object.Tipper.rotation_angle.mean(), 45)
        with self.subTest("PT rotation_angle"):
            self.assertEqual(self.plot_object.pt.rotation_angle.mean(), 45)

    def test_has_tipper(self):
        self.plot_object.plot_tipper = "yri"
        self.plot_object._has_tipper()
        self.assertEqual("yri", self.plot_object.plot_tipper)

    def test_has_pt(self):
        self.plot_pt = True
        self.plot_object._has_pt()
        self.assertEqual(True, self.plot_object.plot_pt)

    def test_setup_subplot_plot_num_1(self):
        self.plot_object.plot_num = 1
        self.assertTupleEqual(
            (-0.095, 0.5), self.plot_object._setup_subplots()
        )

    def test_setup_subplot_plot_num_2(self):
        self.plot_object.plot_num = 2
        self.assertTupleEqual((-0.14, 0.5), self.plot_object._setup_subplots())

    def test_setup_subplot_plot_num_3(self):
        self.plot_object.plot_num = 3
        self.assertTupleEqual(
            (-0.095, 0.5), self.plot_object._setup_subplots()
        )

    def test_plot_resistivity_od(self):
        self.plot_object._setup_subplots()
        eb_list, label_list = self.plot_object._plot_resistivity(
            self.plot_object.axr, self.plot_object.period, self.plot_object.Z
        )
        self.assertEqual(label_list, ["$Z_{xy}$", "$Z_{yx}$"])

    def test_plot_resistivity_od(self):
        self.plot_object._initiate_figure()
        self.plot_object._setup_subplots()
        eb_list, label_list = self.plot_object._plot_resistivity(
            self.plot_object.axr,
            self.plot_object.period,
            self.plot_object.Z,
            mode="d",
        )
        self.assertEqual(label_list, ["$Z_{xx}$", "$Z_{yy}$"])


class TestPlotMTResponsePlot(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        m1 = MT(TF_EDI_CGG)
        m1.read()

        self.z_object = m1.Z.copy()
        self.t_object = m1.Tipper.copy()
        self.pt_object = m1.pt.copy()
        self.station = m1.station

        self.plot_object = plot_mt_response.PlotMTResponse(
            z_object=self.z_object,
            t_object=self.t_object,
            pt_obj=self.pt_object,
            station=self.station,
            show_plot=False,
        )

    @unittest.mock.patch(f"{__name__}.plot_mt_response.plt")
    def test_plot(self, mock_plt):
        self.plot_object.plot()

        assert mock_plt.figure.called


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
