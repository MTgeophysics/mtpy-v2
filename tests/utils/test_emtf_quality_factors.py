# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:39:08 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

from mt_metadata import (
    TF_AVG,
    TF_AVG_NEWER,
    TF_AVG_TIPPER,
    TF_EDI_CGG,
    TF_EDI_EMPOWER,
    TF_EDI_METRONIX,
    TF_EDI_NO_ERROR,
    TF_EDI_PHOENIX,
    TF_EDI_QUANTEC,
    TF_EDI_RHO_ONLY,
    TF_EDI_SPECTRA,
    TF_JFILE,
    TF_POOR_XML,
    TF_XML,
    TF_XML_COMPLETE_REMOTE_INFO,
    TF_XML_MULTIPLE_ATTACHMENTS,
    TF_XML_NO_SITE_LAYOUT,
    TF_XML_WITH_DERIVED_QUANTITIES,
    TF_ZMM,
    TF_ZSS_TIPPER,
)

from mtpy import MT


# =============================================================================


class TestTFQualityFactory(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tf_list = [
            {"fn": TF_AVG, "rounded_qf": 3.0, "qf": 3.35},
            {"fn": TF_AVG_NEWER, "rounded_qf": 4.0, "qf": 3.525},
            {"fn": TF_AVG_TIPPER, "rounded_qf": 2.0, "qf": 1.5},
            {"fn": TF_EDI_CGG, "rounded_qf": 5.0, "qf": 4.85},
            {"fn": TF_EDI_EMPOWER, "rounded_qf": 5.0, "qf": 4.9},
            {"fn": TF_EDI_METRONIX, "rounded_qf": 4.0, "qf": 3.9},
            {"fn": TF_EDI_NO_ERROR, "rounded_qf": 2.0, "qf": 2.35},
            {"fn": TF_EDI_PHOENIX, "rounded_qf": 4.0, "qf": 4.2},
            {"fn": TF_EDI_QUANTEC, "rounded_qf": 5.0, "qf": 4.75},
            {"fn": TF_EDI_RHO_ONLY, "rounded_qf": 4.0, "qf": 3.75},
            {"fn": TF_EDI_SPECTRA, "rounded_qf": 4.0, "qf": 4.0},
            {"fn": TF_JFILE, "rounded_qf": 2.0, "qf": 2.0},
            {"fn": TF_POOR_XML, "rounded_qf": 3.0, "qf": 2.8},
            {"fn": TF_XML, "rounded_qf": 4.0, "qf": 4.4},
            {"fn": TF_XML_COMPLETE_REMOTE_INFO, "rounded_qf": 4.0, "qf": 4.0},
            {"fn": TF_XML_MULTIPLE_ATTACHMENTS, "rounded_qf": 3.0, "qf": 3.35},
            {"fn": TF_XML_NO_SITE_LAYOUT, "rounded_qf": 4.0, "qf": 3.5},
            {
                "fn": TF_XML_WITH_DERIVED_QUANTITIES,
                "rounded_qf": 4.0,
                "qf": 4.0,
            },
            {"fn": TF_ZMM, "rounded_qf": 4.0, "qf": 3.65},
            {"fn": TF_ZSS_TIPPER, "rounded_qf": 4.0, "qf": 3.75},
        ]

    def test_rounded_tf_quality_factor(self):
        for tf_dict in self.tf_list:
            m1 = MT()
            m1.read(tf_dict["fn"])
            qf = m1.estimate_tf_quality(round_qf=True)
            with self.subTest(tf_dict["fn"].name):
                self.assertEqual(tf_dict["rounded_qf"], qf)

    def test_tf_quality_factor(self):
        for tf_dict in self.tf_list:
            m1 = MT()
            m1.read(tf_dict["fn"])
            qf = m1.estimate_tf_quality()
            with self.subTest(tf_dict["fn"].name):
                self.assertAlmostEqual(tf_dict["qf"], qf)


# =============================================================================
# run
# =============================================================================
if __name__ in "__main__":
    unittest.main()
