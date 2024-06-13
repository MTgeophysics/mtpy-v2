# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:39:08 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

from mtpy import MT
from mtpy.utils.estimate_tf_quality_factor import EMTFStats

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

# =============================================================================


# class TestTFQualityFactory(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.tf_list = [
#             {"fn": TF_AVG, "qf": 4.0},
#             {"fn": TF_AVG_NEWER,  "qf": 3.0},
#             {"fn": TF_AVG_TIPPER, "qf": 2.0},
#             {"fn": TF_EDI_CGG, "qf": 5.0},
#             {"fn": TF_EDI_EMPOWER, "qf": 5.0},
#             {"fn": TF_EDI_METRONIX, "qf": 4.0},
#             {"fn": TF_EDI_NO_ERROR, "qf": 2.0},
#             {"fn": TF_EDI_PHOENIX, "qf": 4.0},
#             {"fn": TF_EDI_QUANTEC, "qf": 5.0},
#             {"fn": TF_EDI_RHO_ONLY, "qf": 4.0},
#             {"fn": TF_EDI_SPECTRA, "qf": 4.0},
#             {"fn": TF_JFILE, "qf": 2.0},
#             {"fn": TF_POOR_XML, "qf": 3.0},
#             {"fn": TF_XML, "qf": 4.0},
#             {"fn": TF_XML_COMPLETE_REMOTE_INFO, "qf": 4.0},
#             {"fn": TF_XML_MULTIPLE_ATTACHMENTS, "qf": },
#             {"fn": TF_XML_NO_SITE_LAYOUT, "qf": },
#             {"fn": TF_XML_WITH_DERIVED_QUANTITIES, "qf": },
#             {"fn": TF_ZMM, "qf": },
#             {"fn": TF_ZSS_TIPPER, "qf": },
#         ]


m1 = MT()
m1.read(TF_XML_MULTIPLE_ATTACHMENTS)
m1.estimate_tf_quality()
