# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:39:47 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import unittest

from mtpy.modeling.modem import Data

# =============================================================================


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.d = Data()

    def test_change_comp(self):
        comp_dict = {
            "zxx": "z_xx",
            "zxy": "z_xy",
            "zyx": "z_yx",
            "zyy": "z_yy",
            "tzx": "t_zx",
            "tzy": "t_zy",
            "ptxx": "pt_xx",
            "ptxy": "pt_xy",
            "ptyx": "pt_yx",
            "ptyy": "pt_yy",
        }

        for og, new in comp_dict.items():
            with self.subTest(og):
                self.assertEqual(new, self.d._change_comp(og))


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
