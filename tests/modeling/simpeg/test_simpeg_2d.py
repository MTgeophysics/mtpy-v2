# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:09:02 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
import pandas as pd
import numpy as np
from mtpy.core import MTDataFrame
from mtpy import MTData

from mtpy_data import PROFILE_LIST
from mtpy.modeling.simpeg.data import Simpeg2DData

# =============================================================================


class TestSimpeg2DData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.md = MTData()
        self.md.add_station(PROFILE_LIST)

        self.mt_df = self.md.to_dataframe()

    def setUp(self):
        self.simpeg_data = Simpeg2DData(self.mt_df)

    def test_get_locations(self):
        self.mt_df


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
