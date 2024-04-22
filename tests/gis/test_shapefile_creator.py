# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:06:57 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mtpy import MTData
from mtpy.core import MTDataFrame
from mtpy.gis.shapefile_creator import ShapefileCreator

from mtpy_data.mtpy_data import FWD_CONDUCTIVE_CUBE_GRID_LIST

# =============================================================================


class TestShapefileCreator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.md = MTData()
        self.md.add_station(FWD_CONDUCTIVE_CUBE_GRID_LIST, survey="test")
        self.md.utm_epsg = 32754

        self.mt_df = self.md.to_dataframe()

        self.sc = ShapefileCreator(self.mt_df, self.md.utm_epsg)

    def test_mt_dataframe(self):
        self.assertEqual(self.mt_df, self.sc.mt_dataframe)


# =============================================================================
#
# =============================================================================
if __name__ in "__main__":
    unittest.main()
