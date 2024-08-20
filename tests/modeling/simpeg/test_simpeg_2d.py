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
        self.md.add_station(
            [fn for fn in PROFILE_LIST if fn.name.startswith("16")]
        )
        self.md.utm_epsg = 4462

        self.profile = self.md.get_profile(
            149.15, -22.3257, 149.20, -22.3257, 1000
        )

        self.mt_df = self.profile.to_dataframe()

    def setUp(self):
        self.simpeg_data = Simpeg2DData(self.mt_df)

    def test_get_locations_fail(self):
        df = self.md.to_dataframe()
        df.profile_offset = 0
        s = Simpeg2DData(df)
        self.assertRaises(ValueError, getattr, s, "station_locations")

    def test_station_locations(self):
        with self.subTest("shape"):
            self.assertEqual((6, 2), self.simpeg_data.station_locations.shape)

        with self.subTest("offset"):
            self.assertTrue(
                np.allclose(
                    self.simpeg_data.station_locations[:, 0],
                    np.array(
                        [
                            0.0,
                            479.36423899,
                            1032.47570849,
                            1526.02107079,
                            2005.38361755,
                            2501.76393224,
                        ]
                    ),
                )
            )
        with self.subTest("elevation"):
            self.assertTrue(
                np.allclose(
                    self.simpeg_data.station_locations[:, 1],
                    np.array([210.0, 213.0, 212.0, 219.0, 214.0, 220.0]),
                )
            )

    def test_station_locations_no_elevation(self):
        self.simpeg_data.include_elevation = False
        with self.subTest("shape"):
            self.assertEqual((6, 2), self.simpeg_data.station_locations.shape)

        with self.subTest("offset"):
            self.assertTrue(
                np.allclose(
                    self.simpeg_data.station_locations[:, 0],
                    np.array(
                        [
                            0.0,
                            479.36423899,
                            1032.47570849,
                            1526.02107079,
                            2005.38361755,
                            2501.76393224,
                        ]
                    ),
                )
            )
        with self.subTest("elevation"):
            self.assertTrue(
                np.allclose(
                    self.simpeg_data.station_locations[:, 1],
                    np.zeros((6)),
                )
            )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
