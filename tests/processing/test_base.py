# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:20:41 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from pathlib import Path

from mtpy.processing.base import BaseProcessing

# =============================================================================


class TestBaseProcessing(unittest.TestCase):
    def setUp(self):
        self.fn = Path(__file__)
        self.base = BaseProcessing()

    def test_set_local_station_id(self):
        local = "mt01"
        self.base.local_station_id = local
        self.assertEqual(local, self.base.local_station_id)

    def test_set_local_station_id_none(self):
        local = None
        self.base.local_station_id = local
        self.assertEqual(local, self.base.local_station_id)

    def test_set_remote_station_id(self):
        remote = "mt01"
        self.base.remote_station_id = remote
        self.assertEqual(remote, self.base.remote_station_id)

    def test_set_remote_station_id_none(self):
        remote = None
        self.base.remote_station_id = remote
        self.assertEqual(remote, self.base.remote_station_id)

    # def test_set_local_station_id_fail(self):
    #     self.assertRaise(self.base.local_station_id.setter, )

    def test_set_local_mth5_path(self):
        self.base.local_mth5_path = self.fn
        self.assertEqual(self.fn, self.base.local_mth5_path)

    def test_set_remote_mth5_path(self):
        self.base.remote_mth5_path = self.fn
        self.assertEqual(self.fn, self.base.remote_mth5_path)

    def test_set_path_bad_type(self):
        self.assertRaises(ValueError, self.base.set_path, 10)

    def test_set_path_bad_path(self):
        self.assertRaises(IOError, self.base.set_path, "/home/bad/file")

    def test_set_path_none(self):
        self.assertEqual(None, self.base.set_path(None))


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
