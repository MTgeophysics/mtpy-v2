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
