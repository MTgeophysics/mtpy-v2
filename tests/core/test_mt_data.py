# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:40:56 2024

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

from mtpy import MT, MTData

# =============================================================================


class TestMTData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.utm_epsg = 3216
        self.datum_epsg = 4236
        self.mt_list_01 = [
            MT(
                survey="a",
                station=f"mt{ii:02}",
                latitude=40 + ii,
                longitude=-118,
            )
            for ii in range(4)
        ]
        self.mt_list_02 = [
            MT(
                survey="b",
                station=f"mt{ii:02}",
                latitude=45 + ii,
                longitude=-118,
            )
            for ii in range(4)
        ]

        self.md = MTData(
            mt_list=self.mt_list_01 + self.mt_list_02, utm_epsg=self.utm_epsg
        )

    def test_validate_item_fail(self):
        self.assertRaises(TypeError, self.md._validate_item, 10)

    def test_eq(self):
        md = MTData(
            mt_list=self.mt_list_01 + self.mt_list_02, utm_epsg=self.utm_epsg
        )

        self.assertEqual(self.md, md)

    def test_neq(self):
        md = MTData(mt_list=self.mt_list_01, utm_epsg=self.utm_epsg)

        self.assertNotEqual(self.md, md)

    def test_deep_copy(self):
        md = self.md.copy()
        self.assertEqual(self.md, md)

    def test_utm_epsg(self):
        self.assertEqual(self.md.utm_epsg, self.utm_epsg)

    def test_clone_empty(self):
        md_empty = self.md.clone_empty()

        for attr in self.md._copy_attrs:
            with self.subTest(attr):
                self.assertEqual(
                    getattr(self.md, attr), getattr(md_empty, attr)
                )

    def test_initialization_utm_epsg_no_mt_list(self):
        md = MTData(utm_epsg=self.utm_epsg)
        self.assertEqual(self.md.utm_epsg, self.utm_epsg)

    def test_initialization_datum_epsg_no_mt_list(self):
        md = MTData(datum_epsg=self.datum_epsg)
        self.assertEqual(md.datum_epsg, self.datum_epsg)

    def test_survey_ids(self):
        self.assertListEqual(["a", "b"], sorted(self.md.survey_ids))

    def test_get_survey(self):
        a = self.md.get_survey("a")

        with self.subTest("length"):
            self.assertEqual(4, len(a))

        for attr in self.md._copy_attrs:
            with self.subTest(attr):
                self.assertEqual(getattr(self.md, attr), getattr(a, attr))


# =============================================================================
#
# =============================================================================
if __name__ == "__main__":
    unittest.main()
