# -*- coding: utf-8 -*-
"""Pytest suite for MTDataTree scaffold behavior."""

from pathlib import Path

import pytest
import xarray as xr

from mtpy.core import MTDataTree
from mtpy.core.mt import MT


class TestMTDataTreeInitialization:
    def test_init_creates_root_and_surveys_node(self):
        tree = MTDataTree()

        assert tree.tree.name == MTDataTree.ROOT_NAME
        assert MTDataTree.SURVEYS_NODE in tree.tree.children
        assert tree.tree.attrs["schema_name"] == "mtpy.mt_data_tree"
        assert tree.tree.attrs["schema_version"] == "0.1.0"

    def test_init_applies_custom_attrs(self):
        tree = MTDataTree(coordinate_reference_frame="ned", impedance_units="mt")

        assert tree.tree.attrs["coordinate_reference_frame"] == "ned"
        assert tree.tree.attrs["impedance_units"] == "mt"
        assert tree.attrs is tree.tree.attrs


class TestMTDataTreeAddStation:
    @pytest.mark.parametrize(
        "survey, station, expected_path",
        [
            ("survey_a", "station_01", "surveys/survey_a/stations/station_01"),
            ("survey_b", "station_02", "surveys/survey_b/stations/station_02"),
        ],
    )
    def test_add_station_path_and_attrs(self, survey, station, expected_path):
        mt = MT()
        mt.survey = survey
        mt.station = station

        tree = MTDataTree()
        station_path = tree.add_station(mt)

        assert station_path == expected_path

        ds = tree.get_station(station_path)
        assert isinstance(ds, xr.Dataset)
        assert ds.attrs["survey"] == survey.replace("/", "_")
        assert ds.attrs["station"] == station.replace("/", "_")
        assert "survey_metadata" in ds.attrs
        assert "station_metadata" in ds.attrs

    def test_clean_name_replaces_path_separators(self):
        assert MTDataTree._clean_name("survey/a", "default") == "survey_a"
        assert MTDataTree._clean_name("station/01", "unknown_station") == "station_01"

    def test_add_station_uses_metadata_id_fallback(self):
        mt = MT()
        mt.survey = None
        mt.station = None
        mt.survey_metadata.id = "survey_from_metadata"
        mt.station_metadata.id = "station_from_metadata"

        tree = MTDataTree()
        station_path = tree.add_station(mt)

        assert (
            station_path
            == "surveys/survey_from_metadata/stations/station_from_metadata"
        )

    def test_add_station_default_name_fallbacks(self):
        mt = MT()
        mt.survey = ""
        mt.station = ""
        mt.survey_metadata.id = ""
        mt.station_metadata.id = ""

        tree = MTDataTree()
        station_path = tree.add_station(mt)

        assert station_path == "surveys/default/stations/unknown_station"

    def test_add_station_none_raises(self):
        tree = MTDataTree()
        with pytest.raises(TypeError, match="mt_obj cannot be None"):
            tree.add_station(None)

    def test_add_station_from_filename(self, tmp_path, monkeypatch):
        test_fn = tmp_path / "file_station_01.edi"
        test_fn.write_text("dummy")

        def _fake_read(self):
            stem = Path(self.fn).stem
            self.survey = "survey_from_file"
            self.station = stem

        monkeypatch.setattr(MT, "read", _fake_read)

        tree = MTDataTree()
        station_path = tree.add_station(str(test_fn))

        assert station_path == "surveys/survey_from_file/stations/file_station_01"

    def test_add_station_from_path(self, tmp_path, monkeypatch):
        test_fn = tmp_path / "file_station_02.edi"
        test_fn.write_text("dummy")

        def _fake_read(self):
            stem = Path(self.fn).stem
            self.survey = "survey_from_file"
            self.station = stem

        monkeypatch.setattr(MT, "read", _fake_read)

        tree = MTDataTree()
        station_path = tree.add_station(test_fn)

        assert station_path == "surveys/survey_from_file/stations/file_station_02"

    def test_add_station_list_mixed_inputs(self, basic_mt, tmp_path, monkeypatch):
        test_fn = tmp_path / "file_station_03.edi"
        test_fn.write_text("dummy")

        def _fake_read(self):
            stem = Path(self.fn).stem
            self.survey = "survey_from_file"
            self.station = stem

        monkeypatch.setattr(MT, "read", _fake_read)

        tree = MTDataTree()
        out_paths = tree.add_station([basic_mt, test_fn])

        assert isinstance(out_paths, list)
        assert len(out_paths) == 2
        assert out_paths[0] == "surveys/big/stations/test_01"
        assert out_paths[1] == "surveys/survey_from_file/stations/file_station_03"

    def test_add_station_invalid_type_raises(self):
        tree = MTDataTree()
        with pytest.raises(TypeError, match="mt_obj must be an MT instance"):
            tree.add_station(42)

    def test_add_station_overwrite_false_raises(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        with pytest.raises(KeyError, match="Station path already exists"):
            tree.add_station(basic_mt, overwrite=False)

        # Sanity check existing node remains accessible
        ds = tree.get_station(station_path)
        assert isinstance(ds, xr.Dataset)


class TestMTDataTreeNodeOperations:
    def test_remove_station(self, basic_mt):
        tree = MTDataTree()
        station_path = tree.add_station(basic_mt)

        assert tree._path_exists(station_path)
        tree.remove_station(station_path)
        assert not tree._path_exists(station_path)

    def test_keys_returns_top_level_children(self, basic_mt):
        tree = MTDataTree()
        tree.add_station(basic_mt)

        keys = tree.keys()
        assert MTDataTree.SURVEYS_NODE in keys
