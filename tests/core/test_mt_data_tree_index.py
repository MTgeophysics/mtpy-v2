# -*- coding: utf-8 -*-
"""Tests for MTDataTreeIndexStore and MTData integration."""

import mtpy_data
import pytest

from mtpy.core import MTData
from mtpy.core.mt import MT
from mtpy.core.mt_data_tree_index import (
    MTDataTreeIndexStore,
    StationPeriodRow,
    StationRow,
    SurveyRow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def edi_files():
    return sorted(mtpy_data.PROFILE_LIST)[:4]


@pytest.fixture(scope="session")
def mt_objects(edi_files):
    objs = []
    for fn in edi_files:
        mt_obj = MT(fn)
        mt_obj.read()
        objs.append(mt_obj)
    return objs


@pytest.fixture(scope="session")
def indexed_tree(mt_objects):
    tree = MTData(use_index=True)
    tree.add_stations(mt_objects)
    return tree


@pytest.fixture()
def empty_store():
    return MTDataTreeIndexStore()  # in-memory


# ---------------------------------------------------------------------------
# Unit tests for MTDataTreeIndexStore
# ---------------------------------------------------------------------------


class TestMTDataTreeIndexStoreDDL:
    def test_tables_created(self, empty_store):
        tables = {
            row[0]
            for row in empty_store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"surveys", "stations", "station_period_index"} <= tables

    def test_initial_counts_zero(self, empty_store):
        assert empty_store.n_surveys() == 0
        assert empty_store.n_stations() == 0


class TestUpsertSurvey:
    @pytest.fixture(autouse=True)
    def store(self):
        self._store = MTDataTreeIndexStore()

    def test_insert_new_survey(self):
        row = SurveyRow(name="survey_a", lat_min=30.0, lat_max=40.0)
        sid = self._store.upsert_survey(row)
        assert isinstance(sid, int)
        assert self._store.n_surveys() == 1

    def test_upsert_is_idempotent(self):
        row = SurveyRow(name="survey_a", lat_min=30.0, lat_max=40.0)
        self._store.upsert_survey(row)
        row2 = SurveyRow(name="survey_a", lat_min=35.0, lat_max=45.0)
        self._store.upsert_survey(row2)
        assert self._store.n_surveys() == 1
        sv = self._store.all_surveys()[0]
        assert sv.lat_min == pytest.approx(35.0)


class TestUpsertStation:
    @pytest.fixture(autouse=True)
    def store(self):
        self._store = MTDataTreeIndexStore()

    def test_insert_station_creates_survey_if_missing(self):
        row = StationRow(
            tree_path="surveys/s1/stations/stn1",
            survey_name="s1",
            name="stn1",
            latitude=35.0,
            longitude=-110.0,
        )
        sid = self._store.upsert_station(row)
        assert isinstance(sid, int)
        assert self._store.n_surveys() == 1
        assert self._store.n_stations() == 1

    def test_upsert_station_updates_coords(self):
        row = StationRow(
            tree_path="surveys/s1/stations/stn1",
            survey_name="s1",
            name="stn1",
            latitude=35.0,
            longitude=-110.0,
        )
        self._store.upsert_station(row)
        row2 = StationRow(
            tree_path="surveys/s1/stations/stn1",
            survey_name="s1",
            name="stn1",
            latitude=36.0,
            longitude=-111.0,
        )
        self._store.upsert_station(row2)
        rec = self._store.station_record("surveys/s1/stations/stn1")
        assert rec is not None
        assert rec.latitude == pytest.approx(36.0)

    def test_station_record_returns_none_for_missing(self):
        assert self._store.station_record("nonexistent/path") is None


class TestPeriodIndex:
    @pytest.fixture(autouse=True)
    def store_with_station(self):
        self._store = MTDataTreeIndexStore()
        self._store.upsert_station(
            StationRow(
                tree_path="surveys/s1/stations/stn1",
                survey_name="s1",
                name="stn1",
                latitude=35.0,
                longitude=-110.0,
            )
        )

    def test_replace_period_rows(self):
        period_row = StationPeriodRow(
            station_path="surveys/s1/stations/stn1",
            period_min=0.01,
            period_max=1000.0,
            n_periods=60,
        )
        self._store.replace_station_period_rows(period_row)
        paths = self._store.query_station_paths(period_min=0.5, period_max=10.0)
        assert "surveys/s1/stations/stn1" in paths

    def test_period_filter_outside_range_excluded(self):
        period_row = StationPeriodRow(
            station_path="surveys/s1/stations/stn1",
            period_min=0.01,
            period_max=1000.0,
            n_periods=60,
        )
        self._store.replace_station_period_rows(period_row)
        paths = self._store.query_station_paths(period_min=2000.0, period_max=3000.0)
        assert "surveys/s1/stations/stn1" not in paths


class TestDeleteStation:
    @pytest.fixture(autouse=True)
    def store_with_station(self):
        self._store = MTDataTreeIndexStore()
        self._store.upsert_station(
            StationRow(
                tree_path="surveys/s1/stations/stn1",
                survey_name="s1",
                name="stn1",
            )
        )

    def test_delete_existing(self):
        self._store.upsert_station(
            StationRow(
                tree_path="surveys/s1/stations/tmp",
                survey_name="s1",
                name="tmp",
            )
        )
        deleted = self._store.delete_station_by_tree_path("surveys/s1/stations/tmp")
        assert deleted is True
        assert self._store.station_record("surveys/s1/stations/tmp") is None

    def test_delete_nonexistent_returns_false(self):
        assert self._store.delete_station_by_tree_path("surveys/x/stations/y") is False


class TestQueryStationPaths:
    @pytest.fixture(autouse=True)
    def store_with_station(self):
        self._store = MTDataTreeIndexStore()
        self._store.upsert_station(
            StationRow(
                tree_path="surveys/s1/stations/stn1",
                survey_name="s1",
                name="stn1",
                latitude=36.0,
                longitude=-111.0,
            )
        )

    def test_no_filters_returns_all(self):
        paths = self._store.all_station_paths()
        assert isinstance(paths, list)
        assert len(paths) == 1

    def test_geographic_filter(self):
        paths = self._store.query_station_paths(lat_min=35.5, lat_max=37.0)
        assert "surveys/s1/stations/stn1" in paths

    def test_survey_filter(self):
        paths = self._store.query_station_paths(survey="s1")
        assert len(paths) == 1


# ---------------------------------------------------------------------------
# Integration tests with MTData
# ---------------------------------------------------------------------------


class TestMTDataTreeIndexIntegration:
    def test_use_index_flag_creates_index(self):
        tree = MTData(use_index=True)
        assert tree._index is not None

    def test_no_index_by_default(self):
        tree = MTData()
        assert tree._index is None

    def test_add_station_populates_index(self, mt_objects):
        tree = MTData(use_index=True)
        path = tree.add_station(mt_objects[0])
        assert tree._index.n_stations() == 1
        rec = tree._index.station_record(path)
        assert rec is not None
        assert rec.latitude is not None

    def test_add_stations_bulk_populates_index(self, indexed_tree, mt_objects):
        assert indexed_tree._index.n_stations() == len(mt_objects)

    def test_get_station_accepts_short_path_with_index(self, mt_objects):
        tree = MTData(use_index=True)
        path = tree.add_station(mt_objects[0])
        short_path = f"{mt_objects[0].survey}/{mt_objects[0].station}"

        out = tree.get_station(short_path)

        assert out.identical(tree.get_station(path))

    def test_remove_station_updates_index(self, mt_objects):
        tree = MTData(use_index=True)
        path = tree.add_station(mt_objects[0])
        assert tree._index.n_stations() == 1
        tree.remove_station(path)
        assert tree._index.n_stations() == 0

    def test_remove_station_accepts_short_path_with_index(self, mt_objects):
        tree = MTData(use_index=True)
        path = tree.add_station(mt_objects[0])
        short_path = f"{mt_objects[0].survey}/{mt_objects[0].station}"

        tree.remove_station(short_path)

        assert tree._index.n_stations() == 0
        assert tree._index.station_record(path) is None

    def test_rebuild_index_builds_from_existing_tree(self, mt_objects):
        # Start without index, then rebuild
        tree = MTData(use_index=False)
        tree.add_stations(mt_objects)
        tree.rebuild_index()
        assert tree._index is not None
        assert tree._index.n_stations() == len(mt_objects)

    def test_apply_bounding_box_uses_index(self, indexed_tree):
        """apply_bounding_box should delegate to index when available."""
        all_paths = indexed_tree._index.all_station_paths()
        if not all_paths:
            pytest.skip("No indexed stations available")
        # Get the bounding box from station records.
        lats = [indexed_tree._index.station_record(p).latitude for p in all_paths]
        lons = [indexed_tree._index.station_record(p).longitude for p in all_paths]
        lats = [v for v in lats if v is not None]
        lons = [v for v in lons if v is not None]
        if not lats or not lons:
            pytest.skip("Station records missing coordinate data")
        result = indexed_tree.apply_bounding_box(
            lon_min=min(lons) - 1,
            lon_max=max(lons) + 1,
            lat_min=min(lats) - 1,
            lat_max=max(lats) + 1,
        )
        assert result._iter_station_paths()

    def test_query_station_paths_raises_without_index(self):
        tree = MTData(use_index=False)
        with pytest.raises(RuntimeError, match="Index not enabled"):
            tree.query_station_paths(survey="any")

    def test_query_station_paths_with_index(self, indexed_tree):
        paths = indexed_tree.query_station_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0
