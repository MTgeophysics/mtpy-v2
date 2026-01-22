# -*- coding: utf-8 -*-
"""
Pytest suite for MTData class.

Created on Thu Oct 17 16:40:56 2024
Converted to pytest and optimized for pytest-xdist.

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import pytest

from mtpy import MT, MTData


# =============================================================================
# Session-scoped fixtures for read-only data
# =============================================================================


@pytest.fixture(scope="session")
def utm_epsg():
    """Session-scoped UTM EPSG code."""
    return 3216


@pytest.fixture(scope="session")
def datum_epsg():
    """Session-scoped datum EPSG code."""
    return 4236


@pytest.fixture(scope="session")
def mt_list_01():
    """Session-scoped MT list for survey 'a'."""
    return [
        MT(
            survey="a",
            station=f"mt{ii:02}",
            latitude=40 + ii,
            longitude=-118,
        )
        for ii in range(4)
    ]


@pytest.fixture(scope="session")
def mt_list_02():
    """Session-scoped MT list for survey 'b'."""
    return [
        MT(
            survey="b",
            station=f"mt{ii:02}",
            latitude=45 + ii,
            longitude=-118,
        )
        for ii in range(4)
    ]


@pytest.fixture(scope="session")
def mt_data_ro(mt_list_01, mt_list_02, utm_epsg):
    """
    Session-scoped read-only MTData object.

    This fixture is shared across all tests that only read data.
    Tests that modify MTData should use the function-scoped `mt_data` fixture.
    """
    return MTData(mt_list=mt_list_01 + mt_list_02, utm_epsg=utm_epsg)


# =============================================================================
# Function-scoped fixtures for tests that modify
# =============================================================================


@pytest.fixture
def mt_data(mt_list_01, mt_list_02, utm_epsg):
    """Function-scoped MTData object for tests that modify data."""
    return MTData(mt_list=mt_list_01 + mt_list_02, utm_epsg=utm_epsg)


@pytest.fixture
def empty_mt_data():
    """Empty MTData object for testing add/remove operations."""
    return MTData()


# =============================================================================
# Tests for MTData basic properties and methods
# =============================================================================


class TestMTDataBasicProperties:
    """Test basic properties of MTData."""

    def test_utm_epsg(self, mt_data_ro, utm_epsg):
        """Test UTM EPSG property."""
        assert mt_data_ro.utm_epsg == utm_epsg

    def test_coordinate_reference_frame_default(self, mt_data_ro):
        """Test default coordinate reference frame."""
        assert mt_data_ro.coordinate_reference_frame == "NED"

    def test_coordinate_reference_frame_set(self, mt_list_01):
        """Test setting coordinate reference frame."""
        md = MTData(mt_list=mt_list_01, coordinate_reference_frame="enu")

        assert md.coordinate_reference_frame == "ENU"

        # Verify all MT objects have the same reference frame
        for mt_obj in md.values():
            assert mt_obj.coordinate_reference_frame == "ENU"

    def test_survey_ids(self, mt_data_ro):
        """Test survey IDs extraction."""
        assert sorted(mt_data_ro.survey_ids) == ["a", "b"]

    def test_n_station(self, mt_data_ro):
        """Test station count."""
        assert mt_data_ro.n_stations == 8


class TestMTDataInitialization:
    """Test MTData initialization scenarios."""

    def test_initialization_utm_epsg_no_mt_list(self, utm_epsg):
        """Test initialization with UTM EPSG but no MT list."""
        md = MTData(utm_epsg=utm_epsg)
        assert md.utm_epsg == utm_epsg

    def test_initialization_datum_epsg_no_mt_list(self, datum_epsg):
        """Test initialization with datum EPSG but no MT list."""
        md = MTData(datum_epsg=datum_epsg)
        assert md.datum_epsg == datum_epsg

    def test_initialization_empty(self):
        """Test empty initialization."""
        md = MTData()
        assert md.n_stations == 0
        assert len(md) == 0


class TestMTDataEquality:
    """Test MTData equality and copying."""

    def test_eq(self, mt_list_01, mt_list_02, utm_epsg):
        """Test equality of two MTData objects."""
        md1 = MTData(mt_list=mt_list_01 + mt_list_02, utm_epsg=utm_epsg)
        md2 = MTData(mt_list=mt_list_01 + mt_list_02, utm_epsg=utm_epsg)
        assert md1 == md2

    def test_neq(self, mt_list_01, mt_list_02, utm_epsg):
        """Test inequality of MTData objects."""
        md1 = MTData(mt_list=mt_list_01 + mt_list_02, utm_epsg=utm_epsg)
        md2 = MTData(mt_list=mt_list_01, utm_epsg=utm_epsg)
        assert md1 != md2

    def test_deep_copy(self, mt_data_ro):
        """Test deep copy of MTData."""
        md_copy = mt_data_ro.copy()
        assert mt_data_ro == md_copy
        # Verify it's a deep copy, not a reference
        assert md_copy is not mt_data_ro


class TestMTDataClone:
    """Test MTData cloning operations."""

    @pytest.mark.parametrize(
        "attr",
        [
            "utm_epsg",
            "datum_epsg",
            "coordinate_reference_frame",
            "data_rotation_angle",
        ],
    )
    def test_clone_empty_attributes(self, mt_data_ro, attr):
        """Test that clone_empty preserves key attributes."""
        md_empty = mt_data_ro.clone_empty()
        assert getattr(mt_data_ro, attr) == getattr(md_empty, attr)

    def test_clone_empty_is_empty(self, mt_data_ro):
        """Test that clone_empty creates an empty MTData object."""
        md_empty = mt_data_ro.clone_empty()
        assert md_empty.n_stations == 0
        assert len(md_empty) == 0


class TestMTDataValidation:
    """Test MTData validation methods."""

    def test_validate_item_fail(self, mt_data_ro):
        """Test that _validate_item raises TypeError for invalid item."""
        with pytest.raises(TypeError):
            mt_data_ro._validate_item(10)

    def test_validate_item_success(self, mt_data_ro):
        """Test that _validate_item accepts MT objects."""
        mt_obj = MT(station="test", latitude=40, longitude=-118)
        # Should not raise
        mt_data_ro._validate_item(mt_obj)


class TestMTDataGetSurvey:
    """Test getting data by survey."""

    def test_get_survey(self, mt_data_ro):
        """Test getting stations from a specific survey."""
        survey_a = mt_data_ro.get_survey("a")

        assert len(survey_a) == 4

        # Verify attributes are preserved
        assert survey_a.utm_epsg == mt_data_ro.utm_epsg
        assert survey_a.datum_epsg == mt_data_ro.datum_epsg
        assert (
            survey_a.coordinate_reference_frame == mt_data_ro.coordinate_reference_frame
        )

    def test_get_survey_nonexistent(self, mt_data_ro):
        """Test getting a non-existent survey returns empty MTData."""
        survey_x = mt_data_ro.get_survey("x")
        assert len(survey_x) == 0


class TestMTDataRotation:
    """Test rotation operations on MTData."""

    def test_rotate_inplace(self, mt_data):
        """Test in-place rotation."""
        mt_data.rotate(30)

        assert mt_data.data_rotation_angle == 30
        assert mt_data["a.mt01"].rotation_angle == 30

    def test_rotate_not_inplace(self, utm_epsg):
        """Test rotation without modifying original."""
        # Create fresh MT objects to avoid session fixture interference
        mt_list_a = [
            MT(survey="a", station=f"mt{ii:02}", latitude=40 + ii, longitude=-118)
            for ii in range(4)
        ]
        mt_list_b = [
            MT(survey="b", station=f"mt{ii:02}", latitude=45 + ii, longitude=-118)
            for ii in range(4)
        ]
        md = MTData(mt_list=mt_list_a + mt_list_b, utm_epsg=utm_epsg)
        original_angle = md.data_rotation_angle
        md_rot = md.rotate(30, inplace=False)

        # Verify rotated copy has correct rotation
        assert md_rot.data_rotation_angle == 30
        assert md_rot["a.mt01"].rotation_angle == 30

        # Verify original is completely unchanged
        assert md.data_rotation_angle == original_angle
        assert md["a.mt01"].rotation_angle == 0

    @pytest.mark.parametrize("angle", [0, 30, 45, 90, 180, 270, 360, -45, -90])
    def test_rotate_various_angles(self, mt_data, angle):
        """Test rotation with various angles."""
        mt_data.rotate(angle)
        assert mt_data.data_rotation_angle == angle


class TestMTDataGetStation:
    """Test getting individual stations."""

    def test_get_station_from_key(self, mt_data_ro):
        """Test getting station by station key."""
        station = mt_data_ro.get_station(station_key="a.mt01")
        assert station.station == "mt01"
        assert station.survey == "a"

    def test_get_station_missing_key(self, mt_data_ro):
        """Test getting station with non-existent key raises KeyError."""
        with pytest.raises(KeyError):
            mt_data_ro.get_station(station_key="x.mt99")


class TestMTDataGetSubset:
    """Test getting subsets of stations."""

    def test_get_subset_from_keys(self, mt_data_ro):
        """Test getting subset using station keys."""
        station_keys = ["a.mt01", "b.mt02"]
        md_subset = mt_data_ro.get_subset(station_keys)

        assert list(md_subset.keys()) == station_keys
        assert md_subset.utm_epsg == mt_data_ro.utm_epsg
        assert md_subset.datum_epsg == mt_data_ro.datum_epsg

    def test_get_subset_from_ids_fail(self, mt_data_ro):
        """Test that station IDs without survey prefix fail."""
        station_list = ["mt01", "mt02"]
        with pytest.raises(KeyError):
            mt_data_ro.get_subset(station_list)

    def test_get_subset_empty_list(self, mt_data_ro):
        """Test getting subset with empty list returns empty MTData."""
        md_subset = mt_data_ro.get_subset([])
        assert len(md_subset) == 0


# =============================================================================
# Tests for MTData methods (add, remove, etc.)
# =============================================================================


class TestMTDataAddStation:
    """Test adding stations to MTData."""

    def test_add_station_single(self, empty_mt_data, mt_list_01):
        """Test adding a single station."""
        empty_mt_data.add_station(mt_list_01[0])
        assert list(empty_mt_data.keys()) == ["a.mt00"]
        assert empty_mt_data.n_stations == 1

    def test_add_station_list(self, empty_mt_data, mt_list_01):
        """Test adding a list of stations."""
        empty_mt_data.add_station(mt_list_01)
        assert empty_mt_data.n_stations == 4
        assert "a.mt00" in empty_mt_data
        assert "a.mt03" in empty_mt_data

    def test_add_station_duplicate(self, empty_mt_data, mt_list_01):
        """Test adding duplicate station (should replace)."""
        empty_mt_data.add_station(mt_list_01[0])
        empty_mt_data.add_station(mt_list_01[0])
        assert empty_mt_data.n_stations == 1


class TestMTDataRemoveStation:
    """Test removing stations from MTData."""

    def test_remove_station(self, empty_mt_data, mt_list_01):
        """Test removing a station."""
        empty_mt_data.add_station(mt_list_01)
        empty_mt_data.remove_station("mt00", "a")

        assert "a.mt00" not in empty_mt_data.keys()
        assert empty_mt_data.n_stations == 3

    def test_remove_station_nonexistent(self, mt_data):
        """Test removing non-existent station."""
        initial_count = mt_data.n_stations
        # Should not raise error, just do nothing
        mt_data.remove_station("mt99", "x")
        assert mt_data.n_stations == initial_count


class TestMTDataGetStationKey:
    """Test _get_station_key helper method."""

    def test_get_station_key_no_survey(self, empty_mt_data, mt_list_01):
        """Test getting station key without specifying survey."""
        empty_mt_data.add_station(mt_list_01)
        key = empty_mt_data._get_station_key("mt01", None)
        assert key == "a.mt01"

    def test_get_station_key_with_survey(self, empty_mt_data, mt_list_01):
        """Test getting station key with survey specified."""
        empty_mt_data.add_station(mt_list_01)
        key = empty_mt_data._get_station_key("mt01", "a")
        assert key == "a.mt01"

    def test_get_station_key_fail(self, empty_mt_data, mt_list_01):
        """Test that _get_station_key fails with None station."""
        empty_mt_data.add_station(mt_list_01)
        with pytest.raises(KeyError):
            empty_mt_data._get_station_key(None, "a")


class TestMTDataImpedanceUnits:
    """Test impedance units setting."""

    def test_impedance_units_bad_type(self, empty_mt_data):
        """Test that setting impedance_units to wrong type raises TypeError."""
        with pytest.raises(TypeError):
            empty_mt_data.impedance_units = 4

    def test_impedance_units_bad_choice(self, empty_mt_data):
        """Test that setting invalid impedance_units raises ValueError."""
        with pytest.raises(ValueError):
            empty_mt_data.impedance_units = "ants"

    @pytest.mark.parametrize("units", ["mt", "ohm"])
    def test_set_impedance_units(self, empty_mt_data, mt_list_01, units):
        """Test setting various valid impedance units."""
        empty_mt_data.add_station(mt_list_01)
        empty_mt_data.impedance_units = units

        for mt_obj in empty_mt_data.values():
            assert mt_obj.impedance_units == units


# =============================================================================
# Additional tests for functionality not covered by unittest
# =============================================================================


class TestMTDataIterationAndIndexing:
    """Test iteration and indexing operations."""

    def test_iteration(self, mt_data_ro):
        """Test iterating over MTData keys."""
        keys = list(mt_data_ro.keys())
        assert len(keys) == 8
        assert "a.mt00" in keys
        assert "b.mt03" in keys

    def test_indexing(self, mt_data_ro):
        """Test indexing MTData with station key."""
        mt_obj = mt_data_ro["a.mt01"]
        assert mt_obj.station == "mt01"
        assert mt_obj.survey == "a"

    def test_contains(self, mt_data_ro):
        """Test 'in' operator for station keys."""
        assert "a.mt01" in mt_data_ro
        assert "x.mt99" not in mt_data_ro

    def test_len(self, mt_data_ro):
        """Test len() on MTData."""
        assert len(mt_data_ro) == 8

    def test_values(self, mt_data_ro):
        """Test values() returns MT objects."""
        values = list(mt_data_ro.values())
        assert len(values) == 8
        assert all(isinstance(v, MT) for v in values)

    def test_items(self, mt_data_ro):
        """Test items() returns key-value pairs."""
        items = list(mt_data_ro.items())
        assert len(items) == 8
        for key, value in items:
            assert isinstance(key, str)
            assert isinstance(value, MT)


class TestMTDataEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_mt_data_properties(self):
        """Test properties on empty MTData."""
        md = MTData()
        assert md.n_stations == 0
        assert md.survey_ids == []
        assert len(md) == 0

    def test_single_station_mtdata(self):
        """Test MTData with single station."""
        mt_obj = MT(survey="test", station="mt01", latitude=40, longitude=-118)
        md = MTData(mt_list=[mt_obj])
        assert md.n_stations == 1
        assert md.survey_ids == ["test"]

    def test_rotation_preserves_station_count(self, mt_data):
        """Test that rotation doesn't change station count."""
        initial_count = mt_data.n_stations
        mt_data.rotate(45)
        assert mt_data.n_stations == initial_count

    def test_get_subset_preserves_order(self, mt_data_ro):
        """Test that get_subset preserves station order."""
        keys = ["b.mt02", "a.mt01", "b.mt00"]
        subset = mt_data_ro.get_subset(keys)
        assert list(subset.keys()) == keys


class TestMTDataMultipleSurveys:
    """Test handling of multiple surveys."""

    def test_multiple_surveys_station_keys(self, mt_data_ro):
        """Test that station keys properly distinguish surveys."""
        # Both surveys have stations named mt00-mt03
        assert "a.mt00" in mt_data_ro
        assert "b.mt00" in mt_data_ro
        assert mt_data_ro["a.mt00"].survey == "a"
        assert mt_data_ro["b.mt00"].survey == "b"

    def test_get_survey_returns_correct_stations(self, mt_data_ro):
        """Test that get_survey returns only stations from specified survey."""
        survey_a = mt_data_ro.get_survey("a")
        for mt_obj in survey_a.values():
            assert mt_obj.survey == "a"

    def test_multiple_surveys_count(self, mt_data_ro):
        """Test station count with multiple surveys."""
        survey_a = mt_data_ro.get_survey("a")
        survey_b = mt_data_ro.get_survey("b")
        assert len(survey_a) + len(survey_b) == mt_data_ro.n_stations


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
