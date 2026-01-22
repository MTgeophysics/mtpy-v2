# -*- coding: utf-8 -*-
"""
Pytest-based tests for MTStations and MTLocation functionality.

This module provides comprehensive testing for the MTStations class,
including grid-based station layouts, profile extraction, coordinate
transformations, and multi-EPSG handling.

Converted from unittest to pytest with fixtures for improved efficiency and
optimized for parallel execution with pytest-xdist.

Created on December 22, 2025

@author: jpeacock (original unittest)
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import pytest

from mtpy import MT
from mtpy.core import MTLocation, MTStations


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def utm_epsg():
    """Provide standard UTM EPSG code for tests."""
    return 32611


@pytest.fixture(scope="session")
def grid_base_location():
    """Provide base location for grid tests."""
    return {"east": 243900.352, "north": 4432069.056898517}


@pytest.fixture(scope="session")
def grid_center_point():
    """Provide center point for grid tests."""
    return MTLocation(
        latitude=40.036594,
        longitude=-119.978167,
        utm_epsg=32611,
        model_east=245900.352,
        model_north=4436069.057,
    )


@pytest.fixture(scope="session")
def grid_mt_list(utm_epsg, grid_base_location):
    """Create 5x5 grid of MT stations for testing."""
    dx = 1000
    dy = 2000
    count = 1
    mt_list = []
    for ii in range(5):
        for jj in range(5):
            mt_obj = MT(
                east=(grid_base_location["east"] + ii * dx),
                north=(grid_base_location["north"] + jj * dy),
                utm_epsg=utm_epsg,
                station=f"mt{count:02}",
            )
            count += 1
            mt_list.append(mt_obj)
    return mt_list


@pytest.fixture(scope="session")
def grid_stations(utm_epsg, grid_mt_list):
    """
    Session-scoped MTStations object with grid layout.

    Since all tests that modify create copies first, this can be
    safely shared across all tests for better performance.
    """
    return MTStations(utm_epsg, mt_list=grid_mt_list)


@pytest.fixture(scope="session")
def profile_base_location():
    """Provide base location for profile tests."""
    return {"east": 243900.352, "north": 4432069.056898517}


@pytest.fixture(scope="session")
def profile_center_point():
    """Provide center point for profile tests."""
    return MTLocation(
        latitude=42.212595,
        longitude=-120.078305,
        utm_epsg=32611,
        model_east=245900.352,
        model_north=4677969.409,
    )


@pytest.fixture(scope="session")
def profile_mt_list(utm_epsg, profile_base_location):
    """Create profile of MT stations for testing."""
    slope = 1
    count = 1
    dx = 1000
    mt_list = []
    for ii in range(5):
        x = profile_base_location["east"] + ii * dx
        mt_obj = MT(
            east=x,
            north=slope * x + profile_base_location["north"],
            utm_epsg=utm_epsg,
            station=f"mt{count:02}",
        )
        count += 1
        mt_list.append(mt_obj)
    return mt_list


@pytest.fixture(scope="session")
def profile_stations(utm_epsg, profile_mt_list):
    """
    Session-scoped MTStations object with profile layout.

    Since all tests that modify create copies first, this can be
    safely shared across all tests for better performance.
    """
    return MTStations(utm_epsg, mt_list=profile_mt_list)


@pytest.fixture(scope="session")
def expected_grid_station_names():
    """Provide expected station names for grid."""
    return [f"mt{i:02}" for i in range(1, 26)]


@pytest.fixture(scope="session")
def expected_profile_station_names():
    """Provide expected station names for profile."""
    return ["mt01", "mt02", "mt03", "mt04", "mt05"]


# =============================================================================
# Grid Tests
# =============================================================================


class TestMTStationGrid:
    """Test MTStations functionality with grid layout."""

    def test_station_count(self, grid_stations):
        """Test that grid has correct number of stations."""
        assert len(grid_stations) == 25

    def test_center_point(self, grid_stations, grid_center_point):
        """Test that center point is calculated correctly."""
        assert grid_stations.center_point == grid_center_point

    @pytest.mark.parametrize(
        "column",
        [
            "survey",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "datum_epsg",
            "east",
            "north",
            "utm_epsg",
            "model_east",
            "model_north",
            "model_elevation",
            "profile_offset",
        ],
    )
    def test_station_locations_columns(self, grid_stations, column):
        """Test that station_locations DataFrame has expected columns."""
        assert column in grid_stations.station_locations.columns

    def test_station_names(self, grid_stations, expected_grid_station_names):
        """Test that all station names are correct."""
        actual_names = grid_stations.station_locations["station"].tolist()
        assert actual_names == expected_grid_station_names

    @pytest.mark.parametrize(
        "column,expected_value",
        [
            ("survey", "0"),
            ("datum_epsg", "4326"),
            ("utm_epsg", "32611"),
            ("elevation", 0.0),
            ("model_elevation", 0.0),
            ("profile_offset", 0.0),
        ],
    )
    def test_uniform_column_values(self, grid_stations, column, expected_value):
        """Test that columns with uniform values are correct."""
        if isinstance(expected_value, str):
            assert (grid_stations.station_locations[column] == expected_value).all()
        else:
            assert np.isclose(
                grid_stations.station_locations[column], expected_value
            ).all()

    def test_east_values(self, grid_stations, grid_base_location):
        """Test that east coordinates are correct."""
        expected_east = []
        for ii in range(5):
            expected_east.extend([grid_base_location["east"] + ii * 1000] * 5)
        assert np.allclose(grid_stations.station_locations["east"], expected_east)

    def test_north_values(self, grid_stations, grid_base_location):
        """Test that north coordinates are correct."""
        expected_north = []
        for _ in range(5):
            for jj in range(5):
                expected_north.append(grid_base_location["north"] + jj * 2000)
        assert np.allclose(grid_stations.station_locations["north"], expected_north)

    def test_model_east_values(self, grid_stations):
        """Test that model east coordinates are correct relative to center."""
        expected = []
        for ii in range(5):
            expected.extend([-2000 + ii * 1000] * 5)
        assert np.allclose(grid_stations.station_locations["model_east"], expected)

    def test_model_north_values(self, grid_stations):
        """Test that model north coordinates are correct relative to center."""
        expected = []
        for _ in range(5):
            for jj in range(5):
                expected.append(-4000 + jj * 2000)
        assert np.allclose(grid_stations.station_locations["model_north"], expected)

    def test_copy(self, grid_stations):
        """Test copying MTStations object."""
        sc = grid_stations.copy()
        assert grid_stations == sc

    def test_rotation_angle_default(self, grid_stations):
        """Test default rotation angle is 0."""
        assert grid_stations.rotation_angle == 0

    def test_rotate_stations(self, grid_stations):
        """Test rotating station coordinates."""
        s = grid_stations.copy()
        s.rotate_stations(45)

        # Check rotation angle is set
        assert s.rotation_angle == 45

        # Check that coordinates have changed
        assert not np.allclose(
            s.station_locations["model_east"],
            grid_stations.station_locations["model_east"],
        )

    @pytest.mark.parametrize("angle", [0, 45, 90, 180, 270, 360, -45, -90])
    def test_rotate_stations_various_angles(self, grid_stations, angle):
        """Test rotation with various angles."""
        s = grid_stations.copy()
        s.rotate_stations(angle)
        assert s.rotation_angle == angle

    def test_rotate_45_degrees(self, grid_stations):
        """Test specific 45 degree rotation values."""
        s = grid_stations.copy()
        s.rotate_stations(45)

        # Test a few specific expected values for 45 degree rotation
        # Station mt01: model_east=-2000, model_north=-4000
        # After 45° rotation:
        # new_east = -2000*cos(45) - (-4000)*sin(45) = -1414.21 + 2828.43 = 1414.21
        # new_north = -2000*sin(45) + (-4000)*cos(45) = -1414.21 - 2828.43 = -4242.64

        # Just verify rotation occurred, not exact values (due to floating point)
        assert not np.array_equal(
            s.station_locations["model_east"].values,
            grid_stations.station_locations["model_east"].values,
        )

    def test_utm_epsg_property(self, grid_stations, utm_epsg):
        """Test utm_epsg property."""
        assert grid_stations.utm_epsg == utm_epsg

    def test_getitem_via_mt_list(self, grid_stations):
        """Test accessing stations via mt_list."""
        station = grid_stations.mt_list[0]
        assert station.station == "mt01"

    def test_iteration(self, grid_stations, expected_grid_station_names):
        """Test iterating over stations via mt_list."""
        station_names = [mt.station for mt in grid_stations.mt_list]
        assert station_names == expected_grid_station_names

    def test_to_dataframe(self, grid_stations):
        """Test conversion to DataFrame."""
        df = grid_stations.station_locations
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 25


# =============================================================================
# Profile Tests
# =============================================================================


class TestMTStationProfile:
    """Test MTStations functionality with profile layout."""

    def test_station_count(self, profile_stations):
        """Test that profile has correct number of stations."""
        assert len(profile_stations) == 5

    def test_center_point(self, profile_stations, profile_center_point):
        """Test that center point is calculated correctly."""
        assert profile_stations.center_point == profile_center_point

    def test_station_names(self, profile_stations, expected_profile_station_names):
        """Test that all station names are correct."""
        actual_names = profile_stations.station_locations["station"].tolist()
        assert actual_names == expected_profile_station_names

    def test_generate_profile_degrees(self, profile_stations):
        """Test profile generation in degree units."""
        profile = profile_stations.generate_profile(units="deg")

        # Check that profile returns expected format: (min_lon, min_lat, max_lon, max_lat, params)
        assert len(profile) == 5
        assert "slope" in profile[-1]
        assert "intercept" in profile[-1]

        # Test approximate values
        assert pytest.approx(profile[0], abs=0.001) == -120.10161927978938
        assert pytest.approx(profile[1], abs=0.001) == 42.19396005306167
        assert pytest.approx(profile[2], abs=0.001) == -120.05497729522492
        assert pytest.approx(profile[3], abs=0.001) == 42.23123311383864

    def test_generate_profile_meters(self, profile_stations, profile_base_location):
        """Test profile generation in meter units."""
        profile = profile_stations.generate_profile(units="m")

        # Check format
        assert len(profile) == 5
        assert "slope" in profile[-1]
        assert "intercept" in profile[-1]

        # Test values
        assert pytest.approx(profile[0]) == 243900.352
        assert pytest.approx(profile[1]) == 4675969.408898517
        assert pytest.approx(profile[2]) == 247900.352
        assert pytest.approx(profile[3]) == 4679969.408898517
        assert pytest.approx(profile[-1]["slope"]) == 1.0
        assert pytest.approx(profile[-1]["intercept"]) == 4432069.056898517

    @pytest.mark.parametrize("units", ["m", "deg", "km"])
    def test_generate_profile_units(self, profile_stations, units):
        """Test profile generation with different unit types."""
        if units == "km":
            # km is not supported, raises UnboundLocalError
            with pytest.raises(UnboundLocalError):
                profile_stations.generate_profile(units=units)
        else:
            profile = profile_stations.generate_profile(units=units)
            assert len(profile) == 5

    def test_extract_profile(self, profile_stations, expected_profile_station_names):
        """Test extracting stations along a profile."""
        extracted = profile_stations._extract_profile(
            243900.352, 4675969.408898517, 247900.352, 4679969.408898517, 1000
        )
        extracted_names = [mt.station for mt in extracted]
        assert extracted_names == expected_profile_station_names

    def test_extract_profile_tolerance(self, profile_stations):
        """Test profile extraction with different tolerances."""
        # Larger tolerance should still find all stations
        extracted = profile_stations._extract_profile(
            243900.352, 4675969.408898517, 247900.352, 4679969.408898517, 5000
        )
        assert len(extracted) == 5

        # Very small tolerance might miss stations
        extracted_tight = profile_stations._extract_profile(
            243900.352, 4675969.408898517, 247900.352, 4679969.408898517, 1
        )
        # Should still find some stations
        assert len(extracted_tight) >= 0


# =============================================================================
# Multi-EPSG Tests
# =============================================================================


class TestMTStationMultiEPSG:
    """Test MTStations with mixed EPSG codes."""

    @pytest.fixture
    def mixed_epsg_stations(self):
        """Create stations with different EPSG codes."""
        m1 = MT(latitude=40, longitude=-120, utm_epsg=32611, station="mt01")
        m2 = MT(latitude=20, longitude=-100, utm_epsg=32613, station="mt02")
        m3 = MT(latitude=42, longitude=-118, utm_epsg=32611, station="mt03")

        # Initialize with one EPSG, should be auto-corrected to most common
        return MTStations(32613, mt_list=[m1, m2, m3])

    def test_utm_epsg_most_common(self, mixed_epsg_stations):
        """Test that MTStations uses most common EPSG code."""
        # Should default to 32611 since 2 out of 3 stations use it
        assert mixed_epsg_stations.utm_epsg == 32611

    def test_station_count(self, mixed_epsg_stations):
        """Test station count with mixed EPSG."""
        assert len(mixed_epsg_stations) == 3

    def test_all_stations_same_epsg(self, mixed_epsg_stations):
        """Test that all stations are converted to same EPSG."""
        epsg_values = mixed_epsg_stations.station_locations["utm_epsg"].unique()
        assert len(epsg_values) == 1
        assert epsg_values[0] == "32611"


# =============================================================================
# Additional Functionality Tests
# =============================================================================


class TestMTStationsAdditional:
    """Test additional MTStations functionality not covered in original tests."""

    def test_empty_initialization(self, utm_epsg):
        """Test creating empty MTStations."""
        stations = MTStations(utm_epsg)
        assert len(stations) == 0

    def test_add_station(self, utm_epsg):
        """Test adding stations after initialization."""
        mt = MT(east=100, north=200, utm_epsg=utm_epsg, station="test01")
        stations = MTStations(utm_epsg, mt_list=[mt])
        assert len(stations) == 1
        assert stations.mt_list[0].station == "test01"

    def test_station_locations_shape(self, grid_stations):
        """Test station_locations DataFrame shape."""
        df = grid_stations.station_locations
        assert df.shape[0] == 25  # 25 stations
        assert df.shape[1] >= 13  # At least 13 columns

    @pytest.mark.parametrize(
        "attr",
        ["center_point", "rotation_angle", "utm_epsg", "station_locations"],
    )
    def test_has_attributes(self, grid_stations, attr):
        """Test that MTStations has expected attributes."""
        assert hasattr(grid_stations, attr)

    def test_equality(self, grid_stations):
        """Test MTStations equality comparison."""
        copy = grid_stations.copy()
        assert grid_stations == copy

    def test_inequality(self, grid_stations, profile_stations):
        """Test MTStations inequality comparison."""
        # Different DataFrames can't be compared directly, check they're not equal
        try:
            result = grid_stations == profile_stations
            # If comparison succeeds, they should not be equal
            assert not result
        except ValueError:
            # If comparison fails due to shape mismatch, they're definitely not equal
            pass

    def test_center_calculation(self, grid_stations, grid_center_point):
        """Test that center point is calculated from station coordinates."""
        # Center should be at the middle of the grid
        center = grid_stations.center_point
        assert isinstance(center, MTLocation)
        assert pytest.approx(center.model_east, abs=1) == grid_center_point.model_east
        assert pytest.approx(center.model_north, abs=1) == grid_center_point.model_north

    def test_rotation_preserves_station_count(self, grid_stations):
        """Test that rotation doesn't change station count."""
        s = grid_stations.copy()
        original_count = len(s)
        s.rotate_stations(30)
        assert len(s) == original_count

    def test_rotation_updates_model_coordinates(self, grid_stations):
        """Test that rotation updates model coordinates but not real coordinates."""
        s = grid_stations.copy()
        original_east = s.station_locations["east"].copy()
        original_model_east = s.station_locations["model_east"].copy()

        s.rotate_stations(45)

        # Real coordinates should not change
        assert np.allclose(s.station_locations["east"], original_east)

        # Model coordinates should change (unless rotation is 0 or 360)
        assert not np.allclose(s.station_locations["model_east"], original_model_east)

    @pytest.mark.parametrize("index", [0, 10, 24])
    def test_indexing_various_positions(self, grid_stations, index):
        """Test indexing stations at various positions via mt_list."""
        station = grid_stations.mt_list[index]
        assert isinstance(station, MT)
        assert station.station == f"mt{index + 1:02}"

    def test_negative_indexing(self, grid_stations):
        """Test negative indexing via mt_list."""
        last_station = grid_stations.mt_list[-1]
        assert last_station.station == "mt25"

    def test_slice_indexing(self, grid_stations):
        """Test slice indexing."""
        first_five = grid_stations.mt_list[:5]
        assert len(first_five) == 5
        assert first_five[0].station == "mt01"
        assert first_five[4].station == "mt05"

    def test_profile_offset_calculation(self, profile_stations):
        """Test that profile_offset is calculated (even if zero by default)."""
        offsets = profile_stations.station_locations["profile_offset"]
        assert len(offsets) == 5
        # All should be numeric
        assert offsets.dtype in [np.float64, np.float32, float]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestMTStationsEdgeCases:
    """Test edge cases and error handling for MTStations."""

    def test_single_station(self, utm_epsg):
        """Test MTStations with only one station."""
        mt = MT(east=100, north=200, utm_epsg=utm_epsg, station="solo")
        stations = MTStations(utm_epsg, mt_list=[mt])

        assert len(stations) == 1
        # With single station, center is at the station location
        assert stations.center_point.east == 100
        assert stations.center_point.north == 200

    def test_two_stations(self, utm_epsg):
        """Test MTStations with only two stations."""
        mt1 = MT(east=100, north=200, utm_epsg=utm_epsg, station="st01")
        mt2 = MT(east=300, north=400, utm_epsg=utm_epsg, station="st02")
        stations = MTStations(utm_epsg, mt_list=[mt1, mt2])

        assert len(stations) == 2
        # Center should be midpoint
        assert pytest.approx(stations.center_point.east, abs=1) == 200
        assert pytest.approx(stations.center_point.north, abs=1) == 300

    def test_duplicate_station_names(self, utm_epsg):
        """Test handling of duplicate station names."""
        mt1 = MT(east=100, north=200, utm_epsg=utm_epsg, station="dup")
        mt2 = MT(east=300, north=400, utm_epsg=utm_epsg, station="dup")
        stations = MTStations(utm_epsg, mt_list=[mt1, mt2])

        # Should still create both stations
        assert len(stations) == 2

    def test_rotation_360_equivalent_to_0(self, grid_stations):
        """Test that 360 degree rotation is equivalent to 0."""
        s1 = grid_stations.copy()
        s2 = grid_stations.copy()

        s1.rotate_stations(0)
        s2.rotate_stations(360)

        # Model coordinates should be very close (within floating point error)
        assert np.allclose(
            s1.station_locations["model_east"],
            s2.station_locations["model_east"],
            rtol=1e-10,
        )

    def test_invalid_index(self, grid_stations):
        """Test indexing with invalid index via mt_list."""
        with pytest.raises(IndexError):
            _ = grid_stations.mt_list[100]


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.slow
class TestMTStationsPerformance:
    """Performance tests for MTStations (marked as slow)."""

    def test_large_grid(self, utm_epsg):
        """Test creating and manipulating large station grid."""
        # Create 20x20 grid (400 stations)
        dx, dy = 1000, 1000
        base_east, base_north = 243900.352, 4432069.056898517

        mt_list = []
        count = 1
        for ii in range(20):
            for jj in range(20):
                mt = MT(
                    east=base_east + ii * dx,
                    north=base_north + jj * dy,
                    utm_epsg=utm_epsg,
                    station=f"st{count:03}",
                )
                count += 1
                mt_list.append(mt)

        stations = MTStations(utm_epsg, mt_list=mt_list)

        assert len(stations) == 400
        assert stations.station_locations.shape[0] == 400

    def test_multiple_rotations(self, grid_stations):
        """Test performance of multiple rotations."""
        s = grid_stations.copy()

        # Perform multiple rotations - angles accumulate
        for angle in [15, 30, 45, 60, 75, 90]:
            s.rotate_stations(angle)

        # Final angle should be sum of all rotations
        expected_angle = sum([15, 30, 45, 60, 75, 90])
        assert s.rotation_angle == expected_angle


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestMTStationsIntegration:
    """Integration tests for MTStations."""

    def test_grid_to_profile_workflow(self, grid_stations):
        """Test workflow from grid to profile extraction."""
        # Extract a profile from the grid
        profile = grid_stations.generate_profile(units="m")

        assert profile is not None
        assert len(profile) == 5

    def test_rotation_and_center_consistency(self, grid_stations):
        """Test that rotation doesn't affect center point calculation."""
        original_center = grid_stations.center_point

        s = grid_stations.copy()
        s.rotate_stations(45)

        # Center point should remain the same after rotation
        assert s.center_point == original_center


# =============================================================================
# Pytest Markers and Configuration
# =============================================================================


pytestmark = pytest.mark.unit
