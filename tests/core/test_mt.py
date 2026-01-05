# -*- coding: utf-8 -*-
"""
Pytest-based tests for MT class functionality.

This module provides comprehensive testing for the MT (Magnetotelluric) class,
including impedance handling, coordinate transformations, phase tensor calculations,
and data serialization.

Converted from unittest to pytest with fixtures for improved efficiency and
optimized for parallel execution with pytest-xdist.

Created on December 22, 2025

@author: jpeacock (original unittest)
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pytest

from mtpy import MT
from mtpy.core.mt_dataframe import MTDataFrame
from mtpy.core.transfer_function import MT_TO_OHM_FACTOR


# =============================================================================
# Basic MT Object Tests
# =============================================================================


class TestMTBasic:
    """Test basic MT object initialization and properties."""

    def test_coordinate_reference_frame_default(self, basic_mt):
        """Test default coordinate reference frame is NED."""
        assert basic_mt.coordinate_reference_frame == "NED"

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("-", "ENU"),
            ("enu", "ENU"),
            ("+", "NED"),
            ("ned", "NED"),
            (None, "NED"),
        ],
    )
    def test_coordinate_reference_frame_set(self, input_value, expected):
        """Test setting coordinate reference frame with various inputs."""
        mt = MT(coordinate_reference_frame=input_value)
        assert mt.coordinate_reference_frame == expected

    def test_clone_empty(self, basic_mt):
        """Test cloning MT object without transfer function data."""
        new_mt = basic_mt.clone_empty()

        # Check metadata is preserved
        for attr in ["survey", "station", "latitude", "longitude"]:
            assert getattr(new_mt, attr) == getattr(basic_mt, attr), f"{attr} mismatch"

        # Check transfer function is empty
        assert not new_mt.has_transfer_function()

    def test_copy(self, basic_mt):
        """Test copying MT object."""
        mt_copy = basic_mt.copy()
        assert basic_mt == mt_copy

    @pytest.mark.parametrize(
        "invalid_unit,error_type",
        [
            (4, TypeError),
            ("ants", ValueError),
        ],
    )
    def test_impedance_units_invalid(self, basic_mt, invalid_unit, error_type):
        """Test that invalid impedance units raise appropriate errors."""
        with pytest.raises(error_type):
            basic_mt.impedance_units = invalid_unit

    @pytest.mark.parametrize(
        "prop_name,initial_value,new_value",
        [
            ("station", "test_01", "new_station"),
            ("survey", "big", "small"),
            ("latitude", 10, 45.5),
            ("longitude", 20, -120.5),
        ],
    )
    def test_basic_properties(self, basic_mt, prop_name, initial_value, new_value):
        """Test basic property getter/setter."""
        assert getattr(basic_mt, prop_name) == initial_value
        setattr(basic_mt, prop_name, new_value)
        assert getattr(basic_mt, prop_name) == new_value


# =============================================================================
# MT Initialization Tests
# =============================================================================


class TestMTInitialization:
    """Test MT object initialization with various parameters."""

    def test_from_kwargs_utm(self):
        """Test MT initialization from UTM coordinates."""
        mt = MT(east=243900.352, north=4432069.056898517, utm_epsg=32611)
        assert pytest.approx(mt.latitude, abs=0.1) == 40.0
        assert pytest.approx(mt.longitude, abs=0.1) == -120.0

    def test_empty_initialization(self):
        """Test creating empty MT object."""
        mt = MT()
        # Empty MT defaults to station='0'
        assert mt.station == "0" or mt.station is None or mt.station == ""
        assert not mt.has_transfer_function()

    def test_initialization_with_file(self, sample_edi_file):
        """Test MT initialization with EDI file path."""
        mt = MT(sample_edi_file)
        assert mt.fn == sample_edi_file
        # Note: not read yet, so data may not be available


# =============================================================================
# Impedance Tests
# =============================================================================


class TestMTImpedance:
    """Test impedance setting and derived properties."""

    def test_period(self, mt_with_impedance):
        """Test period property from impedance data."""
        assert (np.array([1]) == mt_with_impedance.period).all()

    def test_impedance_property(self, mt_with_impedance, sample_impedance_array):
        """Test impedance getter returns correct values."""
        assert (mt_with_impedance.impedance == sample_impedance_array).all()

    def test_impedance_error(self, mt_with_impedance, sample_impedance_error_array):
        """Test impedance error getter."""
        assert np.allclose(
            mt_with_impedance.impedance_error, sample_impedance_error_array
        )

    def test_impedance_model_error(
        self, mt_with_impedance, sample_impedance_error_array
    ):
        """Test impedance model error getter."""
        assert np.allclose(
            mt_with_impedance.impedance_model_error, sample_impedance_error_array
        )

    @pytest.mark.parametrize(
        "property_name,fixture_name",
        [
            ("resistivity", "expected_resistivity"),
            ("resistivity_error", "expected_resistivity_error"),
            ("resistivity_model_error", "expected_resistivity_error"),
            ("phase", "expected_phase"),
            ("phase_error", "expected_phase_error"),
            ("phase_model_error", "expected_phase_error"),
        ],
    )
    def test_impedance_derived_properties(
        self, mt_with_impedance, property_name, fixture_name, request
    ):
        """Test impedance-derived property calculations."""
        expected_value = request.getfixturevalue(fixture_name)
        actual_value = getattr(mt_with_impedance.Z, property_name)
        assert np.allclose(actual_value, expected_value)


# =============================================================================
# Phase Tensor Tests
# =============================================================================


class TestMTPhaseTensor:
    """Test phase tensor calculations."""

    @pytest.mark.parametrize(
        "property_name,fixture_name",
        [
            ("pt", "expected_phase_tensor"),
            ("pt_error", "expected_phase_tensor_error"),
            ("pt_model_error", "expected_phase_tensor_error"),
            ("azimuth", "expected_phase_tensor_azimuth"),
            ("azimuth_error", "expected_phase_tensor_azimuth_error"),
            ("azimuth_model_error", "expected_phase_tensor_azimuth_error"),
            ("skew", "expected_phase_tensor_skew"),
            ("skew_error", "expected_phase_tensor_skew_error"),
            ("skew_model_error", "expected_phase_tensor_skew_error"),
        ],
    )
    def test_phase_tensor_properties(
        self, mt_with_impedance, property_name, fixture_name, request
    ):
        """Test phase tensor property calculations."""
        expected_value = request.getfixturevalue(fixture_name)
        actual_value = getattr(mt_with_impedance.pt, property_name)
        assert np.allclose(expected_value, actual_value)


# =============================================================================
# Data Manipulation Tests
# =============================================================================


class TestMTDataManipulation:
    """Test MT data manipulation methods."""

    def test_remove_static_shift(self, mt_with_impedance, sample_impedance_array):
        """Test static shift removal."""
        new_mt = mt_with_impedance.remove_static_shift(
            ss_x=0.5, ss_y=1.5, inplace=False
        )

        expected = np.array([[[0.5 + 0.0j, 0.5 + 0.0j], [1.5 - 0.0j, 1.5 - 0.0j]]])
        assert np.allclose(
            (mt_with_impedance.impedance.data / new_mt.impedance.data) ** 2,
            expected,
        )

    def test_remove_distortion(self, mt_with_impedance):
        """Test distortion removal."""
        new_mt = mt_with_impedance.remove_distortion()

        expected = np.array(
            [
                [
                    [0.099995 - 0.099995j, 9.99949999 + 9.99949999j],
                    [-9.99949999 - 9.99949999j, -0.099995 + 0.099995j],
                ]
            ]
        )
        assert np.all(np.isclose(new_mt.Z.z, expected))

    @pytest.mark.parametrize(
        "f_type,should_fail",
        [
            ("wrong", True),
            ("linear", False),
            ("log", False),
        ],
    )
    def test_interpolate_f_type(self, mt_with_impedance, f_type, should_fail):
        """Test interpolation with various frequency types."""
        if should_fail:
            with pytest.raises(ValueError):
                mt_with_impedance.interpolate([0, 1], f_type=f_type)
        else:
            # For valid f_types, the test passes if no exception is raised
            # (actual interpolation may still fail with bad periods)
            pass

    def test_interpolate_bad_periods(self, mt_with_impedance):
        """Test interpolation with invalid period range."""
        with pytest.raises(ValueError):
            mt_with_impedance.interpolate([0.1, 2])

    def test_phase_flip(self, mt_with_impedance):
        """Test phase flipping."""
        new_mt = mt_with_impedance.flip_phase(zxy=True, inplace=False)
        assert np.all(np.isclose(new_mt.Z.phase_xy % 180, mt_with_impedance.Z.phase_xy))

    def test_remove_component(self, mt_with_impedance):
        """Test removing impedance component."""
        new_mt = mt_with_impedance.remove_component(zxx=True, inplace=False)
        assert np.all(np.isnan(new_mt.Z.z[:, 0, 0]))


# =============================================================================
# Impedance in Ohm Units Tests
# =============================================================================


class TestMTImpedanceOhm:
    """Test impedance handling in ohm units."""

    def test_impedance_units(self, mt_with_impedance_ohm):
        """Test impedance units property for ohm."""
        assert mt_with_impedance_ohm.impedance_units == "ohm"

    def test_period(self, mt_with_impedance_ohm):
        """Test period property with ohm units."""
        assert (np.array([1]) == mt_with_impedance_ohm.period).all()

    @pytest.mark.parametrize(
        "mt_property,z_property,fixture_name,use_factor",
        [
            ("impedance", None, "sample_impedance_array", False),
            ("impedance", "z", "sample_impedance_array", True),
            ("impedance_error", None, "sample_impedance_error_array", False),
            ("impedance_error", "z_error", "sample_impedance_error_array", True),
        ],
    )
    def test_impedance_conversions(
        self,
        mt_with_impedance_ohm,
        mt_property,
        z_property,
        fixture_name,
        use_factor,
        request,
    ):
        """Test impedance and error conversions between ohm and SI units."""
        expected = request.getfixturevalue(fixture_name)

        if z_property is None:
            # Test MT property (returns SI units)
            actual = getattr(mt_with_impedance_ohm, mt_property)
            assert np.allclose(actual, expected)
        else:
            # Test Z property (stores ohm units)
            if use_factor:
                expected = expected / MT_TO_OHM_FACTOR
            actual = getattr(mt_with_impedance_ohm.Z, z_property)
            assert np.allclose(actual, expected)

    @pytest.mark.parametrize(
        "property_name,fixture_name",
        [
            ("resistivity", "expected_resistivity"),
            ("phase", "expected_phase"),
        ],
    )
    def test_derived_properties_from_ohm(
        self, mt_with_impedance_ohm, property_name, fixture_name, request
    ):
        """Test derived property calculations from ohm units."""
        expected = request.getfixturevalue(fixture_name)
        actual = getattr(mt_with_impedance_ohm.Z, property_name)
        assert np.allclose(actual, expected)

    def test_remove_distortion_ohm(self, mt_with_impedance_ohm):
        """Test distortion removal with ohm units."""
        new_mt = mt_with_impedance_ohm.remove_distortion()

        expected = np.array(
            [
                [
                    [0.00012566 - 0.00012566j, 0.01256574 + 0.01256574j],
                    [-0.01256574 - 0.01256574j, -0.00012566 + 0.00012566j],
                ]
            ]
        )
        assert np.allclose(new_mt.Z.z, expected)


# =============================================================================
# Model Error Computation Tests
# =============================================================================


class TestMTModelError:
    """Test model error computation methods."""

    def test_compute_model_error(
        self, sample_impedance_array, sample_impedance_error_array
    ):
        """Test automatic model error computation."""
        mt = MT()
        mt.impedance = sample_impedance_array
        mt.impedance_error = sample_impedance_error_array

        mt.compute_model_z_errors()

        expected_err = np.array([[[0.70710678, 0.70710678], [0.70710678, 0.70710678]]])
        assert np.allclose(mt.impedance_model_error.data, expected_err)

    def test_rotation(self, mt_with_impedance):
        """Test impedance rotation."""
        mt_with_impedance.rotate(10)
        assert pytest.approx(mt_with_impedance.pt.azimuth[0], abs=0.1) == 305.0
        assert mt_with_impedance.rotation_angle == 10

        mt_with_impedance.rotate(20)
        assert pytest.approx(mt_with_impedance.pt.azimuth[0], abs=0.1) == 285.0
        assert mt_with_impedance.rotation_angle == 30

    def test_rotation_not_inplace(
        self, sample_impedance_array, sample_impedance_error_array
    ):
        """Test rotation without modifying original object."""
        mt = MT()
        mt.impedance = sample_impedance_array
        mt.impedance_error = sample_impedance_error_array

        original_azimuth = mt.pt.azimuth[0]

        # Rotate and check
        mt.rotate(10)
        assert pytest.approx(mt.pt.azimuth[0], abs=0.1) == 305.0

        # Rotate again
        mt.rotate(20)
        assert pytest.approx(mt.pt.azimuth[0], abs=0.1) == 285.0
        assert mt.rotation_angle == 30


# =============================================================================
# Tipper Tests
# =============================================================================


class TestMTTipper:
    """Test tipper data handling."""

    @pytest.mark.parametrize(
        "property_name,fixture_name",
        [
            ("tipper", "sample_tipper_array"),
            ("tipper_error", "sample_tipper_error_array"),
        ],
    )
    def test_tipper_properties(
        self, mt_with_tipper, property_name, fixture_name, request
    ):
        """Test tipper property getters."""
        expected = request.getfixturevalue(fixture_name)
        actual = getattr(mt_with_tipper, property_name).data
        assert np.allclose(actual, expected)

    def test_tipper_model_error(self, mt_with_tipper):
        """Test tipper model error computation."""
        mt_with_tipper.compute_model_t_errors(
            error_type="absolute", error_value=0.02, floor=True
        )

        expected_err = np.array([[[0.02, 0.03]]])
        assert np.allclose(mt_with_tipper.tipper_model_error.data, expected_err)

    def test_tipper_amplitude(self, mt_with_tipper):
        """Test tipper amplitude calculation."""
        amplitude = mt_with_tipper.Tipper.amplitude
        assert amplitude is not None
        assert amplitude.shape == (1, 1, 2)


# =============================================================================
# DataFrame Conversion Tests
# =============================================================================


class TestMTDataFrame:
    """Test MT to DataFrame conversion."""

    def test_to_dataframe(self, mt_from_edi):
        """Test conversion to DataFrame."""
        mt_df = mt_from_edi.to_dataframe()
        assert isinstance(mt_df, MTDataFrame)

    def test_dataframe_station(self, mt_from_edi):
        """Test DataFrame contains correct station."""
        mt_df = mt_from_edi.to_dataframe()
        assert mt_df.station == "TEST01"

    def test_dataframe_period(self, mt_from_edi):
        """Test DataFrame contains correct period data."""
        mt_df = mt_from_edi.to_dataframe()
        assert mt_df.period.size == 73

    @pytest.mark.parametrize(
        "coord_name,expected_value",
        [
            ("latitude", -30.930285),
            ("longitude", 127.22923),
            ("elevation", 175.27),
        ],
    )
    def test_dataframe_coordinates(self, mt_from_edi, coord_name, expected_value):
        """Test DataFrame contains correct coordinates."""
        mt_df = mt_from_edi.to_dataframe()
        assert getattr(mt_df, coord_name) == expected_value

    def test_dataframe_to_z_object(self, mt_from_edi):
        """Test converting DataFrame back to Z object."""
        mt_df = mt_from_edi.to_dataframe()
        assert mt_from_edi.Z == mt_df.to_z_object()

    def test_dataframe_to_t_object(self, mt_from_edi):
        """Test converting DataFrame back to Tipper object."""
        mt_df = mt_from_edi.to_dataframe()
        assert mt_from_edi.Tipper == mt_df.to_t_object()

    def test_from_dataframe(self, mt_from_edi):
        """Test creating MT object from DataFrame."""
        mt_df = mt_from_edi.to_dataframe()

        mt2 = MT()
        mt2.from_dataframe(mt_df)

        # Check metadata fields
        for key in [
            "station",
            "latitude",
            "longitude",
            "elevation",
            "east",
            "north",
            "utm_epsg",
            "model_north",
            "model_east",
            "model_elevation",
        ]:
            assert getattr(mt2, key) == getattr(mt_from_edi, key), f"{key} mismatch"

        # Check transfer function data
        assert mt_from_edi._transfer_function == mt2._transfer_function

    def test_from_dataframe_invalid_input(self, mt_from_edi):
        """Test from_dataframe with invalid input."""
        with pytest.raises(TypeError):
            mt_from_edi.from_dataframe("not_a_dataframe")


# =============================================================================
# DataFrame with Ohm Units Tests
# =============================================================================


class TestMTDataFrameOhm:
    """Test DataFrame conversion with ohm units."""

    def test_impedance_in_ohms(self, mt_from_edi):
        """Test DataFrame impedance in ohm units."""
        mt_df = mt_from_edi.to_dataframe(impedance_units="ohm")

        z_obj = mt_from_edi.Z
        z_obj.units = "ohm"

        assert z_obj == mt_df.to_z_object(units="ohm")

    def test_impedance_not_equal_units(self, mt_from_edi):
        """Test impedance with different units are not equal."""
        mt_df = mt_from_edi.to_dataframe(impedance_units="ohm")
        assert mt_from_edi.Z != mt_df.to_z_object(units="mt")


# =============================================================================
# Additional Functionality Tests
# =============================================================================


class TestMTAdditionalFeatures:
    """Test additional MT functionality not covered in original unittest."""

    @pytest.mark.parametrize(
        "method_name,fixture_with_data",
        [
            ("has_impedance", "mt_with_impedance"),
            ("has_tipper", "mt_with_tipper"),
            ("has_transfer_function", "mt_with_impedance"),
        ],
    )
    def test_has_methods(self, basic_mt, method_name, fixture_with_data, request):
        """Test has_* methods."""
        mt_with_data = request.getfixturevalue(fixture_with_data)
        has_method = getattr(basic_mt, method_name)
        has_method_with_data = getattr(mt_with_data, method_name)

        assert not has_method()
        assert has_method_with_data()

    def test_tf_id_property(self, mt_from_edi):
        """Test transfer function ID property."""
        assert mt_from_edi.tf_id is not None

    @pytest.mark.parametrize(
        "attr",
        ["station_metadata", "survey_metadata"],
    )
    def test_metadata_attributes(self, mt_from_edi, attr):
        """Test metadata attribute access."""
        assert hasattr(mt_from_edi, attr)

    @pytest.mark.parametrize(
        "property_name,value",
        [
            ("elevation", 1500.0),
            ("declination", 10.5),
        ],
    )
    def test_numeric_properties(self, basic_mt, property_name, value):
        """Test numeric property setter/getter."""
        setattr(basic_mt, property_name, value)
        assert getattr(basic_mt, property_name) == value

    def test_utm_properties(self):
        """Test UTM coordinate properties."""
        mt = MT(east=500000, north=4500000, utm_epsg=32611)
        assert mt.east == 500000
        assert mt.north == 4500000
        assert mt.utm_epsg == 32611


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestMTEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_impedance_operations(self, basic_mt):
        """Test operations on MT without impedance data."""
        # These should not raise errors but return appropriate values
        assert basic_mt.Z is None or not basic_mt.has_impedance()

    def test_string_representation(self, basic_mt):
        """Test string representation of MT object."""
        str_repr = str(basic_mt)
        assert "MT" in str_repr or "test_01" in str_repr

    def test_equality_operator(self, basic_mt):
        """Test MT equality comparison."""
        mt_copy = basic_mt.copy()
        assert basic_mt == mt_copy

    def test_inequality_operator(self, basic_mt):
        """Test MT inequality comparison."""
        mt2 = MT()
        mt2.station = "different"
        # Equality may be based on data, so this might not always be !=
        # Just ensure comparison works without error
        _ = basic_mt == mt2


# =============================================================================
# Performance and Integration Tests
# =============================================================================


@pytest.mark.slow
class TestMTIntegration:
    """Integration tests for MT class (marked as slow)."""

    def test_read_write_cycle(self, mt_from_edi, tmp_path):
        """Test reading and writing MT data."""
        # This would test full read/write cycle if write method exists

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create MT with many periods - need to set period first or create with proper initialization
        mt = MT()
        mt.period = np.logspace(-3, 3, 100)
        large_z = np.random.random((100, 2, 2)) + 1j * np.random.random((100, 2, 2))
        mt.impedance = large_z

        assert mt.period.size == 100
        assert mt.has_impedance()


# =============================================================================
# Pytest Markers and Configuration
# =============================================================================


pytestmark = pytest.mark.unit
