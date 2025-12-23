# -*- coding: utf-8 -*-
"""
Pytest version of Tipper tests

Created on Tue Nov  8 13:04:38 2022

@author: jpeacock
"""
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mtpy.core.transfer_function.tipper import Tipper


# =============================================================================
# Session Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def empty_tipper():
    """Empty Tipper object for testing initialization."""
    return Tipper()


@pytest.fixture(scope="session")
def tipper_data():
    """Sample tipper data for testing."""
    t = np.ones((1, 1, 2)) + 0.25j * np.ones((1, 1, 2))
    t_error = np.ones((1, 1, 2)) * 0.01
    t_model_error = np.ones((1, 1, 2)) * 0.03
    return {
        "tipper": t,
        "tipper_error": t_error,
        "tipper_model_error": t_model_error,
        "frequency": np.array([1]),
    }


@pytest.fixture(scope="session")
def tipper_with_data(tipper_data):
    """Tipper object initialized with data."""
    return Tipper(
        tipper=tipper_data["tipper"],
        tipper_error=tipper_data["tipper_error"],
        tipper_model_error=tipper_data["tipper_model_error"],
        frequency=tipper_data["frequency"],
    )


# =============================================================================
# Tests for Empty Tipper Initialization
# =============================================================================


class TestTipperInitialize:
    """Tests for empty Tipper initialization."""

    def test_n_periods(self, empty_tipper):
        assert empty_tipper.n_periods == 0

    def test_is_empty(self, empty_tipper):
        assert empty_tipper._is_empty()

    def test_has_tf(self, empty_tipper):
        assert not empty_tipper._has_tf()

    def test_has_tf_error(self, empty_tipper):
        assert not empty_tipper._has_tf_error()

    def test_has_tf_model_error(self, empty_tipper):
        assert not empty_tipper._has_tf_model_error()

    @pytest.mark.parametrize(
        "attr",
        [
            "mag_real",
            "mag_imag",
            "angle_real",
            "angle_imag",
            "mag_error",
            "mag_model_error",
            "angle_error",
            "angle_model_error",
            "amplitude",
            "phase",
            "amplitude_error",
            "phase_error",
            "amplitude_model_error",
            "phase_model_error",
        ],
    )
    def test_empty_properties(self, empty_tipper, attr):
        """Test that all tipper properties are None when empty."""
        assert getattr(empty_tipper, attr) is None


# =============================================================================
# Tests for Tipper with Data
# =============================================================================


class TestSetTipper:
    """Tests for Tipper with data."""

    def test_is_empty(self, tipper_with_data):
        assert not tipper_with_data._is_empty()

    def test_has_tf(self, tipper_with_data):
        assert tipper_with_data._has_tf()

    def test_has_tf_error(self, tipper_with_data):
        assert tipper_with_data._has_tf_error()

    def test_has_tf_model_error(self, tipper_with_data):
        assert tipper_with_data._has_tf_model_error()

    def test_tipper(self, tipper_with_data, tipper_data):
        assert np.isclose(tipper_with_data.tipper, tipper_data["tipper"]).all()

    def test_tipper_error(self, tipper_with_data, tipper_data):
        assert np.isclose(
            tipper_with_data.tipper_error, tipper_data["tipper_error"]
        ).all()

    def test_tipper_model_error(self, tipper_with_data, tipper_data):
        assert np.isclose(
            tipper_with_data.tipper_model_error, tipper_data["tipper_model_error"]
        ).all()

    def test_mag_real(self, tipper_with_data):
        assert np.isclose(tipper_with_data.mag_real, np.array([1.41421356])).all()

    def test_mag_imag(self, tipper_with_data):
        assert np.isclose(tipper_with_data.mag_imag, np.array([0.35355339])).all()

    def test_angle_real(self, tipper_with_data):
        assert np.isclose(tipper_with_data.angle_real, np.array([45.0])).all()

    def test_angle_imag(self, tipper_with_data):
        assert np.isclose(tipper_with_data.angle_imag, np.array([45.0])).all()

    def test_mag_error(self, tipper_with_data):
        assert np.isclose(tipper_with_data.mag_error, np.array([0.01414214])).all()

    def test_angle_error(self, tipper_with_data):
        assert np.isclose(tipper_with_data.angle_error, np.array([0])).all()

    def test_mag_model_error(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.mag_model_error, np.array([0.04242641])
        ).all()

    def test_angle_model_error(self, tipper_with_data):
        assert np.isclose(tipper_with_data.angle_model_error, np.array([0])).all()

    def test_amplitude(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.amplitude, np.array([[[1.03077641, 1.03077641]]])
        ).all()

    def test_amplitude_error(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.amplitude_error,
            np.array([[[0.01212648, 0.01212648]]]),
        ).all()

    def test_amplitude_model_error(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.amplitude_model_error,
            np.array([[[0.03637218, 0.03637218]]]),
        ).all()

    def test_phase(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.phase, np.array([[[14.03624347, 14.03624347]]])
        ).all()

    def test_phase_error(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.phase_error,
            np.array([[[0.67407048, 0.67407048]]]),
        ).all()

    def test_phase_model_error(self, tipper_with_data):
        assert np.isclose(
            tipper_with_data.phase_model_error,
            np.array([[[2.02226993, 2.02226993]]]),
        ).all()

    def test_rotate(self, tipper_with_data):
        """Test tipper rotation."""
        b = tipper_with_data.rotate(45)

        assert np.isclose(b.angle_real, np.array([0])).all()
        assert np.isclose(b.angle_imag, np.array([0])).all()
        assert np.isclose(b.mag_error, np.array([0.02])).all()
        assert np.isclose(b.mag_model_error, np.array([0.06])).all()


# =============================================================================
# Additional Tests for Enhanced Coverage
# =============================================================================


class TestTipperMultiplePeriods:
    """Tests for Tipper with multiple periods."""

    @pytest.fixture(scope="class")
    def multi_period_tipper(self):
        """Tipper with multiple periods."""
        n_periods = 5
        t = np.ones((n_periods, 1, 2)) + 0.25j * np.ones((n_periods, 1, 2))
        t_error = np.ones((n_periods, 1, 2)) * 0.01
        t_model_error = np.ones((n_periods, 1, 2)) * 0.03
        frequency = np.logspace(-3, 3, n_periods)

        return Tipper(
            tipper=t,
            tipper_error=t_error,
            tipper_model_error=t_model_error,
            frequency=frequency,
        )

    def test_n_periods(self, multi_period_tipper):
        assert multi_period_tipper.n_periods == 5

    def test_tipper_shape(self, multi_period_tipper):
        assert multi_period_tipper.tipper.shape == (5, 1, 2)

    def test_mag_real_shape(self, multi_period_tipper):
        assert multi_period_tipper.mag_real.shape == (5,)

    def test_amplitude_shape(self, multi_period_tipper):
        assert multi_period_tipper.amplitude.shape == (5, 1, 2)

    def test_phase_shape(self, multi_period_tipper):
        assert multi_period_tipper.phase.shape == (5, 1, 2)


class TestTipperRotation:
    """Tests for tipper rotation functionality."""

    @pytest.fixture(scope="class")
    def rotation_tipper(self):
        """Tipper for rotation tests."""
        t = np.ones((1, 1, 2)) + 0.25j * np.ones((1, 1, 2))
        t_error = np.ones((1, 1, 2)) * 0.01
        return Tipper(
            tipper=t,
            tipper_error=t_error,
            frequency=np.array([1]),
        )

    @pytest.mark.parametrize("angle", [0, 30, 45, 90, 180, -45])
    def test_rotation_various_angles(self, rotation_tipper, angle):
        """Test rotation at various angles produces valid results."""
        rotated = rotation_tipper.rotate(angle)

        assert rotated is not None
        assert rotated._has_tf()
        assert rotated.tipper is not None

    def test_rotation_zero_preserves_data(self, rotation_tipper):
        """Test that 0 degree rotation preserves original data."""
        rotated = rotation_tipper.rotate(0)

        # Should be very close to original (within numerical precision)
        assert np.allclose(rotation_tipper.tipper, rotated.tipper, rtol=1e-10)

    def test_rotation_360_equivalent_to_zero(self, rotation_tipper):
        """Test that 360 degree rotation is equivalent to 0."""
        rotated_0 = rotation_tipper.rotate(0)
        rotated_360 = rotation_tipper.rotate(360)

        assert np.allclose(rotated_0.tipper, rotated_360.tipper, rtol=1e-10)


class TestTipperDifferentErrorModes:
    """Tests for Tipper with different error configurations."""

    def test_tipper_with_only_tf_error(self, tipper_data):
        """Test Tipper with only tipper_error, no model error."""
        tipper = Tipper(
            tipper=tipper_data["tipper"],
            tipper_error=tipper_data["tipper_error"],
            frequency=tipper_data["frequency"],
        )

        assert tipper._has_tf()
        assert tipper._has_tf_error()
        assert not tipper._has_tf_model_error()
        assert tipper.mag_error is not None
        assert tipper.mag_model_error is None

    def test_tipper_with_only_model_error(self, tipper_data):
        """Test Tipper with only model error, no tipper_error."""
        tipper = Tipper(
            tipper=tipper_data["tipper"],
            tipper_model_error=tipper_data["tipper_model_error"],
            frequency=tipper_data["frequency"],
        )

        assert tipper._has_tf()
        assert not tipper._has_tf_error()
        assert tipper._has_tf_model_error()
        assert tipper.mag_error is None
        assert tipper.mag_model_error is not None

    def test_tipper_without_errors(self, tipper_data):
        """Test Tipper without any errors."""
        tipper = Tipper(
            tipper=tipper_data["tipper"],
            frequency=tipper_data["frequency"],
        )

        assert tipper._has_tf()
        assert not tipper._has_tf_error()
        assert not tipper._has_tf_model_error()
        assert tipper.mag_real is not None
        assert tipper.mag_error is None
        assert tipper.mag_model_error is None


class TestTipperVaryingAmplitudes:
    """Tests for Tipper with varying amplitudes."""

    @pytest.mark.parametrize("amplitude", [0.1, 0.5, 1.0, 2.0, 10.0])
    def test_different_amplitudes(self, amplitude):
        """Test Tipper with different amplitude values."""
        t = np.ones((1, 1, 2)) * amplitude + 0.25j * np.ones((1, 1, 2)) * amplitude
        tipper = Tipper(tipper=t, frequency=np.array([1]))

        assert tipper._has_tf()
        assert tipper.mag_real is not None
        assert tipper.amplitude is not None
        # Check that magnitude scales appropriately
        assert np.all(tipper.mag_real > 0)


class TestTipperCopy:
    """Tests for Tipper copy functionality."""

    def test_copy_creates_independent_object(self, tipper_with_data):
        """Test that copy creates an independent object."""
        tipper_copy = tipper_with_data.copy()

        # Should be equal but not the same object
        assert tipper_copy == tipper_with_data
        assert tipper_copy is not tipper_with_data

        # Modifying copy should not affect original
        tipper_copy.frequency = np.array([10])
        assert not np.allclose(tipper_copy.frequency, tipper_with_data.frequency)


class TestTipperFrequencyPeriod:
    """Tests for frequency and period relationship."""

    def test_frequency_period_relationship(self):
        """Test that frequency and period are reciprocals."""
        frequency = np.logspace(-3, 3, 10)
        tipper = Tipper(frequency=frequency)

        assert np.allclose(tipper.frequency, 1.0 / tipper.period)
        assert np.allclose(tipper.period, 1.0 / tipper.frequency)

    def test_set_frequency(self):
        """Test setting frequency updates period."""
        tipper = Tipper(frequency=np.array([1, 2, 3]))
        new_freq = np.array([10, 20, 30])
        tipper.frequency = new_freq

        assert np.allclose(tipper.frequency, new_freq)
        assert np.allclose(tipper.period, 1.0 / new_freq)

    def test_set_period(self):
        """Test setting period updates frequency."""
        tipper = Tipper(frequency=np.array([1, 2, 3]))
        new_period = np.array([0.1, 0.2, 0.3])
        tipper.period = new_period

        assert np.allclose(tipper.period, new_period)
        assert np.allclose(tipper.frequency, 1.0 / new_period)


@pytest.mark.parametrize(
    "n_periods,expected_shape",
    [
        (1, (1, 1, 2)),
        (5, (5, 1, 2)),
        (10, (10, 1, 2)),
    ],
)
def test_tipper_various_sizes(n_periods, expected_shape):
    """Test Tipper with various period sizes."""
    t = np.ones((n_periods, 1, 2)) + 0.25j * np.ones((n_periods, 1, 2))
    frequency = np.logspace(-3, 3, n_periods)

    tipper = Tipper(tipper=t, frequency=frequency)

    assert tipper.tipper.shape == expected_shape
    assert tipper.n_periods == n_periods
    assert tipper._has_tf()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
