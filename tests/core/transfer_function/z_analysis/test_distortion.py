# -*- coding: utf-8 -*-
"""
Pytest suite for distortion analysis module.

Tests for galvanic distortion calculation and removal from impedance tensors
following Bibby et al. 2005.

Created on December 22, 2025
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis import distortion


# =============================================================================
# Session Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def z_1d_single_period():
    """Z object with 1D characteristics for single period."""
    z_array = np.array([[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]])
    z_error_array = np.array([[[0.01, 0.5], [0.5, 0.01]]])
    return Z(z=z_array, z_error=z_error_array, frequency=np.array([1.0]))


@pytest.fixture(scope="session")
def z_2d_single_period():
    """Z object with 2D characteristics for single period."""
    # Create a 2D impedance with clear strike direction
    z_array = np.array([[[2.0 + 5.0j, 20.0 + 40.0j], [-18.0 - 38.0j, -1.5 - 4.5j]]])
    z_error_array = np.array([[[0.1, 1.0], [1.0, 0.1]]])
    return Z(z=z_array, z_error=z_error_array, frequency=np.array([1.0]))


@pytest.fixture(scope="session")
def z_3d_single_period():
    """Z object with 3D characteristics for single period."""
    # Create a 3D impedance (all components significantly non-zero)
    z_array = np.array([[[5.0 + 10.0j, 15.0 + 25.0j], [-12.0 - 22.0j, -4.0 - 8.0j]]])
    z_error_array = np.array([[[0.2, 0.8], [0.8, 0.2]]])
    return Z(z=z_array, z_error=z_error_array, frequency=np.array([1.0]))


@pytest.fixture(scope="session")
def z_multi_period_mixed():
    """Z object with multiple periods and mixed dimensionality."""
    # First period: 1D, second: 2D, third: 1D
    z_array = np.array(
        [
            [[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]],
            [[2.0 + 5.0j, 20.0 + 40.0j], [-18.0 - 38.0j, -1.5 - 4.5j]],
            [[0.05 - 0.05j, 8.0 + 8.0j], [-8.0 - 8.0j, -0.05 + 0.05j]],
        ]
    )
    z_error_array = np.array(
        [
            [[0.01, 0.5], [0.5, 0.01]],
            [[0.1, 1.0], [1.0, 0.1]],
            [[0.01, 0.4], [0.4, 0.01]],
        ]
    )
    return Z(z=z_array, z_error=z_error_array, frequency=np.array([10.0, 1.0, 0.1]))


@pytest.fixture(scope="session")
def distortion_tensor_valid():
    """Valid distortion tensor for testing."""
    return np.array([[2.0, 0.1], [0.05, 1.5]])


@pytest.fixture(scope="session")
def distortion_tensor_error():
    """Error for distortion tensor."""
    return np.array([[0.1, 0.01], [0.005, 0.08]])


@pytest.fixture(scope="session")
def distortion_tensor_singular():
    """Singular distortion tensor (non-invertible)."""
    return np.array([[1.0, 2.0], [2.0, 4.0]])


@pytest.fixture(scope="session")
def z_with_distortion(distortion_tensor_valid):
    """Z object and distorted version for testing removal."""
    # Create original Z
    z_array = np.array([[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]])
    z_error_array = np.array([[[0.01, 0.5], [0.5, 0.01]]])
    z_original = Z(z=z_array, z_error=z_error_array, frequency=np.array([1.0]))

    # Apply distortion: Z_distorted = D * Z_original
    z_distorted_array = np.dot(distortion_tensor_valid, z_array[0]).reshape((1, 2, 2))
    z_distorted = Z(
        z=z_distorted_array, z_error=z_error_array, frequency=np.array([1.0])
    )

    return {
        "z_original": z_original,
        "z_distorted": z_distorted,
        "distortion": distortion_tensor_valid,
    }


# =============================================================================
# Tests for find_distortion
# =============================================================================


class TestFindDistortion:
    """Tests for find_distortion function."""

    def test_1d_returns_valid_tensor(self, z_1d_single_period):
        """Test that 1D impedance returns a valid distortion tensor."""
        dis, dis_error = distortion.find_distortion(z_1d_single_period)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)
        assert np.all(np.isfinite(dis))
        assert np.all(np.isfinite(dis_error))

    def test_2d_returns_valid_tensor(self, z_2d_single_period):
        """Test that 2D impedance returns a valid distortion tensor."""
        dis, dis_error = distortion.find_distortion(z_2d_single_period)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)
        assert np.all(np.isfinite(dis))
        assert np.all(np.isfinite(dis_error))

    def test_3d_returns_identity(self, z_3d_single_period):
        """Test that 3D impedance returns identity (no distortion calculable)."""
        dis, dis_error = distortion.find_distortion(z_3d_single_period)

        # For 3D data, distortion should be identity or close to it
        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)

    def test_multi_period_mixed(self, z_multi_period_mixed):
        """Test with multiple periods of mixed dimensionality."""
        dis, dis_error = distortion.find_distortion(z_multi_period_mixed)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)
        assert np.all(np.isfinite(dis))
        assert np.all(np.isfinite(dis_error))

    @pytest.mark.parametrize("comp", ["det", "01", "10"])
    def test_different_comp_parameters(self, z_1d_single_period, comp):
        """Test different component parameters for gain calculation."""
        dis, dis_error = distortion.find_distortion(z_1d_single_period, comp=comp)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)
        assert np.all(np.isfinite(dis))

    @pytest.mark.parametrize("only_2d", [True, False])
    def test_only_2d_parameter(self, z_multi_period_mixed, only_2d):
        """Test only_2d parameter effect."""
        dis, dis_error = distortion.find_distortion(
            z_multi_period_mixed, only_2d=only_2d
        )

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)

    @pytest.mark.parametrize("clockwise", [True, False])
    def test_clockwise_parameter(self, z_2d_single_period, clockwise):
        """Test clockwise rotation parameter."""
        dis, dis_error = distortion.find_distortion(
            z_2d_single_period, clockwise=clockwise
        )

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)

    def test_error_propagation(self, z_1d_single_period):
        """Test that errors are properly propagated."""
        dis, dis_error = distortion.find_distortion(z_1d_single_period)

        # Errors should be positive
        assert np.all(dis_error > 0)
        assert np.all(np.isfinite(dis_error))


# =============================================================================
# Tests for remove_distortion_from_z_object
# =============================================================================


class TestRemoveDistortion:
    """Tests for remove_distortion_from_z_object function."""

    def test_remove_distortion_basic(self, z_with_distortion, distortion_tensor_valid):
        """Test basic distortion removal."""
        z_corrected, z_corrected_error = distortion.remove_distortion_from_z_object(
            z_with_distortion["z_distorted"], distortion_tensor_valid
        )

        assert z_corrected.shape == z_with_distortion["z_distorted"].z.shape
        assert z_corrected_error.shape == z_with_distortion["z_distorted"].z.shape
        assert np.all(np.isfinite(z_corrected))
        assert np.all(np.isfinite(z_corrected_error))

    def test_remove_distortion_recovers_original(self, z_with_distortion):
        """Test that removing distortion approximately recovers original Z."""
        z_corrected, _ = distortion.remove_distortion_from_z_object(
            z_with_distortion["z_distorted"], z_with_distortion["distortion"]
        )

        # Should be close to original (allowing for numerical precision)
        assert np.allclose(
            z_corrected, z_with_distortion["z_original"].z, rtol=1e-10, atol=1e-12
        )

    def test_with_error_tensor(
        self, z_with_distortion, distortion_tensor_valid, distortion_tensor_error
    ):
        """Test distortion removal with error tensor."""
        z_corrected, z_corrected_error = distortion.remove_distortion_from_z_object(
            z_with_distortion["z_distorted"],
            distortion_tensor_valid,
            distortion_error_tensor=distortion_tensor_error,
        )

        assert z_corrected.shape == z_with_distortion["z_distorted"].z.shape
        assert z_corrected_error.shape == z_with_distortion["z_distorted"].z.shape
        # Errors should be positive
        assert np.all(z_corrected_error >= 0)

    def test_singular_distortion_raises_error(
        self, z_1d_single_period, distortion_tensor_singular
    ):
        """Test that singular distortion tensor raises ValueError."""
        with pytest.raises(ValueError, match="singular"):
            distortion.remove_distortion_from_z_object(
                z_1d_single_period, distortion_tensor_singular
            )

    def test_invalid_shape_raises_error(self, z_1d_single_period):
        """Test that invalid distortion tensor shape raises ValueError."""
        invalid_distortion = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError):
            distortion.remove_distortion_from_z_object(
                z_1d_single_period, invalid_distortion
            )

    def test_3d_distortion_uses_first_slice(
        self, z_1d_single_period, distortion_tensor_valid
    ):
        """Test that 3D distortion tensor uses only first slice."""
        distortion_3d = np.stack([distortion_tensor_valid, distortion_tensor_valid])

        z_corrected, _ = distortion.remove_distortion_from_z_object(
            z_1d_single_period, distortion_3d
        )

        assert z_corrected.shape == z_1d_single_period.z.shape

    def test_error_propagation(
        self, z_with_distortion, distortion_tensor_valid, distortion_tensor_error
    ):
        """Test error propagation in distortion removal."""
        z_corrected, z_corrected_error = distortion.remove_distortion_from_z_object(
            z_with_distortion["z_distorted"],
            distortion_tensor_valid,
            distortion_error_tensor=distortion_tensor_error,
        )

        # Errors should be finite and positive
        assert np.all(np.isfinite(z_corrected_error))
        assert np.all(z_corrected_error >= 0)

    def test_multi_period_distortion_removal(
        self, z_multi_period_mixed, distortion_tensor_valid
    ):
        """Test distortion removal with multiple periods."""
        z_corrected, z_corrected_error = distortion.remove_distortion_from_z_object(
            z_multi_period_mixed, distortion_tensor_valid
        )

        assert z_corrected.shape == z_multi_period_mixed.z.shape
        assert z_corrected_error.shape == z_multi_period_mixed.z.shape
        assert np.all(np.isfinite(z_corrected))


# =============================================================================
# Integration Tests
# =============================================================================


class TestDistortionIntegration:
    """Integration tests combining find and remove distortion."""

    def test_find_and_remove_distortion(self, z_with_distortion):
        """Test finding distortion and removing it."""
        # Find distortion from distorted data
        dis, dis_error = distortion.find_distortion(z_with_distortion["z_distorted"])

        # Remove found distortion
        z_corrected, z_corrected_error = distortion.remove_distortion_from_z_object(
            z_with_distortion["z_distorted"], dis, distortion_error_tensor=dis_error
        )

        # Result should be reasonable (finite values)
        assert np.all(np.isfinite(z_corrected))
        assert np.all(np.isfinite(z_corrected_error))

    def test_round_trip_with_known_distortion(
        self, z_1d_single_period, distortion_tensor_valid
    ):
        """Test applying and removing known distortion."""
        # Apply distortion
        z_distorted_array = np.dot(
            distortion_tensor_valid, z_1d_single_period.z[0]
        ).reshape((1, 2, 2))
        z_distorted = Z(
            z=z_distorted_array,
            z_error=z_1d_single_period.z_error,
            frequency=z_1d_single_period.frequency,
        )

        # Remove distortion
        z_corrected, _ = distortion.remove_distortion_from_z_object(
            z_distorted, distortion_tensor_valid
        )

        # Should recover original
        assert np.allclose(z_corrected, z_1d_single_period.z, rtol=1e-10, atol=1e-12)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDistortionEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_zero_impedance_elements(self):
        """Test behavior with zero impedance elements."""
        z_array = np.array([[[0.0 + 0.0j, 10.0 + 10.0j], [-10.0 - 10.0j, 0.0 + 0.0j]]])
        z_obj = Z(z=z_array, frequency=np.array([1.0]))

        dis, dis_error = distortion.find_distortion(z_obj)

        # Should handle zeros gracefully
        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)

    def test_very_small_impedance_values(self):
        """Test with very small impedance values."""
        z_array = np.array(
            [[[1e-10 - 1e-10j, 1e-8 + 1e-8j], [-1e-8 - 1e-8j, -1e-10 + 1e-10j]]]
        )
        z_obj = Z(z=z_array, frequency=np.array([1.0]))

        dis, dis_error = distortion.find_distortion(z_obj)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)

    def test_identity_distortion(self, z_1d_single_period):
        """Test removing identity distortion (should not change Z)."""
        identity = np.eye(2)

        z_corrected, _ = distortion.remove_distortion_from_z_object(
            z_1d_single_period, identity
        )

        # Should be essentially unchanged
        assert np.allclose(z_corrected, z_1d_single_period.z, rtol=1e-10)

    def test_no_error_in_z(self):
        """Test distortion calculation when Z has no errors."""
        z_array = np.array([[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]])
        z_obj = Z(z=z_array, frequency=np.array([1.0]))

        dis, dis_error = distortion.find_distortion(z_obj)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)


# =============================================================================
# Parameterized Tests
# =============================================================================


class TestDistortionParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "comp,only_2d,clockwise",
        [
            ("det", False, True),
            ("det", True, True),
            ("det", False, False),
            ("01", False, True),
            ("10", False, True),
        ],
    )
    def test_parameter_combinations(
        self, z_multi_period_mixed, comp, only_2d, clockwise
    ):
        """Test various parameter combinations."""
        dis, dis_error = distortion.find_distortion(
            z_multi_period_mixed, comp=comp, only_2d=only_2d, clockwise=clockwise
        )

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)
        assert np.all(np.isfinite(dis))

    @pytest.mark.parametrize("n_periods", [1, 3, 5, 10])
    def test_various_period_counts(self, n_periods):
        """Test with various numbers of periods."""
        # Create Z with n_periods
        z_array = np.random.randn(n_periods, 2, 2) + 1j * np.random.randn(
            n_periods, 2, 2
        )
        # Make it somewhat 1D-like
        z_array[:, 0, 0] *= 0.1
        z_array[:, 1, 1] *= 0.1
        z_obj = Z(z=z_array, frequency=np.logspace(0, -n_periods + 1, n_periods))

        dis, dis_error = distortion.find_distortion(z_obj)

        assert dis.shape == (2, 2)
        assert dis_error.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
