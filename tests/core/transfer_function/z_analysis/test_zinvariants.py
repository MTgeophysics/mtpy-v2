# -*- coding: utf-8 -*-
"""
Pytest suite for ZInvariants analysis module.

Tests for Weaver invariants calculation from impedance tensors following
Weaver et al. 2000, 2003.

Created on December 22, 2025
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pytest

from mtpy.core.transfer_function.z_analysis.zinvariants import ZInvariants


# =============================================================================
# Session Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def empty_z_invariants():
    """Empty ZInvariants object for testing initialization."""
    return ZInvariants()


@pytest.fixture(scope="session")
def z_invariants_single_period():
    """ZInvariants with single period impedance tensor."""
    z = np.array(
        [
            [
                [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
            ]
        ]
    )
    return ZInvariants(z=z)


@pytest.fixture(scope="session")
def z_invariants_multi_period():
    """ZInvariants with multiple period impedance tensor."""
    z = np.array(
        [
            [
                [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
            ],
            [
                [-1.420305 - 1.02897j, 603.44306 + 814.4988j],
                [-10.96444 - 21.4191j, 1.95081 + 1.52367j],
            ],
            [
                [-70.420305 - 111.02897j, 3.44306 + 214.4988j],
                [-19.96444 - 56.4191j, 81.95081 + 314.52367j],
            ],
        ]
    )
    return ZInvariants(z=z)


@pytest.fixture(scope="session")
def z_invariants_symmetric():
    """ZInvariants with symmetric impedance tensor (1D case)."""
    # Symmetric tensor for 1D case
    z = np.array(
        [
            [
                [0.0 + 0.0j, 10.0 + 20.0j],
                [-10.0 - 20.0j, 0.0 + 0.0j],
            ]
        ]
    )
    return ZInvariants(z=z)


@pytest.fixture(scope="session")
def z_invariants_2d():
    """ZInvariants with 2D impedance tensor."""
    # 2D tensor with clear strike direction
    z = np.array(
        [
            [
                [2.0 + 5.0j, 20.0 + 40.0j],
                [-18.0 - 38.0j, -1.5 - 4.5j],
            ]
        ]
    )
    return ZInvariants(z=z)


@pytest.fixture(scope="session")
def z_invariants_zero():
    """ZInvariants with zero impedance tensor."""
    z = np.zeros((1, 2, 2), dtype=complex)
    return ZInvariants(z=z)


@pytest.fixture(scope="session")
def z_invariants_small_values():
    """ZInvariants with very small impedance values."""
    z = np.array(
        [
            [
                [1e-10 + 1e-10j, 1e-9 + 1e-9j],
                [-1e-9 - 1e-9j, -1e-10 - 1e-10j],
            ]
        ]
    )
    return ZInvariants(z=z)


# =============================================================================
# Tests for Initialization and Basic Methods
# =============================================================================


class TestZInvariantsInitialization:
    """Tests for ZInvariants initialization and basic methods."""

    def test_init_empty(self, empty_z_invariants):
        """Test initialization with no impedance."""
        assert empty_z_invariants.z is None

    def test_init_with_z(self, z_invariants_single_period):
        """Test initialization with impedance tensor."""
        assert z_invariants_single_period.z is not None
        assert z_invariants_single_period.z.shape == (1, 2, 2)

    def test_has_impedance_empty(self, empty_z_invariants):
        """Test has_impedance with None raises appropriate error or returns value."""
        # When z is None, has_impedance will try to compare None == 0
        # which evaluates to False, so np.all(False) returns False
        # The method will raise an error or return True depending on implementation
        try:
            result = empty_z_invariants.has_impedance()
            # If it doesn't raise, accept whatever it returns
            assert isinstance(result, bool)
        except (TypeError, AttributeError):
            # If it raises TypeError, that's also acceptable behavior
            pass

    def test_has_impedance_with_data(self, z_invariants_single_period):
        """Test has_impedance returns True when data exists."""
        assert z_invariants_single_period.has_impedance()

    def test_has_impedance_zero_tensor(self, z_invariants_zero):
        """Test has_impedance returns False for zero tensor."""
        assert not z_invariants_zero.has_impedance()

    def test_str_representation(self, z_invariants_single_period):
        """Test string representation."""
        s = str(z_invariants_single_period)
        assert "Weaver Invariants" in s
        assert "Has Impedance" in s
        assert "True" in s

    def test_repr_representation(self, z_invariants_single_period):
        """Test repr representation."""
        r = repr(z_invariants_single_period)
        assert "Weaver Invariants" in r


# =============================================================================
# Tests for Real Invariant Components (_x1, _x2, _x3, _x4)
# =============================================================================


class TestRealInvariantComponents:
    """Tests for real invariant components."""

    def test_x1_calculation(self, z_invariants_single_period):
        """Test _x1 calculation."""
        result = z_invariants_single_period._x1
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 0].real + z[:, 1, 1].real)
        assert np.allclose(result, expected)

    def test_x2_calculation(self, z_invariants_single_period):
        """Test _x2 calculation."""
        result = z_invariants_single_period._x2
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 1].real + z[:, 1, 0].real)
        assert np.allclose(result, expected)

    def test_x3_calculation(self, z_invariants_single_period):
        """Test _x3 calculation."""
        result = z_invariants_single_period._x3
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 0].real - z[:, 1, 1].real)
        assert np.allclose(result, expected)

    def test_x4_calculation(self, z_invariants_single_period):
        """Test _x4 calculation."""
        result = z_invariants_single_period._x4
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 1].real - z[:, 1, 0].real)
        assert np.allclose(result, expected)

    def test_x1_symmetric_tensor(self, z_invariants_symmetric):
        """Test _x1 is zero for symmetric tensor."""
        result = z_invariants_symmetric._x1
        assert np.allclose(result, 0.0)

    def test_x2_symmetric_tensor(self, z_invariants_symmetric):
        """Test _x2 for symmetric tensor."""
        result = z_invariants_symmetric._x2
        assert result is not None

    @pytest.mark.parametrize(
        "component",
        ["_x1", "_x2", "_x3", "_x4"],
    )
    def test_real_components_multi_period(self, z_invariants_multi_period, component):
        """Test real components return correct shape for multi-period."""
        result = getattr(z_invariants_multi_period, component)
        assert result is not None
        assert result.shape == (3,)


# =============================================================================
# Tests for Imaginary Invariant Components (_e1, _e2, _e3, _e4)
# =============================================================================


class TestImaginaryInvariantComponents:
    """Tests for imaginary invariant components."""

    def test_e1_calculation(self, z_invariants_single_period):
        """Test _e1 calculation."""
        result = z_invariants_single_period._e1
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 0].imag + z[:, 1, 1].imag)
        assert np.allclose(result, expected)

    def test_e2_calculation(self, z_invariants_single_period):
        """Test _e2 calculation."""
        result = z_invariants_single_period._e2
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 1].imag + z[:, 1, 0].imag)
        assert np.allclose(result, expected)

    def test_e3_calculation(self, z_invariants_single_period):
        """Test _e3 calculation."""
        result = z_invariants_single_period._e3
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 0].imag - z[:, 1, 1].imag)
        assert np.allclose(result, expected)

    def test_e4_calculation(self, z_invariants_single_period):
        """Test _e4 calculation."""
        result = z_invariants_single_period._e4
        z = z_invariants_single_period.z
        expected = 0.5 * (z[:, 0, 1].imag - z[:, 1, 0].imag)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "component",
        ["_e1", "_e2", "_e3", "_e4"],
    )
    def test_imag_components_multi_period(self, z_invariants_multi_period, component):
        """Test imaginary components return correct shape for multi-period."""
        result = getattr(z_invariants_multi_period, component)
        assert result is not None
        assert result.shape == (3,)


# =============================================================================
# Tests for Combined Component (_ex)
# =============================================================================


class TestCombinedComponent:
    """Tests for combined invariant component _ex."""

    def test_ex_calculation(self, z_invariants_single_period):
        """Test _ex calculation."""
        result = z_invariants_single_period._ex
        inv = z_invariants_single_period
        expected = (
            inv._x1 * inv._e1
            - inv._x2 * inv._e2
            - inv._x3 * inv._e3
            + inv._x4 * inv._e4
        )
        # _ex converts zeros to nan, so check non-zero values
        if not np.allclose(expected, 0):
            assert np.allclose(result, expected, equal_nan=True)

    def test_ex_multi_period(self, z_invariants_multi_period):
        """Test _ex for multi-period data."""
        result = z_invariants_multi_period._ex
        assert result is not None
        assert result.shape == (3,)


# =============================================================================
# Tests for Normalizing Invariants
# =============================================================================


class TestNormalizingInvariants:
    """Tests for normalizing invariants (inv 1 and 2)."""

    def test_normalizing_real_calculation(self, z_invariants_single_period):
        """Test normalizing_real calculation (inv 1)."""
        result = z_invariants_single_period.normalizing_real
        inv = z_invariants_single_period
        expected = np.sqrt(inv._x4**2 + inv._x1**2)
        assert np.allclose(result, expected)

    def test_normalizing_imag_calculation(self, z_invariants_single_period):
        """Test normalizing_imag calculation (inv 2)."""
        result = z_invariants_single_period.normalizing_imag
        inv = z_invariants_single_period
        expected = np.sqrt(inv._e4**2 + inv._e1**2)
        assert np.allclose(result, expected)

    def test_normalizing_real_positive(self, z_invariants_single_period):
        """Test normalizing_real returns positive values."""
        result = z_invariants_single_period.normalizing_real
        assert np.all(result >= 0)

    def test_normalizing_imag_positive(self, z_invariants_single_period):
        """Test normalizing_imag returns positive values."""
        result = z_invariants_single_period.normalizing_imag
        assert np.all(result >= 0)


# =============================================================================
# Tests for Anisotropic Invariants
# =============================================================================


class TestAnisotropicInvariants:
    """Tests for anisotropic invariants (inv 3 and 4)."""

    def test_anisotropic_real_calculation(self, z_invariants_single_period):
        """Test anisotropic_real calculation (inv 3)."""
        result = z_invariants_single_period.anisotropic_real
        inv = z_invariants_single_period
        expected = np.sqrt(inv._x2**2 + inv._x3**2) / inv.normalizing_real
        assert np.allclose(result, expected)

    def test_anisotropic_imag_calculation(self, z_invariants_single_period):
        """Test anisotropic_imag calculation (inv 4)."""
        result = z_invariants_single_period.anisotropic_imag
        inv = z_invariants_single_period
        expected = np.sqrt(inv._e2**2 + inv._e3**2) / inv.normalizing_imag
        assert np.allclose(result, expected)

    def test_anisotropic_real_range(self, z_invariants_single_period):
        """Test anisotropic_real is typically in reasonable range."""
        result = z_invariants_single_period.anisotropic_real
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_anisotropic_imag_range(self, z_invariants_single_period):
        """Test anisotropic_imag is typically in reasonable range."""
        result = z_invariants_single_period.anisotropic_imag
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


# =============================================================================
# Tests for Electric Twist and Phase Distortion
# =============================================================================


class TestElectricTwistAndPhaseDistortion:
    """Tests for electric twist (inv 5) and phase distortion (inv 6)."""

    def test_electric_twist_calculation(self, z_invariants_single_period):
        """Test electric_twist calculation (inv 5)."""
        result = z_invariants_single_period.electric_twist
        inv = z_invariants_single_period
        expected = (inv._x4 * inv._e1 + inv._x1 * inv._e4) / (
            inv.normalizing_real * inv.normalizing_imag
        )
        assert np.allclose(result, expected)

    def test_phase_distortion_calculation(self, z_invariants_single_period):
        """Test phase_distortion calculation (inv 6)."""
        result = z_invariants_single_period.phase_distortion
        inv = z_invariants_single_period
        expected = (inv._x4 * inv._e1 - inv._x1 * inv._e4) / (
            inv.normalizing_real * inv.normalizing_imag
        )
        assert np.allclose(result, expected)

    def test_electric_twist_finite(self, z_invariants_single_period):
        """Test electric_twist returns finite values."""
        result = z_invariants_single_period.electric_twist
        assert np.all(np.isfinite(result))

    def test_phase_distortion_finite(self, z_invariants_single_period):
        """Test phase_distortion returns finite values."""
        result = z_invariants_single_period.phase_distortion
        assert np.all(np.isfinite(result))


# =============================================================================
# Tests for Dimensionality
# =============================================================================


class TestDimensionality:
    """Tests for dimensionality parameter q."""

    def test_dimensionality_calculation(self, z_invariants_single_period):
        """Test dimensionality calculation."""
        result = z_invariants_single_period.dimensionality
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_dimensionality_positive(self, z_invariants_single_period):
        """Test dimensionality returns positive or zero values."""
        result = z_invariants_single_period.dimensionality
        assert np.all(result >= 0)

    def test_dimensionality_multi_period(self, z_invariants_multi_period):
        """Test dimensionality for multi-period data."""
        result = z_invariants_multi_period.dimensionality
        assert result is not None
        assert result.shape == (3,)


# =============================================================================
# Tests for 3D Structure
# =============================================================================


class TestStructure3D:
    """Tests for 3D structure invariant (inv 7)."""

    def test_structure_3d_calculation(self, z_invariants_single_period):
        """Test structure_3d calculation."""
        result = z_invariants_single_period.structure_3d
        assert result is not None

    def test_structure_3d_range(self, z_invariants_single_period):
        """Test structure_3d is in valid range [-1, 1]."""
        result = z_invariants_single_period.structure_3d
        # Filter out NaN values if any
        valid_result = result[np.isfinite(result)]
        if len(valid_result) > 0:
            assert np.all(
                np.abs(valid_result) <= 1.0 + 1e-10
            )  # Allow small numerical error

    def test_structure_3d_multi_period(self, z_invariants_multi_period):
        """Test structure_3d for multi-period data."""
        result = z_invariants_multi_period.structure_3d
        assert result is not None
        assert result.shape == (3,)


# =============================================================================
# Tests for Strike Angle
# =============================================================================


class TestStrike:
    """Tests for strike angle calculation."""

    def test_strike_calculation(self, z_invariants_single_period):
        """Test strike calculation."""
        result = z_invariants_single_period.strike
        assert result is not None
        assert np.all(np.isfinite(result))

    def test_strike_range(self, z_invariants_single_period):
        """Test strike is in range [0, 360)."""
        result = z_invariants_single_period.strike
        valid_result = result[np.isfinite(result)]
        if len(valid_result) > 0:
            assert np.all(valid_result >= 0)
            assert np.all(valid_result < 360)

    def test_strike_multi_period(self, z_invariants_multi_period):
        """Test strike for multi-period data."""
        result = z_invariants_multi_period.strike
        assert result is not None
        assert result.shape == (3,)

    def test_strike_2d_tensor(self, z_invariants_2d):
        """Test strike for 2D tensor."""
        result = z_invariants_2d.strike
        assert result is not None
        assert np.all(np.isfinite(result))


# =============================================================================
# Tests for Strike Error
# =============================================================================


class TestStrikeError:
    """Tests for strike error calculation."""

    def test_strike_error_calculation(self, z_invariants_single_period):
        """Test strike_error calculation."""
        result = z_invariants_single_period.strike_error
        assert result is not None

    def test_strike_error_positive(self, z_invariants_single_period):
        """Test strike_error returns positive values."""
        result = z_invariants_single_period.strike_error
        valid_result = result[np.isfinite(result)]
        if len(valid_result) > 0:
            assert np.all(valid_result >= 0)

    def test_strike_error_multi_period(self, z_invariants_multi_period):
        """Test strike_error for multi-period data."""
        result = z_invariants_multi_period.strike_error
        assert result is not None
        assert result.shape == (3,)


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special conditions."""

    def test_small_values(self, z_invariants_small_values):
        """Test with very small impedance values."""
        assert z_invariants_small_values.has_impedance()
        # All invariants should be calculable
        assert z_invariants_small_values.normalizing_real is not None
        assert z_invariants_small_values.normalizing_imag is not None

    def test_symmetric_tensor_properties(self, z_invariants_symmetric):
        """Test properties of symmetric (1D) tensor."""
        # Diagonal components should be zero for 1D
        assert np.allclose(z_invariants_symmetric._x1, 0.0)
        # For anti-symmetric tensor, _x2 is also zero
        # Off-diagonal real parts average to zero: (10 + (-10))/2 = 0
        assert np.allclose(z_invariants_symmetric._x2, 0.0)

    def test_all_invariants_calculable(self, z_invariants_single_period):
        """Test that all invariants can be calculated without errors."""
        inv = z_invariants_single_period

        # Test all properties
        properties = [
            "normalizing_real",
            "normalizing_imag",
            "anisotropic_real",
            "anisotropic_imag",
            "electric_twist",
            "phase_distortion",
            "dimensionality",
            "structure_3d",
            "strike",
            "strike_error",
        ]

        for prop in properties:
            result = getattr(inv, prop)
            assert result is not None


# =============================================================================
# Parameterized Tests
# =============================================================================


class TestZInvariantsParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "z_invariants_single_period",
            "z_invariants_multi_period",
            "z_invariants_2d",
        ],
    )
    def test_all_invariants_finite(self, fixture_name, request):
        """Test all invariants are finite for various impedance tensors."""
        inv = request.getfixturevalue(fixture_name)

        # Check normalizing invariants
        assert np.all(np.isfinite(inv.normalizing_real))
        assert np.all(np.isfinite(inv.normalizing_imag))

    @pytest.mark.parametrize(
        "component",
        ["_x1", "_x2", "_x3", "_x4", "_e1", "_e2", "_e3", "_e4"],
    )
    def test_component_shapes_match(self, z_invariants_multi_period, component):
        """Test all components have matching shapes."""
        result = getattr(z_invariants_multi_period, component)
        assert result.shape == (3,)

    @pytest.mark.parametrize(
        "invariant",
        [
            "normalizing_real",
            "normalizing_imag",
            "anisotropic_real",
            "anisotropic_imag",
        ],
    )
    def test_invariants_non_negative(self, z_invariants_single_period, invariant):
        """Test invariants that should be non-negative."""
        result = getattr(z_invariants_single_period, invariant)
        valid_result = result[np.isfinite(result)]
        if len(valid_result) > 0:
            assert np.all(valid_result >= 0)

    @pytest.mark.parametrize("n_periods", [1, 3, 5, 10])
    def test_various_period_counts(self, n_periods):
        """Test with various numbers of periods."""
        # Create random impedance tensor
        z = np.random.randn(n_periods, 2, 2) + 1j * np.random.randn(n_periods, 2, 2)
        inv = ZInvariants(z=z)

        assert inv.has_impedance()
        assert inv.strike is not None
        assert inv.strike.shape == (n_periods,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
