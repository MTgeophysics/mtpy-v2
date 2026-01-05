"""
Pytest suite for niblettbostick.py module.

Tests Niblett-Bostick transformation functions for impedance tensors.
Optimized for pytest-xdist parallel execution using session-scoped fixtures.

Created: 2025-12-22
"""

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis import niblettbostick as nb
from mtpy.utils import MU0


# =============================================================================
# Session-scoped fixtures for reuse across test classes
# =============================================================================


@pytest.fixture(scope="session")
def periods_single():
    """Single period value for basic tests."""
    return np.array([1.0])


@pytest.fixture(scope="session")
def periods_multi():
    """Multiple period values spanning several decades."""
    return np.logspace(-2, 3, 20)


@pytest.fixture(scope="session")
def resistivity_uniform():
    """Uniform resistivity values (100 Ohm-m)."""
    return np.array([100.0])


@pytest.fixture(scope="session")
def resistivity_multi():
    """Multiple resistivity values for multi-period tests."""
    return np.ones(20) * 100.0


@pytest.fixture(scope="session")
def phase_45deg():
    """Phase values at 45 degrees."""
    return np.array([45.0])


@pytest.fixture(scope="session")
def phase_multi():
    """Multiple phase values ranging from 0 to 90 degrees."""
    return np.linspace(10, 80, 20)


@pytest.fixture(scope="session")
def depth_values():
    """Depth values for sensitivity calculations."""
    return np.linspace(0, 10000, 50)


@pytest.fixture(scope="session")
def z_object_1d():
    """Create a simple 1D impedance tensor object."""
    freq = np.logspace(-3, 3, 10)
    z = np.zeros((freq.size, 2, 2), dtype=complex)

    # Create 1D-like impedance: only off-diagonal elements
    resistivity = 100.0
    phase_deg = 45.0
    for ii, ff in enumerate(freq):
        z_abs = np.sqrt(2 * np.pi * ff * MU0 * resistivity)
        z[ii, 0, 1] = z_abs * np.exp(1j * np.deg2rad(phase_deg))
        z[ii, 1, 0] = -z_abs * np.exp(1j * np.deg2rad(phase_deg))

    return Z(z=z, frequency=freq)


@pytest.fixture(scope="session")
def z_object_2d():
    """Create a 2D impedance tensor object with distinct XY and YX components."""
    freq = np.logspace(-3, 3, 15)
    z = np.zeros((freq.size, 2, 2), dtype=complex)

    # Create 2D impedance with different XY and YX
    rho_xy = 100.0
    rho_yx = 200.0
    phase_deg = 45.0

    for ii, ff in enumerate(freq):
        z_abs_xy = np.sqrt(2 * np.pi * ff * MU0 * rho_xy)
        z_abs_yx = np.sqrt(2 * np.pi * ff * MU0 * rho_yx)
        z[ii, 0, 1] = z_abs_xy * np.exp(1j * np.deg2rad(phase_deg))
        z[ii, 1, 0] = -z_abs_yx * np.exp(1j * np.deg2rad(phase_deg))

    return Z(z=z, frequency=freq)


@pytest.fixture(scope="session")
def z_object_varying_phase():
    """Create impedance tensor with varying phase."""
    freq = np.logspace(-3, 3, 12)
    z = np.zeros((freq.size, 2, 2), dtype=complex)

    resistivity = 100.0
    # Phase varies from 15 to 75 degrees
    phases = np.linspace(15, 75, freq.size)

    for ii, ff in enumerate(freq):
        z_abs = np.sqrt(2 * np.pi * ff * MU0 * resistivity)
        z[ii, 0, 1] = z_abs * np.exp(1j * np.deg2rad(phases[ii]))
        z[ii, 1, 0] = -z_abs * np.exp(1j * np.deg2rad(phases[ii]))

    return Z(z=z, frequency=freq)


# =============================================================================
# Test Classes
# =============================================================================


class TestNiblettBostickDepth:
    """Tests for calculate_niblett_bostick_depth function."""

    def test_depth_single_period(self, resistivity_uniform, periods_single):
        """Test depth calculation for single period."""
        depth = nb.calculate_niblett_bostick_depth(resistivity_uniform, periods_single)

        assert isinstance(depth, np.ndarray)
        assert depth.shape == periods_single.shape
        assert np.all(depth > 0)

    def test_depth_multi_period(self, resistivity_multi, periods_multi):
        """Test depth calculation for multiple periods."""
        depth = nb.calculate_niblett_bostick_depth(resistivity_multi, periods_multi)

        assert depth.shape == periods_multi.shape
        assert np.all(depth > 0)
        # Depth should increase with period
        assert np.all(np.diff(depth) > 0)

    def test_depth_formula_validation(self):
        """Validate depth calculation formula."""
        resistivity = np.array([100.0])
        period = np.array([1.0])

        expected_depth = np.sqrt(resistivity * period / (2.0 * np.pi * MU0))
        calculated_depth = nb.calculate_niblett_bostick_depth(resistivity, period)

        assert np.allclose(calculated_depth, expected_depth)

    def test_depth_scales_with_resistivity(self):
        """Test that depth scales with square root of resistivity."""
        period = np.array([1.0])
        rho1 = np.array([100.0])
        rho2 = np.array([400.0])  # 4x resistivity

        depth1 = nb.calculate_niblett_bostick_depth(rho1, period)
        depth2 = nb.calculate_niblett_bostick_depth(rho2, period)

        # Depth should be 2x (sqrt(4) = 2)
        assert np.allclose(depth2 / depth1, 2.0)

    def test_depth_scales_with_period(self):
        """Test that depth scales with square root of period."""
        resistivity = np.array([100.0])
        t1 = np.array([1.0])
        t2 = np.array([4.0])  # 4x period

        depth1 = nb.calculate_niblett_bostick_depth(resistivity, t1)
        depth2 = nb.calculate_niblett_bostick_depth(resistivity, t2)

        # Depth should be 2x (sqrt(4) = 2)
        assert np.allclose(depth2 / depth1, 2.0)


class TestNiblettBostickResistivityWeidelt:
    """Tests for calculate_niblett_bostick_resistivity_weidelt function."""

    def test_weidelt_single_value(self, resistivity_uniform, phase_45deg):
        """Test Weidelt transformation for single value."""
        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(
            resistivity_uniform, phase_45deg
        )

        assert isinstance(rho_nb, np.ndarray)
        assert rho_nb.shape == resistivity_uniform.shape

    def test_weidelt_multi_value(self, resistivity_multi, phase_multi):
        """Test Weidelt transformation for multiple values."""
        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(
            resistivity_multi, phase_multi
        )

        assert rho_nb.shape == resistivity_multi.shape

    def test_weidelt_phase_45_degrees(self):
        """Test Weidelt at 45 degrees (should give rho * (pi/4 - 1))."""
        resistivity = np.array([100.0])
        phase = np.array([45.0])

        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(resistivity, phase)
        expected = resistivity * ((np.pi / 2) * np.deg2rad(45.0) - 1)

        assert np.allclose(rho_nb, expected)

    def test_weidelt_phase_0_degrees(self):
        """Test Weidelt at 0 degrees (should give negative rho)."""
        resistivity = np.array([100.0])
        phase = np.array([0.0])

        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(resistivity, phase)

        assert rho_nb[0] < 0  # Should be negative

    def test_weidelt_phase_90_degrees(self):
        """Test Weidelt at 90 degrees (should give rho * (pi/4 - 1))."""
        resistivity = np.array([100.0])
        phase = np.array([90.0])

        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(resistivity, phase)
        # 90 % 90 = 0, so same as 0 degrees
        expected = resistivity * ((np.pi / 2) * np.deg2rad(0.0) - 1)

        assert np.allclose(rho_nb, expected)

    def test_weidelt_phase_modulo(self):
        """Test that phase modulo 90 is applied correctly."""
        resistivity = np.array([100.0])
        phase1 = np.array([45.0])
        phase2 = np.array([135.0])  # 135 % 90 = 45

        rho_nb1 = nb.calculate_niblett_bostick_resistivity_weidelt(resistivity, phase1)
        rho_nb2 = nb.calculate_niblett_bostick_resistivity_weidelt(resistivity, phase2)

        assert np.allclose(rho_nb1, rho_nb2)


class TestNiblettBostickResistivityDerivatives:
    """Tests for calculate_niblett_bostick_resistivity_derivatives function."""

    def test_derivatives_constant_resistivity(self):
        """Test derivatives for constant resistivity (m=0)."""
        resistivity = np.ones(10) * 100.0
        period = np.logspace(0, 2, 10)

        rho_nb = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        # With m=0, should return resistivity * (1+0)/(1-0) = resistivity
        assert np.allclose(rho_nb, resistivity, rtol=0.1)

    def test_derivatives_increasing_resistivity(self):
        """Test derivatives for increasing resistivity (m>0)."""
        period = np.logspace(0, 2, 10)
        resistivity = period * 10  # Linear increase in log space

        rho_nb = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        # With positive m, rho_nb should be > resistivity
        assert np.all(rho_nb[~np.isnan(rho_nb)] >= resistivity[~np.isnan(rho_nb)])

    def test_derivatives_decreasing_resistivity(self):
        """Test derivatives for decreasing resistivity (m<0)."""
        period = np.logspace(0, 2, 10)
        resistivity = 1000 / period  # Decreasing

        rho_nb = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        # With negative m, rho_nb should be < resistivity
        assert np.all(rho_nb[~np.isnan(rho_nb)] <= resistivity[~np.isnan(rho_nb)])

    def test_derivatives_gradient_too_large(self):
        """Test that gradients outside [-1, 1] return NaN."""
        period = np.logspace(0, 1, 5)
        # Create resistivity with steep gradient (m > 1)
        resistivity = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

        rho_nb = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        # Some values should be NaN due to steep gradient
        assert np.any(np.isnan(rho_nb))

    def test_derivatives_shape_preservation(self):
        """Test that output shape matches input shape."""
        resistivity = np.ones(15) * 100.0
        period = np.logspace(-1, 2, 15)

        rho_nb = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        assert rho_nb.shape == resistivity.shape


class TestDepthSensitivity:
    """Tests for calculate_depth_sensitivity function."""

    def test_sensitivity_single_depth(self):
        """Test sensitivity calculation for single depth."""
        depth = np.array([1000.0])
        period = np.array([1.0])

        sensitivity = nb.calculate_depth_sensitivity(depth, period)

        assert isinstance(sensitivity, np.ndarray)
        assert sensitivity.shape[0] == depth.shape[0]

    def test_sensitivity_multi_depth(self, depth_values, periods_single):
        """Test sensitivity for multiple depths."""
        sensitivity = nb.calculate_depth_sensitivity(depth_values, periods_single)

        assert sensitivity.shape[0] == depth_values.shape[0]
        assert np.all(sensitivity >= 0)

    def test_sensitivity_positive_values(self, depth_values, periods_multi):
        """Test that sensitivity values are non-negative."""
        for period in periods_multi[:5]:  # Test a few periods
            sensitivity = nb.calculate_depth_sensitivity(
                depth_values, np.array([period])
            )
            assert np.all(sensitivity >= 0)

    def test_sensitivity_decreases_with_depth(self):
        """Test that sensitivity generally decreases with depth."""
        depth = np.linspace(100, 10000, 50)
        period = np.array([1.0])

        sensitivity = nb.calculate_depth_sensitivity(depth, period)

        # After initial increase, sensitivity should decrease
        # Find peak and check decay after
        peak_idx = np.argmax(sensitivity)
        if peak_idx < len(sensitivity) - 1:
            assert np.all(np.diff(sensitivity[peak_idx:]) <= 0)

    def test_sensitivity_rho_parameter(self):
        """Test sensitivity with different resistivity values."""
        depth = np.linspace(0, 5000, 30)
        period = np.array([1.0])

        sens1 = nb.calculate_depth_sensitivity(depth, period, rho=100)
        sens2 = nb.calculate_depth_sensitivity(depth, period, rho=200)

        # According to docstring, sensitivity is independent of sigma and frequency
        # So results should be the same
        assert np.allclose(sens1, sens2)


class TestDepthOfInvestigation:
    """Tests for calculate_depth_of_investigation function."""

    def test_doi_1d_impedance(self, z_object_1d):
        """Test depth of investigation for 1D impedance."""
        depth_array = nb.calculate_depth_of_investigation(z_object_1d)

        assert isinstance(depth_array, np.ndarray)
        assert len(depth_array) == len(z_object_1d.period)

        # Check that all required fields exist
        required_fields = [
            "period",
            "depth_xy",
            "depth_yx",
            "depth_det",
            "depth_min",
            "depth_max",
            "resistivity_xy",
            "resistivity_yx",
            "resistivity_det",
            "resistivity_min",
            "resistivity_max",
        ]
        for field in required_fields:
            assert field in depth_array.dtype.names

    def test_doi_2d_impedance(self, z_object_2d):
        """Test depth of investigation for 2D impedance."""
        depth_array = nb.calculate_depth_of_investigation(z_object_2d)

        assert len(depth_array) == len(z_object_2d.period)

        # For 2D, XY and YX should differ
        assert not np.allclose(depth_array["depth_xy"], depth_array["depth_yx"])

    def test_doi_periods_match(self, z_object_1d):
        """Test that output periods match input periods."""
        depth_array = nb.calculate_depth_of_investigation(z_object_1d)

        assert np.allclose(depth_array["period"], z_object_1d.period)

    def test_doi_depths_positive(self, z_object_1d):
        """Test that all depth values are positive."""
        depth_array = nb.calculate_depth_of_investigation(z_object_1d)

        for field in ["depth_xy", "depth_yx", "depth_det"]:
            assert np.all(depth_array[field] > 0)

    def test_doi_min_max_bounds(self, z_object_2d):
        """Test that min/max depths bound the component depths."""
        depth_array = nb.calculate_depth_of_investigation(z_object_2d)

        for ii in range(len(depth_array)):
            depths = [
                depth_array["depth_xy"][ii],
                depth_array["depth_yx"][ii],
                depth_array["depth_det"][ii],
            ]
            if not np.any(np.isnan(depths)):
                assert depth_array["depth_min"][ii] <= np.min(depths)
                assert depth_array["depth_max"][ii] >= np.max(depths)

    def test_doi_resistivity_calculated(self, z_object_1d):
        """Test that resistivity values are calculated."""
        depth_array = nb.calculate_depth_of_investigation(z_object_1d)

        # For single period (1D), Weidelt method is used
        assert np.all(np.isfinite(depth_array["resistivity_xy"]))
        assert np.all(np.isfinite(depth_array["resistivity_yx"]))

    def test_doi_varying_phase(self, z_object_varying_phase):
        """Test DOI with varying phase values."""
        depth_array = nb.calculate_depth_of_investigation(z_object_varying_phase)

        assert len(depth_array) == len(z_object_varying_phase.period)
        # Should have valid depths and resistivities
        assert np.all(depth_array["depth_xy"] > 0)


class TestNiblettBostickEdgeCases:
    """Test edge cases and special conditions."""

    def test_depth_zero_period(self):
        """Test depth calculation with zero period."""
        resistivity = np.array([100.0])
        period = np.array([0.0])

        depth = nb.calculate_niblett_bostick_depth(resistivity, period)
        assert depth[0] == 0.0

    def test_depth_zero_resistivity(self):
        """Test depth calculation with zero resistivity."""
        resistivity = np.array([0.0])
        period = np.array([1.0])

        depth = nb.calculate_niblett_bostick_depth(resistivity, period)
        assert depth[0] == 0.0

    def test_weidelt_extreme_phases(self):
        """Test Weidelt with extreme phase values."""
        resistivity = np.array([100.0, 100.0, 100.0])
        phases = np.array([0.0, 45.0, 89.9])

        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(resistivity, phases)

        # All should be finite
        assert np.all(np.isfinite(rho_nb))

    def test_sensitivity_zero_depth(self):
        """Test sensitivity at zero depth."""
        depth = np.array([0.0])
        period = np.array([1.0])

        sensitivity = nb.calculate_depth_sensitivity(depth, period)

        # At zero depth, sensitivity should be zero
        assert np.allclose(sensitivity, 0.0)


class TestNiblettBostickParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize("resistivity", [10.0, 100.0, 1000.0, 10000.0])
    @pytest.mark.parametrize("period", [0.01, 1.0, 100.0])
    def test_depth_various_parameters(self, resistivity, period):
        """Test depth calculation with various parameter combinations."""
        rho = np.array([resistivity])
        per = np.array([period])

        depth = nb.calculate_niblett_bostick_depth(rho, per)

        assert depth[0] > 0
        assert np.isfinite(depth[0])

    @pytest.mark.parametrize("phase", [15.0, 30.0, 45.0, 60.0, 75.0])
    def test_weidelt_various_phases(self, phase):
        """Test Weidelt transformation at various phases."""
        resistivity = np.array([100.0])
        phase_arr = np.array([phase])

        rho_nb = nb.calculate_niblett_bostick_resistivity_weidelt(
            resistivity, phase_arr
        )

        assert np.isfinite(rho_nb[0])

    @pytest.mark.parametrize("n_periods", [5, 10, 20, 50])
    def test_derivatives_various_lengths(self, n_periods):
        """Test derivatives with various array lengths."""
        resistivity = np.ones(n_periods) * 100.0
        period = np.logspace(-1, 2, n_periods)

        rho_nb = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        assert rho_nb.shape == resistivity.shape

    @pytest.mark.parametrize("rho", [50.0, 100.0, 200.0])
    def test_sensitivity_various_rho(self, rho):
        """Test sensitivity with various resistivity values."""
        depth = np.linspace(0, 5000, 20)
        period = np.array([1.0])

        sensitivity = nb.calculate_depth_sensitivity(depth, period, rho=rho)

        assert sensitivity.shape[0] == depth.shape[0]
        assert np.all(sensitivity >= 0)


class TestNiblettBostickIntegration:
    """Integration tests combining multiple functions."""

    def test_depth_and_resistivity_consistency(self):
        """Test that depth and resistivity calculations are consistent."""
        period = np.logspace(-1, 2, 15)
        resistivity = np.ones_like(period) * 100.0
        phase = np.ones_like(period) * 45.0

        # Calculate depth
        depth = nb.calculate_niblett_bostick_depth(resistivity, period)

        # Calculate resistivity using both methods
        rho_nb_weidelt = nb.calculate_niblett_bostick_resistivity_weidelt(
            resistivity, phase
        )
        rho_nb_deriv = nb.calculate_niblett_bostick_resistivity_derivatives(
            resistivity, period
        )

        # All should be finite and positive
        assert np.all(depth > 0)
        assert np.all(np.isfinite(rho_nb_weidelt))
        assert np.all(np.isfinite(rho_nb_deriv))

    def test_full_doi_workflow(self, z_object_2d):
        """Test complete depth of investigation workflow."""
        # Calculate DOI
        depth_array = nb.calculate_depth_of_investigation(z_object_2d)

        # Verify all components are present and finite
        assert len(depth_array) == len(z_object_2d.period)

        # Periods come from frequencies, so they're in decreasing order
        # Depths should decrease as period decreases (frequency increases)
        # Check that min <= max for all
        assert np.all(depth_array["depth_min"] <= depth_array["depth_max"])

        # Check that all depths are positive
        assert np.all(depth_array["depth_min"] > 0)
        assert np.all(depth_array["depth_max"] > 0)

    def test_doi_with_rotation(self, z_object_2d):
        """Test DOI calculation after Z rotation."""
        # Rotate impedance
        z_object_2d.rotate(30)

        # Calculate DOI
        depth_array = nb.calculate_depth_of_investigation(z_object_2d)

        # Should still produce valid results
        assert len(depth_array) == len(z_object_2d.period)
        assert np.all(depth_array["depth_xy"] > 0)
        assert np.all(depth_array["depth_yx"] > 0)
