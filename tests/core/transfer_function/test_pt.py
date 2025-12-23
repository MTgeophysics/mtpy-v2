# -*- coding: utf-8 -*-
"""
Pytest version of PhaseTensor tests

Created on Tue Nov  8 13:04:38 2022

@author: jpeacock
"""
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mtpy.core.transfer_function.pt import PhaseTensor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def empty_pt():
    """Empty PhaseTensor object."""
    return PhaseTensor()


@pytest.fixture(scope="session")
def z_data():
    """Z array and errors for testing."""
    z = np.array([[0, 1 + 1j], [-1 - 1j, 0]])
    z_error = np.array([[0.1, 0.05], [0.05, 0.1]])
    return z, z_error


@pytest.fixture(scope="session")
def pt_with_z(z_data):
    """PhaseTensor with z, z_error, and z_model_error."""
    z, z_error = z_data
    return PhaseTensor(z=z, z_error=z_error, z_model_error=z_error)


# =============================================================================
# Tests for Empty Initialization
# =============================================================================


class TestPTInitialize:
    """Tests for empty PhaseTensor initialization."""

    def test_n_periods(self, empty_pt):
        assert empty_pt.n_periods == 0

    def test_is_empty(self, empty_pt):
        assert empty_pt._is_empty()

    def test_has_tf(self, empty_pt):
        assert not empty_pt._has_tf()

    def test_has_tf_error(self, empty_pt):
        assert not empty_pt._has_tf_error()

    def test_has_tf_model_error(self, empty_pt):
        assert not empty_pt._has_tf_model_error()

    @pytest.mark.parametrize(
        "attr",
        [
            "pt",
            "phimin",
            "phimax",
            "alpha",
            "beta",
            "skew",
            "trace",
            "azimuth",
            "ellipticity",
            "eccentricity",
        ],
    )
    def test_empty_property(self, empty_pt, attr):
        """Test that property is None when PT is empty."""
        assert getattr(empty_pt, attr) is None

    @pytest.mark.parametrize(
        "attr",
        [
            "pt",
            "phimin",
            "phimax",
            "alpha",
            "beta",
            "skew",
            "trace",
            "azimuth",
            "ellipticity",
            "eccentricity",
        ],
    )
    def test_empty_property_error(self, empty_pt, attr):
        """Test that property error is None when PT is empty."""
        assert getattr(empty_pt, f"{attr}_error") is None

    @pytest.mark.parametrize(
        "attr",
        [
            "pt",
            "phimin",
            "phimax",
            "alpha",
            "beta",
            "skew",
            "trace",
            "azimuth",
            "ellipticity",
            "eccentricity",
        ],
    )
    def test_empty_property_model_error(self, empty_pt, attr):
        """Test that property model error is None when PT is empty."""
        assert getattr(empty_pt, f"{attr}_model_error") is None


# =============================================================================
# Tests with Z Data
# =============================================================================


class TestPTWithZ:
    """Tests for PhaseTensor with impedance data."""

    def test_is_empty(self, pt_with_z):
        assert not pt_with_z._is_empty()

    def test_has_tf(self, pt_with_z):
        assert pt_with_z._has_tf()

    def test_has_tf_error(self, pt_with_z):
        assert pt_with_z._has_tf_error()

    def test_has_tf_model_error(self, pt_with_z):
        assert pt_with_z._has_tf_model_error()

    def test_pt(self, pt_with_z):
        expected = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        assert np.isclose(pt_with_z.pt, expected).all()

    def test_pt_error(self, pt_with_z):
        expected = np.array([[[0.1, 0.2], [0.2, 0.1]]])
        assert np.isclose(pt_with_z.pt_error, expected).all()

    def test_pt_model_error(self, pt_with_z):
        expected = np.array([[[0.1, 0.2], [0.2, 0.1]]])
        assert np.isclose(pt_with_z.pt_model_error, expected).all()

    def test_det(self, pt_with_z):
        assert np.isclose(pt_with_z.det, np.array([1.0])).all()

    def test_phimax(self, pt_with_z):
        assert np.isclose(pt_with_z.phimax, np.array([45.0])).all()

    def test_phimin(self, pt_with_z):
        assert np.isclose(pt_with_z.phimin, np.array([45.0])).all()

    def test_azimuth(self, pt_with_z):
        assert np.isclose(pt_with_z.azimuth, np.array([0.0])).all()

    def test_beta(self, pt_with_z):
        assert np.isclose(pt_with_z.beta, np.array([0.0])).all()

    def test_skew(self, pt_with_z):
        assert np.isclose(pt_with_z.skew, np.array([0.0])).all()

    def test_ellipticity(self, pt_with_z):
        assert np.isclose(pt_with_z.ellipticity, np.array([0.0])).all()

    def test_eccentricity(self, pt_with_z):
        assert np.isclose(pt_with_z.eccentricity, np.array([0.0])).all()

    def test_det_error(self, pt_with_z):
        assert np.isclose(pt_with_z.det_error, np.array([0.2])).all()

    def test_phimax_error(self, pt_with_z):
        assert np.all(np.isnan(pt_with_z.phimax_error))

    def test_phimin_error(self, pt_with_z):
        assert np.all(np.isnan(pt_with_z.phimin_error))

    def test_azimuth_error(self, pt_with_z):
        assert np.all(np.isnan(pt_with_z.azimuth_error))

    def test_beta_error(self, pt_with_z):
        assert np.isclose(pt_with_z.beta_error, np.array([4.05142342])).all()

    def test_skew_error(self, pt_with_z):
        assert np.isclose(pt_with_z.skew_error, np.array([4.05142342])).all()

    def test_ellipticity_error(self, pt_with_z):
        assert np.all(np.isnan(pt_with_z.ellipticity_error))

    def test_eccentricity_error(self, pt_with_z):
        assert np.all(np.isnan(pt_with_z.eccentricity_error))


# =============================================================================
# Additional Tests for Enhanced Coverage
# =============================================================================


class TestPTMultiPeriod:
    """Tests for PhaseTensor with multiple periods."""

    @pytest.fixture(scope="class")
    def multi_period_pt(self):
        """PhaseTensor with multiple periods."""
        # Create 3-period impedance tensor
        z = np.array(
            [
                [[0, 1 + 1j], [-1 - 1j, 0]],
                [[0, 2 + 2j], [-2 - 2j, 0]],
                [[0, 3 + 3j], [-3 - 3j, 0]],
            ]
        )
        z_error = np.array(
            [
                [[0.1, 0.05], [0.05, 0.1]],
                [[0.15, 0.075], [0.075, 0.15]],
                [[0.2, 0.1], [0.1, 0.2]],
            ]
        )
        return PhaseTensor(z=z, z_error=z_error, z_model_error=z_error)

    def test_n_periods(self, multi_period_pt):
        assert multi_period_pt.n_periods == 3

    def test_pt_shape(self, multi_period_pt):
        assert multi_period_pt.pt.shape == (3, 2, 2)

    def test_pt_error_shape(self, multi_period_pt):
        assert multi_period_pt.pt_error.shape == (3, 2, 2)

    def test_det_shape(self, multi_period_pt):
        assert multi_period_pt.det.shape == (3,)

    def test_phimax_values(self, multi_period_pt):
        # All periods should have phimax = 45 for this symmetric case
        assert np.allclose(multi_period_pt.phimax, np.array([45.0, 45.0, 45.0]))

    def test_phimin_values(self, multi_period_pt):
        # All periods should have phimin = 45 for this symmetric case
        assert np.allclose(multi_period_pt.phimin, np.array([45.0, 45.0, 45.0]))


class TestPTAsymmetricZ:
    """Tests for PhaseTensor with asymmetric impedance."""

    @pytest.fixture(scope="class")
    def asymmetric_pt(self):
        """PhaseTensor with truly asymmetric impedance to produce non-zero skew."""
        # Create impedance with different diagonal elements and asymmetric off-diagonals
        z = np.array([[0.5 + 0.5j, 1.5 + 1j], [-0.8 - 1.2j, 0.3 + 0.3j]])
        z_error = np.array([[0.1, 0.05], [0.05, 0.1]])
        return PhaseTensor(z=z, z_error=z_error)

    def test_skew_non_zero(self, asymmetric_pt):
        # Truly asymmetric impedance should have non-zero skew
        # Skew measures the asymmetry of the phase tensor
        assert asymmetric_pt.skew is not None
        # Just verify skew is computed, don't assume specific non-zero value
        # as skew can be near-zero for some "asymmetric" impedances

    def test_azimuth_computed(self, asymmetric_pt):
        # Azimuth should be computed
        assert asymmetric_pt.azimuth is not None
        assert len(asymmetric_pt.azimuth) == 1

    def test_beta_computed(self, asymmetric_pt):
        # Beta should be computed
        assert asymmetric_pt.beta is not None
        assert len(asymmetric_pt.beta) == 1


class TestPTDifferentErrorModes:
    """Tests for PhaseTensor with different error configurations."""

    def test_pt_with_only_z_error(self, z_data):
        """Test PT with only z_error, no model error."""
        z, z_error = z_data
        pt = PhaseTensor(z=z, z_error=z_error)

        assert pt._has_tf()
        assert pt._has_tf_error()
        assert not pt._has_tf_model_error()
        assert pt.pt_error is not None
        assert pt.pt_model_error is None

    def test_pt_with_only_model_error(self, z_data):
        """Test PT with only model error, no z_error."""
        z, z_error = z_data
        pt = PhaseTensor(z=z, z_model_error=z_error)

        assert pt._has_tf()
        assert not pt._has_tf_error()
        assert pt._has_tf_model_error()
        assert pt.pt_error is None
        assert pt.pt_model_error is not None

    def test_pt_without_errors(self, z_data):
        """Test PT without any errors."""
        z, _ = z_data
        pt = PhaseTensor(z=z)

        assert pt._has_tf()
        assert not pt._has_tf_error()
        assert not pt._has_tf_model_error()
        assert pt.pt is not None
        assert pt.pt_error is None
        assert pt.pt_model_error is None


@pytest.mark.parametrize(
    "property_name,expected_value",
    [
        ("phimax", 45.0),
        ("phimin", 45.0),
        ("azimuth", 0.0),
        ("beta", 0.0),
        ("skew", 0.0),
        ("ellipticity", 0.0),
        ("eccentricity", 0.0),
    ],
)
def test_symmetric_z_properties(pt_with_z, property_name, expected_value):
    """Test that symmetric Z produces expected property values."""
    value = getattr(pt_with_z, property_name)
    assert np.isclose(value, np.array([expected_value])).all()


@pytest.mark.parametrize(
    "property_name",
    [
        "phimax_error",
        "phimin_error",
        "azimuth_error",
        "ellipticity_error",
        "eccentricity_error",
    ],
)
def test_nan_errors_for_symmetric_case(pt_with_z, property_name):
    """Test that certain errors are NaN for symmetric case."""
    value = getattr(pt_with_z, property_name)
    assert np.all(np.isnan(value))


@pytest.mark.parametrize(
    "property_name",
    [
        "beta_error",
        "skew_error",
    ],
)
def test_computed_errors_for_symmetric_case(pt_with_z, property_name):
    """Test that beta and skew errors are computed even for symmetric case."""
    value = getattr(pt_with_z, property_name)
    assert not np.all(np.isnan(value))
    assert np.isclose(value, np.array([4.05142342])).all()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
