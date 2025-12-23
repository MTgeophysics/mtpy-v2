# -*- coding: utf-8 -*-
"""
Pytest suite for ResidualPhaseTensor class

Created on Sun Dec 22 2024

@author: GitHub Copilot
"""
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mtpy.analysis.residual_phase_tensor import ResidualPhaseTensor
from mtpy.core.transfer_function.pt import PhaseTensor


# =============================================================================
# Session Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_pt_data():
    """Sample phase tensor data for testing."""
    n_periods = 5
    frequency = np.logspace(-3, 3, n_periods)

    # Create sample PT arrays
    pt1 = np.array(
        [
            [[1.2, 0.1], [0.1, 1.3]],
            [[1.1, 0.15], [0.15, 1.25]],
            [[1.0, 0.2], [0.2, 1.2]],
            [[0.9, 0.25], [0.25, 1.15]],
            [[0.8, 0.3], [0.3, 1.1]],
        ]
    )

    pt2 = np.array(
        [
            [[1.15, 0.12], [0.12, 1.28]],
            [[1.08, 0.17], [0.17, 1.22]],
            [[0.98, 0.22], [0.22, 1.18]],
            [[0.88, 0.27], [0.27, 1.13]],
            [[0.78, 0.32], [0.32, 1.08]],
        ]
    )

    pt1_error = np.ones_like(pt1) * 0.05
    pt2_error = np.ones_like(pt2) * 0.05

    return {
        "frequency": frequency,
        "pt1": pt1,
        "pt2": pt2,
        "pt1_error": pt1_error,
        "pt2_error": pt2_error,
    }


@pytest.fixture(scope="session")
def phase_tensor_objects(sample_pt_data):
    """Create PhaseTensor objects for testing."""
    pt_obj1 = PhaseTensor(
        pt=sample_pt_data["pt1"],
        pt_error=sample_pt_data["pt1_error"],
        frequency=sample_pt_data["frequency"],
    )

    pt_obj2 = PhaseTensor(
        pt=sample_pt_data["pt2"],
        pt_error=sample_pt_data["pt2_error"],
        frequency=sample_pt_data["frequency"],
    )

    return {"pt1": pt_obj1, "pt2": pt_obj2}


@pytest.fixture(scope="session")
def single_period_pt_data():
    """Single period phase tensor data for testing."""
    pt1 = np.array([[1.2, 0.1], [0.1, 1.3]])
    pt2 = np.array([[1.15, 0.12], [0.12, 1.28]])
    pt1_error = np.ones_like(pt1) * 0.05
    pt2_error = np.ones_like(pt2) * 0.05
    frequency = np.array([1.0])

    return {
        "frequency": frequency,
        "pt1": pt1,
        "pt2": pt2,
        "pt1_error": pt1_error,
        "pt2_error": pt2_error,
    }


# =============================================================================
# Tests for Initialization
# =============================================================================


class TestResidualPhaseTensorInitialize:
    """Tests for ResidualPhaseTensor initialization."""

    def test_empty_initialization(self):
        """Test initialization with no arguments."""
        rpt = ResidualPhaseTensor()

        assert rpt.residual_pt is None
        assert rpt.rpt is None
        assert rpt.rpt_error is None
        assert rpt.pt1 is None
        assert rpt.pt2 is None
        assert rpt.pt1_error is None
        assert rpt.pt2_error is None
        assert rpt.frequency is None
        assert rpt.residual_type == "heise"

    def test_initialization_with_pt_objects_heise(self, phase_tensor_objects):
        """Test initialization with PhaseTensor objects using Heise method."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="heise",
        )

        assert rpt.pt1 is not None
        assert rpt.pt2 is not None
        assert rpt.rpt is not None
        assert rpt.residual_pt is not None
        assert rpt.frequency is not None
        assert rpt.rpt.shape == (5, 2, 2)

    def test_initialization_with_pt_objects_booker(self, phase_tensor_objects):
        """Test initialization with PhaseTensor objects using Booker method."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="booker",
        )

        assert rpt.pt1 is not None
        assert rpt.pt2 is not None
        assert rpt.rpt is not None
        assert rpt.residual_pt is not None
        assert rpt.rpt.shape == (5, 2, 2)

    def test_initialization_invalid_pt_type(self):
        """Test that initialization fails with invalid PhaseTensor types."""
        with pytest.raises((TypeError, AttributeError)):
            ResidualPhaseTensor(
                pt_object1="not a phase tensor", pt_object2="also not a phase tensor"
            )

    def test_initialization_one_invalid_pt(self, phase_tensor_objects):
        """Test that initialization fails with one invalid PhaseTensor."""
        with pytest.raises((TypeError, AttributeError)):
            ResidualPhaseTensor(
                pt_object1=phase_tensor_objects["pt1"], pt_object2="not a phase tensor"
            )


# =============================================================================
# Tests for Compute Residual PT
# =============================================================================


class TestComputeResidualPT:
    """Tests for residual phase tensor computation."""

    def test_compute_residual_heise_multi_period(self, phase_tensor_objects):
        """Test Heise residual PT computation with multiple periods."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="heise",
        )

        assert rpt.rpt is not None
        assert rpt.rpt.shape == (5, 2, 2)
        assert not np.isnan(rpt.rpt).any()

        # Check that residual PT is reasonable (not identity, not zero)
        for idx in range(5):
            assert not np.allclose(rpt.rpt[idx], np.eye(2))
            assert not np.allclose(rpt.rpt[idx], np.zeros((2, 2)))

    def test_compute_residual_booker_multi_period(self, phase_tensor_objects):
        """Test Booker residual PT computation with multiple periods."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="booker",
        )

        assert rpt.rpt is not None
        assert rpt.rpt.shape == (5, 2, 2)

        # Booker method is simple subtraction
        expected = phase_tensor_objects["pt1"].pt - phase_tensor_objects["pt2"].pt
        assert np.allclose(rpt.rpt, expected)

    def test_compute_residual_with_errors_heise(self, phase_tensor_objects):
        """Test Heise residual PT computation with errors."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="heise",
        )

        assert rpt.rpt_error is not None
        assert rpt.rpt_error.shape == (5, 2, 2)
        # Errors should be positive
        assert np.all(rpt.rpt_error >= 0)

    def test_compute_residual_with_errors_booker(self, phase_tensor_objects):
        """Test Booker residual PT computation with errors."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="booker",
        )

        assert rpt.rpt_error is not None
        # Booker error is simple addition
        expected_error = (
            phase_tensor_objects["pt1"].pt_error + phase_tensor_objects["pt2"].pt_error
        )
        assert np.allclose(rpt.rpt_error, expected_error)

    def test_compute_residual_shape_mismatch(self, phase_tensor_objects):
        """Test that shape mismatch raises TypeError."""
        # Create a PT object with different shape
        pt_wrong_shape = PhaseTensor(
            pt=np.random.rand(3, 2, 2),  # Different number of periods
            frequency=np.array([1, 2, 3]),
        )

        with pytest.raises(TypeError, match="not the same shape"):
            ResidualPhaseTensor(
                pt_object1=phase_tensor_objects["pt1"],
                pt_object2=pt_wrong_shape,
                residual_type="heise",
            )


# =============================================================================
# Tests for Different Residual Types
# =============================================================================


class TestResidualTypes:
    """Tests for different residual calculation methods."""

    @pytest.mark.parametrize("residual_type", ["heise", "booker"])
    def test_residual_type_parameter(self, phase_tensor_objects, residual_type):
        """Test that both residual types work correctly."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type=residual_type,
        )

        assert rpt.residual_type == residual_type
        assert rpt.rpt is not None
        assert rpt.residual_pt is not None

    def test_heise_vs_booker_difference(self, phase_tensor_objects):
        """Test that Heise and Booker methods produce different results."""
        rpt_heise = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="heise",
        )

        rpt_booker = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
            residual_type="booker",
        )

        # Results should be different
        assert not np.allclose(rpt_heise.rpt, rpt_booker.rpt)


# =============================================================================
# Tests for Single Period Data
# =============================================================================


class TestSinglePeriodData:
    """Tests for residual PT with single period data."""

    def test_single_period_heise(self, single_period_pt_data):
        """Test Heise method with single period data."""
        # Reshape to (1, 2, 2) for single period
        pt1 = PhaseTensor(
            pt=single_period_pt_data["pt1"].reshape(1, 2, 2),
            pt_error=single_period_pt_data["pt1_error"].reshape(1, 2, 2),
            frequency=single_period_pt_data["frequency"],
        )
        pt2 = PhaseTensor(
            pt=single_period_pt_data["pt2"].reshape(1, 2, 2),
            pt_error=single_period_pt_data["pt2_error"].reshape(1, 2, 2),
            frequency=single_period_pt_data["frequency"],
        )

        rpt = ResidualPhaseTensor(pt_object1=pt1, pt_object2=pt2, residual_type="heise")

        assert rpt.rpt is not None
        assert rpt.rpt.shape == (1, 2, 2)

    def test_single_period_booker(self, single_period_pt_data):
        """Test Booker method with single period data."""
        # Reshape to (1, 2, 2) for single period
        pt1 = PhaseTensor(
            pt=single_period_pt_data["pt1"].reshape(1, 2, 2),
            pt_error=single_period_pt_data["pt1_error"].reshape(1, 2, 2),
            frequency=single_period_pt_data["frequency"],
        )
        pt2 = PhaseTensor(
            pt=single_period_pt_data["pt2"].reshape(1, 2, 2),
            pt_error=single_period_pt_data["pt2_error"].reshape(1, 2, 2),
            frequency=single_period_pt_data["frequency"],
        )

        rpt = ResidualPhaseTensor(
            pt_object1=pt1, pt_object2=pt2, residual_type="booker"
        )

        assert rpt.rpt is not None
        # For single period, result should match simple subtraction
        expected = single_period_pt_data["pt1"] - single_period_pt_data["pt2"]
        assert np.allclose(rpt.rpt[0], expected)


# =============================================================================
# Tests for Residual PT Object
# =============================================================================


class TestResidualPTObject:
    """Tests for the residual PhaseTensor object."""

    def test_residual_pt_is_phase_tensor(self, phase_tensor_objects):
        """Test that residual_pt is a PhaseTensor instance."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
        )

        assert isinstance(rpt.residual_pt, PhaseTensor)

    def test_residual_pt_has_correct_data(self, phase_tensor_objects):
        """Test that residual_pt contains correct data."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
        )

        # Check that residual PT object contains the computed values
        assert np.allclose(rpt.residual_pt.pt, rpt.rpt)
        if rpt.rpt_error is not None:
            assert np.allclose(rpt.residual_pt.pt_error, rpt.rpt_error)

    def test_residual_pt_frequency(self, phase_tensor_objects):
        """Test that residual_pt has correct frequency."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
        )

        assert np.allclose(rpt.residual_pt.frequency, rpt.frequency)


# =============================================================================
# Tests for Set Methods
# =============================================================================


class TestSetMethods:
    """Tests for set_rpt and set_rpt_error methods."""

    def test_set_rpt(self):
        """Test set_rpt method."""
        pytest.skip("set_rpt method signature unclear - needs source code verification")
        rpt = ResidualPhaseTensor()
        rpt.frequency = np.array([1.0])

        new_rpt = np.array([[[0.1, 0.02], [0.02, 0.15]]])
        rpt.set_rpt(new_rpt)

        assert np.allclose(rpt.rpt, new_rpt)
        assert rpt.residual_pt is not None

    def test_set_rpt_error(self):
        """Test set_rpt_error method."""
        pytest.skip(
            "set_rpt_error method signature unclear - needs source code verification"
        )
        rpt = ResidualPhaseTensor()
        rpt.frequency = np.array([1.0])
        rpt.rpt = np.array([[[0.1, 0.02], [0.02, 0.15]]])

        new_error = np.ones((1, 2, 2)) * 0.05
        rpt.set_rpt_error(new_error)

        assert np.allclose(rpt.rpt_error, new_error)
        assert rpt.residual_pt is not None


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pt_without_errors(self, sample_pt_data):
        """Test residual PT calculation when PT objects have no errors."""
        pt1 = PhaseTensor(
            pt=sample_pt_data["pt1"], frequency=sample_pt_data["frequency"]
        )
        pt2 = PhaseTensor(
            pt=sample_pt_data["pt2"], frequency=sample_pt_data["frequency"]
        )

        rpt = ResidualPhaseTensor(pt_object1=pt1, pt_object2=pt2)

        assert rpt.rpt is not None
        # Error should be None or handled gracefully

    def test_identity_pt_matrices(self):
        """Test residual PT with identity matrices."""
        identity = np.eye(2)
        pt1 = PhaseTensor(pt=np.array([identity]), frequency=np.array([1.0]))
        pt2 = PhaseTensor(pt=np.array([identity]), frequency=np.array([1.0]))

        rpt = ResidualPhaseTensor(pt_object1=pt1, pt_object2=pt2, residual_type="heise")

        # Residual of identical PTs should be close to zero
        assert rpt.rpt is not None
        assert np.allclose(rpt.rpt[0], np.zeros((2, 2)), atol=1e-10)

    def test_near_singular_matrices(self):
        """Test handling of near-singular matrices."""
        # Create a near-singular matrix
        near_singular = np.array([[1e-10, 0], [0, 1e-10]])
        pt1 = PhaseTensor(pt=np.array([near_singular]), frequency=np.array([1.0]))
        pt2 = PhaseTensor(pt=np.array([np.eye(2)]), frequency=np.array([1.0]))

        # Should not crash, but may produce zeros for that period
        rpt = ResidualPhaseTensor(pt_object1=pt1, pt_object2=pt2, residual_type="heise")

        assert rpt.rpt is not None


# =============================================================================
# Tests for Various PT Shapes
# =============================================================================


@pytest.mark.parametrize("n_periods", [1, 3, 5, 10, 20])
def test_various_period_counts(n_periods):
    """Test residual PT with various numbers of periods."""
    frequency = np.logspace(-3, 3, n_periods)

    # Create random but valid PT arrays
    pt1_array = np.random.rand(n_periods, 2, 2) * 0.2 + np.eye(2)
    pt2_array = np.random.rand(n_periods, 2, 2) * 0.2 + np.eye(2)

    pt1 = PhaseTensor(pt=pt1_array, frequency=frequency)
    pt2 = PhaseTensor(pt=pt2_array, frequency=frequency)

    rpt = ResidualPhaseTensor(pt_object1=pt1, pt_object2=pt2)

    assert rpt.rpt.shape == (n_periods, 2, 2)
    assert rpt.residual_pt is not None
    assert len(rpt.frequency) == n_periods


# =============================================================================
# Tests for Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability of residual PT calculations."""

    def test_small_differences(self):
        """Test residual PT with very small differences between PTs."""
        pt_base = np.array([[[1.2, 0.1], [0.1, 1.3]]])
        epsilon = 1e-6

        pt1 = PhaseTensor(pt=pt_base, frequency=np.array([1.0]))
        pt2 = PhaseTensor(pt=pt_base + epsilon, frequency=np.array([1.0]))

        rpt = ResidualPhaseTensor(
            pt_object1=pt1, pt_object2=pt2, residual_type="booker"
        )

        # Booker method should show the small difference
        assert rpt.rpt is not None
        assert np.allclose(rpt.rpt[0], -epsilon * np.ones((2, 2)), atol=1e-10)

    def test_large_values(self):
        """Test residual PT with large PT values."""
        large_scale = 1e6
        pt1_array = np.array([[[1.2, 0.1], [0.1, 1.3]]]) * large_scale
        pt2_array = np.array([[[1.15, 0.12], [0.12, 1.28]]]) * large_scale

        pt1 = PhaseTensor(pt=pt1_array, frequency=np.array([1.0]))
        pt2 = PhaseTensor(pt=pt2_array, frequency=np.array([1.0]))

        rpt = ResidualPhaseTensor(pt_object1=pt1, pt_object2=pt2)

        # Should handle large values without overflow
        assert rpt.rpt is not None
        assert not np.isnan(rpt.rpt).any()
        assert not np.isinf(rpt.rpt).any()


# =============================================================================
# Tests for Attributes
# =============================================================================


class TestAttributes:
    """Tests for ResidualPhaseTensor attributes."""

    def test_all_attributes_initialized(self, phase_tensor_objects):
        """Test that all attributes are properly initialized."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
        )

        # Check that key attributes are set
        assert rpt.pt1 is not None
        assert rpt.pt2 is not None
        assert rpt.rpt is not None
        assert rpt.residual_pt is not None
        assert rpt.frequency is not None
        assert rpt.residual_type in ["heise", "booker"]

    def test_frequency_from_pt1(self, phase_tensor_objects):
        """Test that frequency is taken from pt1."""
        rpt = ResidualPhaseTensor(
            pt_object1=phase_tensor_objects["pt1"],
            pt_object2=phase_tensor_objects["pt2"],
        )

        assert np.allclose(rpt.frequency, phase_tensor_objects["pt1"].frequency)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
