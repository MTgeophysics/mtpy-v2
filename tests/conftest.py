# -*- coding: utf-8 -*-
"""
Pytest configuration and global fixtures for mtpy-v2 tests.

This module provides shared fixtures that can be used across all test modules
to improve test efficiency and reduce redundant setup operations.

Created on December 22, 2025
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.core.transfer_function import MT_TO_OHM_FACTOR, Z


# =============================================================================
# Session-level fixtures (created once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def sample_edi_file():
    """Provide path to sample EDI file for testing."""
    return TF_EDI_CGG


@pytest.fixture(scope="session")
def sample_impedance_array():
    """
    Provide a sample impedance tensor array for testing.

    Returns a 1x2x2 complex impedance array.
    """
    return np.array([[0.1 - 0.1j, 10 + 10j], [-10 - 10j, -0.1 + 0.1j]]).reshape(
        (1, 2, 2)
    )


@pytest.fixture(scope="session")
def sample_impedance_error_array():
    """Provide a sample impedance error array for testing."""
    return np.array([[0.1, 0.05], [0.05, 0.1]]).reshape((1, 2, 2))


@pytest.fixture(scope="session")
def sample_tipper_array():
    """Provide a sample tipper array for testing."""
    return np.array([[[0.25 - 0.2j, 0.25 + 0.2j]]])


@pytest.fixture(scope="session")
def sample_tipper_error_array():
    """Provide a sample tipper error array for testing."""
    return np.array([[[0.02, 0.03]]])


@pytest.fixture(scope="session")
def expected_resistivity():
    """Provide expected resistivity values for sample impedance."""
    return np.array([[[4.0e-03, 4.0e01], [4.0e01, 4.0e-03]]])


@pytest.fixture(scope="session")
def expected_resistivity_error():
    """Provide expected resistivity error values."""
    return np.array([[[0.00565685, 0.28284271], [0.28284271, 0.00565685]]])


@pytest.fixture(scope="session")
def expected_phase():
    """Provide expected phase values for sample impedance."""
    return np.array([[[-45.0, 45.0], [-135.0, 135.0]]])


@pytest.fixture(scope="session")
def expected_phase_error():
    """Provide expected phase error values."""
    return np.array([[[35.26438968, 0.20257033], [0.20257033, 35.26438968]]])


@pytest.fixture(scope="session")
def expected_phase_tensor():
    """Provide expected phase tensor values."""
    return np.array([[[1.00020002, -0.020002], [-0.020002, 1.00020002]]])


@pytest.fixture(scope="session")
def expected_phase_tensor_error():
    """Provide expected phase tensor error values."""
    return np.array([[[0.01040308, 0.02020604], [0.02020604, 0.01040308]]])


@pytest.fixture(scope="session")
def expected_phase_tensor_azimuth():
    """Provide expected phase tensor azimuth."""
    return np.array([315.0])


@pytest.fixture(scope="session")
def expected_phase_tensor_azimuth_error():
    """Provide expected phase tensor azimuth error."""
    return np.array([3.30832308])


@pytest.fixture(scope="session")
def expected_phase_tensor_skew():
    """Provide expected phase tensor skew."""
    return np.array([0])


@pytest.fixture(scope="session")
def expected_phase_tensor_skew_error():
    """Provide expected phase tensor skew error."""
    return np.array([0.40923428])


# =============================================================================
# Function-level fixtures (created for each test function)
# =============================================================================


@pytest.fixture
def basic_mt():
    """
    Create a basic MT object with minimal setup.

    Returns a new MT instance for each test to ensure test isolation.
    """
    mt = MT()
    mt.station = "test_01"
    mt.survey = "big"
    mt.latitude = 10
    mt.longitude = 20
    return mt


@pytest.fixture
def mt_with_impedance(sample_impedance_array, sample_impedance_error_array):
    """
    Create an MT object with impedance data.

    Returns an MT instance with impedance and error data set.
    """
    mt = MT()
    mt.station = "mt001"
    mt.impedance = sample_impedance_array
    mt.impedance_error = sample_impedance_error_array
    mt.impedance_model_error = sample_impedance_error_array
    return mt


@pytest.fixture
def mt_with_impedance_ohm(sample_impedance_array, sample_impedance_error_array):
    """
    Create an MT object with impedance data in ohm units.

    Returns an MT instance with impedance in ohm units.
    """
    z_ohm = sample_impedance_array / MT_TO_OHM_FACTOR
    z_err_ohm = sample_impedance_error_array / MT_TO_OHM_FACTOR

    z_object = Z(
        z=z_ohm,
        z_error=z_err_ohm,
        z_model_error=z_err_ohm,
        units="ohm",
    )

    mt = MT()
    mt.station = "mt001"
    mt.Z = z_object
    return mt


@pytest.fixture
def mt_with_tipper(sample_tipper_array, sample_tipper_error_array):
    """
    Create an MT object with tipper data.

    Returns an MT instance with tipper and error data set.
    """
    mt = MT()
    mt.tipper = sample_tipper_array
    mt.tipper_error = sample_tipper_error_array
    return mt


@pytest.fixture
def mt_from_edi(sample_edi_file):
    """
    Create an MT object from EDI file.

    Returns an MT instance loaded from the sample EDI file.
    """
    mt = MT(sample_edi_file)
    mt.read()
    return mt


# =============================================================================
# Pytest configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to optimize for pytest-xdist.

    This ensures tests are properly isolated and can run in parallel.
    """
    # Add markers based on test names
    for item in items:
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)
