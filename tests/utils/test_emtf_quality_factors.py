# -*- coding: utf-8 -*-
"""
Pytest test suite for MT transfer function quality factor estimation.

Created on Wed Jun 12 15:39:08 2024
Converted to pytest format on Dec 31, 2025

@author: jpeacock
"""
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata import (
    TF_AVG,
    TF_AVG_NEWER,
    TF_AVG_TIPPER,
    TF_EDI_CGG,
    TF_EDI_EMPOWER,
    TF_EDI_METRONIX,
    TF_EDI_NO_ERROR,
    TF_EDI_PHOENIX,
    TF_EDI_QUANTEC,
    TF_EDI_RHO_ONLY,
    TF_EDI_SPECTRA,
    TF_JFILE,
    TF_POOR_XML,
    TF_XML,
    TF_XML_COMPLETE_REMOTE_INFO,
    TF_XML_MULTIPLE_ATTACHMENTS,
    TF_XML_NO_SITE_LAYOUT,
    TF_XML_WITH_DERIVED_QUANTITIES,
    TF_ZMM,
    TF_ZSS_TIPPER,
)

from mtpy import MT


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tf_test_data():
    """Session-scoped fixture providing TF file test data.

    Returns a list of dictionaries with TF file paths and expected quality factors.
    This is session-scoped for performance since the data doesn't change.
    """
    return [
        {"fn": TF_AVG, "rounded_qf": 3.0, "qf": 3.35},
        {"fn": TF_AVG_NEWER, "rounded_qf": 4.0, "qf": 3.525},
        # {"fn": TF_AVG_TIPPER, "rounded_qf": 2.0, "qf": 1.45},
        {"fn": TF_EDI_CGG, "rounded_qf": 5.0, "qf": 4.85},
        {"fn": TF_EDI_EMPOWER, "rounded_qf": 5.0, "qf": 4.9},
        {"fn": TF_EDI_METRONIX, "rounded_qf": 4.0, "qf": 3.9},
        {"fn": TF_EDI_NO_ERROR, "rounded_qf": 2.0, "qf": 2.35},
        {"fn": TF_EDI_PHOENIX, "rounded_qf": 4.0, "qf": 4.2},
        {"fn": TF_EDI_QUANTEC, "rounded_qf": 5.0, "qf": 4.75},
        {"fn": TF_EDI_RHO_ONLY, "rounded_qf": 4.0, "qf": 4.0},
        {"fn": TF_EDI_SPECTRA, "rounded_qf": 4.0, "qf": 4.0},
        {"fn": TF_JFILE, "rounded_qf": 1.0, "qf": 1.35},
        {"fn": TF_POOR_XML, "rounded_qf": 3.0, "qf": 2.8},
        {"fn": TF_XML, "rounded_qf": 4.0, "qf": 4.4},
        {"fn": TF_XML_COMPLETE_REMOTE_INFO, "rounded_qf": 4.0, "qf": 4.0},
        {"fn": TF_XML_MULTIPLE_ATTACHMENTS, "rounded_qf": 3.0, "qf": 3.35},
        {"fn": TF_XML_NO_SITE_LAYOUT, "rounded_qf": 4.0, "qf": 3.5},
        {"fn": TF_XML_WITH_DERIVED_QUANTITIES, "rounded_qf": 4.0, "qf": 4.0},
        {"fn": TF_ZMM, "rounded_qf": 4.0, "qf": 3.65},
        {"fn": TF_ZSS_TIPPER, "rounded_qf": 4.0, "qf": 3.75},
    ]


@pytest.fixture
def mt_object():
    """Function-scoped fixture providing a fresh MT object for each test.

    Function-scoped for parallel test isolation.
    """
    return MT()


@pytest.fixture
def loaded_mt_object(mt_object, request):
    """Function-scoped fixture providing an MT object loaded with a TF file.

    Uses indirect parametrization to load different TF files.
    """
    if hasattr(request, "param"):
        mt_object.read(request.param)
    return mt_object


# =============================================================================
# Parametrized Test Data
# =============================================================================


def get_tf_file_params(tf_test_data):
    """Helper to create parametrize arguments from test data."""
    return [
        pytest.param(
            tf_dict["fn"], tf_dict["rounded_qf"], tf_dict["qf"], id=tf_dict["fn"].name
        )
        for tf_dict in tf_test_data
    ]


# =============================================================================
# Test Classes
# =============================================================================


class TestTFQualityFactorRounded:
    """Test rounded quality factor estimation for various TF file formats."""

    @pytest.mark.parametrize(
        "tf_file,expected_rounded_qf,expected_qf",
        [
            pytest.param(TF_AVG, 3.0, 3.35, id="TF_AVG"),
            pytest.param(TF_AVG_NEWER, 4.0, 3.525, id="TF_AVG_NEWER"),
            pytest.param(TF_AVG_TIPPER, 2.0, 1.5, id="TF_AVG_TIPPER"),
            pytest.param(TF_EDI_CGG, 5.0, 4.85, id="TF_EDI_CGG"),
            pytest.param(TF_EDI_EMPOWER, 5.0, 4.9, id="TF_EDI_EMPOWER"),
            pytest.param(TF_EDI_METRONIX, 4.0, 3.9, id="TF_EDI_METRONIX"),
            pytest.param(TF_EDI_NO_ERROR, 2.0, 2.35, id="TF_EDI_NO_ERROR"),
            pytest.param(TF_EDI_PHOENIX, 4.0, 4.2, id="TF_EDI_PHOENIX"),
            pytest.param(TF_EDI_QUANTEC, 5.0, 4.75, id="TF_EDI_QUANTEC"),
            pytest.param(TF_EDI_RHO_ONLY, 4.0, 3.75, id="TF_EDI_RHO_ONLY"),
            pytest.param(TF_EDI_SPECTRA, 4.0, 4.0, id="TF_EDI_SPECTRA"),
            pytest.param(TF_JFILE, 1.0, 1.35, id="TF_JFILE"),
            pytest.param(TF_POOR_XML, 3.0, 2.8, id="TF_POOR_XML"),
            pytest.param(TF_XML, 4.0, 4.4, id="TF_XML"),
            pytest.param(
                TF_XML_COMPLETE_REMOTE_INFO, 4.0, 4.0, id="TF_XML_COMPLETE_REMOTE_INFO"
            ),
            pytest.param(
                TF_XML_MULTIPLE_ATTACHMENTS, 3.0, 3.35, id="TF_XML_MULTIPLE_ATTACHMENTS"
            ),
            pytest.param(TF_XML_NO_SITE_LAYOUT, 4.0, 3.5, id="TF_XML_NO_SITE_LAYOUT"),
            pytest.param(
                TF_XML_WITH_DERIVED_QUANTITIES,
                4.0,
                4.0,
                id="TF_XML_WITH_DERIVED_QUANTITIES",
            ),
            pytest.param(TF_ZMM, 4.0, 3.65, id="TF_ZMM"),
            pytest.param(TF_ZSS_TIPPER, 4.0, 3.75, id="TF_ZSS_TIPPER"),
        ],
    )
    def test_rounded_quality_factor(
        self, mt_object, tf_file, expected_rounded_qf, expected_qf
    ):
        """Test that rounded quality factor matches expected value."""
        mt_object.read(tf_file)
        qf = mt_object.estimate_tf_quality(round_qf=True)
        assert qf == expected_rounded_qf


class TestTFQualityFactorUnrounded:
    """Test unrounded quality factor estimation for various TF file formats."""

    @pytest.mark.parametrize(
        "tf_file,expected_rounded_qf,expected_qf",
        [
            pytest.param(TF_AVG, 3.0, 3.35, id="TF_AVG"),
            pytest.param(TF_AVG_NEWER, 4.0, 3.525, id="TF_AVG_NEWER"),
            pytest.param(TF_AVG_TIPPER, 2.0, 1.5, id="TF_AVG_TIPPER"),
            pytest.param(TF_EDI_CGG, 5.0, 4.85, id="TF_EDI_CGG"),
            pytest.param(TF_EDI_EMPOWER, 5.0, 4.9, id="TF_EDI_EMPOWER"),
            pytest.param(TF_EDI_METRONIX, 4.0, 3.9, id="TF_EDI_METRONIX"),
            pytest.param(TF_EDI_NO_ERROR, 2.0, 2.35, id="TF_EDI_NO_ERROR"),
            pytest.param(TF_EDI_PHOENIX, 4.0, 4.2, id="TF_EDI_PHOENIX"),
            pytest.param(TF_EDI_QUANTEC, 5.0, 4.75, id="TF_EDI_QUANTEC"),
            pytest.param(TF_EDI_RHO_ONLY, 4.0, 3.75, id="TF_EDI_RHO_ONLY"),
            pytest.param(TF_EDI_SPECTRA, 4.0, 4.0, id="TF_EDI_SPECTRA"),
            pytest.param(TF_JFILE, 1.0, 1.35, id="TF_JFILE"),
            pytest.param(TF_POOR_XML, 3.0, 2.8, id="TF_POOR_XML"),
            pytest.param(TF_XML, 4.0, 4.4, id="TF_XML"),
            pytest.param(
                TF_XML_COMPLETE_REMOTE_INFO, 4.0, 4.0, id="TF_XML_COMPLETE_REMOTE_INFO"
            ),
            pytest.param(
                TF_XML_MULTIPLE_ATTACHMENTS, 3.0, 3.35, id="TF_XML_MULTIPLE_ATTACHMENTS"
            ),
            pytest.param(TF_XML_NO_SITE_LAYOUT, 4.0, 3.5, id="TF_XML_NO_SITE_LAYOUT"),
            pytest.param(
                TF_XML_WITH_DERIVED_QUANTITIES,
                4.0,
                4.0,
                id="TF_XML_WITH_DERIVED_QUANTITIES",
            ),
            pytest.param(TF_ZMM, 4.0, 3.65, id="TF_ZMM"),
            pytest.param(TF_ZSS_TIPPER, 4.0, 3.75, id="TF_ZSS_TIPPER"),
        ],
    )
    def test_unrounded_quality_factor(
        self, mt_object, tf_file, expected_rounded_qf, expected_qf
    ):
        """Test that unrounded quality factor matches expected value."""
        mt_object.read(tf_file)
        qf = mt_object.estimate_tf_quality(round_qf=False)
        assert qf == pytest.approx(expected_qf, rel=5e-2)


class TestTFQualityFactorBothModes:
    """Test both rounded and unrounded modes in a single test."""

    @pytest.mark.parametrize(
        "tf_file,expected_rounded_qf,expected_qf",
        [
            pytest.param(TF_AVG, 3.0, 3.35, id="TF_AVG"),
            pytest.param(TF_EDI_CGG, 5.0, 4.85, id="TF_EDI_CGG"),
            pytest.param(TF_JFILE, 1.0, 1.35, id="TF_JFILE"),
            pytest.param(TF_XML, 4.0, 4.4, id="TF_XML"),
        ],
    )
    def test_rounded_vs_unrounded(
        self, mt_object, tf_file, expected_rounded_qf, expected_qf, subtests
    ):
        """Test that both rounded and unrounded modes produce expected results."""
        mt_object.read(tf_file)

        with subtests.test(msg="rounded"):
            qf_rounded = mt_object.estimate_tf_quality(round_qf=True)
            assert qf_rounded == expected_rounded_qf

        with subtests.test(msg="unrounded"):
            qf_unrounded = mt_object.estimate_tf_quality(round_qf=False)
            assert qf_unrounded == pytest.approx(expected_qf, rel=5e-2)

        with subtests.test(msg="rounding relationship"):
            # Rounded value should be close to unrounded (within 0.5 due to rounding)
            assert abs(qf_rounded - qf_unrounded) <= 0.5


class TestTFQualityFactorRange:
    """Test that quality factors are within valid ranges."""

    @pytest.mark.parametrize(
        "tf_file",
        [
            TF_AVG,
            TF_AVG_NEWER,
            TF_AVG_TIPPER,
            TF_EDI_CGG,
            TF_EDI_EMPOWER,
            TF_EDI_METRONIX,
            TF_EDI_NO_ERROR,
            TF_EDI_PHOENIX,
            TF_EDI_QUANTEC,
            TF_EDI_RHO_ONLY,
            TF_EDI_SPECTRA,
            TF_JFILE,
            TF_POOR_XML,
            TF_XML,
            TF_XML_COMPLETE_REMOTE_INFO,
            TF_XML_MULTIPLE_ATTACHMENTS,
            TF_XML_NO_SITE_LAYOUT,
            TF_XML_WITH_DERIVED_QUANTITIES,
            TF_ZMM,
            TF_ZSS_TIPPER,
        ],
        ids=[
            "TF_AVG",
            "TF_AVG_NEWER",
            "TF_AVG_TIPPER",
            "TF_EDI_CGG",
            "TF_EDI_EMPOWER",
            "TF_EDI_METRONIX",
            "TF_EDI_NO_ERROR",
            "TF_EDI_PHOENIX",
            "TF_EDI_QUANTEC",
            "TF_EDI_RHO_ONLY",
            "TF_EDI_SPECTRA",
            "TF_JFILE",
            "TF_POOR_XML",
            "TF_XML",
            "TF_XML_COMPLETE_REMOTE_INFO",
            "TF_XML_MULTIPLE_ATTACHMENTS",
            "TF_XML_NO_SITE_LAYOUT",
            "TF_XML_WITH_DERIVED_QUANTITIES",
            "TF_ZMM",
            "TF_ZSS_TIPPER",
        ],
    )
    def test_quality_factor_range(self, mt_object, tf_file, subtests):
        """Test that quality factors are within valid range (1-5)."""
        mt_object.read(tf_file)

        with subtests.test(msg="unrounded_range"):
            qf = mt_object.estimate_tf_quality(round_qf=False)
            assert 1.0 <= qf <= 5.0, f"Quality factor {qf} out of range [1, 5]"

        with subtests.test(msg="rounded_range"):
            qf_rounded = mt_object.estimate_tf_quality(round_qf=True)
            assert qf_rounded in [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ], f"Rounded quality factor {qf_rounded} not in [1, 2, 3, 4, 5]"

        with subtests.test(msg="is_numeric"):
            qf = mt_object.estimate_tf_quality()
            assert isinstance(
                qf, (int, float, np.number)
            ), f"Quality factor should be numeric, got {type(qf)}"


class TestTFQualityFactorConsistency:
    """Test consistency of quality factor estimation."""

    def test_multiple_calls_same_result(self, mt_object):
        """Test that multiple calls to estimate_tf_quality return same result."""
        mt_object.read(TF_EDI_CGG)

        qf1 = mt_object.estimate_tf_quality()
        qf2 = mt_object.estimate_tf_quality()
        qf3 = mt_object.estimate_tf_quality()

        assert qf1 == qf2 == qf3, "Multiple calls should return same quality factor"

    def test_rounded_idempotent(self, mt_object):
        """Test that rounding is idempotent."""
        mt_object.read(TF_EDI_CGG)

        qf_rounded1 = mt_object.estimate_tf_quality(round_qf=True)
        # Shouldn't need to round again, but verify it's stable
        qf_rounded2 = mt_object.estimate_tf_quality(round_qf=True)

        assert qf_rounded1 == qf_rounded2


class TestTFQualityFactorDefaultBehavior:
    """Test default behavior of estimate_tf_quality."""

    def test_default_is_unrounded(self, mt_object):
        """Test that default behavior returns unrounded quality factor."""
        mt_object.read(TF_EDI_CGG)

        qf_default = mt_object.estimate_tf_quality()
        qf_explicit_false = mt_object.estimate_tf_quality(round_qf=False)

        assert (
            qf_default == qf_explicit_false
        ), "Default should be same as round_qf=False"

    def test_default_vs_rounded(self, mt_object):
        """Test that default differs from rounded when appropriate."""
        mt_object.read(TF_AVG)  # Has qf=3.35, rounded=3.0

        qf_default = mt_object.estimate_tf_quality()
        qf_rounded = mt_object.estimate_tf_quality(round_qf=True)

        # For this file, they should differ
        assert qf_default != qf_rounded, "Default and rounded should differ for TF_AVG"


class TestTFQualityFactorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_quality_file(self, mt_object):
        """Test file with lowest quality factor (TF_JFILE)."""
        mt_object.read(TF_JFILE)

        qf = mt_object.estimate_tf_quality()
        qf_rounded = mt_object.estimate_tf_quality(round_qf=True)

        assert qf_rounded == 1.0, "Lowest quality should round to 1"
        assert qf >= 1.0, "Quality factor should be at least 1"

    def test_maximum_quality_files(self, subtests):
        """Test files with highest quality factors."""
        high_quality_files = [
            (TF_EDI_CGG, 4.85),
            (TF_EDI_EMPOWER, 4.9),
        ]

        for tf_file, expected_qf in high_quality_files:
            with subtests.test(msg=tf_file.name):
                # Create fresh MT object for each file
                mt_obj = MT()
                mt_obj.read(tf_file)
                qf = mt_obj.estimate_tf_quality()
                qf_rounded = mt_obj.estimate_tf_quality(round_qf=True)

                assert qf_rounded == 5.0, f"{tf_file.name} should round to 5"
                assert qf == pytest.approx(expected_qf, rel=5e-2)


class TestTFQualityFactorSpecialCases:
    """Test special cases from the TF file list."""

    def test_poor_quality_xml(self, mt_object):
        """Test TF_POOR_XML which has lower quality."""
        mt_object.read(TF_POOR_XML)

        qf = mt_object.estimate_tf_quality()

        assert qf == pytest.approx(
            2.8, rel=5e-2
        ), "TF_POOR_XML should have quality factor around 2.8"

    def test_no_error_edi(self, mt_object):
        """Test TF_EDI_NO_ERROR which has no error information."""
        mt_object.read(TF_EDI_NO_ERROR)

        qf = mt_object.estimate_tf_quality()
        qf_rounded = mt_object.estimate_tf_quality(round_qf=True)

        assert qf == pytest.approx(
            2.35, rel=5e-2
        ), "TF_EDI_NO_ERROR should have quality factor around 2.35"
        assert qf_rounded == 2.0, "TF_EDI_NO_ERROR should round to 2"

    def test_tipper_files(self, subtests):
        """Test files that include tipper data."""
        tipper_files = [
            (TF_AVG_TIPPER, 1.45, 2.0),
            (TF_ZSS_TIPPER, 3.75, 4.0),
        ]

        for tf_file, expected_qf, expected_rounded in tipper_files:
            with subtests.test(msg=tf_file.name):
                # Create fresh MT object for each file
                mt_obj = MT()
                mt_obj.read(tf_file)
                qf = mt_obj.estimate_tf_quality()
                qf_rounded = mt_obj.estimate_tf_quality(round_qf=True)

                assert qf == pytest.approx(expected_qf, rel=5e-2)
                assert qf_rounded == expected_rounded


class TestTFQualityFactorRobustness:
    """Test robustness and error handling."""

    def test_repeated_loading(self):
        """Test that loading different files produces different quality factors."""
        # Load first file
        mt1 = MT()
        mt1.read(TF_EDI_CGG)
        qf1 = mt1.estimate_tf_quality()

        # Load second file with fresh object (different period sizes)
        mt2 = MT()
        mt2.read(TF_JFILE)
        qf2 = mt2.estimate_tf_quality()

        # Quality factors should be different
        assert qf1 != qf2, "Different files should have different quality factors"

        # Should match expected values
        assert qf1 == pytest.approx(4.85, rel=1e-2)
        assert qf2 == pytest.approx(1.35, rel=1e-2)


class TestTFQualityFactorParallelSafety:
    """Test thread safety and parallel execution compatibility."""

    def test_independent_mt_objects(self):
        """Test that separate MT objects are independent."""
        mt1 = MT()
        mt2 = MT()

        mt1.read(TF_EDI_CGG)
        mt2.read(TF_JFILE)

        qf1 = mt1.estimate_tf_quality()
        qf2 = mt2.estimate_tf_quality()

        # Should have different quality factors
        assert qf1 != qf2

        # Should match expected values
        assert qf1 == pytest.approx(4.85, rel=5e-2)
        assert qf2 == pytest.approx(1.35, rel=5e-2)


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
