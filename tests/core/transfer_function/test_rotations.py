"""
tests/test_mt_rotation.py

Pytest suite validating MT impedance tensor rotation in mtpy-v2.

Strategy
--------
We define a synthetic impedance tensor with geoelectric strike = 0 degrees
(aligned with geographic north). For a perfect 2D tensor aligned with the
measurement axes, Zxx = Zyy = 0 and all structural information lives in
Zxy / Zyx.

We then simulate "as measured in geomagnetic coordinates" by rotating that
tensor by +declination (geomagnetic frame is rotated `declination` degrees
clockwise from geographic north, by the standard positive-east convention).
Rotating by -declination should recover the original geographic (strike=0)
tensor, and rotating by +declination again should recover the geomagnetic
version — a full round trip.

Two layers of tests:

1. Low-level tests against `mtpy.utils.calculator` functions directly,
   checked against an INDEPENDENT reference implementation (re-derived by
   hand from the explicit rotation formulas, not copy-pasted from the
   module under test) to guard against e.g. sign or transpose bugs that
   would otherwise be invisible to a self-referential test.

2. Integration tests against `TFBase.rotate()`, exercising the full
   object-oriented rotation path including bookkeeping of
   `rotation_angle` and propagation of tf_error.

IMPORTANT: adjust TFBASE_IMPORT below to match your actual mtpy-v2 module
layout if it differs.
"""

import numpy as np
import pytest

from mtpy.utils.calculator import (
    get_rotation_matrix,
    rotate_matrix_with_errors,
    rotate_matrix_errors,
)

# --- adjust this import to match your package layout ---
from mtpy.core.transfer_function.base import TFBase

ATOL_CENTRAL = 1e-8  # tolerance for central-value (impedance) comparisons
ATOL_STRIKE0 = 1e-6  # tolerance for "should be exactly zero" diagonal terms


# =============================================================================
# Independent reference implementation (NOT importing code under test)
# =============================================================================
def _reference_rotate(Z, theta_deg):
    """
    Hand-derived reference rotation, independent of calculator.py, used to
    cross-check the implementation under test. Implements the clockwise MT
    convention: Z' = R(theta) Z R(theta)^T with
    R(theta) = [[cos, sin], [-sin, cos]].

    Expanded explicitly (not via matrix multiplication) so that a bug in
    the matrix-multiplication implementation under test (e.g. a transpose
    error) would not be masked by reusing the same linear-algebra approach.
    """
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    Zxx, Zxy = Z[..., 0, 0], Z[..., 0, 1]
    Zyx, Zyy = Z[..., 1, 0], Z[..., 1, 1]

    Zxx_p = Zxx * c**2 + (Zxy + Zyx) * s * c + Zyy * s**2
    Zxy_p = Zxy * c**2 - (Zxx - Zyy) * s * c - Zyx * s**2
    Zyx_p = Zyx * c**2 - (Zxx - Zyy) * s * c - Zxy * s**2
    Zyy_p = Zyy * c**2 - (Zxy + Zyx) * s * c + Zxx * s**2

    out = np.zeros_like(Z)
    out[..., 0, 0] = Zxx_p
    out[..., 0, 1] = Zxy_p
    out[..., 1, 0] = Zyx_p
    out[..., 1, 1] = Zyy_p
    return out


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def declination_deg():
    """N15E: magnetic north is 15 degrees east of true north."""
    return 15.0


@pytest.fixture
def n_periods():
    return 4


@pytest.fixture
def periods(n_periods):
    return np.logspace(-2, 2, n_periods)


@pytest.fixture
def strike0_impedance(n_periods):
    """
    Synthetic 2D impedance tensor, strike = 0 (aligned with geographic
    north). Zxx = Zyy = 0 exactly; Zxy/Zyx vary with period to exercise
    batch rotation across frequencies, not just a single static tensor.
    """
    Z = np.zeros((n_periods, 2, 2), dtype=complex)
    for k in range(n_periods):
        Z[k, 0, 0] = 0.0 + 0.0j
        Z[k, 0, 1] = (50 + 10 * k) + (30 + 5 * k) * 1j
        Z[k, 1, 0] = -(40 + 8 * k) - (25 + 4 * k) * 1j
        Z[k, 1, 1] = 0.0 + 0.0j
    return Z


@pytest.fixture
def strike0_errors(strike0_impedance):
    """
    Absolute errors: 5% of magnitude on the structural (off-diagonal)
    terms, and a small noise floor on the diagonal terms (whose true
    value is zero, so a relative error would be degenerate).
    """
    err = np.zeros(strike0_impedance.shape, dtype=float)
    err[..., 0, 1] = 0.05 * np.abs(strike0_impedance[..., 0, 1])
    err[..., 1, 0] = 0.05 * np.abs(strike0_impedance[..., 1, 0])
    err[..., 0, 0] = 0.5
    err[..., 1, 1] = 0.5
    return err


@pytest.fixture
def geomagnetic_impedance(strike0_impedance, declination_deg):
    """
    Simulated "as measured in geomagnetic coordinates" tensor: the fixed
    physical (geographic, strike=0) tensor expressed in a frame rotated
    +declination clockwise from geographic north.
    """
    out = np.zeros_like(strike0_impedance)
    for k in range(strike0_impedance.shape[0]):
        rotated, _ = rotate_matrix_with_errors(
            strike0_impedance[k], declination_deg, clockwise=True
        )
        out[k] = rotated
    return out


# =============================================================================
# Low-level tests: rotation matrix / calculator functions
# =============================================================================
class TestRotationMatrixLowLevel:
    def test_clockwise_matrix_structure(self):
        """R(theta) clockwise = [[cos, sin], [-sin, cos]]."""
        R = get_rotation_matrix(30, clockwise=True)
        expected = np.array(
            [
                [np.cos(np.radians(30)), np.sin(np.radians(30))],
                [-np.sin(np.radians(30)), np.cos(np.radians(30))],
            ]
        )
        np.testing.assert_allclose(R, expected, atol=ATOL_CENTRAL)

    def test_rotation_matrix_is_orthogonal(self):
        """R R^T = I for any angle (required for the similarity transform
        to preserve the tensor's physical invariants, e.g. determinant)."""
        for angle in [0, 15, 37, 90, 180, 270, -15]:
            R = get_rotation_matrix(angle, clockwise=True)
            np.testing.assert_allclose(R @ R.T, np.eye(2), atol=ATOL_CENTRAL)

    @pytest.mark.parametrize("angle", [0, 7.5, 15, 33, -15, 45, 90])
    def test_rotate_matches_independent_reference(self, strike0_impedance, angle):
        """Cross-check rotate_matrix_with_errors against an independently
        derived formula, not the matrix-multiplication code path itself."""
        for k in range(strike0_impedance.shape[0]):
            Z = strike0_impedance[k]
            rotated, _ = rotate_matrix_with_errors(Z, angle, clockwise=True)
            expected = _reference_rotate(Z, angle)
            np.testing.assert_allclose(rotated, expected, atol=ATOL_CENTRAL)

    def test_determinant_invariant_under_rotation(
        self, strike0_impedance, declination_deg
    ):
        """det(Z) must be unchanged by rotation since det(R) = 1. This is
        a strong, convention-independent sanity check."""
        for k in range(strike0_impedance.shape[0]):
            Z = strike0_impedance[k]
            rotated, _ = rotate_matrix_with_errors(Z, declination_deg, clockwise=True)
            np.testing.assert_allclose(
                np.linalg.det(Z), np.linalg.det(rotated), atol=ATOL_CENTRAL
            )


# =============================================================================
# Core requested behavior: forward rotation, undo declination, round trip
# =============================================================================
class TestDeclinationRotation:
    def test_geomagnetic_fixture_is_not_strike_zero(self, geomagnetic_impedance):
        """Sanity check on the fixture itself: rotating a strike-0 tensor
        by a nonzero declination should genuinely mix the components, i.e.
        Zxx/Zyy should no longer be ~0 in the geomagnetic frame. If this
        fails, the fixture isn't exercising anything meaningful."""
        for k in range(geomagnetic_impedance.shape[0]):
            assert abs(geomagnetic_impedance[k, 0, 0]) > 1.0
            assert abs(geomagnetic_impedance[k, 1, 1]) > 1.0

    def test_undo_declination_recovers_geographic_strike_zero(
        self, geomagnetic_impedance, strike0_impedance, declination_deg
    ):
        """Core test: rotating the geomagnetic-frame tensor by -declination
        should recover the known geoelectric-strike-0 tensor, including
        Zxx ~ 0 and Zyy ~ 0."""
        for k in range(geomagnetic_impedance.shape[0]):
            recovered, _ = rotate_matrix_with_errors(
                geomagnetic_impedance[k], -declination_deg, clockwise=True
            )
            np.testing.assert_allclose(
                recovered, strike0_impedance[k], atol=ATOL_CENTRAL
            )
            assert abs(recovered[0, 0]) < ATOL_STRIKE0
            assert abs(recovered[1, 1]) < ATOL_STRIKE0

    def test_full_round_trip_recovers_geomagnetic_tensor(
        self, geomagnetic_impedance, declination_deg
    ):
        """Rotate geomagnetic -> geographic -> geomagnetic and confirm we
        recover the original geomagnetic-frame tensor exactly (to float
        precision). This validates that -D and +D are true inverses of
        one another under this rotation convention."""
        for k in range(geomagnetic_impedance.shape[0]):
            geo, _ = rotate_matrix_with_errors(
                geomagnetic_impedance[k], -declination_deg, clockwise=True
            )
            back_to_mag, _ = rotate_matrix_with_errors(
                geo, declination_deg, clockwise=True
            )
            np.testing.assert_allclose(
                back_to_mag, geomagnetic_impedance[k], atol=ATOL_CENTRAL
            )

    @pytest.mark.parametrize("declination", [0, 5, 15, -10, 45, 90, -33.3])
    def test_round_trip_holds_for_various_declinations(
        self, strike0_impedance, declination
    ):
        """Broader coverage: round trip should hold regardless of the sign
        or magnitude of declination, including 0 (no rotation) and 90
        (geomagnetic frame perpendicular to geographic, an edge case)."""
        for k in range(strike0_impedance.shape[0]):
            Z = strike0_impedance[k]
            mag, _ = rotate_matrix_with_errors(Z, declination, clockwise=True)
            geo, _ = rotate_matrix_with_errors(mag, -declination, clockwise=True)
            np.testing.assert_allclose(geo, Z, atol=ATOL_CENTRAL)


# =============================================================================
# Error propagation
# =============================================================================
class TestErrorPropagation:
    def test_error_propagation_matches_manual_formula(
        self, strike0_errors, declination_deg
    ):
        """Cross-check rotate_matrix_errors against the quadrature-sum
        formula worked out by hand, independent of the implementation."""
        for k in range(strike0_errors.shape[0]):
            err = strike0_errors[k]
            result = rotate_matrix_errors(err, declination_deg)

            phi = np.radians(declination_deg % 360)
            c, s = np.cos(phi), np.sin(phi)
            expected = np.zeros_like(err)
            expected[0, 0] = np.sqrt(
                (c**2 * err[0, 0]) ** 2
                + (c * s * err[0, 1]) ** 2
                + (c * s * err[1, 0]) ** 2
                + (s**2 * err[1, 1]) ** 2
            )
            expected[0, 1] = np.sqrt(
                (c**2 * err[0, 1]) ** 2
                + (c * s * err[1, 1]) ** 2
                + (c * s * err[0, 0]) ** 2
                + (s**2 * err[1, 0]) ** 2
            )
            expected[1, 0] = np.sqrt(
                (c**2 * err[1, 0]) ** 2
                + (c * s * err[1, 1]) ** 2
                + (c * s * err[0, 0]) ** 2
                + (s**2 * err[0, 1]) ** 2
            )
            expected[1, 1] = np.sqrt(
                (c**2 * err[1, 1]) ** 2
                + (c * s * err[0, 1]) ** 2
                + (c * s * err[1, 0]) ** 2
                + (s**2 * err[0, 0]) ** 2
            )
            np.testing.assert_allclose(result, expected, atol=ATOL_CENTRAL)

    def test_error_propagation_is_angle_sign_symmetric(self, strike0_errors):
        """The error formula depends on angle only through squared cos/sin
        terms, so rotating by +D or -D of the same magnitude must give the
        identical propagated error. This is a documented property, not a
        bug, and a regression here would signal an accidental formula
        change."""
        D = 15.0
        for k in range(strike0_errors.shape[0]):
            err = strike0_errors[k]
            plus = rotate_matrix_errors(err, D)
            minus = rotate_matrix_errors(err, -D)
            np.testing.assert_allclose(plus, minus, atol=ATOL_CENTRAL)

    def test_error_round_trip_is_NOT_exact(self, strike0_errors):
        """
        Known limitation (documented in calculator.py's comment block):
        rotating the error matrix forward and then backward does NOT
        exactly recover the original error, because the propagation
        formula uses quadrature sums (absolute magnitudes) rather than a
        true linear invertible transform. This test documents and locks
        in that expected behavior so a future "fix" that silently changes
        it doesn't go unnoticed. We assert the discrepancy stays within a
        sane bound rather than being wildly off.
        """
        D = 15.0
        for k in range(strike0_errors.shape[0]):
            err = strike0_errors[k]
            forward = rotate_matrix_errors(err, D)
            back = rotate_matrix_errors(forward, -D)

            # Confirm it's close but not exact -- sanity on the bound,
            # not a false claim of exact invertibility.
            assert not np.allclose(back, err, atol=1e-10)
            np.testing.assert_allclose(back, err, rtol=2.0)

    def test_errors_remain_nonnegative_and_finite(
        self, strike0_errors, declination_deg
    ):
        """Basic sanity: propagated errors should never be negative or
        NaN/inf, regardless of rotation angle."""
        for angle in [0, declination_deg, -declination_deg, 90, 180]:
            for k in range(strike0_errors.shape[0]):
                result = rotate_matrix_errors(strike0_errors[k], angle)
                assert np.all(result >= 0)
                assert np.all(np.isfinite(result))


# =============================================================================
# Integration tests against TFBase.rotate()
# =============================================================================
class TestTFBaseRotationIntegration:
    @pytest.fixture
    def tf_geomagnetic(self, geomagnetic_impedance, strike0_errors, periods):
        """A TFBase object holding data 'as collected' in geomagnetic
        coordinates."""
        freq = 1.0 / periods
        return TFBase(tf=geomagnetic_impedance, tf_error=strike0_errors, frequency=freq)

    def test_rotate_to_geographic_recovers_strike_zero(
        self, tf_geomagnetic, strike0_impedance, declination_deg
    ):
        tf_geo = tf_geomagnetic.rotate(
            -declination_deg, coordinate_reference_frame="ned"
        )
        values = tf_geo._dataset.transfer_function.values
        np.testing.assert_allclose(values, strike0_impedance, atol=ATOL_CENTRAL)
        assert np.all(np.abs(values[:, 0, 0]) < ATOL_STRIKE0)
        assert np.all(np.abs(values[:, 1, 1]) < ATOL_STRIKE0)

    def test_rotate_does_not_mutate_original_when_not_inplace(
        self, tf_geomagnetic, geomagnetic_impedance, declination_deg
    ):
        original = tf_geomagnetic._dataset.transfer_function.values.copy()
        _ = tf_geomagnetic.rotate(
            -declination_deg, coordinate_reference_frame="ned", inplace=False
        )
        np.testing.assert_allclose(
            tf_geomagnetic._dataset.transfer_function.values, original
        )

    def test_full_round_trip_through_tfbase(
        self, tf_geomagnetic, geomagnetic_impedance, declination_deg
    ):
        """Rotate to geographic, then back to geomagnetic, via the public
        TFBase API, and confirm we recover the original data."""
        tf_geo = tf_geomagnetic.rotate(
            -declination_deg, coordinate_reference_frame="ned"
        )
        tf_back = tf_geo.rotate(declination_deg, coordinate_reference_frame="ned")
        np.testing.assert_allclose(
            tf_back._dataset.transfer_function.values,
            geomagnetic_impedance,
            atol=ATOL_CENTRAL,
        )

    def test_rotation_angle_bookkeeping(self, tf_geomagnetic, declination_deg):
        """
        rotation_angle should accumulate additively. Note: the angle
        passed through `validate_angle` is first reduced mod 360, so a
        rotation of -15 is stored as 345, not -15. A full round trip
        (-D then +D) should therefore sum to 360, not 0 -- check modulo
        360 rather than asserting a literal zero.
        """
        tf_geo = tf_geomagnetic.rotate(
            -declination_deg, coordinate_reference_frame="ned", inplace=False
        )
        tf_back = tf_geo.rotate(
            declination_deg, coordinate_reference_frame="ned", inplace=False
        )
        net_angle = np.asarray(tf_back.rotation_angle) % 360
        np.testing.assert_allclose(net_angle, 0.0, atol=1e-6)

    def test_inplace_rotation_mutates_self(self, tf_geomagnetic, declination_deg):
        result = tf_geomagnetic.rotate(
            -declination_deg, coordinate_reference_frame="ned", inplace=True
        )
        assert result is None
        values = tf_geomagnetic._dataset.transfer_function.values
        assert np.all(np.abs(values[:, 0, 0]) < ATOL_STRIKE0)

    def test_per_period_angle_array(
        self, tf_geomagnetic, strike0_impedance, declination_deg, n_periods
    ):
        """Exercise the branch where a different rotation angle is given
        per period (e.g. frequency-dependent strike correction). Using a
        uniform array of -declination should match the scalar case."""
        angles = np.full(n_periods, -declination_deg)
        tf_geo = tf_geomagnetic.rotate(angles, coordinate_reference_frame="ned")
        np.testing.assert_allclose(
            tf_geo._dataset.transfer_function.values,
            strike0_impedance,
            atol=ATOL_CENTRAL,
        )

    def test_enu_frame_uses_inverse_sense(
        self, strike0_impedance, strike0_errors, periods, declination_deg
    ):
        """coordinate_reference_frame='enu' should rotate with the inverse
        (counterclockwise-form) matrix relative to 'ned'. Rotating a
        strike-0 tensor by +D in 'ned' then by +D again in 'enu' should
        return to the original, since enu uses R(D)^-1 for the same angle."""
        freq = 1.0 / periods
        tf = TFBase(tf=strike0_impedance, tf_error=strike0_errors, frequency=freq)
        tf_ned = tf.rotate(declination_deg, coordinate_reference_frame="ned")
        tf_back = tf_ned.rotate(declination_deg, coordinate_reference_frame="enu")
        np.testing.assert_allclose(
            tf_back._dataset.transfer_function.values,
            strike0_impedance,
            atol=ATOL_CENTRAL,
        )
