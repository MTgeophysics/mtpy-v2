"""Shared impedance-domain calculations used by Z and Dataset accessors."""

from __future__ import annotations

import numpy as np

import mtpy.utils.calculator as MTcc


def compute_resistivity(
    z_mt: np.ndarray | None, frequency: np.ndarray
) -> np.ndarray | None:
    """Compute apparent resistivity from impedance in mt units."""
    if z_mt is None:
        return None

    return np.apply_along_axis(
        lambda values: np.abs(values) ** 2 / frequency * 0.2,
        0,
        z_mt,
    )


def compute_phase(z_mt: np.ndarray | None) -> np.ndarray | None:
    """Compute impedance phase in degrees from impedance in mt units."""
    if z_mt is None:
        return None

    return np.rad2deg(np.angle(z_mt))


def compute_resistivity_error(
    z_mt: np.ndarray | None,
    z_error_mt: np.ndarray | None,
    frequency: np.ndarray,
) -> np.ndarray | None:
    """Compute apparent resistivity error from impedance and standard deviation."""
    if z_mt is None or z_error_mt is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.apply_along_axis(
            lambda values: values / frequency * 0.2,
            0,
            2 * z_error_mt * np.abs(z_mt),
        )


def compute_phase_error(
    z_mt: np.ndarray | None,
    z_error: np.ndarray | None,
) -> np.ndarray | None:
    """Compute impedance phase error in degrees from complex impedance error."""
    if z_mt is None or z_error is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.degrees(np.arctan(z_error / np.abs(z_mt)))


def compute_impedance_error(
    res_error: np.ndarray | None,
    phase_error: np.ndarray | None,
    frequency: np.ndarray,
) -> np.ndarray | None:
    """Compute impedance error from apparent resistivity and phase errors."""
    if res_error is None:
        return None

    return np.abs(
        np.sqrt(frequency * (res_error.T) * 250).T * np.tan(np.radians(phase_error))
    )


def compute_tipper_amplitude(tipper: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper amplitude."""
    if tipper is None:
        return None

    return np.abs(tipper)


def compute_tipper_phase(tipper: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper phase in degrees."""
    if tipper is None:
        return None

    return np.rad2deg(np.angle(tipper))


def compute_tipper_amp_phase_error(
    tipper: np.ndarray | None,
    error: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute tipper amplitude and phase errors."""
    if tipper is None or error is None:
        return None, None

    amplitude_error = np.zeros(error.shape)
    phase_error = np.zeros(error.shape)

    for idx_f in range(len(tipper)):
        for jj in range(2):
            if isinstance(tipper, np.ma.core.MaskedArray) and tipper.mask[idx_f, 0, jj]:
                continue

            r_error, phi_error = MTcc.propagate_error_rect2polar(
                np.real(tipper[idx_f, 0, jj]),
                error[idx_f, 0, jj],
                np.imag(tipper[idx_f, 0, jj]),
                error[idx_f, 0, jj],
            )
            amplitude_error[idx_f, 0, jj] = r_error
            phase_error[idx_f, 0, jj] = phi_error

    return amplitude_error, phase_error


def compute_tipper_magnitude_real(tipper: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper real-component magnitude."""
    if tipper is None:
        return None

    return np.sqrt(tipper[:, 0, 0].real ** 2 + tipper[:, 0, 1].real ** 2)


def compute_tipper_magnitude_imag(tipper: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper imaginary-component magnitude."""
    if tipper is None:
        return None

    return np.sqrt(tipper[:, 0, 0].imag ** 2 + tipper[:, 0, 1].imag ** 2)


def compute_tipper_angle_real(tipper: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper real-component angle in degrees."""
    if tipper is None:
        return None

    return np.rad2deg(np.arctan2(tipper[:, 0, 1].real, tipper[:, 0, 0].real))


def compute_tipper_angle_imag(tipper: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper imaginary-component angle in degrees."""
    if tipper is None:
        return None

    return np.rad2deg(np.arctan2(tipper[:, 0, 1].imag, tipper[:, 0, 0].imag))


def compute_tipper_magnitude_error(error: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper magnitude error."""
    if error is None:
        return None

    return np.sqrt(error[:, 0, 0] ** 2 + error[:, 0, 1] ** 2)


def compute_tipper_angle_error(error: np.ndarray | None) -> np.ndarray | None:
    """Compute tipper angle error in degrees."""
    if error is None:
        return None

    return np.abs(np.rad2deg(np.arctan(error[:, 0, 0] / error[:, 0, 1])) - 45)


def compute_phase_tensor(z: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor from impedance."""
    if z is None:
        return None

    pt_array = np.zeros_like(z, dtype=float)
    z_real = np.real(z)
    z_imag = np.imag(z)

    with np.errstate(divide="ignore", invalid="ignore"):
        det_real = np.linalg.det(z_real)
        pt_array[:, 0, 0] = (
            z_real[:, 1, 1] * z_imag[:, 0, 0] - z_real[:, 0, 1] * z_imag[:, 1, 0]
        )
        pt_array[:, 0, 1] = (
            z_real[:, 1, 1] * z_imag[:, 0, 1] - z_real[:, 0, 1] * z_imag[:, 1, 1]
        )
        pt_array[:, 1, 0] = (
            z_real[:, 0, 0] * z_imag[:, 1, 0] - z_real[:, 1, 0] * z_imag[:, 0, 0]
        )
        pt_array[:, 1, 1] = (
            z_real[:, 0, 0] * z_imag[:, 1, 1] - z_real[:, 1, 0] * z_imag[:, 0, 1]
        )
        pt_array = np.apply_along_axis(lambda values: values / det_real, 0, pt_array)

    return pt_array


def compute_phase_tensor_error(
    z: np.ndarray | None,
    z_error: np.ndarray | None,
) -> np.ndarray | None:
    """Compute phase tensor error from impedance and impedance error."""
    if z is None or z_error is None:
        return None

    pt_array = compute_phase_tensor(z)
    pt_error = np.zeros_like(pt_array)
    z_real = np.real(z)
    z_imag = np.imag(z)

    with np.errstate(divide="ignore", invalid="ignore"):
        det_real = np.abs(np.linalg.det(z_real))
        pt_error[:, 0, 0] = (
            np.abs(-pt_array[:, 0, 0] * z_real[:, 1, 1] * z_error[:, 0, 0])
            + np.abs(pt_array[:, 0, 0] * z_real[:, 0, 1] * z_error[:, 1, 0])
            + np.abs(
                (z_imag[:, 0, 0] - pt_array[:, 0, 0] * z_real[:, 0, 0])
                * z_error[:, 1, 1]
            )
            + np.abs(
                (-z_imag[:, 1, 0] + pt_array[:, 0, 0] * z_real[:, 1, 0])
                * z_error[:, 0, 1]
            )
            + np.abs(z_real[:, 1, 1] * z_error[:, 0, 0])
            + np.abs(z_real[:, 0, 1] * z_error[:, 1, 0])
        ) / det_real

        pt_error[:, 0, 1] = (
            np.abs(-pt_array[:, 0, 1] * z_real[:, 1, 1] * z_error[:, 0, 0])
            + np.abs(pt_array[:, 0, 1] * z_real[:, 0, 1] * z_error[:, 1, 0])
            + np.abs(
                (z_imag[:, 0, 1] - pt_array[:, 0, 1] * z_real[:, 0, 0])
                * z_error[:, 1, 1]
            )
            + np.abs(
                (-z_imag[:, 1, 1] + pt_array[:, 0, 1] * z_real[:, 1, 0])
                * z_error[:, 0, 1]
            )
            + np.abs(z_real[:, 1, 1] * z_error[:, 0, 1])
            + np.abs(z_real[:, 0, 1] * z_error[:, 1, 1])
        ) / det_real

        pt_error[:, 1, 0] = (
            np.abs(
                (z_imag[:, 1, 0] - pt_array[:, 1, 0] * z_real[:, 1, 1])
                * z_error[:, 0, 0]
            )
            + np.abs(pt_array[:, 1, 0] * z_real[:, 1, 0] * z_error[:, 0, 1])
            + np.abs(
                (-z_imag[:, 0, 0] + pt_array[:, 1, 0] * z_real[:, 0, 1])
                * z_error[:, 1, 0]
            )
            + np.abs(-pt_array[:, 1, 0] * z_real[:, 0, 0] * z_error[:, 1, 1])
            + np.abs(z_real[:, 0, 0] * z_error[:, 1, 0])
            + np.abs(-z_real[:, 1, 0] * z_error[:, 0, 0])
        ) / det_real

        pt_error[:, 1, 1] = (
            np.abs(
                (z_imag[:, 1, 1] - pt_array[:, 1, 1] * z_real[:, 1, 1])
                * z_error[:, 0, 0]
            )
            + np.abs(pt_array[:, 1, 1] * z_real[:, 1, 0] * z_error[:, 0, 1])
            + np.abs(
                (-z_imag[:, 0, 1] + pt_array[:, 1, 1] * z_real[:, 0, 1])
                * z_error[:, 1, 0]
            )
            + np.abs(-pt_array[:, 1, 1] * z_real[:, 0, 0] * z_error[:, 1, 1])
            + np.abs(z_real[:, 0, 0] * z_error[:, 1, 1])
            + np.abs(-z_real[:, 1, 0] * z_error[:, 0, 1])
        ) / det_real

    return pt_error


def compute_pt_alpha(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor alpha in degrees."""
    if pt is None:
        return None

    return np.degrees(
        0.5 * np.arctan2(pt[:, 0, 1] + pt[:, 1, 0], pt[:, 0, 0] - pt[:, 1, 1])
    )


def compute_pt_beta(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor beta in degrees."""
    if pt is None:
        return None

    return np.degrees(
        0.5 * np.arctan2(pt[:, 0, 1] - pt[:, 1, 0], pt[:, 0, 0] + pt[:, 1, 1])
    )


def compute_pt_pi1(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute the first principal phase tensor invariant."""
    if pt is None:
        return None

    return 0.5 * np.sqrt(
        (pt[:, 0, 0] - pt[:, 1, 1]) ** 2 + (pt[:, 0, 1] + pt[:, 1, 0]) ** 2
    )


def compute_pt_pi2(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute the second principal phase tensor invariant."""
    if pt is None:
        return None

    return 0.5 * np.sqrt(
        (pt[:, 0, 0] + pt[:, 1, 1]) ** 2 + (pt[:, 0, 1] - pt[:, 1, 0]) ** 2
    )


def compute_pt_phimin(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute minimum phase tensor angle in degrees."""
    pi1 = compute_pt_pi1(pt)
    pi2 = compute_pt_pi2(pt)
    if pi1 is None or pi2 is None:
        return None

    return np.degrees(np.arctan(pi2 - pi1))


def compute_pt_phimax(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute maximum phase tensor angle in degrees."""
    pi1 = compute_pt_pi1(pt)
    pi2 = compute_pt_pi2(pt)
    if pi1 is None or pi2 is None:
        return None

    return np.degrees(np.arctan(pi2 + pi1))


def compute_pt_skew(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor skew in degrees."""
    return compute_pt_beta(pt)


def compute_pt_azimuth(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor azimuth in degrees."""
    alpha = compute_pt_alpha(pt)
    beta = compute_pt_beta(pt)
    if alpha is None or beta is None:
        return None

    return (alpha - beta) % 360


def compute_pt_ellipticity(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor ellipticity."""
    phimax = compute_pt_phimax(pt)
    phimin = compute_pt_phimin(pt)
    if phimax is None or phimin is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        return (phimax - phimin) / (phimax + phimin)


def compute_pt_eccentricity(pt: np.ndarray | None) -> np.ndarray | None:
    """Compute phase tensor eccentricity."""
    pi1 = compute_pt_pi1(pt)
    pi2 = compute_pt_pi2(pt)
    if pi1 is None or pi2 is None:
        return None

    return pi1 / pi2
