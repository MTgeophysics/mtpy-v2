"""Shared impedance-domain calculations used by Z and Dataset accessors."""

from __future__ import annotations

import numpy as np


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
