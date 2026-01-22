# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 23:25:57 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import cmath
import copy
import math

import numpy as np

import mtpy.utils.calculator as MTcc

from .base import TFBase


# =============================================================================


class Tipper(TFBase):
    """
    Tipper class.

    Errors are given as standard deviations (sqrt(VAR)).

    Parameters
    ----------
    tipper : np.ndarray, optional
        Tipper array in the shape of [Tx, Ty] (nf, 1, 2), by default None
    tipper_error : np.ndarray, optional
        Array of estimated tipper errors in the shape of [Tx, Ty] (nf, 1, 2),
        by default None
    frequency : np.ndarray, optional
        Array of frequencies corresponding to the tipper elements (nf),
        by default None
    tipper_model_error : np.ndarray, optional
        Array of model errors in the shape of [Tx, Ty] (nf, 1, 2),
        by default None

    """

    def __init__(
        self,
        tipper: np.ndarray | None = None,
        tipper_error: np.ndarray | None = None,
        frequency: np.ndarray | None = None,
        tipper_model_error: np.ndarray | None = None,
    ) -> None:
        """Initialize tipper object."""
        super().__init__(
            tf=tipper,
            tf_error=tipper_error,
            tf_model_error=tipper_model_error,
            frequency=frequency,
            _name="tipper",
            _expected_shape=(1, 2),
            inputs=["x", "y"],
            outputs=["z"],
        )

    # --- tipper ----
    @property
    def tipper(self) -> np.ndarray | None:
        """Tipper array."""
        if self._has_tf():
            return self._dataset.transfer_function.values

    @tipper.setter
    def tipper(self, tipper: np.ndarray | None) -> None:
        """
        Set tipper array.

        Parameters
        ----------
        tipper : np.ndarray or None
            Tipper array in the shape of [Tx, Ty] (nf, 1, 2)

        """

        old_shape = None
        if self._has_tf():
            old_shape = self._dataset.transfer_function.shape
        elif self._has_frequency():
            old_shape = (
                self.frequency.size,
                self._expected_shape[0],
                self._expected_shape[1],
            )

        tipper = self._validate_array_input(tipper, "complex", old_shape)
        if tipper is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf=tipper)
        else:
            self._dataset["transfer_function"].loc[self.comps] = tipper

    # ----tipper error---------------
    @property
    def tipper_error(self) -> np.ndarray | None:
        """Tipper error."""
        if self._has_tf_error():
            return self._dataset.transfer_function_error.values

    @tipper_error.setter
    def tipper_error(self, tipper_error: np.ndarray | None) -> None:
        """
        Set tipper error array.

        Parameters
        ----------
        tipper_error : np.ndarray or None
            Array of estimated tipper errors in the shape of [Tx, Ty] (nf, 1, 2)

        """

        old_shape = None
        if not self._has_tf_error():
            old_shape = self._dataset.transfer_function_error.shape
        elif self._has_frequency():
            old_shape = (
                self.frequency.size,
                self._expected_shape[0],
                self._expected_shape[1],
            )

        tipper_error = self._validate_array_input(tipper_error, "float", old_shape)
        if tipper_error is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf_error=tipper_error)
        else:
            self._dataset["transfer_function_error"].loc[self.comps] = tipper_error

    # ----tipper model error---------------------------------------------------------
    @property
    def tipper_model_error(self) -> np.ndarray | None:
        """Tipper model error."""
        if self._has_tf_model_error():
            return self._dataset.transfer_function_model_error.values

    @tipper_model_error.setter
    def tipper_model_error(self, tipper_model_error: np.ndarray | None) -> None:
        """
        Set tipper model error array.

        Parameters
        ----------
        tipper_model_error : np.ndarray or None
            Array of estimated tipper model errors in the shape of [Tx, Ty] (nf, 1, 2)

        """

        old_shape = None
        if not self._has_tf_error():
            old_shape = self._dataset.transfer_function_error.shape
        elif self._has_frequency():
            old_shape = (
                self.frequency.size,
                self._expected_shape[0],
                self._expected_shape[1],
            )
        tipper_model_error = self._validate_array_input(
            tipper_model_error, "float", old_shape
        )
        if tipper_model_error is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf_error=tipper_model_error)
        else:
            self._dataset["transfer_function_model_error"].loc[
                self.comps
            ] = tipper_model_error

    # ----amplitude and phase
    def _compute_amp_phase_error(
        self, error: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Compute amplitude and phase errors.

        Parameters
        ----------
        error : np.ndarray or None
            Error array

        Returns
        -------
        tuple of np.ndarray or None
            Amplitude error and phase error arrays

        """
        amplitude_error = None
        phase_error = None
        if error is not None:
            amplitude_error = np.zeros(self.tipper_error.shape)
            phase_error = np.zeros(self.tipper_error.shape)
            for idx_f in range(len(self.tipper)):
                for jj in range(2):
                    if type(self.tipper) == np.ma.core.MaskedArray:
                        if self.tipper.mask[idx_f, 0, jj]:
                            continue
                    r_error, phi_error = MTcc.propagate_error_rect2polar(
                        np.real(self.tipper[idx_f, 0, jj]),
                        error[idx_f, 0, jj],
                        np.imag(self.tipper[idx_f, 0, jj]),
                        error[idx_f, 0, jj],
                    )

                    amplitude_error[idx_f, 0, jj] = r_error
                    phase_error[idx_f, 0, jj] = phi_error

        return amplitude_error, phase_error

    def set_amp_phase(self, r: np.ndarray, phi: np.ndarray) -> None:
        """
        Set values for amplitude (r) and phase (phi).

        Parameters
        ----------
        r : np.ndarray
            Amplitude array
        phi : np.ndarray
            Phase array in degrees

        Notes
        -----
        Updates the attributes tipper and tipper_error.

        """

        if self.tipper is not None:
            tipper_new = copy.copy(self.tipper)

            if self.tipper.shape != r.shape:
                self.logger.error(
                    'Error - shape of "r" array does not match shape of '
                    + "tipper array: %s ; %s" % (str(r.shape), str(self.tipper.shape))
                )
                return
            if self.tipper.shape != phi.shape:
                self.logger.error(
                    'Error - shape of "phi" array does not match shape of '
                    + "tipper array: %s ; %s" % (str(phi.shape), str(self.tipper.shape))
                )
                return
        else:
            tipper_new = np.zeros(r.shape, "complex")

            if r.shape != phi.shape:
                self.logger.error(
                    'Error - shape of "phi" array does not match shape '
                    + 'of "r" array: %s ; %s' % (str(phi.shape), str(r.shape))
                )
                return
        # assert real array:
        if np.linalg.norm(np.imag(r)) != 0:
            self.logger.error('Error - array "r" is not real valued !')
            return
        if np.linalg.norm(np.imag(phi)) != 0:
            self.logger.error('Error - array "phi" is not real valued !')
            return
        for idx_f in range(len(r)):
            for jj in range(2):
                tipper_new[idx_f, 0, jj] = cmath.rect(
                    r[idx_f, 0, jj],
                    math.radians(phi[idx_f, 0, jj]),
                )
        self.tipper = tipper_new

    # ---------------------------------
    # properties
    @property
    def amplitude(self) -> np.ndarray | None:
        """Amplitude of tipper."""
        if self._has_tf():
            return np.abs(self.tipper)

    @property
    def phase(self) -> np.ndarray | None:
        """Phase of tipper in degrees."""
        if self._has_tf():
            return np.rad2deg(np.angle(self.tipper))

    @property
    def amplitude_error(self) -> np.ndarray | None:
        """Amplitude error."""
        if self._has_tf_error():
            return self._compute_amp_phase_error(self.tipper_error)[0]

    @property
    def phase_error(self) -> np.ndarray | None:
        """Phase error in degrees."""
        if self._has_tf_error():
            return self._compute_amp_phase_error(self.tipper_error)[1]

    @property
    def amplitude_model_error(self) -> np.ndarray | None:
        """Amplitude model error."""
        if self._has_tf_model_error():
            return self._compute_amp_phase_error(self.tipper_model_error)[0]

    @property
    def phase_model_error(self) -> np.ndarray | None:
        """Phase model error in degrees."""
        if self._has_tf_model_error():
            return self._compute_amp_phase_error(self.tipper_model_error)[1]

    # ----magnitude and direction----------------------------------------------

    def set_mag_direction(
        self,
        mag_real: np.ndarray,
        ang_real: np.ndarray,
        mag_imag: np.ndarray,
        ang_imag: np.ndarray,
    ) -> None:
        """
        Compute tipper from magnitude and direction of real and imaginary components.

        Parameters
        ----------
        mag_real : np.ndarray
            Magnitude of real component
        ang_real : np.ndarray
            Angle of real component
        mag_imag : np.ndarray
            Magnitude of imaginary component
        ang_imag : np.ndarray
            Angle of imaginary component

        Notes
        -----
        Updates tipper. No error propagation yet.

        """

        self.tipper[:, 0, 0].real = np.sqrt(
            (mag_real**2 * np.arctan(ang_real) ** 2) / (1 - np.arctan(ang_real) ** 2)
        )

        self.tipper[:, 0, 1].real = np.sqrt(
            mag_real**2 / (1 - np.arctan(ang_real) ** 2)
        )

        self.tipper[:, 0, 0].imag = np.sqrt(
            (mag_imag**2 * np.arctan(ang_imag) ** 2) / (1 - np.arctan(ang_imag) ** 2)
        )

        self.tipper[:, 0, 1].imag = np.sqrt(
            mag_imag**2 / (1 - np.arctan(ang_imag) ** 2)
        )
        # for consistency recalculate mag and angle
        self.compute_mag_direction()
        self.compute_amp_phase()

    @property
    def mag_real(self) -> np.ndarray | None:
        """Magnitude of real component."""
        if self._has_tf():
            return np.sqrt(
                self.tipper[:, 0, 0].real ** 2 + self.tipper[:, 0, 1].real ** 2
            )

    @property
    def mag_imag(self) -> np.ndarray | None:
        """Magnitude of imaginary component."""
        if self._has_tf():
            return np.sqrt(
                self.tipper[:, 0, 0].imag ** 2 + self.tipper[:, 0, 1].imag ** 2
            )

    @property
    def angle_real(self) -> np.ndarray | None:
        """Angle of real component in degrees."""
        if self._has_tf():
            return np.rad2deg(
                np.arctan2(self.tipper[:, 0, 1].real, self.tipper[:, 0, 0].real)
            )

    @property
    def angle_imag(self) -> np.ndarray | None:
        """Angle of imaginary component in degrees."""
        if self._has_tf():
            return np.rad2deg(
                np.arctan2(self.tipper[:, 0, 1].imag, self.tipper[:, 0, 0].imag)
            )

    @property
    def mag_error(self) -> np.ndarray | None:
        """Magnitude error."""
        if self._has_tf_error():
            return np.sqrt(
                self.tipper_error[:, 0, 0] ** 2 + self.tipper_error[:, 0, 1] ** 2
            )

    @property
    def angle_error(self) -> np.ndarray | None:
        """Angle error in degrees."""
        if self._has_tf_error():
            return np.abs(
                np.rad2deg(
                    np.arctan(self.tipper_error[:, 0, 0] / self.tipper_error[:, 0, 1])
                )
                - 45
            )

    @property
    def mag_model_error(self) -> np.ndarray | None:
        """Magnitude model error."""
        if self._has_tf_model_error():
            return np.sqrt(
                self.tipper_model_error[:, 0, 0] ** 2
                + self.tipper_model_error[:, 0, 1] ** 2
            )

    @property
    def angle_model_error(self) -> np.ndarray | None:
        """Angle model error in degrees."""
        if self._has_tf_model_error():
            return np.abs(
                np.rad2deg(
                    np.arctan(
                        self.tipper_model_error[:, 0, 0]
                        / self.tipper_model_error[:, 0, 1]
                    )
                )
                - 45
            )
