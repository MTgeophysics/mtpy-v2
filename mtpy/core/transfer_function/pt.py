#!/usr/bin/env python

"""
======================
Phase Tensor
======================

Following Caldwell et al, 2004


Originally written by Stephan Thiel in Matlab
translated to Python by Lars Krieger

Revised by J. Peacock 2022 to fit with version 2.

"""
# =============================================================================
# Imports
# =============================================================================
import copy

import numpy as np

from .base import TFBase


# =============================================================================


class PhaseTensor(TFBase):
    """
    Phase Tensor class.

    Methods include reading and writing from and to edi-objects, rotations,
    combinations of Z instances, as well as calculation of invariants,
    inverse, amplitude/phase.

    Phase tensor is a real array of the form (n_freq, 2, 2) with indices:
        - phase_tensor_xx: (0,0)
        - phase_tensor_xy: (0,1)
        - phase_tensor_yx: (1,0)
        - phase_tensor_yy: (1,1)

    All internal methods are based on Caldwell et al. (2004) and
    Bibby et al. (2005), which use the canonical cartesian 2D
    reference (x1, x2). However, all components, coordinates,
    and angles for in- and outputs are given in the geographical
    reference frame:
        - x-axis = North
        - y-axis = East
        - z-axis = Down

    Therefore, all results from using those methods are consistent
    (angles are referenced from North rather than x1).

    Parameters
    ----------
    z : np.ndarray, optional
        Impedance tensor array (n_frequency, 2, 2), by default None
    z_error : np.ndarray, optional
        Impedance tensor error array (n_frequency, 2, 2), by default None
    z_model_error : np.ndarray, optional
        Impedance tensor model error array (n_frequency, 2, 2), by default None
    frequency : np.ndarray, optional
        Frequency array (n_frequency), by default None
    pt : np.ndarray, optional
        Phase tensor array (n_frequency, 2, 2), by default None
    pt_error : np.ndarray, optional
        Phase tensor error array (n_frequency, 2, 2), by default None
    pt_model_error : np.ndarray, optional
        Phase tensor model error array (n_frequency, 2, 2), by default None

    """

    def __init__(
        self,
        z: np.ndarray | None = None,
        z_error: np.ndarray | None = None,
        z_model_error: np.ndarray | None = None,
        frequency: np.ndarray | None = None,
        pt: np.ndarray | None = None,
        pt_error: np.ndarray | None = None,
        pt_model_error: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            tf=pt,
            tf_error=pt_error,
            tf_model_error=pt_model_error,
            frequency=frequency,
            _name="phase_tensor",
            _tf_dtypes={
                "tf": float,
                "tf_error": float,
                "tf_model_error": float,
            },
        )

        if z is not None:
            self.pt = self._pt_from_z(z)
            if z_error is not None:
                self.pt_error = self._pt_error_from_z(z, z_error)
            if z_model_error is not None:
                self.pt_model_error = self._pt_error_from_z(z, z_model_error)

    def _pt_from_z(self, z: np.ndarray) -> np.ndarray | None:
        """
        Create phase tensor from impedance.

        Parameters
        ----------
        z : np.ndarray
            Impedance tensor array (n_frequency, 2, 2)

        Returns
        -------
        np.ndarray or None
            Phase tensor array (n_frequency, 2, 2)

        """
        old_shape = None
        if self._has_tf():
            old_shape = self._dataset.transfer_function.shape
        z = self._validate_array_input(z, "complex", old_shape)
        if z is None:
            return

        pt_array = np.zeros_like(z, dtype=float)

        z_real = np.real(z)
        z_imag = np.imag(z)

        with np.errstate(divide="ignore", invalid="ignore"):
            det_real = np.linalg.det(z_real)
            det_zero = np.where(det_real == 0)[0]
            if det_zero.shape[0] > 0:
                self.logger.debug(
                    f"z at index {det_zero} contains a singular matrix,"
                    " thus it cannot be converted into a phase tensor, setting to 0."
                )

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

            pt_array = np.apply_along_axis(lambda x: x / det_real, 0, pt_array)

        return pt_array

    def _pt_error_from_z(self, z: np.ndarray, z_error: np.ndarray) -> np.ndarray | None:
        """
        Calculate phase tensor error from impedance error.

        Parameters
        ----------
        z : np.ndarray
            Impedance tensor array (n_frequency, 2, 2)
        z_error : np.ndarray
            Impedance tensor error array (n_frequency, 2, 2)

        Returns
        -------
        np.ndarray or None
            Phase tensor error array (n_frequency, 2, 2)

        """

        pt_array = self._pt_from_z(z)

        old_shape = None
        if self._has_tf():
            old_shape = self._dataset.transfer_function.shape
        z = self._validate_array_input(z, "complex", old_shape)

        old_shape = None
        if not self._has_tf_error():
            old_shape = self._dataset.transfer_function_error.shape

        z_error = self._validate_array_input(z_error, "float", old_shape)
        if z_error is None:
            return

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

    @property
    def pt(self) -> np.ndarray | None:
        """Phase tensor array."""
        if self._has_tf():
            return self._dataset.transfer_function.values

    @pt.setter
    def pt(self, pt: np.ndarray | None) -> None:
        """
        Set phase tensor.

        Parameters
        ----------
        pt : np.ndarray or None
            Phase tensor array (n_frequencies, 2, 2)

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
        pt = self._validate_array_input(pt, self._tf_dtypes["tf"], old_shape)
        if pt is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf=pt)
        else:
            self._dataset["transfer_function"].loc[self.comps] = pt

    @property
    def pt_error(self) -> np.ndarray | None:
        """Phase tensor error."""
        if self._has_tf_error():
            return self._dataset.transfer_function_error.values

    @pt_error.setter
    def pt_error(self, pt_error: np.ndarray | None) -> None:
        """
        Set phase tensor error.

        Parameters
        ----------
        pt_error : np.ndarray or None
            Phase tensor error array (n_frequencies, 2, 2)

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

        pt_error = self._validate_array_input(pt_error, "float", old_shape)
        if pt_error is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf_error=pt_error)
        else:
            self._dataset["transfer_function_error"].loc[self.comps] = pt_error

    @property
    def pt_model_error(self) -> np.ndarray | None:
        """Phase tensor model error."""

        if self._has_tf_model_error():
            return self._dataset.transfer_function_model_error.values

    @pt_model_error.setter
    def pt_model_error(self, pt_model_error: np.ndarray | None) -> None:
        """
        Set phase tensor model error.

        Parameters
        ----------
        pt_model_error : np.ndarray or None
            Phase tensor model error array (n_frequencies, 2, 2)

        """
        old_shape = None
        if not self._has_tf_error():
            old_shape = self._dataset.transfer_function_model_error.shape

        pt_model_error = self._validate_array_input(pt_model_error, "float", old_shape)
        if pt_model_error is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf_model_error=pt_model_error)
        else:
            self._dataset["transfer_function_model_error"].loc[
                self.comps
            ] = pt_model_error

    # ==========================================================================
    #  define get methods for read only properties
    # ==========================================================================

    # ---trace-------------------------------------------------------------
    @property
    def trace(self) -> np.ndarray | None:
        """Trace of phase tensor."""
        if self.pt is None:
            return None
        return np.array([np.trace(i) for i in self.pt])

    @property
    def trace_error(self) -> np.ndarray | None:
        """Trace error of phase tensor."""

        if self._has_tf_error():
            tr_error = self.pt_error[:, 0, 0] + self.pt_error[:, 1, 1]
            return tr_error

    @property
    def trace_model_error(self) -> np.ndarray | None:
        """Trace model error of phase tensor."""

        if self._has_tf_model_error():
            tr_model_error = self.pt_error[:, 0, 0] + self.pt_error[:, 1, 1]
            return tr_model_error

    # ---alpha-------------------------------------------------------------
    @property
    def alpha(self) -> np.ndarray | None:
        """Principal axis angle (strike) of phase tensor in degrees."""

        if self.pt is None:
            return None
        return np.degrees(
            0.5
            * np.arctan2(
                self.pt[:, 0, 1] + self.pt[:, 1, 0],
                self.pt[:, 0, 0] - self.pt[:, 1, 1],
            )
        )

    @property
    def alpha_error(self) -> np.ndarray | None:
        """Principal axis angle error of phase tensor in degrees."""

        if self._has_tf_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                y = self.pt[:, 0, 1] + self.pt[:, 1, 0]
                yerr = np.sqrt(
                    self.pt_error[:, 0, 1] ** 2 + self.pt_error[:, 1, 0] ** 2
                )
                x = self.pt[:, 0, 0] - self.pt[:, 1, 1]
                xerr = np.sqrt(
                    self.pt_error[:, 0, 0] ** 2 + self.pt_error[:, 1, 1] ** 2
                )

                alpha_error = np.degrees(
                    0.5
                    / (x**2 + y**2)
                    * np.sqrt(y**2 * xerr**2 + x**2 * yerr**2)
                )
                return alpha_error

    @property
    def alpha_model_error(self) -> np.ndarray | None:
        """Principal axis angle model error of phase tensor in degrees."""

        if self._has_tf_model_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                y = self.pt[:, 0, 1] + self.pt[:, 1, 0]
                yerr = np.sqrt(
                    self.pt_model_error[:, 0, 1] ** 2
                    + self.pt_model_error[:, 1, 0] ** 2
                )
                x = self.pt[:, 0, 0] - self.pt[:, 1, 1]
                xerr = np.sqrt(
                    self.pt_model_error[:, 0, 0] ** 2
                    + self.pt_model_error[:, 1, 1] ** 2
                )

                alpha_model_error = np.degrees(
                    0.5
                    / (x**2 + y**2)
                    * np.sqrt(y**2 * xerr**2 + x**2 * yerr**2)
                )
                return alpha_model_error

    # ---beta-------------------------------------------------------------
    @property
    def beta(self) -> np.ndarray | None:
        """3D-dimensionality angle Beta (invariant) of phase tensor in degrees."""

        if self.pt is None:
            return None
        return np.degrees(
            0.5
            * np.arctan2(
                self.pt[:, 0, 1] - self.pt[:, 1, 0],
                self.pt[:, 0, 0] + self.pt[:, 1, 1],
            )
        )

    @property
    def beta_error(self) -> np.ndarray | None:
        """3D-dimensionality angle error Beta of phase tensor in degrees."""

        if self._has_tf_error():
            y = self.pt[:, 0, 1] - self.pt[:, 1, 0]
            yerr = np.sqrt(self.pt_error[:, 0, 1] ** 2 + self.pt_error[:, 1, 0] ** 2)
            x = self.pt[:, 0, 0] + self.pt[:, 1, 1]
            xerr = np.sqrt(self.pt_error[:, 0, 0] ** 2 + self.pt_error[:, 1, 1] ** 2)

            beta_error = np.degrees(
                0.5
                / (x**2 + y**2)
                * np.sqrt(y**2 * xerr**2 + x**2 * yerr**2)
            )
            return beta_error

    @property
    def beta_model_error(self) -> np.ndarray | None:
        """3D-dimensionality angle model error Beta of phase tensor in degrees."""

        if self._has_tf_error():
            y = self.pt[:, 0, 1] - self.pt[:, 1, 0]
            yerr = np.sqrt(
                self.pt_model_error[:, 0, 1] ** 2 + self.pt_model_error[:, 1, 0] ** 2
            )
            x = self.pt[:, 0, 0] + self.pt[:, 1, 1]
            xerr = np.sqrt(
                self.pt_model_error[:, 0, 0] ** 2 + self.pt_model_error[:, 1, 1] ** 2
            )

            beta_model_error = np.degrees(
                0.5
                / (x**2 + y**2)
                * np.sqrt(y**2 * xerr**2 + x**2 * yerr**2)
            )
            return beta_model_error

    # ---skew-------------------------------------------------------------
    @property
    def skew(self) -> np.ndarray | None:
        """3D-dimensionality skew angle of phase tensor in degrees."""
        return self.beta

    @property
    def skew_error(self) -> np.ndarray | None:
        """3D-dimensionality skew angle error of phase tensor in degrees."""
        return self.beta_error

    @property
    def skew_model_error(self) -> np.ndarray | None:
        """3D-dimensionality skew angle model error of phase tensor in degrees."""
        return self.beta_model_error

    # ---azimuth (strike angle)-------------------------------------------------
    @property
    def azimuth(self) -> np.ndarray | None:
        """Azimuth angle related to geoelectric strike in degrees."""

        if self.pt is None:
            return None
        return (self.alpha - self.beta) % 360

    @property
    def azimuth_error(self) -> np.ndarray | None:
        """Azimuth angle error related to geoelectric strike in degrees."""
        if self._has_tf_error():
            return np.sqrt(abs(self.alpha_error + self.beta_error))

    @property
    def azimuth_model_error(self) -> np.ndarray | None:
        """Azimuth angle model error related to geoelectric strike in degrees."""
        if self._has_tf_model_error():
            return np.sqrt(abs(self.alpha_model_error + self.beta_model_error))

    # ---ellipticity----------------------------------------------------
    @property
    def ellipticity(self) -> np.ndarray | None:
        """Ellipticity of the phase tensor, related to dimensionality."""

        if self.pt is None:
            return None
        result = None
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (self.phimax - self.phimin) / (self.phimax + self.phimin)
        return result

    @property
    def ellipticity_error(self) -> np.ndarray | None:
        """Ellipticity error of the phase tensor, related to dimensionality."""
        if self._has_tf_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    self.ellipticity
                    * np.sqrt(self.phimax_error + self.phimin_error)
                    * np.sqrt(
                        (1 / (self.phimax - self.phimin)) ** 2
                        + (1 / (self.phimax + self.phimin)) ** 2
                    )
                )

    @property
    def ellipticity_model_error(self) -> np.ndarray | None:
        """Ellipticity model error of the phase tensor, related to dimensionality."""
        if self._has_tf_model_error():
            return (
                self.ellipticity
                * np.sqrt(self.phimax_model_error + self.phimin_model_error)
                * np.sqrt(
                    (1 / (self.phimax - self.phimin)) ** 2
                    + (1 / (self.phimax + self.phimin)) ** 2
                )
            )

    # ---det-------------------------------------------------------------
    @property
    def det(self) -> np.ndarray | None:
        """Determinant of phase tensor."""
        if self.pt is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.array([np.linalg.det(pt_arr) for pt_arr in self.pt])

    @property
    def det_error(self) -> np.ndarray | None:
        """Determinant error of phase tensor."""
        if self._has_tf_error():
            return (
                np.abs(self.pt[:, 1, 1] * self.pt_error[:, 0, 0])
                + np.abs(self.pt[:, 0, 0] * self.pt_error[:, 1, 1])
                + np.abs(self.pt[:, 0, 1] * self.pt_error[:, 1, 0])
                + np.abs(self.pt[:, 1, 0] * self.pt_error[:, 0, 1])
            )

    @property
    def det_model_error(self) -> np.ndarray | None:
        """Determinant model error of phase tensor."""
        if self._has_tf_model_error():
            return (
                np.abs(self.pt[:, 1, 1] * self.pt_model_error[:, 0, 0])
                + np.abs(self.pt[:, 0, 0] * self.pt_model_error[:, 1, 1])
                + np.abs(self.pt[:, 0, 1] * self.pt_model_error[:, 1, 0])
                + np.abs(self.pt[:, 1, 0] * self.pt_model_error[:, 0, 1])
            )

    # ---principle component 1----------------------------------------------
    @property
    def _pi1(self) -> np.ndarray:
        """
        Pi1 calculated according to Bibby et al. 2005.

        Pi1 = 0.5 * sqrt(PT[0,0] - PT[1,1])**2 + (PT[0,1] + PT[1,0])**2).

        """
        # after bibby et al. 2005

        return 0.5 * np.sqrt(
            (self.pt[:, 0, 0] - self.pt[:, 1, 1]) ** 2
            + (self.pt[:, 0, 1] + self.pt[:, 1, 0]) ** 2
        )

    @property
    def _pi1_error(self) -> np.ndarray | None:
        """Pi1 error."""
        if self._has_tf_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    1.0
                    / (4 * self._pi1)
                    * np.sqrt(
                        (self.pt[:, 0, 0] - self.pt[:, 1, 1]) ** 2
                        * (self.pt_error[:, 0, 0] ** 2 + self.pt_error[:, 1, 1] ** 2)
                        + (self.pt[:, 0, 1] + self.pt[:, 1, 0]) ** 2
                        * (self.pt_error[:, 0, 1] ** 2 + self.pt_error[:, 1, 0] ** 2)
                    )
                )

    @property
    def _pi1_model_error(self) -> np.ndarray | None:
        """Pi1 model error."""
        if self._has_tf_model_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    1.0
                    / (4 * self._pi1)
                    * np.sqrt(
                        (self.pt[:, 0, 0] - self.pt[:, 1, 1]) ** 2
                        * (self.pt_error[:, 0, 0] ** 2 + self.pt_error[:, 1, 1] ** 2)
                        + (self.pt[:, 0, 1] + self.pt[:, 1, 0]) ** 2
                        * (
                            self.pt_model_error[:, 0, 1] ** 2
                            + self.pt_model_error[:, 1, 0] ** 2
                        )
                    )
                )

    # ---principle component 2----------------------------------------------
    @property
    def _pi2(self) -> np.ndarray:
        """
        Pi2 calculated according to Bibby et al. 2005.

        Pi2 = 0.5 * sqrt(PT[0,0] + PT[1,1])**2 + (PT[0,1] - PT[1,0])**2).

        """
        # after bibby et al. 2005

        return 0.5 * np.sqrt(
            (self.pt[:, 0, 0] + self.pt[:, 1, 1]) ** 2
            + (self.pt[:, 0, 1] - self.pt[:, 1, 0]) ** 2
        )

    @property
    def _pi2_error(self) -> np.ndarray | None:
        """Pi2 error."""

        if self._has_tf_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    1.0
                    / (4 * self._pi2)
                    * np.sqrt(
                        (self.pt[:, 0, 0] + self.pt[:, 1, 1]) ** 2
                        * (self.pt_error[:, 0, 0] ** 2 + self.pt_error[:, 1, 1] ** 2)
                        + (self.pt[:, 0, 1] - self.pt[:, 1, 0]) ** 2
                        * (self.pt_error[:, 0, 1] ** 2 + self.pt_error[:, 1, 0] ** 2)
                    )
                )

    @property
    def _pi2_model_error(self) -> np.ndarray | None:
        """Pi2 model error."""

        if self._has_tf_model_error():
            with np.errstate(divide="ignore", invalid="ignore"):
                return (
                    1.0
                    / (4 * self._pi2)
                    * np.sqrt(
                        (self.pt[:, 0, 0] + self.pt[:, 1, 1]) ** 2
                        * (
                            self.pt_model_error[:, 0, 0] ** 2
                            + self.pt_model_error[:, 1, 1] ** 2
                        )
                        + (self.pt[:, 0, 1] - self.pt[:, 1, 0]) ** 2
                        * (
                            self.pt_model_error[:, 0, 1] ** 2
                            + self.pt_model_error[:, 1, 0] ** 2
                        )
                    )
                )

    # ---phimin----------------------------------------------
    @property
    def phimin(self) -> np.ndarray | None:
        """
        Minimum phase calculated according to Bibby et al. 2005.

        Phi_min = Pi2 - Pi1.

        """

        if self._has_tf():
            return np.degrees(np.arctan(self._pi2 - self._pi1))

    @property
    def phimin_error(self) -> np.ndarray | None:
        """Minimum phase error."""
        if self._has_tf_error():
            return np.degrees(
                np.arctan(np.sqrt(self._pi2_error**2 + self._pi1_error**2))
            )

    @property
    def phimin_model_error(self) -> np.ndarray | None:
        """Minimum phase model error."""
        if self._has_tf_model_error():
            return np.degrees(
                np.arctan(
                    np.sqrt(self._pi2_model_error**2 + self._pi1_model_error**2)
                )
            )

    # ---phimax----------------------------------------------
    @property
    def phimax(self) -> np.ndarray | None:
        """
        Maximum phase calculated according to Bibby et al. 2005.

        Phi_max = Pi2 + Pi1.

        """

        if self._has_tf():
            return np.degrees(np.arctan(self._pi2 + self._pi1))

    @property
    def phimax_error(self) -> np.ndarray | None:
        """Maximum phase error."""
        if self._has_tf_error():
            return np.degrees(
                np.arctan(np.sqrt(self._pi2_error**2 + self._pi1_error**2))
            )

    @property
    def phimax_model_error(self) -> np.ndarray | None:
        """Maximum phase model error."""
        if self._has_tf_model_error():
            return np.degrees(
                np.arctan(
                    np.sqrt(self._pi2_model_error**2 + self._pi1_model_error**2)
                )
            )

    # ---only 1d----------------------------------------------
    @property
    def only1d(self) -> np.ndarray | None:
        """
        Return phase tensor in 1D form.

        If phase tensor is not 1D per se, the diagonal elements are set to zero,
        the off-diagonal elements keep their signs, but their absolute is
        set to the mean of the original phase tensor off-diagonal absolutes.

        """

        if self._has_tf():
            pt_1d = copy.copy(self.pt)
            pt_1d[:, 0, 1] = 0
            pt_1d[:, 1, 0] = 0

            mean_1d = 0.5 * (pt_1d[:, 0, 0] + pt_1d[:, 1, 1])
            pt_1d[:, 0, 0] = mean_1d
            pt_1d[:, 1, 1] = mean_1d
            return pt_1d

    # ---only 2d----------------------------------------------
    @property
    def only2d(self) -> np.ndarray | None:
        """
        Return phase tensor in 2D form.

        If phase tensor is not 2D per se, the diagonal elements are set to zero.

        """
        if self._has_tf():
            pt_2d = copy.copy(self.pt)

            pt_2d[:, 0, 1] = 0
            pt_2d[:, 1, 0] = 0

            pt_2d[:, 0, 0] = self.phimax[:]
            pt_2d[:, 1, 1] = self.phimin[:]
            return pt_2d

    @property
    def eccentricity(self) -> np.ndarray | None:
        """Eccentricity estimation of dimensionality."""

        if self._has_tf():
            return self._pi1 / self._pi2

    @property
    def eccentricity_error(self) -> np.ndarray | None:
        """Error in eccentricity estimation."""
        if self._has_tf_error():
            return (
                np.sqrt(
                    (self._pi1_error / self._pi1) ** 2
                    + (self._pi2_error / self._pi2) ** 2
                )
                * self.eccentricity
            )

    @property
    def eccentricity_model_error(self) -> np.ndarray | None:
        """Model error in eccentricity estimation."""
        if self._has_tf_model_error():
            return (
                np.sqrt(
                    (self._pi1_model_error / self._pi1) ** 2
                    + (self._pi2_model_error / self._pi2) ** 2
                )
                * self.eccentricity
            )
