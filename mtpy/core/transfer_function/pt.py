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
from .tf_helpers import (
    compute_phase_tensor,
    compute_phase_tensor_error,
    compute_pt_alpha,
    compute_pt_alpha_error,
    compute_pt_azimuth,
    compute_pt_azimuth_error,
    compute_pt_beta,
    compute_pt_beta_error,
    compute_pt_det,
    compute_pt_det_error,
    compute_pt_eccentricity,
    compute_pt_eccentricity_error,
    compute_pt_ellipticity,
    compute_pt_ellipticity_error,
    compute_pt_phimax,
    compute_pt_phimax_error,
    compute_pt_phimin,
    compute_pt_phimin_error,
    compute_pt_pi1,
    compute_pt_pi1_error,
    compute_pt_pi2,
    compute_pt_pi2_error,
    compute_pt_skew,
    compute_pt_skew_error,
    compute_pt_trace,
    compute_pt_trace_error,
)


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
        return compute_phase_tensor(z)

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

        old_shape = None
        if self._has_tf():
            old_shape = self._dataset.transfer_function.shape
        z = self._validate_array_input(z, "complex", old_shape)

        old_shape = None
        if not self._has_tf_error():
            old_shape = self._dataset.transfer_function_error.shape

        z_error = self._validate_array_input(z_error, "float", old_shape)
        return compute_phase_tensor_error(z, z_error)

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
        return compute_pt_trace(self.pt)

    @property
    def trace_error(self) -> np.ndarray | None:
        """Trace error of phase tensor."""
        return compute_pt_trace_error(self.pt_error)

    @property
    def trace_model_error(self) -> np.ndarray | None:
        """Trace model error of phase tensor."""
        return compute_pt_trace_error(self.pt_model_error)

    # ---alpha-------------------------------------------------------------
    @property
    def alpha(self) -> np.ndarray | None:
        """Principal axis angle (strike) of phase tensor in degrees."""
        return compute_pt_alpha(self.pt)

    @property
    def alpha_error(self) -> np.ndarray | None:
        """Principal axis angle error of phase tensor in degrees."""
        return compute_pt_alpha_error(self.pt, self.pt_error)

    @property
    def alpha_model_error(self) -> np.ndarray | None:
        """Principal axis angle model error of phase tensor in degrees."""
        return compute_pt_alpha_error(self.pt, self.pt_model_error)

    # ---beta-------------------------------------------------------------
    @property
    def beta(self) -> np.ndarray | None:
        """3D-dimensionality angle Beta (invariant) of phase tensor in degrees."""
        return compute_pt_beta(self.pt)

    @property
    def beta_error(self) -> np.ndarray | None:
        """3D-dimensionality angle error Beta of phase tensor in degrees."""
        return compute_pt_beta_error(self.pt, self.pt_error)

    @property
    def beta_model_error(self) -> np.ndarray | None:
        """3D-dimensionality angle model error Beta of phase tensor in degrees."""
        return compute_pt_beta_error(self.pt, self.pt_model_error)

    # ---skew-------------------------------------------------------------
    @property
    def skew(self) -> np.ndarray | None:
        """3D-dimensionality skew angle of phase tensor in degrees."""
        return compute_pt_skew(self.pt)

    @property
    def skew_error(self) -> np.ndarray | None:
        """3D-dimensionality skew angle error of phase tensor in degrees."""
        return compute_pt_skew_error(self.pt, self.pt_error)

    @property
    def skew_model_error(self) -> np.ndarray | None:
        """3D-dimensionality skew angle model error of phase tensor in degrees."""
        return compute_pt_skew_error(self.pt, self.pt_model_error)

    # ---azimuth (strike angle)-------------------------------------------------
    @property
    def azimuth(self) -> np.ndarray | None:
        """Azimuth angle related to geoelectric strike in degrees."""
        return compute_pt_azimuth(self.pt)

    @property
    def azimuth_error(self) -> np.ndarray | None:
        """Azimuth angle error related to geoelectric strike in degrees."""
        return compute_pt_azimuth_error(self.pt, self.pt_error)

    @property
    def azimuth_model_error(self) -> np.ndarray | None:
        """Azimuth angle model error related to geoelectric strike in degrees."""
        return compute_pt_azimuth_error(self.pt, self.pt_model_error)

    # ---ellipticity----------------------------------------------------
    @property
    def ellipticity(self) -> np.ndarray | None:
        """Ellipticity of the phase tensor, related to dimensionality."""
        return compute_pt_ellipticity(self.pt)

    @property
    def ellipticity_error(self) -> np.ndarray | None:
        """Ellipticity error of the phase tensor, related to dimensionality."""
        return compute_pt_ellipticity_error(self.pt, self.pt_error)

    @property
    def ellipticity_model_error(self) -> np.ndarray | None:
        """Ellipticity model error of the phase tensor, related to dimensionality."""
        return compute_pt_ellipticity_error(self.pt, self.pt_model_error)

    # ---det-------------------------------------------------------------
    @property
    def det(self) -> np.ndarray | None:
        """Determinant of phase tensor."""
        return compute_pt_det(self.pt)

    @property
    def det_error(self) -> np.ndarray | None:
        """Determinant error of phase tensor."""
        return compute_pt_det_error(self.pt, self.pt_error)

    @property
    def det_model_error(self) -> np.ndarray | None:
        """Determinant model error of phase tensor."""
        return compute_pt_det_error(self.pt, self.pt_model_error)

    # ---principle component 1----------------------------------------------
    @property
    def _pi1(self) -> np.ndarray:
        """
        Pi1 calculated according to Bibby et al. 2005.

        Pi1 = 0.5 * sqrt(PT[0,0] - PT[1,1])**2 + (PT[0,1] + PT[1,0])**2).

        """
        # after bibby et al. 2005

        return compute_pt_pi1(self.pt)

    @property
    def _pi1_error(self) -> np.ndarray | None:
        """Pi1 error."""
        return compute_pt_pi1_error(self.pt, self.pt_error)

    @property
    def _pi1_model_error(self) -> np.ndarray | None:
        """Pi1 model error."""
        return compute_pt_pi1_error(self.pt, self.pt_model_error)

    # ---principle component 2----------------------------------------------
    @property
    def _pi2(self) -> np.ndarray:
        """
        Pi2 calculated according to Bibby et al. 2005.

        Pi2 = 0.5 * sqrt(PT[0,0] + PT[1,1])**2 + (PT[0,1] - PT[1,0])**2).

        """
        # after bibby et al. 2005

        return compute_pt_pi2(self.pt)

    @property
    def _pi2_error(self) -> np.ndarray | None:
        """Pi2 error."""
        return compute_pt_pi2_error(self.pt, self.pt_error)

    @property
    def _pi2_model_error(self) -> np.ndarray | None:
        """Pi2 model error."""
        return compute_pt_pi2_error(self.pt, self.pt_model_error)

    # ---phimin----------------------------------------------
    @property
    def phimin(self) -> np.ndarray | None:
        """
        Minimum phase calculated according to Bibby et al. 2005.

        Phi_min = Pi2 - Pi1.

        """
        return compute_pt_phimin(self.pt)

    @property
    def phimin_error(self) -> np.ndarray | None:
        """Minimum phase error."""
        return compute_pt_phimin_error(self.pt, self.pt_error)

    @property
    def phimin_model_error(self) -> np.ndarray | None:
        """Minimum phase model error."""
        return compute_pt_phimin_error(self.pt, self.pt_model_error)

    # ---phimax----------------------------------------------
    @property
    def phimax(self) -> np.ndarray | None:
        """
        Maximum phase calculated according to Bibby et al. 2005.

        Phi_max = Pi2 + Pi1.

        """
        return compute_pt_phimax(self.pt)

    @property
    def phimax_error(self) -> np.ndarray | None:
        """Maximum phase error."""
        return compute_pt_phimax_error(self.pt, self.pt_error)

    @property
    def phimax_model_error(self) -> np.ndarray | None:
        """Maximum phase model error."""
        return compute_pt_phimax_error(self.pt, self.pt_model_error)

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
        return compute_pt_eccentricity(self.pt)

    @property
    def eccentricity_error(self) -> np.ndarray | None:
        """Error in eccentricity estimation."""
        return compute_pt_eccentricity_error(self.pt, self.pt_error)

    @property
    def eccentricity_model_error(self) -> np.ndarray | None:
        """Model error in eccentricity estimation."""
        return compute_pt_eccentricity_error(self.pt, self.pt_model_error)
