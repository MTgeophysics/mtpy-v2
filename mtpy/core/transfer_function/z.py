#!/usr/bin/env python

"""
Z
===

Container for the Impedance Tensor

Originally written by Jared Peacock Lars Krieger
Updated 2022 by J. Peacock to work with new framework

"""

# =============================================================================
# Imports
# =============================================================================
import copy
from typing import Any

import numpy as np

from . import IMPEDANCE_UNITS, MT_TO_OHM_FACTOR
from .base import TFBase
from .pt import PhaseTensor
from .z_analysis import (
    calculate_depth_of_investigation,
    find_distortion,
    remove_distortion_from_z_object,
    ZInvariants,
)


# ==============================================================================
# Impedance Tensor Class
# ==============================================================================
class Z(TFBase):
    """
    Impedance tensor (Z) class.

    Z is a complex array of the form (n_frequency, 2, 2) with indices:
    - Zxx: (0,0)
    - Zxy: (0,1)
    - Zyx: (1,0)
    - Zyy: (1,1)

    All errors are given as standard deviations (sqrt(VAR)).

    Parameters
    ----------
    z : np.ndarray, optional
        Array containing complex impedance values (n_frequency, 2, 2)
    z_error : np.ndarray, optional
        Array containing error values (standard deviation) of impedance
        tensor elements (n_frequency, 2, 2)
    frequency : np.ndarray, optional
        Array of frequency values corresponding to impedance tensor
        elements (n_frequency)
    z_model_error : np.ndarray, optional
        Array containing model error values (n_frequency, 2, 2)
    units : str, optional
        Units for impedance: 'mt' [mV/km/nT] or 'ohm' [Ohms], by default 'mt'

    """

    def __init__(
        self,
        z: np.ndarray | None = None,
        z_error: np.ndarray | None = None,
        frequency: np.ndarray | None = None,
        z_model_error: np.ndarray | None = None,
        units: str = "mt",
    ) -> None:
        """
        Initialize an instance of the Z class.

        Parameters
        ----------
        z : np.ndarray, optional
            Array containing complex impedance values (n_frequency, 2, 2),
            by default None
        z_error : np.ndarray, optional
            Array containing error values (standard deviation) of impedance
            tensor elements (n_frequency, 2, 2), by default None
        frequency : np.ndarray, optional
            Array of frequency values corresponding to impedance tensor
            elements (n_frequency), by default None
        z_model_error : np.ndarray, optional
            Array containing model error values (n_frequency, 2, 2),
            by default None
        units : str, optional
            Units for impedance: 'mt' [mV/km/nT] or 'ohm' [Ohms],
            by default 'mt'

        """

        self._ohm_factor = MT_TO_OHM_FACTOR
        self._unit_factors = IMPEDANCE_UNITS
        self.units = units

        # if units input is ohms, then we want to scale them to mt units that
        # way the underlying data is consistent in [mV/km/nT]
        if z is not None:
            z = z * self._scale_factor
        if z_error is not None:
            z_error = z_error * self._scale_factor
        if z_model_error is not None:
            z_model_error = z_model_error * self._scale_factor

        super().__init__(
            tf=z,
            tf_error=z_error,
            tf_model_error=z_model_error,
            frequency=frequency,
            _name="impedance",
        )

    @property
    def units(self) -> str:
        """Impedance units."""
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        """Set impedance units (options: 'mt' or 'ohm')."""
        if not isinstance(value, str):
            raise TypeError("Units input must be a string.")
        if value.lower() not in self._unit_factors.keys():
            raise ValueError(f"{value} is not an acceptable unit for impedance.")

        self._units = value

    @property
    def _scale_factor(self) -> float:
        """Unit scale factor."""
        return self._unit_factors[self._units]

    @property
    def z(self) -> np.ndarray | None:
        """Impedance tensor array (nfrequency, 2, 2)."""
        if self._has_tf():
            return self._dataset.transfer_function.values / self._scale_factor

    @z.setter
    def z(self, z: np.ndarray | None) -> None:
        """
        Set impedance tensor.

        Parameters
        ----------
        z : np.ndarray or None
            Complex impedance tensor array (nfrequency, 2, 2) in mt units [mV/km/nT]

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
        z = self._validate_array_input(z, "complex", old_shape)
        if z is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf=z)
        else:
            self._dataset["transfer_function"].loc[self.comps] = z

    # ----impedance error-----------------------------------------------------
    @property
    def z_error(self) -> np.ndarray | None:
        """Error of impedance tensor array as standard deviation."""
        if self._has_tf_error():
            return self._dataset.transfer_function_error.values / self._scale_factor

    @z_error.setter
    def z_error(self, z_error: np.ndarray | None) -> None:
        """
        Set impedance tensor error.

        Parameters
        ----------
        z_error : np.ndarray or None
            Error of impedance tensor array as standard deviation (nfrequency, 2, 2)

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

        z_error = self._validate_array_input(z_error, "float", old_shape)
        if z_error is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf_error=z_error)
        else:
            self._dataset["transfer_function_error"].loc[self.comps] = z_error

    # ----impedance model error-----------------------------------------------------
    @property
    def z_model_error(self) -> np.ndarray | None:
        """Model error of impedance tensor array as standard deviation."""
        if self._has_tf_model_error():
            return (
                self._dataset.transfer_function_model_error.values / self._scale_factor
            )

    @z_model_error.setter
    def z_model_error(self, z_model_error: np.ndarray | None) -> None:
        """
        Set impedance tensor model error.

        Parameters
        ----------
        z_model_error : np.ndarray or None
            Model error of impedance tensor array as standard deviation
            (nfrequency, 2, 2)

        """

        old_shape = None
        if not self._has_tf_model_error():
            old_shape = self._dataset.transfer_function_error.shape

        elif self._has_frequency():
            old_shape = (
                self.frequency.size,
                self._expected_shape[0],
                self._expected_shape[1],
            )

        z_model_error = self._validate_array_input(z_model_error, "float", old_shape)

        if z_model_error is None:
            return

        if self._is_empty():
            self._dataset = self._initialize(tf_error=z_model_error)
        else:
            self._dataset["transfer_function_model_error"].loc[
                self.comps
            ] = z_model_error

    def remove_ss(
        self,
        reduce_res_factor_x: float | list | np.ndarray = 1.0,
        reduce_res_factor_y: float | list | np.ndarray = 1.0,
        inplace: bool = False,
    ) -> "Z" | None:
        """
        Remove static shift by providing correction factors.

        Assume the original observed tensor Z is built by a static shift S
        and an unperturbed "correct" Z0:
            Z = S * Z0

        Therefore the correct Z will be:
            Z0 = S^(-1) * Z

        Parameters
        ----------
        reduce_res_factor_x : float or array-like, optional
            Static shift factor to be applied to x components (z[:, 0, :]).
            Assumed to be in resistivity scale, by default 1.0
        reduce_res_factor_y : float or array-like, optional
            Static shift factor to be applied to y components (z[:, 1, :]).
            Assumed to be in resistivity scale, by default 1.0
        inplace : bool, optional
            Update the current object or return a new impedance, by default False

        Returns
        -------
        Z or None
            Corrected Z if inplace is False, None otherwise

        """

        def _validate_factor_single(factor):
            """Validate factor single."""
            try:
                x_factor = float(factor)
            except ValueError:
                msg = f"factor must be a valid number not {factor}"
                self.logger.error(msg)
                raise ValueError(msg)
            return np.repeat(x_factor, len(self.z))

        def _validate_ss_input(factor):
            """Validate ss input."""
            if not np.iterable(factor):
                x_factor = _validate_factor_single(factor)

            elif len(reduce_res_factor_x) == 1:
                x_factor = _validate_factor_single(factor)
            else:
                x_factor = np.array(factor, dtype=float)

            if len(x_factor) != len(self.z):
                msg = (
                    f"Length of reduce_res_factor_x needs to be {len(self.z)} "
                    f" not {len(x_factor)}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
            return x_factor

        x_factors = np.sqrt(_validate_ss_input(reduce_res_factor_x))
        y_factors = np.sqrt(_validate_ss_input(reduce_res_factor_y))

        z_corrected = copy.copy(self.z)

        z_corrected[:, 0, 0] = (self.z[:, 0, 0] * self._scale_factor) / x_factors
        z_corrected[:, 0, 1] = (self.z[:, 0, 1] * self._scale_factor) / x_factors
        z_corrected[:, 1, 0] = (self.z[:, 1, 0] * self._scale_factor) / y_factors
        z_corrected[:, 1, 1] = (self.z[:, 1, 1] * self._scale_factor) / y_factors

        z_corrected = z_corrected / self._scale_factor

        if inplace:
            self.z = z_corrected
        else:
            return Z(
                z=z_corrected,
                z_error=self.z_error,
                frequency=self.frequency,
                z_model_error=self.z_model_error,
                units=self.units,
            )

    def remove_distortion(
        self,
        distortion_tensor: np.ndarray | None = None,
        distortion_error_tensor: np.ndarray | None = None,
        n_frequencies: int | None = None,
        comp: str = "det",
        only_2d: bool = False,
        inplace: bool = False,
    ) -> "Z" | None:
        """
        Remove distortion D from observed impedance tensor Z.

        Obtain the unperturbed "correct" Z0 from:
            Z = D * Z0

        Propagation of errors/uncertainties included.

        Parameters
        ----------
        distortion_tensor : np.ndarray, optional
            Real distortion tensor (2, 2), by default None
        distortion_error_tensor : np.ndarray, optional
            Real distortion error tensor (2, 2), by default None
        n_frequencies : int, optional
            Number of frequencies to use for estimation, by default None
        comp : str, optional
            Component to use for estimation, by default 'det'
        only_2d : bool, optional
            Only use 2D data, by default False
        inplace : bool, optional
            Update the current object or return a new impedance, by default False

        Returns
        -------
        Z or None
            Impedance tensor with distortion removed if inplace is False,
            None otherwise

        """

        if distortion_tensor is None:
            (
                distortion_tensor,
                distortion_error_tensor,
            ) = self.estimate_distortion(
                n_frequencies=n_frequencies, comp=comp, only_2d=only_2d
            )

        z_corrected, z_corrected_error = remove_distortion_from_z_object(
            self, distortion_tensor, distortion_error_tensor, self.logger
        )

        # into mt units
        z_corrected = z_corrected * self._scale_factor
        z_corrected_error = z_corrected_error * self._scale_factor

        if inplace:
            self.z = z_corrected
            self.z_error = z_corrected_error
        else:
            z_object = Z(
                z=z_corrected,
                z_error=z_corrected_error,
                frequency=self.frequency,
                z_model_error=self.z_model_error,
            )
            z_object.units = self.units
            return z_object

    @property
    def resistivity(self) -> np.ndarray | None:
        """Resistivity of impedance."""
        if self.z is not None:
            return np.apply_along_axis(
                lambda x: np.abs(x) ** 2 / self.frequency * 0.2,
                0,
                self.z * self._scale_factor,
            )

    @property
    def phase(self) -> np.ndarray | None:
        """Phase of impedance."""
        if self.z is not None:
            return np.rad2deg(np.angle(self.z * self._scale_factor))

    @property
    def resistivity_error(self) -> np.ndarray | None:
        """
        Resistivity error of impedance.

        By standard error propagation, relative error in resistivity is
        2 * relative error in z amplitude.

        """
        if self.z is not None and self.z_error is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.apply_along_axis(
                    lambda x: x / self.frequency * 0.2,
                    0,
                    2
                    * (self.z_error * self._scale_factor)
                    * np.abs(self.z * self._scale_factor),
                )

    @property
    def phase_error(self) -> np.ndarray | None:
        """
        Phase error of impedance.

        Uncertainty in phase (in degrees) is computed by defining a circle around
        the z vector in the complex plane. The uncertainty is the absolute angle
        between the vector to (x,y) and the vector between the origin and the
        tangent to the circle.

        """
        if self.z is not None and self.z_error is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.degrees(np.arctan(self.z_error / np.abs(self.z)))

    @property
    def resistivity_model_error(self) -> np.ndarray | None:
        """Resistivity model error of impedance."""
        if self.z is not None and self.z_model_error is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.apply_along_axis(
                    lambda x: x / self.frequency * 0.2,
                    0,
                    2
                    * (self.z_model_error * self._scale_factor)
                    * np.abs(self.z * self._scale_factor),
                )

    @property
    def phase_model_error(self) -> np.ndarray | None:
        """Phase model error of impedance."""
        if self.z is not None and self.z_model_error is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.degrees(np.arctan(self.z_model_error / np.abs(self.z)))

    def _compute_z_error(
        self, res_error: np.ndarray | None, phase_error: np.ndarray | None
    ) -> np.ndarray | None:
        """
        Compute impedance error from apparent resistivity and phase errors.

        Parameters
        ----------
        res_error : np.ndarray or None
            Resistivity error array
        phase_error : np.ndarray or None
            Phase error array in degrees

        Returns
        -------
        np.ndarray or None
            Impedance error array

        """
        if res_error is None:
            return None

        # not extremely positive where the 250 comes from it is roughly 5 x 50
        # which is about 5 * (2*pi)**2
        return np.abs(
            np.sqrt(self.frequency * (res_error.T) * 250).T
            * np.tan(np.radians(phase_error))
        )

    def set_resistivity_phase(
        self,
        resistivity: np.ndarray,
        phase: np.ndarray,
        frequency: np.ndarray,
        res_error: np.ndarray | None = None,
        phase_error: np.ndarray | None = None,
        res_model_error: np.ndarray | None = None,
        phase_model_error: np.ndarray | None = None,
    ) -> None:
        """
        Set values for resistivity and phase with error propagation.

        Parameters
        ----------
        resistivity : np.ndarray
            Resistivity array in Ohm-m (num_frequency, 2, 2)
        phase : np.ndarray
            Phase array in degrees (num_frequency, 2, 2)
        frequency : np.ndarray
            Frequency array in Hz (num_frequency)
        res_error : np.ndarray, optional
            Resistivity error array in Ohm-m (num_frequency, 2, 2), by default None
        phase_error : np.ndarray, optional
            Phase error array in degrees (num_frequency, 2, 2), by default None
        res_model_error : np.ndarray, optional
            Resistivity model error array in Ohm-m (num_frequency, 2, 2),
            by default None
        phase_model_error : np.ndarray, optional
            Phase model error array in degrees (num_frequency, 2, 2),
            by default None

        """

        if resistivity is None or phase is None or frequency is None:
            self.logger.debug(
                "Cannot estimate resitivity and phase if resistivity, "
                "phase, or frequency is None."
            )
            return

        self.frequency = self._validate_frequency(frequency)
        resistivity = self._validate_array_input(resistivity, float)
        phase = self._validate_array_input(phase, float)

        res_error = self._validate_array_input(res_error, float)
        phase_error = self._validate_array_input(phase_error, float)
        res_model_error = self._validate_array_input(res_model_error, float)
        phase_model_error = self._validate_array_input(phase_model_error, float)

        abs_z = np.sqrt(5.0 * self.frequency * (resistivity.T)).T
        self.z = abs_z * np.exp(1j * np.radians(phase))

        self.z_error = self._compute_z_error(res_error, phase_error)
        self.z_model_error = self._compute_z_error(res_model_error, phase_model_error)

    @property
    def det(self) -> np.ndarray | None:
        """Determinant of impedance."""
        if self.z is not None:
            det_z = np.array(
                [np.linalg.det(ii * self._scale_factor) ** 0.5 for ii in self.z]
            )

            return det_z

    @property
    def det_error(self) -> np.ndarray | None:
        """Determinant of impedance error."""
        det_z_error = None
        if self.z_error is not None:
            det_z_error = np.zeros_like(self.det, dtype=float)
            with np.errstate(invalid="ignore"):
                # components of the impedance tensor are not independent variables
                # so can't use standard error propagation
                # calculate manually:
                # difference of determinant of z + z_error and z - z_error then divide by 2
                det_z_error[:] = (
                    self._scale_factor
                    * (
                        np.abs(
                            np.linalg.det(self.z + self.z_error)
                            - np.linalg.det(self.z - self.z_error)
                        )
                        / 2.0
                    )
                    ** 0.5
                )
        return det_z_error

    @property
    def det_model_error(self) -> np.ndarray | None:
        """Determinant of impedance model error."""
        det_z_error = None
        if self.z_model_error is not None:
            det_z_error = np.zeros_like(self.det, dtype=float)
            with np.errstate(invalid="ignore"):
                # components of the impedance tensor are not independent variables
                # so can't use standard error propagation
                # calculate manually:
                # difference of determinant of z + z_error and z - z_error then divide by 2
                det_z_error[:] = (
                    np.abs(
                        np.linalg.det(self.z + self.z_model_error)
                        - np.linalg.det(self.z - self.z_model_error)
                    )
                    / 2.0
                ) ** 0.5
        return det_z_error

    @property
    def phase_det(self) -> np.ndarray | None:
        """Phase determinant."""
        if self.det is not None:
            return np.rad2deg(np.arctan2(self.det.imag, self.det.real))

    @property
    def phase_error_det(self) -> np.ndarray | None:
        """Phase error determinant."""
        if self.det is not None:
            return np.rad2deg(np.arcsin(self.det_error / abs(self.det)))

    @property
    def phase_model_error_det(self) -> np.ndarray | None:
        """Phase model error determinant."""
        if self.det is not None:
            return np.rad2deg(np.arcsin(self.det_model_error / abs(self.det)))

    @property
    def res_det(self) -> np.ndarray | None:
        """Resistivity determinant."""
        if self.det is not None:
            return 0.2 * (1.0 / self.frequency) * abs(self.det) ** 2

    @property
    def res_error_det(self) -> np.ndarray | None:
        """Resistivity error determinant."""
        if self.det_error is not None:
            return (
                0.2 * (1.0 / self.frequency) * np.abs(self.det + self.det_error) ** 2
                - self.res_det
            )

    @property
    def res_model_error_det(self) -> np.ndarray | None:
        """Resistivity model error determinant."""
        if self.det_model_error is not None:
            return (
                0.2
                * (1.0 / self.frequency)
                * np.abs(self.det + self.det_model_error) ** 2
                - self.res_det
            )

    def _get_component(self, comp: str, array: np.ndarray | None) -> np.ndarray | None:
        """
        Get the correct component from an array.

        Parameters
        ----------
        comp : str
            Component name: 'xx', 'xy', 'yx', or 'yy'
        array : np.ndarray or None
            Impedance array

        Returns
        -------
        np.ndarray or None
            Array component

        """
        if array is not None:
            index_dict = {"x": 0, "y": 1}
            ii = index_dict[comp[-2]]
            jj = index_dict[comp[-1]]

            return array[:, ii, jj]

    @property
    def res_xx(self) -> np.ndarray | None:
        """Resistivity of xx component."""
        return self._get_component("xx", self.resistivity)

    @property
    def res_xy(self) -> np.ndarray | None:
        """Resistivity of xy component."""
        return self._get_component("xy", self.resistivity)

    @property
    def res_yx(self) -> np.ndarray | None:
        """Resistivity of yx component."""
        return self._get_component("yx", self.resistivity)

    @property
    def res_yy(self) -> np.ndarray | None:
        """Resistivity of yy component."""
        return self._get_component("yy", self.resistivity)

    @property
    def res_error_xx(self) -> np.ndarray | None:
        """Resistivity error of xx component."""
        return self._get_component("xx", self.resistivity_error)

    @property
    def res_error_xy(self) -> np.ndarray | None:
        """Resistivity error of xy component."""
        return self._get_component("xy", self.resistivity_error)

    @property
    def res_error_yx(self) -> np.ndarray | None:
        """Resistivity error of yx component."""
        return self._get_component("yx", self.resistivity_error)

    @property
    def res_error_yy(self) -> np.ndarray | None:
        """Resistivity error of yy component."""
        return self._get_component("yy", self.resistivity_error)

    @property
    def res_model_error_xx(self) -> np.ndarray | None:
        """Resistivity model error of xx component."""
        return self._get_component("xx", self.resistivity_model_error)

    @property
    def res_model_error_xy(self) -> np.ndarray | None:
        """Resistivity model error of xy component."""
        return self._get_component("xy", self.resistivity_model_error)

    @property
    def res_model_error_yx(self) -> np.ndarray | None:
        """Resistivity model error of yx component."""
        return self._get_component("yx", self.resistivity_model_error)

    @property
    def res_model_error_yy(self) -> np.ndarray | None:
        """Resistivity model error of yy component."""
        return self._get_component("yy", self.resistivity_model_error)

    @property
    def phase_xx(self) -> np.ndarray | None:
        """Phase of xx component."""
        return self._get_component("xx", self.phase)

    @property
    def phase_xy(self) -> np.ndarray | None:
        """Phase of xy component."""
        return self._get_component("xy", self.phase)

    @property
    def phase_yx(self) -> np.ndarray | None:
        """Phase of yx component."""
        return self._get_component("yx", self.phase)

    @property
    def phase_yy(self) -> np.ndarray | None:
        """Phase of yy component."""
        return self._get_component("yy", self.phase)

    @property
    def phase_error_xx(self) -> np.ndarray | None:
        """Phase error of xx component."""
        return self._get_component("xx", self.phase_error)

    @property
    def phase_error_xy(self) -> np.ndarray | None:
        """Phase error of xy component."""
        return self._get_component("xy", self.phase_error)

    @property
    def phase_error_yx(self) -> np.ndarray | None:
        """Phase error of yx component."""
        return self._get_component("yx", self.phase_error)

    @property
    def phase_error_yy(self) -> np.ndarray | None:
        """Phase error of yy component."""
        return self._get_component("yy", self.phase_error)

    @property
    def phase_model_error_xx(self) -> np.ndarray | None:
        """Phase model error of xx component."""
        return self._get_component("xx", self.phase_model_error)

    @property
    def phase_model_error_xy(self) -> np.ndarray | None:
        """Phase model error of xy component."""
        return self._get_component("xy", self.phase_model_error)

    @property
    def phase_model_error_yx(self) -> np.ndarray | None:
        """Phase model error of yx component."""
        return self._get_component("yx", self.phase_model_error)

    @property
    def phase_model_error_yy(self) -> np.ndarray | None:
        """Phase model error of yy component."""
        return self._get_component("yy", self.phase_model_error)

    @property
    def phase_tensor(self) -> PhaseTensor:
        """Phase tensor object based on impedance."""
        return PhaseTensor(
            z=self.z,
            z_error=self.z_error,
            z_model_error=self.z_model_error,
            frequency=self.frequency,
        )

    @property
    def invariants(self) -> ZInvariants:
        """Weaver invariants."""
        return ZInvariants(z=self.z)

    def estimate_dimensionality(
        self, skew_threshold: float = 5, eccentricity_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Estimate dimensionality of the impedance tensor.

        Based on parameters such as strike and phase tensor eccentricity.

        Parameters
        ----------
        skew_threshold : float, optional
            Skew threshold for 3D determination, by default 5
        eccentricity_threshold : float, optional
            Eccentricity threshold for 2D determination, by default 0.1

        Returns
        -------
        np.ndarray
            Dimensionality array (1D, 2D, or 3D) for each period

        """

        dimensionality = np.ones(self.period.size, dtype=int)

        # need to get 2D first then 3D
        dimensionality[
            np.where(self.phase_tensor.eccentricity > eccentricity_threshold)
        ] = 2
        dimensionality[np.where(np.abs(self.phase_tensor.skew) > skew_threshold)] = 3

        return dimensionality

    def estimate_distortion(
        self,
        n_frequencies: int | None = None,
        comp: str = "det",
        only_2d: bool = False,
        clockwise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate distortion tensor.

        Parameters
        ----------
        n_frequencies : int, optional
            Number of frequencies to use, by default None (uses all)
        comp : str, optional
            Component to use for estimation, by default 'det'
        only_2d : bool, optional
            Only use 2D data, by default False
        clockwise : bool, optional
            Clockwise rotation, by default True

        Returns
        -------
        tuple of np.ndarray
            Distortion tensor and distortion error tensor

        """
        if n_frequencies is None:
            nf = self.frequency.size
        else:
            nf = n_frequencies

        if self._has_tf():
            new_z_object = Z(
                z=self._dataset.transfer_function.values[0:nf, :, :],
                frequency=self.frequency[0:nf],
            )
            if self._has_tf_error():
                new_z_object.z_error = self._dataset.transfer_function_error.values[
                    0:nf
                ]

        return find_distortion(
            new_z_object, comp=comp, only_2d=only_2d, clockwise=clockwise
        )

    def estimate_depth_of_investigation(self) -> Any:
        """
        Estimate depth of investigation.

        Returns
        -------
        Any
            Depth of investigation results

        """

        return calculate_depth_of_investigation(self)
