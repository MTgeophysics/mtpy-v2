#!/usr/bin/env python

"""
.. module:: TFBase
   :synopsis: Generic Transfer Function object

.. moduleauthor:: Jared Peacock <jpeacock@usgs.gov>

Updated 11/2020 for logging and formating (J. Peacock).
    - ToDo: add functionality for covariance matrix
"""

# =============================================================================
# Imports
# =============================================================================
from copy import deepcopy

import numpy as np
import xarray as xr
from scipy import interpolate
from loguru import logger

from mtpy.utils.calculator import (
    rotate_matrix_with_errors,
    rotate_vector_with_errors,
)


# ==============================================================================
# Impedance Tensor Class
# ==============================================================================
class TFBase:
    """Generic transfer function object that uses xarray as its base container
    for the data.
    """

    def __init__(
        self,
        tf=None,
        tf_error=None,
        frequency=None,
        tf_model_error=None,
        **kwargs,
    ):
        self.logger = logger
        self.rotation_angle = 0.0
        self.inputs = ["x", "y"]
        self.outputs = ["x", "y"]
        self._expected_shape = (2, 2)
        self._name = "base transfer function"
        self._dataset = None
        self._tf_dtypes = {
            "tf": complex,
            "tf_error": float,
            "tf_model_error": float,
        }

        frequency = self._validate_frequency(frequency)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._dataset = self._initialize(
            periods=1.0 / frequency,
            tf=tf,
            tf_error=tf_error,
            tf_model_error=tf_model_error,
        )

    def __str__(self):
        """Str function."""
        lines = [f"Transfer Function {self._name}", "-" * 30]
        if self.frequency is not None:
            lines.append(f"\tNumber of periods:  {self.frequency.size}")
            lines.append(
                f"\tFrequency range:        {self.frequency.min():.5E} -- "
                f"{self.frequency.max():.5E} Hz"
            )
            lines.append(
                f"\tPeriod range:           {1/self.frequency.max():.5E} -- "
                f"{1/self.frequency.min():.5E} s"
            )
            lines.append("")
            lines.append(f"\tHas {self._name}:              {self._has_tf()}")
            lines.append(f"\tHas {self._name}_error:        {self._has_tf_error()}")
            lines.append(
                f"\tHas {self._name}_model_error:  {self._has_tf_model_error()}"
            )
        return "\n".join(lines)

    def __repr__(self):
        """Repr function."""
        return self.__str__()

    def __eq__(self, other):
        """Eq function."""
        if not isinstance(other, TFBase):
            msg = f"Cannot compare {type(other)} with TFBase"
            self.logger.error(msg)
            raise ValueError(msg)

        # loop over variables to make sure they are all the same.
        for var in list(self._dataset.data_vars):
            has_tf_str = f"_has_{var.replace('transfer_function', 'tf')}"
            if getattr(self, has_tf_str):
                if getattr(other, has_tf_str):
                    if not np.allclose(
                        self._dataset[var].data, other._dataset[var].data
                    ):
                        self.logger.info(f"Transfer functions {var} not equal")
                        return False
                else:
                    return False
        return True

    def __deepcopy__(self, memo):
        """Deepcopy function."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["logger"]:
                continue

            setattr(result, k, deepcopy(v, memo))
        return result

    def copy(self):
        """Copy function."""
        return deepcopy(self)

    def _initialize(self, periods=[1], tf=None, tf_error=None, tf_model_error=None):
        """Initialized based on input channels, output channels and period."""

        if tf is not None:
            tf = self._validate_array_input(tf, self._tf_dtypes["tf"])
            periods = self._validate_frequency(periods, tf.shape[0])
            if tf_error is not None:
                self._validate_array_shape(tf_error, tf.shape)
            else:
                tf_error = np.zeros_like(tf, dtype=self._tf_dtypes["tf_error"])

            if tf_model_error is not None:
                self._validate_array_shape(tf_model_error, tf.shape)
            else:
                tf_model_error = np.zeros_like(
                    tf, dtype=self._tf_dtypes["tf_model_error"]
                )

        elif tf_error is not None:
            tf_error = self._validate_array_input(tf_error, self._tf_dtypes["tf_error"])
            periods = self._validate_frequency(periods, tf_error.shape[0])
            tf = np.zeros_like(tf_error, dtype=self._tf_dtypes["tf"])

            if tf_model_error is not None:
                self._validate_array_shape(tf_model_error, tf_error.shape)
            else:
                tf_model_error = np.zeros_like(
                    tf_error, dtype=self._tf_dtypes["tf_model_error"]
                )

        elif tf_model_error is not None:
            tf_model_error = self._validate_array_input(
                tf_model_error, self._tf_dtypes["tf_model_error"]
            )
            tf = np.zeros_like(tf_model_error, dtype=self._tf_dtypes["tf"])
            tf_error = np.zeros_like(tf_model_error, dtype=self._tf_dtypes["tf_error"])
            periods = self._validate_frequency(periods, tf_model_error.shape[0])

        else:
            periods = self._validate_frequency(periods)
            tf_shape = (
                periods.size,
                self._expected_shape[0],
                self._expected_shape[1],
            )
            tf = np.zeros(tf_shape, dtype=self._tf_dtypes["tf"])
            tf_error = np.zeros(tf_shape, dtype=self._tf_dtypes["tf_error"])
            tf_model_error = np.zeros(tf_shape, dtype=self._tf_dtypes["tf_model_error"])

        tf = xr.DataArray(
            data=tf,
            dims=["period", "output", "input"],
            coords={
                "period": periods,
                "output": self.outputs,
                "input": self.inputs,
            },
            name="transfer_function",
        )
        tf_err = xr.DataArray(
            data=tf_error,
            dims=["period", "output", "input"],
            coords={
                "period": periods,
                "output": self.outputs,
                "input": self.inputs,
            },
            name="transfer_function_error",
        )
        tf_model_err = xr.DataArray(
            data=tf_model_error,
            dims=["period", "output", "input"],
            coords={
                "period": periods,
                "output": self.outputs,
                "input": self.inputs,
            },
            name="transfer_function_model_error",
        )

        return xr.Dataset(
            {
                tf.name: tf,
                tf_err.name: tf_err,
                tf_model_err.name: tf_model_err,
            },
            coords={
                "period": periods,
                "output": self.outputs,
                "input": self.inputs,
            },
        )

    def _is_empty(self):
        """Check to see if the data set is empty, default settings.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if self._dataset is None:
            return True

        if (
            self._has_tf() == False
            and self._has_tf_error() == False
            and self._has_tf_model_error() == False
        ):
            if not self._has_frequency():
                return True
            else:
                return False

        return False

    def _has_tf(self):
        """Has tf."""
        return not (self._dataset.transfer_function.values == 0).all()

    def _has_tf_error(self):
        """Has tf error."""
        return not (self._dataset.transfer_function_error.values == 0).all()

    def _has_tf_model_error(self):
        """Has tf model error."""
        return not (self._dataset.transfer_function_model_error.values == 0).all()

    def _has_frequency(self):
        """Has frequency."""
        if (self._dataset.coords["period"].values == np.array([1])).all():
            return False
        return True

    @property
    def comps(self):
        """Comps function."""
        return dict(input=self.inputs, output=self.outputs)

    # ---frequencyuency-------------------------------------------------------------
    @property
    def frequency(self):
        """Frequencyuencies for each impedance tensor element

        Units are Hz..
        """
        return 1.0 / self._dataset.period.values

    @frequency.setter
    def frequency(self, frequency):
        """Set the array of frequency.
        :param frequency: Array of frequencyunecies (Hz).
        :type frequency: np.ndarray
        """

        if frequency is None:
            return

        if self._is_empty():
            frequency = self._validate_frequency(frequency)
            self._dataset = self._initialize(periods=1.0 / frequency)

        else:
            frequency = self._validate_frequency(
                frequency, n_frequencies=self._dataset.period.shape[0]
            )

            self._dataset = self._dataset.assign_coords({"period": 1.0 / frequency})

    @property
    def period(self):
        """Periods in seconds."""

        return 1.0 / self.frequency

    @period.setter
    def period(self, value):
        """Setting periods will set the frequencyuencies."""

        self.frequency = 1.0 / value

    @property
    def n_periods(self):
        """N periods."""
        if self._is_empty():
            return 0

        return self.period.size

    def _validate_frequency(self, frequency, n_frequencies=None):
        """Validate frequency."""

        if frequency is None:
            return np.array([1])

        frequency = np.array(frequency, dtype=float)
        if len(frequency) > 1:
            frequency = frequency.flatten()

        if n_frequencies is not None:
            if frequency.size == 1:
                if (frequency == np.array([1])).all():
                    return np.arange(1, n_frequencies + 1, 1)
            if frequency.size != n_frequencies:
                raise ValueError(
                    f"input frequencies must have shape {n_frequencies} not "
                    f"{frequency.size}. "
                    "Use tf._dataset = TFBase._initialize(1./new_frequencies) "
                    "or make a new transfer function object"
                )

        return frequency

    def _validate_array_input(self, tf_array, expected_dtype, old_shape=None):
        """Validate an input impedance array.
        :param old_shape:
            Defaults to None.
        :param expected_dtype:
        :param tf_array:
        :param array: DESCRIPTION.
        :type array: TYPE
        :param dtype: DESCRIPTION.
        :type dtype: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if tf_array is None:
            return
        if not isinstance(tf_array, np.ndarray):
            if isinstance(tf_array, (float, int, complex)):
                tf_array = [tf_array]
            tf_array = np.array(tf_array, dtype=expected_dtype)
        if tf_array.dtype not in [expected_dtype]:
            tf_array = tf_array.astype(expected_dtype)

        if len(tf_array.shape) == 3:
            if tf_array.shape[1:3] == self._expected_shape:
                if old_shape is not None:
                    self._validate_array_shape(tf_array, old_shape)
                return tf_array
            else:
                msg = (
                    f"Input array must be shape (n, "
                    f"{self.expected_shape[0]}, {self.expected_shape[1]}) "
                    f"not {tf_array.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
        elif len(tf_array.shape) == 2:
            if tf_array.shape == self._expected_shape:
                tf_array = tf_array.reshape(
                    (1, self._expected_shape[0], self._expected_shape[1])
                )
                self.logger.debug(
                    f"setting input tf with shape {self._expected_shape} "
                    f"to (1, self._expected_shape[0], self._expected_shape[1])"
                )
                if old_shape is not None:
                    self._validate_array_shape(tf_array, old_shape)
                return tf_array
            else:
                msg = (
                    f"Input array must be shape (n, "
                    f"{self._expected_shape[0]}, {self._expected_shape[1]}) "
                    f"not {tf_array.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            msg = (
                f"{tf_array.shape} are not the correct dimensions, "
                f"must be (n, {self._expected_shape[0]}, {self._expected_shape[1]})"
            )
            self.logger.error(msg)
            raise ValueError(msg)

    def _validate_array_shape(self, array, expected_shape):
        """Check array for expected shape.
        :param array: DESCRIPTION.
        :type array: TYPE
        :param expected_shape: DESCRIPTION.
        :type expected_shape: TYPE
        :raises ValueError: DESCRIPTION.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        # check to see if the new z array is the same shape as the old
        if array.shape != expected_shape:
            msg = (
                f"Input array shape {array.shape} does not match expected "
                f"shape {expected_shape}. Suggest initiating new dataset "
                f"using {self.__class__.__name__}._initialize() or "
                f"making a new object {self.__class__.__name__}()."
            )
            self.logger.error(msg)
            raise ValueError(msg)

    def _validate_real_valued(self, array):
        """Make sure resistivity is real valued.
        :param array:
        :param res: DESCRIPTION.
        :type res: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        # assert real array:
        if np.linalg.norm(np.imag(array)) != 0:
            msg = "Array is not real valued"
            self.logger.error(msg)
            raise ValueError(msg)
        return array

    @property
    def inverse(self):
        """Return the inverse of transfer function.

        (no error propagtaion included yet)
        """

        if self.has_tf():
            inverse = self._dataset.copy()

            try:
                inverse.transfer_function = np.linalg.inv(inverse.transfer_function)

            except np.linalg.LinAlgError:
                raise ValueError("Transfer Function is a singular matrix cannot invert")

            return inverse

    def rotate(self, alpha, inplace=False, coordinate_reference_frame="ned"):
        """Rotate transfer function array by angle alpha.

        Rotation angle must be given in degrees. All angles are referenced
        to the `coordinate_reference_frame` where the rotation angle is
        clockwise positive, rotating North into East.

        Most transfer functions are referenced an NED coordinate system, which
        is what MTpy uses as the default.

        In the NED coordinate system:

        x=North, y=East, z=+down and a positve clockwise rotation is a positive
        angle. In this coordinate system the rotation matrix is the
        conventional rotation matrix.

        In the ENU coordinate system:

        x=East, y=North, z=+up and a positve clockwise rotation is a positive
        angle. In this coordinate system the rotation matrix is the
        inverser of the conventional rotation matrix.

        :param alpha: Angle to rotate by assuming a clockwise rotation from
         north in the `coordinate_reference_frame`
        :type alpha: float (in degrees)
        :param inplace: rotate in place. Will add alpha to `rotation_angle`
        :type inplace: bool
        :param coordinate_reference_frame: If set to `ned` or `+` then the
         rotation will be clockwise from north (x1).  Therefore, a positive
         angle will rotate eastward and a negative angle will rotate westward.
         If set to `enu` or `-` then rotation will be counter-clockwise in a
         coordinate system of (x1, x2). The angle is then positive from x1 to
         x2.
        :return: rotated transfer function if `inplace` is False
        :rtype: xarray.DataSet
        """

        if not self._has_tf():
            self.logger.warning(
                "transfer function array is empty and cannot be rotated"
            )
            return

        def get_rotate_function(shape):
            """Get rotate function."""
            if shape[0] == 2:
                return rotate_matrix_with_errors
            elif shape[0] == 1:
                return rotate_vector_with_errors

        def validate_angle(self, angle):
            """Validate angle to be a valid float."""
            try:
                return float(angle % 360)
            except ValueError:
                msg = f"Angle must be a valid number (in degrees) not {alpha}"
                self.logger.error(msg)
                raise ValueError(msg)

        def get_clockwise(coordinate_reference_frame):

            if coordinate_reference_frame.lower() in ["ned", "+"]:
                return True
            elif coordinate_reference_frame.lower() in ["enu", "-"]:
                return False
            else:
                raise ValueError(
                    f"coordinate_reference_frame {coordinate_reference_frame} "
                    "not understood."
                )

        if isinstance(alpha, (float, int, str)):
            degree_angle = np.repeat(validate_angle(self, alpha), self.n_periods)

        elif isinstance(alpha, (list, tuple, np.ndarray)):
            if len(alpha) == 1:
                degree_angle = np.repeat(validate_angle(self, alpha[0]), self.n_periods)
            else:
                degree_angle = np.array(alpha, dtype=float) % 360
                if degree_angle.size != self.n_periods:
                    raise ValueError(
                        "angles must be the same size as periods "
                        f"{self.n_periods} not {degree_angle.size}"
                    )

        self.rotation_angle = self.rotation_angle + degree_angle

        ds = self._dataset.copy()
        rot_tf = np.zeros_like(self._dataset.transfer_function.values, dtype=complex)
        rot_tf_error = np.zeros_like(
            self._dataset.transfer_function.values, dtype=float
        )
        rot_tf_model_error = np.zeros_like(
            self._dataset.transfer_function.values, dtype=float
        )

        rotate_func = get_rotate_function(self._expected_shape)
        clockwise = get_clockwise(coordinate_reference_frame)

        for index, angle in enumerate(degree_angle):
            if self._has_tf():
                if self._has_tf_error():
                    (
                        rot_tf[index, :, :],
                        rot_tf_error[index, :, :],
                    ) = rotate_func(
                        ds.transfer_function[index].values,
                        angle,
                        ds.transfer_function_error[index].values,
                        clockwise=clockwise,
                    )
                if self._has_tf_model_error():
                    (
                        rot_tf[index, :, :],
                        rot_tf_model_error[index, :, :],
                    ) = rotate_func(
                        ds.transfer_function[index].values,
                        angle,
                        ds.transfer_function_model_error[index].values,
                        clockwise=clockwise,
                    )
                if not self._has_tf_error() and not self._has_tf_model_error():
                    (rot_tf[index, :, :], _) = rotate_func(
                        ds.transfer_function[index].values,
                        angle,
                        clockwise=clockwise,
                    )
        ds.transfer_function.values = rot_tf
        ds.transfer_function_error.values = rot_tf_error
        ds.transfer_function_model_error.values = rot_tf_model_error

        if inplace:
            self._dataset = ds
        else:
            tb = self.copy()
            tb._dataset = ds
            return tb

    def interpolate_improved(
        self,
        new_periods,
        inplace=False,
        method="slinear",
        extrapolate=False,
        **kwargs,
    ):
        """Interpolate transfer function onto a new period range using SciPy interpolators.

        This method individually interpolates the real and imaginary parts of complex data
        and directly handles each component separately for better accuracy.

        Parameters
        ----------
        new_periods : array_like
            New periods to interpolate onto
        inplace : bool, optional
            Interpolate inplace, defaults to False
        method : str, optional
            Interpolation method, defaults to 'pchip'
            Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', 'pchip', 'akima', 'spline', 'barycentric',
            'polynomial', 'krogh'
        extrapolate : bool, optional
            Whether to extrapolate beyond original period range, defaults to False
        **kwargs :
            Additional kwargs passed to scipy.interpolate methods

        Returns
        -------
        TFBase or None
            Interpolated transfer function object if inplace=False, otherwise None
        """

        # Convert periods to array if not already
        new_periods = np.array(new_periods, dtype=float)

        # Get source periods and prepare target periods array
        source_periods = self._dataset.period.values

        # Check for overlapping ranges if not extrapolating
        if not extrapolate:
            min_period = np.nanmin(source_periods)
            max_period = np.nanmax(source_periods)
            valid_indices = (new_periods >= min_period) & (new_periods <= max_period)
            if not all(valid_indices):
                logger.warning(
                    f"Some target periods outside source range ({min_period:.6g} - {max_period:.6g}s) "
                    f"and extrapolate=False. These values will be set to NaN."
                )

        da_dict = {}
        # Loop through each variable in the dataset
        for var_name, da in self._dataset.data_vars.items():
            # Create output array with same shape but for new periods
            output_shape = (len(new_periods),) + da.shape[1:]
            if np.issubdtype(da.dtype, np.complexfloating):
                output_array = np.zeros(output_shape, dtype=complex)
                is_complex = True
            else:
                output_array = np.zeros(output_shape, dtype=da.dtype)
                is_complex = False

            # Loop through inputs and outputs
            for i_index, inp in enumerate(da.input.values):
                for j_index, outp in enumerate(da.output.values):
                    # Extract data for this component
                    comp_data = da.sel(input=inp, output=outp).values

                    # Find finite (non-NaN) values
                    finite_mask = np.isfinite(comp_data)

                    if not np.any(finite_mask):
                        # If all NaN, keep as NaN
                        if is_complex:
                            output_array[:, j_index, i_index] = np.nan + 1j * np.nan
                        else:
                            output_array[:, j_index, i_index] = np.nan
                        continue

                    # Get periods with valid data
                    valid_periods = source_periods[finite_mask]
                    valid_data = comp_data[finite_mask]

                    if len(valid_periods) < 2:
                        # Need at least 2 points for interpolation
                        if is_complex:
                            output_array[:, j_index, i_index] = np.nan + 1j * np.nan
                        else:
                            output_array[:, j_index, i_index] = np.nan
                        continue

                    try:
                        if is_complex:
                            # Handle complex data - interpolate real and imaginary parts separately
                            real_part = np.zeros(len(new_periods))
                            imag_part = np.zeros(len(new_periods))

                            # Create interpolators for real and imaginary parts
                            real_interp = interpolate.interp1d(
                                valid_periods,
                                np.real(valid_data),
                                kind=method,
                                bounds_error=False,
                                fill_value="extrapolate" if extrapolate else np.nan,
                                **kwargs,
                            )

                            imag_interp = interpolate.interp1d(
                                valid_periods,
                                np.imag(valid_data),
                                kind=method,
                                bounds_error=False,
                                fill_value="extrapolate" if extrapolate else np.nan,
                                **kwargs,
                            )

                            # Apply interpolation
                            real_part = real_interp(new_periods)
                            imag_part = imag_interp(new_periods)

                            # If not extrapolating, ensure values outside range are NaN
                            if not extrapolate:
                                mask = (new_periods < valid_periods.min()) | (
                                    new_periods > valid_periods.max()
                                )
                                real_part[mask] = np.nan
                                imag_part[mask] = np.nan

                            # Store interpolated values
                            output_array[:, j_index, i_index] = (
                                real_part + 1j * imag_part
                            )
                        else:
                            # Handle real data
                            interp_func = interpolate.interp1d(
                                valid_periods,
                                valid_data,
                                kind=method,
                                bounds_error=False,
                                fill_value="extrapolate" if extrapolate else np.nan,
                                **kwargs,
                            )

                            # Apply interpolation
                            result = interp_func(new_periods)

                            # If not extrapolating, ensure values outside range are NaN
                            if not extrapolate:
                                mask = (new_periods < valid_periods.min()) | (
                                    new_periods > valid_periods.max()
                                )
                                result[mask] = np.nan

                            output_array[:, j_index, i_index] = result

                    except Exception as e:
                        logger.warning(
                            f"Interpolation failed for {var_name}[{outp},{inp}]: {str(e)}"
                        )
                        if is_complex:
                            output_array[:, j_index, i_index] = np.nan + 1j * np.nan
                        else:
                            output_array[:, j_index, i_index] = np.nan

            # Create new DataArray and assign to dataset
            new_da = xr.DataArray(
                data=output_array,
                dims=["period", "output", "input"],
                coords={
                    "period": new_periods,
                    "output": da.output.values,
                    "input": da.input.values,
                },
                name=var_name,
            )
            da_dict[var_name] = new_da

        ds = xr.Dataset(da_dict)
        if inplace:
            self._dataset = ds
            return None
        else:
            result = self.copy()
            result._dataset = ds
            return result

    def _create_interpolator(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str,
        extrapolate: bool = False,
        **kwargs,
    ) -> (
        interpolate.InterpolatedUnivariateSpline
        | interpolate.PchipInterpolator
        | interpolate.Akima1DInterpolator
        | interpolate.BarycentricInterpolator
        | interpolate.KroghInterpolator
        | interpolate.CubicSpline
        | interpolate.interp1d
    ):
        """Create a scipy interpolator based on the specified method.

        :param x: x values (periods)
        :param y: y values (data)
        :param method: interpolation method
        :param kwargs: additional arguments for interpolator
        :return: interpolation function
        """

        # For certain interpolators, we need to handle kwargs differently
        if method == "spline":
            # Default k=3 for cubic spline
            k = kwargs.pop("k", 3)
            # Ensure we have enough points for the requested spline order
            if len(x) <= k:
                # Fall back to lower order or linear if not enough points
                k = min(len(x) - 1, 1)
            s = kwargs.pop("s", 0)  # Smoothing factor
            return interpolate.InterpolatedUnivariateSpline(
                x, y, k=k, ext=int(extrapolate)
            )
        elif method == "pchip":
            return interpolate.PchipInterpolator(
                x, y, extrapolate=extrapolate, **kwargs
            )
        elif method == "akima":
            # Akima doesn't support extrapolation directly, need to handle separately
            interp = interpolate.Akima1DInterpolator(x, y, **kwargs)
            if extrapolate:
                # Wrap with extrapolate=True
                return lambda xx: interp(xx, extrapolate=True)
            else:
                return interp
        elif method == "barycentric":
            # Barycentric doesn't support extrapolation, will need custom handling
            interp = interpolate.BarycentricInterpolator(x, y, **kwargs)
            if extrapolate:
                # Create a function that extends beyond the bounds
                def interp_func(xx):
                    mask = (xx < x.min()) | (xx > x.max())
                    result = np.empty_like(xx, dtype=float)
                    result[~mask] = interp(xx[~mask])

                    # Extrapolate using the closest points
                    if np.any(xx < x.min()):
                        left_idx = xx < x.min()
                        slope = (y[1] - y[0]) / (x[1] - x[0])
                        result[left_idx] = y[0] + slope * (xx[left_idx] - x[0])

                    if np.any(xx > x.max()):
                        right_idx = xx > x.max()
                        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
                        result[right_idx] = y[-1] + slope * (xx[right_idx] - x[-1])

                    return result

                return interp_func
            else:
                # Return original interpolator
                return interp
        elif method == "krogh":
            # Krogh doesn't support extrapolation directly
            interp = interpolate.KroghInterpolator(x, y, **kwargs)
            if extrapolate:
                # Create a function that extends beyond the bounds
                def interp_func(xx):
                    mask = (xx < x.min()) | (xx > x.max())
                    result = np.empty_like(xx, dtype=float)
                    result[~mask] = interp(xx[~mask])

                    # Extrapolate using the closest points
                    if np.any(xx < x.min()):
                        left_idx = xx < x.min()
                        slope = (y[1] - y[0]) / (x[1] - x[0])
                        result[left_idx] = y[0] + slope * (xx[left_idx] - x[0])

                    if np.any(xx > x.max()):
                        right_idx = xx > x.max()
                        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
                        result[right_idx] = y[-1] + slope * (xx[right_idx] - x[-1])

                    return result

                return interp_func
            else:
                # Return original interpolator
                return interp
        elif method == "polynomial":
            # Use CubicSpline instead of polynomial for better handling of extrapolation
            return interpolate.CubicSpline(x, y, extrapolate=extrapolate, **kwargs)
        else:
            # Default to general interp1d for methods like linear, cubic, etc.
            fill_value = "extrapolate" if extrapolate else np.nan
            return interpolate.interp1d(
                x, y, kind=method, bounds_error=False, fill_value=fill_value, **kwargs
            )

    def interpolate(
        self,
        new_periods,
        inplace=False,
        method="slinear",
        na_method="pchip",
        log_space=False,
        extrapolate=False,
        **kwargs,
    ):
        """Interpolate onto a new period range.

        The way this works is that NaNs
        are first interpolated using method `na_method` along the original
        period map. This allows us to use xarray tools for interpolation. If we
        drop NaNs using xarray it drops each column or row that has a single
        NaN and removes way too much data. Therefore interpolating NaNs first
        keeps most of the data.  Then a 1D interpolation is done for the
        `new_periods` using method `method`.

        'pchip' seems to work best when using xr.interpolate_na

        Set log_space=True if the object being interpolated is in log space,
        like impedance. It seems that functions that are naturally in log-space
        cause issues with the interpolators so taking the log of the function
        seems to produce better results.
        :param new_periods: New periods to interpolate on to.
        :type new_periods: np.ndarray, list
        :param inplace: Interpolate inplace, defaults to False.
        :type inplace: bool, optional
        :param method: Method for 1D linear interpolation options are
            ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"],, defaults to "slinear".
        :type method: string, optional
        :param na_method: Method to interpolate NaNs along original periods
            options are {"linear", "nearest", "zero", "slinear", "quadratic",
            "cubic", "polynomial", "barycentric", "krogh", "pchip", "spline",
            "akima"}, defaults to "pchip".
        :type na_method: string, optional
        :param log_space: Set to true if function is naturally logarithmic,, defaults to False.
        :type log_space: bool, optional
        :param extrapolate: Extrapolate past original period range, default is
            False. If set to True be careful cause the values are not great, defaults to False.
        :type extrapolate: bool, optional
        :param **kwargs: Keyword args passed to interpolation methods.
        :type **kwargs: dict
        :return: Interpolated object.
        :rtype: :class:`mtpy.core.transfer_fuction.base.TFBase`
        """

        da_dict = {}
        for key in self._dataset.data_vars:
            # need to interpolate over nans first, if use dropna loose a lot
            # of data.  going to interpolate anyway. pchip seems to work best
            # for interpolate na.  If this doesn't work think about
            # interpolating component by component.
            if log_space:
                da_drop_nan = np.log(self._dataset[key]).interpolate_na(
                    dim="period", method=na_method
                )
                da_dict[key] = np.exp(
                    da_drop_nan.interp(period=new_periods, method=method, kwargs=kwargs)
                )
            else:
                da_drop_nan = self._dataset[key].interpolate_na(
                    dim="period", method=na_method
                )
                da_dict[key] = da_drop_nan.interp(
                    period=new_periods, method=method, kwargs=kwargs
                )

            # need to abide by the original data that has nans at the
            # beginning and end of the data.  interpolate_na will remove these
            # and gives terrible values, so fill these back in with nans
            if not extrapolate:
                da_dict[key] = self._backfill_nans(self._dataset[key], da_dict[key])

        ds = xr.Dataset(da_dict)

        if inplace:
            self._dataset = ds
        else:
            tb = self.copy()
            tb._dataset = ds
            return tb

    @staticmethod
    def _find_nans_index(data_array):
        """Find nans at beginning and end of xarray.

                When you interpolate a
        xarray.DataArray we interpolate nans, which removes the original nans
                at the beginning and end of the array.  We need to find these
                indicies and period min and max so we can put them back in.
                :param data_array: DESCRIPTION.
                :type data_array: TYPE
                :return: DESCRIPTION.
                :rtype: TYPE
        """

        index_list = []
        for ch_in in data_array.input.data:
            for ch_out in data_array.output.data:
                index = np.where(
                    np.nan_to_num(data_array.loc[{"input": ch_in, "output": ch_out}])
                    == 0
                )[0]
                if len(index) > 0:
                    entry = {
                        "input": ch_in,
                        "output": ch_out,
                        "beginning": [],
                        "end": [],
                        "period_min": float(data_array.period.min()),
                        "period_max": float(data_array.period.max()),
                    }
                    if index[0] == 0:
                        entry["beginning"] = []
                        ii = 0
                        diff = 1
                        while diff == 1 and ii < len(index) - 1:
                            diff = index[ii + 1] - index[ii]
                            entry["beginning"].append(ii)
                            ii += 1
                        if len(entry["beginning"]) > 0:
                            entry["period_min"] = float(
                                data_array.period[entry["beginning"][-1] + 1]
                            )

                    if index[-1] == (data_array.shape[0] - 1):
                        entry["end"] = [-1]
                        ii = -2
                        diff = 1
                        while diff == 1 and abs(ii) < len(index):
                            diff = abs(index[ii - 1] - index[ii])
                            entry["end"].append(ii)
                            ii -= 1
                        entry["period_max"] = float(
                            data_array.period[entry["end"][-1] - 1]
                        )
                    index_list.append(entry)

        return index_list

    def _backfill_nans(self, original, interpolated):
        """Back fill with nans for extrapolated values.
        :param original: Original data array.
        :type original: xarray.DataArray
        :param interpolated: Interpolated data array.
        :type interpolated: xr.DataArray
        :return: Nan's filled in from beginning and end of original data.
        :rtype: xarray.DataArray
        """

        original_index = self._find_nans_index(original)
        for entry in original_index:
            beginning = np.where(interpolated.period < entry["period_min"])
            end = np.where(interpolated.period > entry["period_max"])
            nan_index = np.append(beginning[0], end[0])
            if "complex" in interpolated.dtype.name:
                interpolated.loc[{"input": entry["input"], "output": entry["output"]}][
                    nan_index
                ] = (np.nan + 1j * np.nan)
            else:
                interpolated.loc[{"input": entry["input"], "output": entry["output"]}][
                    nan_index
                ] = np.nan

        return interpolated

    def to_xarray(self):
        """To an xarray dataset.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        return self._dataset

    def from_xarray(self, dataset):
        """Fill from an xarray dataset.
        :param dataset: DESCRIPTION.
        :type dataset: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        ## Probably need more validation than this
        if isinstance(dataset, xr.Dataset):
            self._dataset = dataset

    def to_dataframe(self):
        """Return a pandas dataframe with the appropriate columns as a single
        index, or multi-index?
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        pass

    def from_dataframe(self, dataframe):
        """Fill from a pandas dataframe with the appropriate columns.
        :param dataframe: DESCRIPTION.
        :type dataframe: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        pass
