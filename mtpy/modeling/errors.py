# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:01:37 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np


# =============================================================================


class ModelErrors:
    def __init__(self, data=None, measurement_error=None, **kwargs):
        """Init function."""

        self._functions = {
            "egbert": self.compute_geometric_mean_error,
            "geometric_mean": self.compute_geometric_mean_error,
            "arithmetic_mean": self.compute_arithmetic_mean_error,
            "row": self.compute_row_error,
            "mean_od": self.compute_arithmetic_mean_error,
            "median": self.compute_median_error,
            "eigen": self.compute_eigen_value_error,
            "percent": self.compute_percent_error,
            "absolute": self.compute_absolute_error,
            "abs": self.compute_absolute_error,
            "data": self.use_measurement_error,
        }

        self._array_shapes = {
            "impedance": (2, 2),
            "z": (2, 2),
            "transfer_function": (3, 2),
            "tipper": (1, 2),
            "t": (1, 2),
        }

        self.error_value = 5
        self.error_type = "percent"
        self.floor = True
        self.mode = "impedance"

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.data = data
        self.measurement_error = measurement_error

    def __str__(self):
        """Str function."""
        lines = ["Model Errors:", "-" * 20]
        lines += [f"\terror_type:    {self.error_type}"]
        lines += [f"\terror_value:   {self.error_value}"]
        lines += [f"\tfloor:         {self.floor}"]
        lines += [f"\tmode:          {self.mode}"]

        return "\n".join(lines)

    def __repr__(self):
        """Repr function."""
        return self.__str__()

    def __eq__(self, other):
        """Eq function."""
        if not isinstance(other, ModelErrors):
            raise TypeError(f"Cannot compare ModelErrors to type {type(other)}")

        for key in ["error_value", "error_type", "floor", "mode"]:
            value_og = getattr(self, key)
            value_other = getattr(other, key)

            if value_og != value_other:
                return False
        return True

    def validate_percent(self, value):
        """Make sure the percent is a decimal.
        :param value: DESCRIPTION.
        :type value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if value >= 1:
            value /= 100.0

        return value

    @property
    def error_parameters(self):
        """Error parameters."""
        return {
            "error_value": self.error_value,
            "error_type": self.error_type,
            "floor": self.floor,
        }

    @property
    def error_type(self):
        """Error type."""
        return self._error_type

    @error_type.setter
    def error_type(self, value):
        """Error type."""
        if value not in self._functions.keys():
            raise NotImplementedError(f"Error Type {value} not supported.")
        self._error_type = value

    @property
    def floor(self):
        """Floor function."""
        return self._floor

    @floor.setter
    def floor(self, value):
        """Floor function."""
        if value not in [False, True]:
            raise ValueError("Floor must be True or False")
        self._floor = value

    @property
    def error_value(self):
        """Error value."""
        return self._error_value

    @error_value.setter
    def error_value(self, value):
        """Error value."""
        self._error_value = self.validate_percent(value)

    @property
    def mode(self):
        """Mode function."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """Mode function."""
        if value not in self._array_shapes.keys():
            raise NotImplementedError(f"Mode {value} not supported.")
        self._mode = value

    def _get_shape(self):
        """Get shape."""
        try:
            return self._array_shapes[self.mode]

        except KeyError:
            raise NotImplementedError(f"Mode {self.mode} not supported.")

    def validate_array_shape(self, data):
        """Validate array shape.
        :param data: DESCRIPTION.
        :type data: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        expected_shape = self._get_shape()
        if data.shape == expected_shape:
            data = data.reshape((1, expected_shape[0], expected_shape[1]))

        if data.shape[1] != expected_shape[0] or data.shape[2] != expected_shape[1]:
            raise ValueError(
                f"Shape {data.shape} is not expected shape of (n, "
                f"{expected_shape[0]}, {expected_shape[1]})"
            )

        return data

    @property
    def data(self):
        """Data function."""
        return self._data

    @data.setter
    def data(self, value):
        """Data function."""
        if value is not None:
            self._data = self.validate_array_shape(value)
        else:
            self._data = None

    @property
    def measurement_error(self):
        """Measurement error."""
        return self._measurement_error

    @measurement_error.setter
    def measurement_error(self, value):
        """Measurement error."""
        if value is not None:
            self._measurement_error = self.validate_array_shape(value)
        else:
            self._measurement_error = None

    def mask_zeros(self, data):
        """Mask zeros.
        :param data: DESCRIPTION.
        :type data: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        dshape = data.shape
        data = np.nan_to_num(data).reshape(dshape)
        return np.ma.masked_equal(data, 0)

    def resize_output(self, error_array):
        """Resize the error estimtion to the same size as the input data.
        :param error_array: DESCRIPTION.
        :type error_array: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if error_array.shape != self.data.shape:
            if error_array.shape[0] == self.data.shape[0]:
                err = np.zeros_like(self.data, dtype=float)
                for index in range(self.data.shape[0]):
                    err[index] = error_array[index]
                return err

        return error_array

    def set_floor(self, error_array):
        """Set error floor.
        :param error_array:
        :param array: DESCRIPTION.
        :type array: TYPE
        :param floor: DESCRIPTION.
        :type floor: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self.measurement_error is not None:
            index = np.where(error_array < self.measurement_error)
            error_array[index] = self.measurement_error[index]

        return error_array

    def use_measurement_error(self):
        """Use measurement error."""
        return self.measurement_error

    def compute_percent_error(self):
        """Percent error.
        :param data: DESCRIPTION.
        :type data: TYPE
        :param percent: DESCRIPTION.
        :type percent: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        err = self.error_value * np.abs(self.data)
        if self.floor:
            err = self.set_floor(err)

        return err

    def compute_arithmetic_mean_error(self):
        """Error_value * (Zxy + Zyx) / 2.
        :param data: DESCRIPTION.
        :type data: TYPE
        :param error_value: DESCRIPTION.
        :type error_value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self.data.shape[1] < 2:
            od = self.mask_zeros(np.array([self.data[:, 0, 0], self.data[:, 0, 1]]))

        else:
            od = self.mask_zeros(np.array([self.data[:, 0, 1], self.data[:, 1, 0]]))
        err = self.resize_output(self.error_value * np.ma.mean(np.ma.abs(od), axis=0))

        if self.floor:
            err = self.set_floor(err)

        if isinstance(err, np.ma.core.MaskedArray):
            return err.data

        return err

    def compute_median_error(self):
        """Median(array) * error_value.
        :param array: DESCRIPTION.
        :type array: TYPE
        :param error_value: DESCRIPTION.
        :type error_value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        data = self.mask_zeros(self.data)
        err = self.resize_output(
            np.abs(np.ma.median(data, axis=(1, 2))) * self.error_value
        )

        if self.floor:
            err = self.set_floor(err)

        if isinstance(err, np.ma.core.MaskedArray):
            return err.data

        return err

    def compute_eigen_value_error(self):
        """Error_value * eigen(data).mean().
        :param data: DESCRIPTION.
        :type data: TYPE
        :param error_value: DESCRIPTION.
        :type error_value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self.data.shape[1] < 2:
            raise IndexError(
                "Cannot compute eigen value error with an array of shape "
                f"{self.data.shape}"
            )

        data = self.mask_zeros(self.data)

        try:
            err = self.error_value * np.abs(np.linalg.eigvals(data)).mean(axis=1)
        except Exception:
            data_shape = data.shape
            err = (
                self.error_value
                * np.abs(
                    np.linalg.eigvals(np.nan_to_num(data).reshape(data_shape))
                ).mean()
            )

        if np.atleast_1d(err).sum(axis=0) == 0:
            err = self.error_value * data[np.nonzero(data)].mean()

        err = self.resize_output(err)

        if self.floor:
            err = self.set_floor(err)
        return err

    def compute_geometric_mean_error(self):
        """Error_value * sqrt(Zxy * Zyx).
        :param data: DESCRIPTION.
        :type data: TYPE
        :param error_value: DESCRIPTION.
        :type error_value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        data = self.data.copy()

        if self.data.shape[1] < 2:
            zero_xy = np.where(data[:, 0, 0] == 0)
            data[zero_xy, 0, 0] = data[zero_xy, 0, 0]

            zero_yx = np.where(data[:, 0, 1] == 0)
            data[zero_yx, 0, 1] = data[zero_yx, 0, 1]

            data = self.mask_zeros(data)

            err = self.resize_output(
                self.error_value * np.ma.sqrt(np.ma.abs(data[:, 0, 0] * data[:, 0, 1]))
            )

        else:
            zero_xy = np.where(data[:, 0, 1] == 0)
            data[zero_xy, 0, 1] = data[zero_xy, 1, 0]

            zero_yx = np.where(data[:, 1, 0] == 0)
            data[zero_yx, 1, 0] = data[zero_yx, 0, 1]

            data = self.mask_zeros(data)

            err = self.resize_output(
                self.error_value * np.ma.sqrt(np.ma.abs(data[:, 0, 1] * data[:, 1, 0]))
            )

        if self.floor:
            err = self.set_floor(err)

        if isinstance(err, np.ma.core.MaskedArray):
            return err.data

        return err

    def compute_row_error(self):
        """Set zxx and zxy the same error and zyy and zyx the same error.
        :param data: DESCRIPTION.
        :type data: TYPE
        :param error_value: DESCRIPTION.
        :type error_value: TYPE
        :param floor: DESCRIPTION, defaults to True.
        :type floor: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self.data.shape[1] < 2:
            err_xy = np.abs(self.data[:, 0, 0]) * self.error_value
            err_yx = np.abs(self.data[:, 0, 1]) * self.error_value

            err = np.zeros_like(self.data, dtype=float)
            err[:, 0, 0] = err_xy
            err[:, 0, 1] = err_yx

        else:
            err_xy = np.abs(self.data[:, 0, 1]) * self.error_value
            err_yx = np.abs(self.data[:, 1, 0]) * self.error_value

            err = np.zeros_like(self.data, dtype=float)
            err[:, 0, 0] = err_xy
            err[:, 0, 1] = err_xy
            err[:, 1, 0] = err_yx
            err[:, 1, 1] = err_yx

        err = self.resize_output(err)
        if self.floor:
            err = self.set_floor(err)

        return err

    def compute_absolute_error(self):
        """Compute absolute error.
        :param data: DESCRIPTION.
        :type data: TYPE
        :param error_value: DESCRIPTION.
        :type error_value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        err = np.ones_like(self.data, dtype=float) * self.error_value

        if self.floor:
            err = self.set_floor(err)
        return err

    def compute_error(self, data=None, error_type=None, error_value=None, floor=None):
        """Compute error.
        :param data: DESCRIPTION, defaults to None.
        :type data: TYPE, optional
        :param error_type: DESCRIPTION, defaults to None.
        :type error_type: TYPE, optional
        :param error_value: DESCRIPTION, defaults to None.
        :type error_value: TYPE, optional
        :param floor: DESCRIPTION, defaults to None.
        :type floor: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if data is not None:
            self.data = data
        if error_type is not None:
            self.error_type = error_type
        if error_value is not None:
            self.error_value = error_value
        if floor is not None:
            self.floor = floor

        return self._functions[self.error_type]()
