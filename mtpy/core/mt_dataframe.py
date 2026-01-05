# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:20:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from . import Tipper, Z


# =============================================================================


class MTDataFrame:
    """
    DataFrame for a single MT station.

    Parameters
    ----------
    data : dict, np.ndarray, pd.DataFrame, MTDataFrame, or None, optional
        Initial data to populate the dataframe, by default None
    n_entries : int, optional
        Number of empty entries to create if data is None, by default 0
    **kwargs : dict
        Additional keyword arguments for setting attributes

    Attributes
    ----------
    dataframe : pd.DataFrame
        Pandas DataFrame containing MT data with standardized columns
    working_survey : str or None
        Current working survey name
    working_station : str or None
        Current working station name

    Notes
    -----
    Tried subclassing pandas.DataFrame, but that turned out to not be
    straightforward, so went with composition instead.

    """

    def __init__(
        self,
        data: dict | np.ndarray | pd.DataFrame | "MTDataFrame" | None = None,
        n_entries: int = 0,
        **kwargs: Any,
    ) -> None:
        self._dtype_list = [
            ("survey", "U25"),
            ("station", "U25"),
            ("latitude", float),
            ("longitude", float),
            ("elevation", float),
            ("datum_epsg", "U6"),
            ("east", float),
            ("north", float),
            ("utm_epsg", "U6"),
            ("model_east", float),
            ("model_north", float),
            ("model_elevation", float),
            ("profile_offset", float),
            ("period", float),
            ("z_xx", complex),
            ("z_xx_error", float),
            ("z_xx_model_error", float),
            ("z_xy", complex),
            ("z_xy_error", float),
            ("z_xy_model_error", float),
            ("z_yx", complex),
            ("z_yx_error", float),
            ("z_yx_model_error", float),
            ("z_yy", complex),
            ("z_yy_error", float),
            ("z_yy_model_error", float),
            ("t_zx", complex),
            ("t_zx_error", float),
            ("t_zx_model_error", float),
            ("t_zy", complex),
            ("t_zy_error", float),
            ("t_zy_model_error", float),
            ("t_mag_real", float),
            ("t_mag_real_error", float),
            ("t_mag_real_model_error", float),
            ("t_mag_imag", float),
            ("t_mag_imag_error", float),
            ("t_mag_imag_model_error", float),
            ("t_angle_real", float),
            ("t_angle_real_error", float),
            ("t_angle_real_model_error", float),
            ("t_angle_imag", float),
            ("t_angle_imag_error", float),
            ("t_angle_imag_model_error", float),
            ("res_xx", float),
            ("res_xx_error", float),
            ("res_xx_model_error", float),
            ("res_xy", float),
            ("res_xy_error", float),
            ("res_xy_model_error", float),
            ("res_yx", float),
            ("res_yx_error", float),
            ("res_yx_model_error", float),
            ("res_yy", float),
            ("res_yy_error", float),
            ("res_yy_model_error", float),
            ("phase_xx", float),
            ("phase_xx_error", float),
            ("phase_xx_model_error", float),
            ("phase_xy", float),
            ("phase_xy_error", float),
            ("phase_xy_model_error", float),
            ("phase_yx", float),
            ("phase_yx_error", float),
            ("phase_yx_model_error", float),
            ("phase_yy", float),
            ("phase_yy_error", float),
            ("phase_yy_model_error", float),
            ("pt_xx", float),
            ("pt_xx_error", float),
            ("pt_xx_model_error", float),
            ("pt_xy", float),
            ("pt_xy_error", float),
            ("pt_xy_model_error", float),
            ("pt_yx", float),
            ("pt_yx_error", float),
            ("pt_yx_model_error", float),
            ("pt_yy", float),
            ("pt_yy_error", float),
            ("pt_yy_model_error", float),
            ("pt_phimin", float),
            ("pt_phimin_error", float),
            ("pt_phimin_model_error", float),
            ("pt_phimax", float),
            ("pt_phimax_error", float),
            ("pt_phimax_model_error", float),
            ("pt_azimuth", float),
            ("pt_azimuth_error", float),
            ("pt_azimuth_model_error", float),
            ("pt_skew", float),
            ("pt_skew_error", float),
            ("pt_skew_model_error", float),
            ("pt_ellipticity", float),
            ("pt_ellipticity_error", float),
            ("pt_ellipticity_model_error", float),
            ("pt_det", float),
            ("pt_det_error", float),
            ("pt_det_model_error", float),
            ("rms_zxx", float),
            ("rms_zxy", float),
            ("rms_zyx", float),
            ("rms_zyy", float),
            ("rms_tzx", float),
            ("rms_tzy", float),
        ]

        self._index_dict = {
            "xx": {"ii": 0, "jj": 0},
            "xy": {"ii": 0, "jj": 1},
            "yx": {"ii": 1, "jj": 0},
            "yy": {"ii": 1, "jj": 1},
            "zx": {"ii": 0, "jj": 0},
            "zy": {"ii": 0, "jj": 1},
        }

        self._key_dict = {
            "z": "z",
            "res": "resistivity",
            "phase": "phase",
            "pt": "phase_tensor",
            "t": "tipper",
        }

        self._station_location_attrs = [
            "survey",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "datum_epsg",
            "east",
            "north",
            "utm_epsg",
            "model_east",
            "model_north",
            "model_elevation",
            "profile_offset",
        ]

        if data is not None:
            self.dataframe = self._validate_data(data)

        else:
            self.dataframe = self._get_initial_df(n_entries)

        self.working_survey = None
        self.working_station = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        """
        String representation of MTDataFrame.

        Returns
        -------
        str
            String representation of the dataframe or empty message

        """
        if self._has_data():
            return self.dataframe.__str__()

        else:
            return "Empty MTStationDataFrame"

    def __repr__(self) -> str:
        """
        Representation of MTDataFrame.

        Returns
        -------
        str
            String representation of the dataframe or empty constructor

        """
        if self._has_data():
            return self.dataframe.__repr__()
        else:
            return "MTStationDataFrame()"

    @property
    def _column_names(self) -> list[str]:
        """
        List of all column names in the dataframe.

        Returns
        -------
        list of str
            Column names extracted from dtype list

        """
        return [col[0] for col in self._dtype_list]

    @property
    def _pt_attrs(self) -> list[str]:
        """
        List of phase tensor attribute column names.

        Returns
        -------
        list of str
            Column names starting with 'pt'

        """
        return [col for col in self._column_names if col.startswith("pt")]

    @property
    def _tipper_attrs(self) -> list[str]:
        """
        List of tipper attribute column names.

        Returns
        -------
        list of str
            Column names starting with 't_'

        """
        return [col for col in self._column_names if col.startswith("t_")]

    def __eq__(self, other: Any) -> bool:
        """
        Compare two MTDataFrame objects for equality.

        Parameters
        ----------
        other : MTDataFrame or compatible data type
            Another dataframe to compare with

        Returns
        -------
        bool
            True if dataframes are equal, False otherwise

        """
        other = self._validate_data(other)
        return (self.dataframe == other).all().all()

    @property
    def nonzero_items(self) -> int:
        """
        Count number of non-zero entries in data columns.

        Returns
        -------
        int
            Number of non-zero entries excluding error columns

        """

        if self._has_data():
            cols = [
                dtype[0] for dtype in self._dtype_list[14:] if "error" not in dtype[0]
            ]

            return np.count_nonzero(self.dataframe[cols])
        else:
            return 0

    def _validate_data(
        self, data: dict | np.ndarray | pd.DataFrame | "MTDataFrame" | None
    ) -> pd.DataFrame | None:
        """
        Validate and convert input data to standardized DataFrame format.

        Parameters
        ----------
        data : dict, np.ndarray, pd.DataFrame, MTDataFrame, or None
            Input data to validate

        Returns
        -------
        pd.DataFrame or None
            Validated and standardized DataFrame

        Raises
        ------
        TypeError
            If data type is not supported

        """

        if data is None:
            return

        if isinstance(data, (dict, np.ndarray, pd.DataFrame)):
            df = pd.DataFrame(data)

        elif isinstance(data, (MTDataFrame)):
            df = data.dataframe

        else:
            raise TypeError(f"Input data must be a pandas.DataFrame not {type(data)}")

        for col in self._dtype_list:
            if col[0] not in df.columns:
                df[col[0]] = np.zeros(df.shape[0], dtype=col[1])

        # resort to the desired column order
        if df.columns.to_list() != self._column_names:
            df = df[self._column_names]

        return df

    def _get_initial_df(self, n_entries: int = 0) -> pd.DataFrame:
        """
        Create an empty DataFrame with the standard MT data structure.

        Parameters
        ----------
        n_entries : int, optional
            Number of empty rows to create, by default 0

        Returns
        -------
        pd.DataFrame
            Empty DataFrame with standard columns

        """
        return pd.DataFrame(np.empty(n_entries, dtype=np.dtype(self._dtype_list)))

    def _has_data(self) -> bool:
        """
        Check if dataframe contains any data.

        Returns
        -------
        bool
            True if dataframe has rows, False otherwise

        """
        if self.dataframe is None:
            return False
        elif self.dataframe.shape[0] > 0:
            return True
        return False

    def get_station_df(self, station: str | None = None) -> pd.DataFrame:
        """
        Get DataFrame for a single station.

        Parameters
        ----------
        station : str, optional
            Station name to retrieve, by default None (uses working_station)

        Returns
        -------
        pd.DataFrame
            DataFrame filtered for the specified station

        Raises
        ------
        ValueError
            If station is not found in dataframe

        """
        if station is not None:
            self.working_station = station
        if self._has_data():
            if self.working_station is None:
                self.working_station = self.dataframe.station.unique()[0]

            if self.working_station not in self.dataframe.station.values:
                raise ValueError(
                    f"Could not find station {self.working_station} in dataframe."
                )

            return self.dataframe[self.dataframe.station == self.working_station]

    @property
    def size(self) -> int | None:
        """
        Number of periods in the dataframe.

        Returns
        -------
        int or None
            Number of unique periods, or None if no data

        """
        if self._has_data():
            return self.period.size

    def _get_index(self, comp: str) -> dict[str, int] | None:
        """
        Get component index values for tensor elements.

        Parameters
        ----------
        comp : str
            Component identifier: 'xx', 'xy', 'yx', 'yy', 'zx', or 'zy'

        Returns
        -------
        dict or None
            Dictionary with 'ii' and 'jj' keys for array indices, or None if invalid

        """
        if comp in self._index_dict.keys():
            return self._index_dict[comp]

    def _get_key_index(self, key: str) -> dict[str, int] | None:
        """
        Get index from a column key name.

        Parameters
        ----------
        key : str
            Column name (e.g., 'z_xy', 't_zx')

        Returns
        -------
        dict or None
            Dictionary with 'ii' and 'jj' keys for array indices

        """

        if key.count("_") > 0:
            comp = key.split("_")[1]
            return self._get_index(comp)

    @property
    def period(self) -> np.ndarray | None:
        """
        Array of unique periods in sorted order.

        Returns
        -------
        np.ndarray or None
            Sorted array of period values, or None if no data

        """

        if self._has_data():
            return np.sort(self.dataframe.period.unique())

    @property
    def frequency(self) -> np.ndarray | None:
        """
        Array of unique frequencies in Hz.

        Returns
        -------
        np.ndarray or None
            Array of frequency values (1/period), or None if no data

        """

        if self._has_data():
            return 1.0 / self.period

    def get_period(self, period: float, tol: float | None = None) -> "MTDataFrame":
        """
        Get data for a specific period or period range.

        Parameters
        ----------
        period : float
            Target period value in seconds
        tol : float, optional
            Tolerance as a fraction (e.g., 0.05 for 5%), by default None

        Returns
        -------
        MTDataFrame
            New MTDataFrame containing only the requested period(s)

        """
        if tol is not None:
            return MTDataFrame(
                self.dataframe[
                    (self.dataframe.period > period * (1 - tol)) & self.dataframe.period
                    < period * (1 + tol)
                ]
            )
        else:
            return MTDataFrame(self.dataframe[self.dataframe.period == period])

    @property
    def survey(self) -> str | None:
        """
        Survey name from the dataframe.

        Returns
        -------
        str or None
            Survey name, or None if no data

        """
        if self._has_data():
            if self.working_survey is None:
                self.working_survey = self.dataframe.survey.unique()[0]
            return self.working_survey

    @survey.setter
    def survey(self, value: str) -> None:
        """
        Set survey name in the dataframe.

        Parameters
        ----------
        value : str
            Survey name to set

        """
        if self._has_data():
            if self.working_survey in [None, ""]:
                self.dataframe.loc[self.dataframe.survey == "", "survey"] = value
                self.working_survey = value

    @property
    def station(self) -> str | None:
        """
        Station name from the dataframe.

        Returns
        -------
        str or None
            Station name, or None if no data

        """
        if self._has_data():
            if self.working_station is None:
                self.working_station = self.dataframe.station.unique()[0]
            return self.working_station

    @station.setter
    def station(self, value: str) -> None:
        """
        Set station name in the dataframe.

        Parameters
        ----------
        value : str
            Station name to set

        """
        if self._has_data():
            if self.working_station in [None, ""]:
                self.dataframe.loc[self.dataframe.station == "", "station"] = value
                self.working_station = value

    @property
    def latitude(self) -> float | None:
        """
        Station latitude in decimal degrees.

        Returns
        -------
        float or None
            Latitude value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "latitude"
            ].unique()[0]

    @latitude.setter
    def latitude(self, value: float) -> None:
        """
        Set station latitude.

        Parameters
        ----------
        value : float
            Latitude in decimal degrees

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "latitude"
            ] = value

    @property
    def longitude(self) -> float | None:
        """
        Station longitude in decimal degrees.

        Returns
        -------
        float or None
            Longitude value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "longitude"
            ].unique()[0]

    @longitude.setter
    def longitude(self, value: float) -> None:
        """
        Set station longitude.

        Parameters
        ----------
        value : float
            Longitude in decimal degrees

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "longitude"
            ] = value

    @property
    def elevation(self) -> float | None:
        """
        Station elevation in meters.

        Returns
        -------
        float or None
            Elevation value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "elevation"
            ].unique()[0]

    @elevation.setter
    def elevation(self, value: float) -> None:
        """
        Set station elevation.

        Parameters
        ----------
        value : float
            Elevation in meters

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "elevation"
            ] = value

    @property
    def datum_epsg(self) -> str | None:
        """
        Datum EPSG code.

        Returns
        -------
        str or None
            EPSG code string, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "datum_epsg"
            ].unique()[0]

    @datum_epsg.setter
    def datum_epsg(self, value: str) -> None:
        """
        Set datum EPSG code.

        Parameters
        ----------
        value : str
            EPSG code string

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "datum_epsg"
            ] = value

    @property
    def east(self) -> float | None:
        """
        Station easting coordinate in meters.

        Returns
        -------
        float or None
            Easting value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "east"
            ].unique()[0]

    @east.setter
    def east(self, value: float) -> None:
        """
        Set station easting coordinate.

        Parameters
        ----------
        value : float
            Easting in meters

        """
        if self._has_data():
            self.dataframe.loc[self.dataframe.station == self.station, "east"] = value

    @property
    def north(self) -> float | None:
        """
        Station northing coordinate in meters.

        Returns
        -------
        float or None
            Northing value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "north"
            ].unique()[0]

    @north.setter
    def north(self, value: float) -> None:
        """
        Set station northing coordinate.

        Parameters
        ----------
        value : float
            Northing in meters

        """
        if self._has_data():
            self.dataframe.loc[self.dataframe.station == self.station, "north"] = value

    @property
    def utm_epsg(self) -> str | None:
        """
        UTM EPSG code.

        Returns
        -------
        str or None
            EPSG code string, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "utm_epsg"
            ].unique()[0]

    @utm_epsg.setter
    def utm_epsg(self, value: str) -> None:
        """
        Set UTM EPSG code.

        Parameters
        ----------
        value : str
            EPSG code string

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "utm_epsg"
            ] = value

    @property
    def model_east(self) -> float | None:
        """
        Model easting coordinate in meters.

        Returns
        -------
        float or None
            Model easting value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_east"
            ].unique()[0]

    @model_east.setter
    def model_east(self, value: float) -> None:
        """
        Set model easting coordinate.

        Parameters
        ----------
        value : float
            Model easting in meters

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "model_east"
            ] = value

    @property
    def model_north(self) -> float | None:
        """
        Model northing coordinate in meters.

        Returns
        -------
        float or None
            Model northing value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_north"
            ].unique()[0]

    @model_north.setter
    def model_north(self, value: float) -> None:
        """
        Set model northing coordinate.

        Parameters
        ----------
        value : float
            Model northing in meters

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "model_north"
            ] = value

    @property
    def model_elevation(self) -> float | None:
        """
        Model elevation in meters.

        Returns
        -------
        float or None
            Model elevation value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_elevation"
            ].unique()[0]

    @model_elevation.setter
    def model_elevation(self, value: float) -> None:
        """
        Set model elevation.

        Parameters
        ----------
        value : float
            Model elevation in meters

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station,
                "model_elevation",
            ] = value

    @property
    def profile_offset(self) -> float | None:
        """
        Distance along profile in meters.

        Returns
        -------
        float or None
            Profile offset value, or None if no data

        """
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "profile_offset"
            ].unique()[0]

    @profile_offset.setter
    def profile_offset(self, value: float) -> None:
        """
        Set distance along profile.

        Parameters
        ----------
        value : float
            Profile offset in meters

        """
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station,
                "profile_offset",
            ] = value

    def _get_empty_impedance_array(self, dtype: type = complex) -> np.ndarray:
        """
        Create an empty impedance tensor array.

        Parameters
        ----------
        dtype : type, optional
            Data type for the array, by default complex

        Returns
        -------
        np.ndarray
            Array of shape (n_periods, 2, 2) filled with zeros

        """
        return np.zeros((self.period.size, 2, 2), dtype=complex)

    @property
    def impedance(self) -> np.ndarray:
        """
        Impedance tensor from dataframe.

        Returns
        -------
        np.ndarray
            Impedance tensor of shape (n_periods, 2, 2)

        """
        z = self._get_empty_impedance_array()

        for key in ["zxx", "zxy", "zyx", "zyy"]:
            index = self._get_index(key)
            z[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, key
            ]
        return z

    def _get_data_array(self, obj: Any, attr: str) -> np.ndarray | None:
        """
        Get data array from an object's attribute.

        Parameters
        ----------
        obj : object
            Object to retrieve attribute from
        attr : str
            Attribute name

        Returns
        -------
        np.ndarray or None
            Data array if attribute exists, None otherwise

        """
        try:
            return getattr(obj, attr)
        except TypeError:
            return None

    def _fill_data(
        self,
        data_array: np.ndarray | None,
        column: str,
        index: dict[str, int] | None,
    ) -> None:
        """
        Fill dataframe column with data from an array.

        Parameters
        ----------
        data_array : np.ndarray or None
            Data to fill into the dataframe
        column : str
            Column name to fill
        index : dict or None
            Dictionary with 'ii' and 'jj' keys for array slicing, or None for 1D data

        """

        if data_array is not None:
            if index is None:
                self.dataframe.loc[
                    self.dataframe.station == self.station,
                    column,
                ] = data_array[:]
            else:
                self.dataframe.loc[
                    self.dataframe.station == self.station,
                    column,
                ] = data_array[:, index["ii"], index["jj"]]

    def from_z_object(self, z_object: Z) -> None:
        """
        Fill dataframe from a Z impedance object.

        Parameters
        ----------
        z_object : Z
            Z object containing impedance tensor data

        Notes
        -----
        Populates impedance, resistivity, phase, and phase tensor columns

        """

        self.dataframe.loc[
            self.dataframe.station == self.station, "period"
        ] = z_object.period

        # should make a copy of the phase tensor otherwise it gets calculated
        # multiple times and becomes a time sink.
        pt_object = z_object.phase_tensor.copy()

        for error in ["", "_error", "_model_error"]:
            if getattr(z_object, f"_has_tf{error}")():
                for key in ["z", "res", "phase"]:
                    obj_key = self._key_dict[key]
                    data_array = getattr(z_object, f"{obj_key}{error}").copy()
                    for comp in ["xx", "xy", "yx", "yy"]:
                        index = self._get_index(comp)
                        # if key in ["pt"]:
                        #     data_array = self._get_data_array(
                        #         obj, f"{key}{error}"
                        #     )
                        # else:
                        #     data_array = self._get_data_array(
                        #         z_object, f"{obj_key}{error}"
                        #     )

                        self._fill_data(data_array, f"{key}_{comp}{error}", index)

                ## phase tensor
                data_array = getattr(pt_object, f"pt{error}")
                for comp in ["xx", "xy", "yx", "yy"]:
                    index = self._get_index(comp)
                    self._fill_data(data_array, f"pt_{comp}{error}", index)
                # PT attributes
                for pt_attr in [
                    "phimin",
                    "phimax",
                    "azimuth",
                    "skew",
                    "ellipticity",
                    "det",
                ]:
                    data_array = self._get_data_array(pt_object, f"{pt_attr}{error}")
                    self._fill_data(data_array, f"pt_{pt_attr}{error}", None)

    def from_t_object(self, t_object: Tipper) -> None:
        """
        Fill dataframe from a Tipper object.

        Parameters
        ----------
        t_object : Tipper
            Tipper object containing tipper data

        Notes
        -----
        Populates tipper magnitude, angle, and component columns

        """
        self.dataframe.loc[
            self.dataframe.station == self.station, "period"
        ] = t_object.period

        for error in ["", "_error", "_model_error"]:
            if getattr(t_object, f"_has_tf{error}")():
                obj_key = self._key_dict["t"]
                for comp in ["zx", "zy"]:
                    index = self._get_index(comp)
                    data_array = self._get_data_array(t_object, f"{obj_key}{error}")
                    self._fill_data(data_array, f"t_{comp}{error}", index)

                if error in [""]:
                    for t_attr in [
                        "mag_real",
                        "mag_imag",
                        "angle_real",
                        "angle_imag",
                    ]:
                        data_array = self._get_data_array(t_object, t_attr)
                        self._fill_data(data_array, f"t_{t_attr}", None)

    def to_z_object(self, units: str = "mt") -> Z:
        """
        Create a Z impedance object from dataframe data.

        Parameters
        ----------
        units : str, optional
            Impedance units ('mt' or 'ohm'), by default 'mt'

        Returns
        -------
        Z
            Z object containing impedance tensor

        Notes
        -----
        If impedance values are zero, attempts to reconstruct from
        resistivity and phase data

        """

        nf = self.period.size
        z = np.zeros((nf, 2, 2), dtype=complex)
        z_err = np.zeros((nf, 2, 2), dtype=float)
        z_model_err = np.zeros((nf, 2, 2), dtype=float)

        res = np.zeros((nf, 2, 2), dtype=float)
        res_err = np.zeros((nf, 2, 2), dtype=float)
        res_model_err = np.zeros((nf, 2, 2), dtype=float)

        phase = np.zeros((nf, 2, 2), dtype=float)
        phase_err = np.zeros((nf, 2, 2), dtype=float)
        phase_model_err = np.zeros((nf, 2, 2), dtype=float)

        for comp in ["xx", "xy", "yx", "yy"]:
            index = self._get_index(comp)

            z[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, f"z_{comp}"
            ]
            z_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, f"z_{comp}_error"
            ]

            z_model_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, f"z_{comp}_model_error"
            ]

        z_object = Z(z, z_err, self.frequency, z_model_err, units=units)

        if (z == 0).all():
            for comp in ["xx", "xy", "yx", "yy"]:
                index = self._get_index(comp)
                ### resistivity
                res[:, index["ii"], index["jj"]] = self.dataframe.loc[
                    self.dataframe.station == self.station, f"res_{comp}"
                ]
                res_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                    self.dataframe.station == self.station, f"res_{comp}_error"
                ]
                res_model_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                    self.dataframe.station == self.station,
                    f"res_{comp}_model_error",
                ]

                ### Phase
                phase[:, index["ii"], index["jj"]] = self.dataframe.loc[
                    self.dataframe.station == self.station, f"phase_{comp}"
                ]
                phase_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                    self.dataframe.station == self.station,
                    f"phase_{comp}_error",
                ]

                phase_model_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                    self.dataframe.station == self.station,
                    f"phase_{comp}_model_error",
                ]

            if not (res == 0).all():
                if not (phase == 0).all():
                    z_object.set_resistivity_phase(
                        res,
                        phase,
                        self.frequency,
                        res_error=res_err,
                        phase_error=phase_err,
                        res_model_error=res_model_err,
                        phase_model_error=phase_model_err,
                    )
                else:
                    raise ValueError("cannot estimate Z without phase information")

        return z_object

    def to_t_object(self) -> Tipper:
        """
        Create a Tipper object from dataframe data.

        Returns
        -------
        Tipper
            Tipper object containing tipper data

        """

        nf = self.period.size

        t = np.zeros((nf, 1, 2), dtype=complex)
        t_err = np.zeros((nf, 1, 2), dtype=float)
        t_model_err = np.zeros((nf, 1, 2), dtype=float)

        for comp in ["zx", "zy"]:
            index = self._get_index(comp)
            t[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, f"t_{comp}"
            ]
            t_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, f"t_{comp}_error"
            ]
            t_model_err[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, f"t_{comp}_model_error"
            ]

        return Tipper(t, t_err, self.frequency, t_model_err)

    @property
    def station_locations(self) -> pd.DataFrame:
        """
        DataFrame of station location information.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per station containing location attributes

        """
        return (
            self.dataframe.groupby("station")
            .nth(0)[self._station_location_attrs]
            .reset_index()
        )

    @property
    def phase_tensor(self) -> pd.DataFrame:
        """
        DataFrame with phase tensor information.

        Returns
        -------
        pd.DataFrame
            DataFrame containing location and phase tensor attributes

        """

        return self.dataframe[
            self._station_location_attrs + self._pt_attrs
        ].reset_index()

    @property
    def tipper(self) -> pd.DataFrame:
        """
        DataFrame with tipper information.

        Returns
        -------
        pd.DataFrame
            DataFrame containing location and tipper attributes

        """

        return self.dataframe[
            self._station_location_attrs + self._tipper_attrs
        ].reset_index()

    @property
    def for_shapefiles(self) -> pd.DataFrame:
        """
        DataFrame formatted for shapefile export.

        Returns
        -------
        pd.DataFrame
            DataFrame with location, phase tensor, and tipper attributes

        """

        return self.dataframe[
            self._station_location_attrs + self._pt_attrs + self._tipper_attrs
        ].reset_index()

    def get_station_distances(self, utm: bool = False) -> pd.Series:
        """
        Calculate pairwise distances between stations.

        Parameters
        ----------
        utm : bool, optional
            If True, use UTM coordinates (east, north), otherwise use
            geographic coordinates (longitude, latitude), by default False

        Returns
        -------
        pd.Series
            Series of non-zero pairwise distances between stations

        """
        if utm:
            x_key = "east"
            y_key = "north"
        else:
            x_key = "longitude"
            y_key = "latitude"
        sdf = self.station_locations
        distances = pdist(sdf[[x_key, y_key]].values, metric="euclidean")
        distances = distances[np.nonzero(distances)]

        return pd.Series(distances)
