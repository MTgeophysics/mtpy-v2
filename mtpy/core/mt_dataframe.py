# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:20:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from . import Z, Tipper

# =============================================================================


class MTDataFrame:
    """Dataframe for a single station

    Tried subclassing pandas.DataFrame, but that turned out to not be straight
    forward, so when with compilation instead.

    Think about having period as an index?.
    """

    def __init__(self, data=None, n_entries=0, **kwargs):
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

    def __str__(self):
        """Str function."""
        if self._has_data():
            return self.dataframe.__str__()

        else:
            return "Empty MTStationDataFrame"

    def __repr__(self):
        """Repr function."""
        if self._has_data():
            return self.dataframe.__repr__()
        else:
            return "MTStationDataFrame()"

    @property
    def _column_names(self):
        """Column names."""
        return [col[0] for col in self._dtype_list]

    @property
    def _pt_attrs(self):
        """Pt attrs."""
        return [col for col in self._column_names if col.startswith("pt")]

    @property
    def _tipper_attrs(self):
        """Tipper attrs."""
        return [col for col in self._column_names if col.startswith("t_")]

    def __eq__(self, other):
        """Eq function."""
        other = self._validate_data(other)
        return (self.dataframe == other).all().all()

    @property
    def nonzero_items(self):
        """Return number of non zero entries."""

        if self._has_data():
            cols = [
                dtype[0]
                for dtype in self._dtype_list[14:]
                if "error" not in dtype[0]
            ]

            return np.count_nonzero(self.dataframe[cols])
        else:
            return 0

    def _validate_data(self, data):
        """Validate data.
        :param data: DESCRIPTION.
        :type data: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if data is None:
            return

        if isinstance(data, (dict, np.ndarray, pd.DataFrame)):
            df = pd.DataFrame(data)

        elif isinstance(data, (MTDataFrame)):
            df = data.dataframe

        else:
            raise TypeError(
                f"Input data must be a pandas.DataFrame not {type(data)}"
            )

        for col in self._dtype_list:
            if col[0] not in df.columns:
                df[col[0]] = np.zeros(df.shape[0], dtype=col[1])

        # resort to the desired column order
        if df.columns.to_list() != self._column_names:
            df = df[self._column_names]

        return df

    def _get_initial_df(self, n_entries=0):
        """Get initial df."""
        return pd.DataFrame(
            np.empty(n_entries, dtype=np.dtype(self._dtype_list))
        )

    def _has_data(self):
        """Has data."""
        if self.dataframe is None:
            return False
        elif self.dataframe.shape[0] > 0:
            return True
        return False

    def get_station_df(self, station=None):
        """Get a single station df.
        :return: DESCRIPTION.
        :rtype: TYPE
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

            return self.dataframe[
                self.dataframe.station == self.working_station
            ]

    @property
    def size(self):
        """Size function."""
        if self._has_data():
            return self.period.size

    def _get_index(self, comp):
        """Get component index values.
        :param comp: | xx | xy | yx | yy | zx | zy |.
        :type comp: string
        :return: Index values for input and output channels.
        :rtype: dict
        """
        if comp in self._index_dict.keys():
            return self._index_dict[comp]

    def _get_key_index(self, key):
        """Get key index."""

        if key.count("_") > 0:
            comp = key.split("_")[1]
            return self._get_index(comp)

    @property
    def period(self):
        """Get frequencies.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self._has_data():
            return np.sort(self.dataframe.period.unique())

    @property
    def frequency(self):
        """Get frequencies.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self._has_data():
            return 1.0 / self.period

    def get_period(self, period, tol=None):
        """Get periods with a percentage based on tol if given.
        :param period: Exact period value to search for.
        :type period: float
        :param tol: Tolerance to search around given period, defaults to None.
        :type tol: float, optional
        :return: Dataframe with periods.
        :rtype: TYPE
        """
        if tol is not None:
            return MTDataFrame(
                self.dataframe[
                    (self.dataframe.period > period * (1 - tol))
                    & self.dataframe.period
                    < period * (1 + tol)
                ]
            )
        else:
            return MTDataFrame(self.dataframe[self.dataframe.period == period])

    @property
    def survey(self):
        """Survey name."""
        if self._has_data():
            if self.working_survey is None:
                self.working_survey = self.dataframe.survey.unique()[0]
            return self.working_survey

    @survey.setter
    def survey(self, value):
        """Survey name."""
        if self._has_data():
            if self.working_survey in [None, ""]:
                self.dataframe.loc[self.dataframe.survey == "", "survey"] = (
                    value
                )
                self.working_survey = value

    @property
    def station(self):
        """Station name."""
        if self._has_data():
            if self.working_station is None:
                self.working_station = self.dataframe.station.unique()[0]
            return self.working_station

    @station.setter
    def station(self, value):
        """Station name."""
        if self._has_data():
            if self.working_station in [None, ""]:
                self.dataframe.loc[self.dataframe.station == "", "station"] = (
                    value
                )
                self.working_station = value

    @property
    def latitude(self):
        """Latitude."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "latitude"
            ].unique()[0]

    @latitude.setter
    def latitude(self, value):
        """Latitude."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "latitude"
            ] = value

    @property
    def longitude(self):
        """Longitude."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "longitude"
            ].unique()[0]

    @longitude.setter
    def longitude(self, value):
        """Longitude."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "longitude"
            ] = value

    @property
    def elevation(self):
        """Elevation."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "elevation"
            ].unique()[0]

    @elevation.setter
    def elevation(self, value):
        """Elevation."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "elevation"
            ] = value

    @property
    def datum_epsg(self):
        """Datum_epsg."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "datum_epsg"
            ].unique()[0]

    @datum_epsg.setter
    def datum_epsg(self, value):
        """Datum_epsg."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "datum_epsg"
            ] = value

    @property
    def east(self):
        """Station."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "east"
            ].unique()[0]

    @east.setter
    def east(self, value):
        """East."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "east"
            ] = value

    @property
    def north(self):
        """North."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "north"
            ].unique()[0]

    @north.setter
    def north(self, value):
        """North."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "north"
            ] = value

    @property
    def utm_epsg(self):
        """Utm_epsg."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "utm_epsg"
            ].unique()[0]

    @utm_epsg.setter
    def utm_epsg(self, value):
        """Utm_epsg."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "utm_epsg"
            ] = value

    @property
    def model_east(self):
        """Model_east."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_east"
            ].unique()[0]

    @model_east.setter
    def model_east(self, value):
        """Model_east."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "model_east"
            ] = value

    @property
    def model_north(self):
        """Model_north."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_north"
            ].unique()[0]

    @model_north.setter
    def model_north(self, value):
        """Model_north."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "model_north"
            ] = value

    @property
    def model_elevation(self):
        """Model_elevation."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_elevation"
            ].unique()[0]

    @model_elevation.setter
    def model_elevation(self, value):
        """Model_elevation."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station,
                "model_elevation",
            ] = value

    @property
    def profile_offset(self):
        """Profile_offset."""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "profile_offset"
            ].unique()[0]

    @profile_offset.setter
    def profile_offset(self, value):
        """Profile_offset."""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station,
                "profile_offset",
            ] = value

    def _get_empty_impedance_array(self, dtype=complex):
        """Get empty impedance array."""
        return np.zeros((self.period.size, 2, 2), dtype=complex)

    @property
    def impedance(self):
        """Impedance elements."""
        z = self._get_empty_impedance_array()

        for key in ["zxx", "zxy", "zyx", "zyy"]:
            index = self._get_index(key)
            z[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, key
            ]
        return z

    def _get_data_array(self, obj, attr):
        """Get data array from object given the attribute.
        :param obj: DESCRIPTION.
        :type obj: TYPE
        :param attr: DESCRIPTION.
        :type attr: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        try:
            return getattr(obj, attr)
        except TypeError:
            return None

    def _fill_data(self, data_array, column, index):
        """Fill data frame column with data array spliced by index.
        :param column:
        :param data_array: DESCRIPTION.
        :type data_array: TYPE
        :param attr: DESCRIPTION.
        :type attr: TYPE
        :param index: DESCRIPTION.
        :type index: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
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

    def from_z_object(self, z_object):
        """Fill impedance.
        :param z_object:
        :param impedance: DESCRIPTION.
        :type impedance: TYPE
        :param index: DESCRIPTION.
        :type index: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        self.dataframe.loc[self.dataframe.station == self.station, "period"] = (
            z_object.period
        )

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

                        self._fill_data(
                            data_array, f"{key}_{comp}{error}", index
                        )

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
                    data_array = self._get_data_array(
                        pt_object, f"{pt_attr}{error}"
                    )
                    self._fill_data(data_array, f"pt_{pt_attr}{error}", None)

    def from_t_object(self, t_object):
        """Fill tipper.
        :param t_object:
        :param tipper: DESCRIPTION.
        :type tipper: TYPE
        :param tipper_error: DESCRIPTION.
        :type tipper_error: TYPE
        :param index: DESCRIPTION.
        :type index: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        self.dataframe.loc[self.dataframe.station == self.station, "period"] = (
            t_object.period
        )

        for error in ["", "_error", "_model_error"]:
            if getattr(t_object, f"_has_tf{error}")():
                obj_key = self._key_dict["t"]
                for comp in ["zx", "zy"]:
                    index = self._get_index(comp)
                    data_array = self._get_data_array(
                        t_object, f"{obj_key}{error}"
                    )
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

    def to_z_object(self, units="mt"):
        """Fill z_object from dataframe

        Need to have the components this way for transposing the elements so
        that the shape is (nf, 2, 2).
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

                phase_model_err[:, index["ii"], index["jj"]] = (
                    self.dataframe.loc[
                        self.dataframe.station == self.station,
                        f"phase_{comp}_model_error",
                    ]
                )

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
                    raise ValueError(
                        "cannot estimate Z without phase information"
                    )

        return z_object

    def to_t_object(self):
        """To a tipper object.
        :return: DESCRIPTION.
        :rtype: TYPE
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
    def station_locations(self):
        """Station locations."""
        return (
            self.dataframe.groupby("station")
            .nth(0)[self._station_location_attrs]
            .reset_index()
        )

    @property
    def phase_tensor(self):
        """Phase tensor information."""

        return self.dataframe[
            self._station_location_attrs + self._pt_attrs
        ].reset_index()

    @property
    def tipper(self):
        """Phase tensor information."""

        return self.dataframe[
            self._station_location_attrs + self._tipper_attrs
        ].reset_index()

    @property
    def for_shapefiles(self):
        r"""For shape files includes phase tensor and tippe."""

        return self.dataframe[
            self._station_location_attrs + self._pt_attrs + self._tipper_attrs
        ].reset_index()

    def get_station_distances(self, utm=False):
        """Get distance information between stations.
        :return: DESCRIPTION.
        :rtype: TYPE
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
