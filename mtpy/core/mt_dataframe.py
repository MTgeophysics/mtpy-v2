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

from . import Z, Tipper

# =============================================================================


class MTDataFrame:
    """
    Dataframe for a single station

    Tried subclassing pandas.DataFrame, but that turned out to not be straight
    forward, so when with compilation instead.

    Think about having period as an index?
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
            "latitude",
            "longitude",
            "latitude",
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
        if self._has_data():
            return self.dataframe.__str__()

        else:
            return "Empty MTStationDataFrame"

    def __repr__(self):
        if self._has_data():
            return self.dataframe.__repr__()
        else:
            return "MTStationDataFrame()"

    @property
    def _column_names(self):
        return [col[0] for col in self._dtype_list]

    @property
    def _pt_attrs(self):
        return [col for col in self._column_names if col.startswith("pt")]

    @property
    def _tipper_attrs(self):
        return [col for col in self._column_names if col.startswith("t_")]

    def __eq__(self, other):
        other = self._validata_data(other)
        return self.dataframe == other

    @property
    def nonzero_items(self):
        """return number of non zero entries"""

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
        """

        :param data: DESCRIPTION
        :type data: TYPE
        :return: DESCRIPTION
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
        return pd.DataFrame(
            np.empty(n_entries, dtype=np.dtype(self._dtype_list))
        )

    def _has_data(self):
        if self.dataframe is None:
            return False
        elif self.dataframe.shape[0] > 0:
            return True
        return False

    def get_station_df(self, station=None):
        """
        get a single station df

        :return: DESCRIPTION
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
        if self._has_data():
            return self.period.size

    def _get_index(self, comp):
        """
        get component index values

        :param comp: | xx | xy | yx | yy | zx | zy |
        :type comp: string
        :return: index values for input and output channels
        :rtype: dict

        """
        if comp in self._index_dict.keys():
            return self._index_dict[comp]

    def _get_key_index(self, key):
        """ """

        if key.count("_") > 0:
            comp = key.split("_")[1]
            return self._get_index(comp)

    @property
    def period(self):
        """
        Get frequencies

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self._has_data():
            return np.sort(self.dataframe.period.unique())

    @property
    def frequency(self):
        """
        Get frequencies

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self._has_data():
            return 1.0 / self.period

    @property
    def survey(self):
        """survey name"""
        if self._has_data():
            if self.working_survey is None:
                self.working_survey = self.dataframe.survey.unique()[0]
            return self.working_survey

    @survey.setter
    def survey(self, value):
        """survey name"""
        if self._has_data():
            if self.working_survey in [None, ""]:
                self.dataframe.loc[
                    self.dataframe.survey == "", "survey"
                ] = value
                self.working_survey = value

    @property
    def station(self):
        """station name"""
        if self._has_data():
            if self.working_station is None:
                self.working_station = self.dataframe.station.unique()[0]
            return self.working_station

    @station.setter
    def station(self, value):
        """station name"""
        if self._has_data():
            if self.working_station in [None, ""]:
                self.dataframe.loc[
                    self.dataframe.station == "", "station"
                ] = value
                self.working_station = value

    @property
    def latitude(self):
        """latitude"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "latitude"
            ].unique()[0]

    @latitude.setter
    def latitude(self, value):
        """latitude"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "latitude"
            ] = value

    @property
    def longitude(self):
        """longitude"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "longitude"
            ].unique()[0]

    @longitude.setter
    def longitude(self, value):
        """longitude"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "longitude"
            ] = value

    @property
    def elevation(self):
        """elevation"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "elevation"
            ].unique()[0]

    @elevation.setter
    def elevation(self, value):
        """elevation"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "elevation"
            ] = value

    @property
    def datum_epsg(self):
        """datum_epsg"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "datum_epsg"
            ].unique()[0]

    @datum_epsg.setter
    def datum_epsg(self, value):
        """datum_epsg"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "datum_epsg"
            ] = value

    @property
    def east(self):
        """station"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "east"
            ].unique()[0]

    @east.setter
    def east(self, value):
        """east"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "east"
            ] = value

    @property
    def north(self):
        """north"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "north"
            ].unique()[0]

    @north.setter
    def north(self, value):
        """north"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "north"
            ] = value

    @property
    def utm_epsg(self):
        """utm_epsg"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "utm_epsg"
            ].unique()[0]

    @utm_epsg.setter
    def utm_epsg(self, value):
        """utm_epsg"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "utm_epsg"
            ] = value

    @property
    def model_east(self):
        """model_east"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_east"
            ].unique()[0]

    @model_east.setter
    def model_east(self, value):
        """model_east"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "model_east"
            ] = value

    @property
    def model_north(self):
        """model_north"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_north"
            ].unique()[0]

    @model_north.setter
    def model_north(self, value):
        """model_north"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station, "model_north"
            ] = value

    @property
    def model_elevation(self):
        """model_elevation"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "model_elevation"
            ].unique()[0]

    @model_elevation.setter
    def model_elevation(self, value):
        """model_elevation"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station,
                "model_elevation",
            ] = value

    @property
    def profile_offset(self):
        """profile_offset"""
        if self._has_data():
            return self.dataframe.loc[
                self.dataframe.station == self.station, "profile_offset"
            ].unique()[0]

    @profile_offset.setter
    def profile_offset(self, value):
        """profile_offset"""
        if self._has_data():
            self.dataframe.loc[
                self.dataframe.station == self.station,
                "profile_offset",
            ] = value

    def _get_empty_impedance_array(self, dtype=complex):
        return np.zeros((self.period.size, 2, 2), dtype=complex)

    @property
    def impedance(self):
        """Impedance elements"""
        z = self._get_empty_impedance_array()

        for key in ["zxx", "zxy", "zyx", "zyy"]:
            index = self._get_index(key)
            z[:, index["ii"], index["jj"]] = self.dataframe.loc[
                self.dataframe.station == self.station, key
            ]
        return z

    def from_z_object(self, z_object):
        """
        Fill impedance
        :param impedance: DESCRIPTION
        :type impedance: TYPE
        :param index: DESCRIPTION
        :type index: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.dataframe.loc[
            self.dataframe.station == self.station, "period"
        ] = z_object.period

        for error in ["", "_error", "_model_error"]:
            if getattr(z_object, f"_has_tf{error}")():
                for key in ["z", "res", "phase", "pt"]:
                    obj_key = self._key_dict[key]
                    for comp in ["xx", "xy", "yx", "yy"]:
                        index = self._get_index(comp)
                        if key in ["pt"]:
                            data_array = getattr(
                                z_object.phase_tensor, f"{key}{error}"
                            )
                        else:
                            data_array = getattr(z_object, f"{obj_key}{error}")
                        self.dataframe.loc[
                            self.dataframe.station == self.station,
                            f"{key}_{comp}{error}",
                        ] = data_array[:, index["ii"], index["jj"]]

                    if key in ["pt"]:
                        for pt_attr in [
                            "phimin",
                            "phimax",
                            "azimuth",
                            "skew",
                            "ellipticity",
                            "det",
                        ]:
                            data_array = getattr(
                                z_object.phase_tensor, f"{pt_attr}{error}"
                            )
                            self.dataframe.loc[
                                self.dataframe.station == self.station,
                                f"pt_{pt_attr}{error}",
                            ] = data_array[:]

    def from_t_object(self, t_object):
        """
        Fill tipper
        :param tipper: DESCRIPTION
        :type tipper: TYPE
        :param tipper_error: DESCRIPTION
        :type tipper_error: TYPE
        :param index: DESCRIPTION
        :type index: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.dataframe.loc[
            self.dataframe.station == self.station, "period"
        ] = t_object.period

        for error in ["", "_error", "_model_error"]:
            if getattr(t_object, f"_has_tf{error}")():
                obj_key = self._key_dict["t"]
                for comp in ["zx", "zy"]:
                    index = self._get_index(comp)
                    data_array = getattr(t_object, f"{obj_key}{error}")
                    self.dataframe.loc[
                        self.dataframe.station == self.station,
                        f"t_{comp}{error}",
                    ] = data_array[:, index["ii"], index["jj"]]
                if error in [""]:
                    for t_attr in [
                        "mag_real",
                        "mag_imag",
                        "angle_real",
                        "angle_imag",
                    ]:
                        data_array = getattr(t_object, t_attr)
                        self.dataframe.loc[
                            self.dataframe.station == self.station,
                            f"t_{t_attr}",
                        ] = data_array

    def to_z_object(self):
        """
        fill z_object from dataframe

        Need to have the components this way for transposing the elements so
        that the shape is (nf, 2, 2)
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

        z_object = Z(z, z_err, self.frequency, z_model_err)

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
                res_model_err[
                    :, index["ii"], index["jj"]
                ] = self.dataframe.loc[
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

                phase_model_err[
                    :, index["ii"], index["jj"]
                ] = self.dataframe.loc[
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
                    raise ValueError(
                        "cannot estimate Z without phase information"
                    )

        return z_object

    def to_t_object(self):
        """
        To a tipper object

        :return: DESCRIPTION
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
        return (
            self.dataframe.groupby("station")
            .first()[self._station_location_attrs]
            .reset_index()
        )

    @property
    def phase_tensor(self):
        """phase tensor information"""

        return self.dataframe[
            self._station_location_attrs + self._pt_attrs
        ].reset_index()

    @property
    def tipper(self):
        """phase tensor information"""

        return self.dataframe[
            self._station_location_attrs + self._tipper_attrs
        ].reset_index()
