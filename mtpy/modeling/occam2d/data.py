# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:01:14 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mtpy.core.mt_dataframe import MTDataFrame

# =============================================================================


class Occam2DData:
    """
    Reads and writes data files and more.

    Inherets Profile, so the intended use is to use Data to project stations
    onto a profile, then write the data file.

    ===================== =====================================================
    Model Modes           Description
    ===================== =====================================================
    1 or log_all          Log resistivity of TE and TM plus Tipper
    2 or log_te_tip       Log resistivity of TE plus Tipper
    3 or log_tm_tip       Log resistivity of TM plus Tipper
    4 or log_te_tm        Log resistivity of TE and TM
    5 or log_te           Log resistivity of TE
    6 or log_tm           Log resistivity of TM
    7 or all              TE, TM and Tipper
    8 or te_tip           TE plus Tipper
    9 or tm_tip           TM plus Tipper
    10 or te_tm           TE and TM mode
    11 or te              TE mode
    12 or tm              TM mode
    13 or tip             Only Tipper
    ===================== =====================================================


    :Example Write Data File: ::

        >>> from mtpy.modeling.occam2d import Data
        >>> occam_data_object = Data()
        >>> occam_data_object.read_data_file(r"path/to/data/file.dat")
        >>> occam_data_object.model_mode = 2
        >>> occam_data_object.write_data_file(r"path/to/new/data/file_te.dat")

    """

    def __init__(self, dataframe=None, center_point=None, **kwargs):

        self.logger = logger
        self.dataframe = dataframe
        self.data_filename = None
        self.fn_basename = "OccamDataFile.dat"
        self.save_path = Path()
        self.interpolate_freq = None
        self.model_mode = "1"

        self.res_te_err = 10
        self.res_tm_err = 10
        self.phase_te_err = 5
        self.phase_tm_err = 5
        self.tipper_err = 10
        self.error_type = "floor"
        self.profile_origin = (0, 0)
        self.profile_angle = 0
        self.geoelectric_strike = 0

        self.occam_format = "OCCAM2MTDATA_1.0"
        self.title = "MTpy-OccamDatafile"
        self.masked_data = None
        self._tab = " " * 3

        self._line_keys = [
            "station",
            "frequency",
            "profile_offset",
            "model_east",
            "model_north",
            "res_xy",
            "res_yx",
            "phase_xy",
            "phase_yx",
            "tzx_real",
            "tzx_imag",
            "res_xy_model_error",
            "res_yx_model_error",
            "phase_xy_model_error",
            "phase_yx_model_error",
            "tzx_real_model_error",
            "tzx_imag_model_error",
        ]

        self.occam_dict = {
            "1": "res_xy",
            "2": "phase_xy",
            "3": "tzx_real",
            "4": "tzx_imag",
            "5": "res_yx",
            "6": "phase_yx",
            "9": "res_xy",
            "10": "res_yx",
        }

        self.df_dict = {
            "1": "res_xy",
            "2": "phase_xy",
            "3": "tzx",
            "5": "res_yx",
            "6": "phase_yx",
        }

        self.mode_dict = {
            "log_all": [1, 2, 3, 4, 5, 6],
            "log_te_tip": [1, 2, 3, 4],
            "log_tm_tip": [5, 6, 3, 4],
            "log_te_tm": [1, 2, 5, 6],
            "log_te": [1, 2],
            "log_tm": [5, 6],
            "all": [9, 2, 3, 4, 10, 6],
            "te_tip": [9, 2, 3, 4],
            "tm_tip": [10, 6, 3, 4],
            "te_tm": [9, 2, 10, 6],
            "te": [9, 2],
            "tm": [10, 6],
            "tip": [3, 4],
            "1": [1, 2, 3, 4, 5, 6],
            "2": [1, 2, 3, 4],
            "3": [5, 6, 3, 4],
            "4": [1, 2, 5, 6],
            "5": [1, 2],
            "6": [5, 6],
            "7": [9, 2, 3, 4, 10, 6],
            "8": [9, 2, 3, 4],
            "9": [10, 6, 3, 4],
            "10": [9, 2, 10, 6],
            "11": [9, 2],
            "12": [10, 6],
            "13": [3, 4],
        }

        self._data_header = "{0:<6}{1:<6}{2:<6} {3:<8} {4:<8}".format(
            "SITE", "FREQ", "TYPE", "DATUM", "ERROR"
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        lines = ["Occam2D Data"]
        lines.append(f"\tNumber of Stations:     {self.n_stations}")
        lines.append(f"\tNumber of Frequencies:  {self.n_frequencies}")
        lines.append(f"\tNumber of data:         {self.n_data}")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _has_data(self):
        return self._mt_dataframe._has_data()

    @property
    def n_stations(self):
        if self._has_data():
            return self.dataframe.station.unique().size
        return 0

    @property
    def n_frequencies(self):
        if self._has_data():
            return self._mt_dataframe.period.size
        return 0

    @property
    def n_data(self):
        return self._mt_dataframe.nonzero_items

    @property
    def frequencies(self):
        if self._has_data():
            return pd.Series(self._mt_dataframe.frequency)

    @property
    def stations(self):
        if self._has_data():
            return pd.Series(self.dataframe.station.unique())

    @property
    def offsets(self):
        return np.array(
            [
                self.dataframe.loc[
                    self.dataframe.station == ss, "profile_offset"
                ].iloc[0]
                for ss in self.stations
            ]
        )

    @property
    def data_filename(self):
        return self._data_fn

    @data_filename.setter
    def data_filename(self, value):
        if value is None:
            self._data_fn = None
        else:
            self._data_fn = Path(value)

    @property
    def dataframe(self):
        return self._mt_dataframe.dataframe

    @dataframe.setter
    def dataframe(self, df):
        """
        Set dataframe to an MTDataframe
        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if df is None:
            self._mt_dataframe = MTDataFrame()

        elif isinstance(df, (pd.DataFrame, MTDataFrame, np.ndarray)):
            self._mt_dataframe = MTDataFrame(df)

        else:
            raise TypeError(
                f"Input must be a dataframe or MTDataFrame object not {type(df)}"
            )

    def _line_entry(self):
        return dict([(key, np.nan) for key in self._line_keys])

    def _get_model_locations(self, profile_offset, profile_angle):
        """
        get the origin of the profile in real world coordinates

        Author: Alison Kirkby (2013)

        NEED TO ADAPT THIS TO THE CURRENT SETUP.
        """

        return (
            profile_offset * np.cos(np.deg2rad(profile_angle)),
            profile_offset * np.sin(np.deg2rad(profile_angle)),
        )

    def read_data_file(self, data_fn=None):
        """
        Read in an existing data file and populate appropriate attributes
            * data
            * data_list
            * freq
            * station_list
            * station_locations

        Arguments:
        -----------
            **data_fn** : string
                          full path to data file
                          *default* is None and set to save_path/fn_basename

        :Example: ::

            >>> import mtpy.modeling.occam2d as occam2d
            >>> ocd = occam2d.Data()
            >>> ocd.read_data_file(r"/home/Occam2D/Line1/Inv1/Data.dat")

        """

        if data_fn is not None:
            self.data_filename = data_fn

        if not self.data_filename.is_file():
            raise ValueError(f"Could not find {self.data_filename}")
        if self.data_filename is None:
            raise ValueError("data_filename is None, input filename")

        self.save_path = self.data_filename.parent

        with open(self.data_filename, "r") as dfid:
            dlines = dfid.readlines()

        # get format of input data
        self.occam_format = dlines[0].strip().split(":")[1].strip()

        # get title
        title_str = dlines[1].strip().split(":")[1].strip()

        title_list = title_str.split(",")
        self.title = title_list[0]

        # get strike angle and profile angle
        if len(title_list) > 1:
            for t_str in title_list[1:]:
                t_list = t_str.split("=")
                if len(t_list) > 1:
                    key = t_list[0].strip().lower().replace(" ", "_")
                    if key == "profile":
                        key = "profile_angle"
                    elif key == "strike":
                        key = "geoelectric_strike"
                    value = t_list[1].split("deg")[0].strip()
                    self.logger.debug(f"{key} = {value}")
                    try:
                        setattr(self, key, float(value))
                    except ValueError:
                        setattr(self, key, value)

        # get number of sites
        nsites = int(dlines[2].strip().split(":")[1].strip())
        self.logger.debug(f"number of sites = {nsites}")

        # get station names
        stations = np.array([dlines[ii].strip() for ii in range(3, nsites + 3)])

        # get offsets in meters
        offsets = np.array(
            [
                float(dlines[ii].strip())
                for ii in range(4 + nsites, 4 + 2 * nsites)
            ]
        )

        # get number of frequencies
        nfreq = int(dlines[4 + 2 * nsites].strip().split(":")[1].strip())
        self.logger.debug("number of frequencies = {nfreq}")

        # get frequencies
        frequency = np.array(
            [
                float(dlines[ii].strip())
                for ii in range(5 + 2 * nsites, 5 + 2 * nsites + nfreq)
            ]
        )

        # -----------get data-------------------
        # set zero array size the first row will be the data and second the
        # error

        data_list = dlines[7 + 2 * nsites + nfreq :]
        entries = []
        res_log = False
        for line in data_list:
            try:
                s_index, f_index, comp, odata, oerr = line.split()
                # station index -1 cause python starts at 0
                s_index = int(s_index) - 1

                # frequency index -1 cause python starts at 0
                f_index = int(f_index) - 1
                # data key
                key = self.occam_dict[comp]

                # put into array
                if int(comp) == 1 or int(comp) == 5:
                    res_log = True
                    value = 10 ** float(odata)
                    # error
                    value_error = float(oerr) * np.log(10)
                else:
                    value = float(odata)
                    # error
                    value_error = float(oerr)

                entry = self._line_entry()
                entry["station"] = stations[s_index]
                entry["frequency"] = frequency[f_index]
                entry["profile_offset"] = offsets[s_index]
                (
                    entry["model_east"],
                    entry["model_north"],
                ) = self._get_model_locations(
                    entry["profile_offset"], self.profile_angle
                )
                entry[key] = value
                entry[f"{key}_model_error"] = value_error
                entries.append(entry)
            except ValueError:
                self.logger.debug("Could not read line {0}".format(line))

        # format dataframe
        df = pd.DataFrame(entries)
        df["tzx"] = df.tzx_real + 1j * df.tzx_imag
        df["tzx_model_error"] = df.tzx_real_model_error
        df["period"] = 1.0 / df.frequency
        df = df.drop(
            columns=[
                "tzx_real",
                "tzx_imag",
                "tzx_real_model_error",
                "tzx_imag_model_error",
                "frequency",
            ],
            axis=1,
        )
        df = df.groupby(["station", "period"]).agg("first")
        df = df.sort_values("profile_offset").reset_index()
        self.dataframe = df

        self.inv_mode = self._get_model_mode_from_data(res_log)

    def _get_model_mode_from_data(self, res_log):
        """
        Get inversion mode from the data

        :param res_log: DESCRIPTION
        :type res_log: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        inv_list = []
        for inv_mode, comp in self.df_dict.items():
            if np.count_nonzero(self.dataframe[comp]) > 0:
                if comp == "res_xy":
                    if res_log:
                        inv_list.append(1)
                    else:
                        inv_list.append(9)
                elif comp == "res_yx":
                    if res_log:
                        inv_list.append(5)
                    else:
                        inv_list.append(10)
                elif comp == "tzx":
                    inv_list.append(3)
                    inv_list.append(4)
                else:
                    inv_list.append(int(inv_mode))

        return self._match_inv_list_to_mode(inv_list)

    def _match_inv_list_to_mode(self, inv_list):
        """
        match the modes to the inversion mode
        :param inv_list: DESCRIPTION
        :type inv_list: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for inv_mode, comp_list in self.mode_dict.items():
            if sorted(inv_list) == sorted(comp_list):
                return inv_mode

    def _get_data_block(self):
        """
        Get all the data needed to write a data file.

        """
        if not self._has_data():
            raise ValueError("Cannot write data from an empty dataframe.")
        data_list = []
        for s_index, station in self.stations.iteritems():
            sdf = self.dataframe.loc[self.dataframe.station == station]
            for f_index, frequency in self.frequencies.iteritems():
                fdf = sdf[
                    (sdf.period >= (1.0 / frequency) * 0.99)
                    & (sdf.period <= (1.0 / frequency) * 1.01)
                ]
                for comp_number in self.mode_dict[self.inv_mode]:
                    comp = self.df_dict[str(comp_number)]
                    value = fdf[comp].values[0]
                    if value != 0:
                        if comp_number in [1, 5]:
                            error_value = fdf[f"{comp}_model_error"].values[
                                0
                            ] / np.log(10)
                            value = np.log10(value)

                        elif comp_number in [3]:
                            value = value.real
                            error_value = fdf[f"{comp}_model_error"].values[0]
                        elif comp_number in [4]:
                            value = value.imag
                            error_value = fdf[f"{comp}_model_error"].values[0]
                        else:
                            value = value
                            error_value = fdf[f"{comp}_model_error"].values[0]
                        data_list.append(
                            f"{s_index + 1:^6}{f_index + 1:^6}{comp_number:^6} "
                            f"{value:>8.4f} {error_value:>8.4f}"
                        )
        return data_list

    def mask_from_datafile(self, mask_datafn):
        """
        reads a separate data file and applies mask from this data file.
        mask_datafn needs to have exactly the same frequencies, and station names
        must match exactly.

        """
        ocdm = Occam2DData()
        ocdm.read_data_file(mask_datafn)
        # list of stations, in order, for the mask_datafn and the input data
        # file
        ocdm_stlist = [ocdm.data[i]["station"] for i in range(len(ocdm.data))]
        ocd_stlist = [self.data[i]["station"] for i in range(len(self.data))]

        for i_ocd, stn in enumerate(ocd_stlist):
            i_ocdm = ocdm_stlist.index(stn)
            for dmode in [
                "te_res",
                "tm_res",
                "te_phase",
                "tm_phase",
                "im_tip",
                "re_tip",
            ]:

                for i in range(len(self.freq)):
                    if self.data[i_ocdm][dmode][0][i] == 0:
                        self.data[i_ocd][dmode][0][i] = 0.0
        self.fn_basename = (
            self.fn_basename[:-4] + "Masked" + self.fn_basename[-4:]
        )
        self.write_data_file()

    def _make_title(self):
        """
        make title with profile angle, strike and origin
        """
        return (
            f"{self.title}, Profile={self.profile_angle:.1f} deg, "
            f"Strike={self.geoelectric_strike:.1f} deg, "
            f"Origin={self.profile_origin}"
        )

    def write_data_file(self, data_fn=None):
        """
        Write a data file.

        Arguments:
        -----------
            **data_fn** : string
                          full path to data file.
                          *default* is save_path/fn_basename

        If there data is None, then _fill_data is called to create a profile,
        rotate data and get all the necessary data.  This way you can use
        write_data_file directly without going through the steps of projecting
        the stations, etc.

        :Example: ::
            >>> edipath = r"/home/mt/edi_files"
            >>> slst = ['mt{0:03}'.format(ss) for ss in range(1, 20)]
            >>> ocd = occam2d.Data(edi_path=edipath, station_list=slst)
            >>> ocd.save_path = r"/home/occam/line1/inv1"
            >>> ocd.write_data_file()

        """

        if data_fn is not None:
            self.data_fn = data_fn
        else:
            if self.save_path is None:
                self.save_path = Path()
            if not self.save_path.exists():
                self.save_path.mkdir()

            self.data_fn = self.save_path.joinpath(self.fn_basename)

        data_lines = []

        # --> header line
        data_lines.append(f"{'format:'.upper():<18}{self.occam_format}")
        data_lines.append(f"{'title:'.upper():<18}{self._make_title()}")

        # --> sites
        data_lines.append(f"{'sites:'.upper():<18}{self.n_stations}")
        for station in self.stations:
            data_lines.append(f"{self._tab}{station}")

        # --> offsets
        data_lines.append(f"{'offset (m):'.upper():<18}")
        for offset in self.offsets:
            data_lines.append(f"{self._tab}{offset:.1f}")
        # --> frequencies
        data_lines.append(f"{'frequencies:'.upper():<18}{self.n_frequencies}")
        for ff in self.frequencies:
            data_lines.append(f"{self._tab}{ff:<10.6e}")

        data_block = self._get_data_block()
        # --> data
        data_lines.append(f"{'data blocks:'.upper():<18}{len(data_block)}")
        data_lines.append(self._data_header)
        data_lines += data_block

        with open(self.data_fn, "w") as dfid:
            dfid.write("\n".join(data_lines))

        self.logger.debug(f"Wrote Occam2D data file to {self.data_fn}")
