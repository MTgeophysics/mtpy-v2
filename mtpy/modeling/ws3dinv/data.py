# -*- coding: utf-8 -*-
"""
Merge transfer functions together
"""
# ==============================================================================
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd

from mtpy.core import MTDataFrame

# ==============================================================================


class WSData:
    """
    Includes tools for reading and writing data files intended to be used with
    ws3dinv.

    :Example: ::

        >>> import mtpy.modeling.ws3dinv as ws
        >>> import os
        >>> edi_path = r"/home/EDI_Files"
        >>> edi_list = [os.path.join(edi_path, edi) for edi in edi_path
        >>> ...         if edi.find('.edi') > 0]
        >>> # create an evenly space period list in log space
        >>> p_list = np.logspace(np.log10(.001), np.log10(1000), 12)
        >>> wsdata = ws.WSData(edi_list=edi_list, period_list=p_list,
        >>> ...                station_fn=r"/home/stations.txt")
        >>> wsdata.write_data_file()


    ====================== ====================================================
    Attributes              Description
    ====================== ====================================================
    data                   numpy structured array with keys:
                               * *station* --> station name
                               * *east*    --> relative eastern location in
                                               grid
                               * *north*   --> relative northern location in
                                               grid
                               * *z_data*  --> impedance tensor array with
                                               shape
                                         (n_stations, n_freq, 4, dtype=complex)
                               * *z_data_err--> impedance tensor error without
                                                error map applied
                               * *z_err_map --> error map from data file
    data_fn                full path to data file
    edi_list               list of edi files used to make data file
    n_z                    [ 4 | 8 ] number of impedance tensor elements
                           *default* is 8
    ncol                   number of columns in out file from winglink
                           *default* is 5
    period_list            list of periods to invert for
    ptol                   if periods in edi files don't match period_list
                           then program looks for periods within ptol
                           *defualt* is .15 or 15 percent
    rotation_angle         Angle to rotate the data relative to north.  Here
                           the angle is measure clockwise from North,
                           Assuming North is 0 and East is 90.  Rotating data,
                           and grid to align with regional geoelectric strike
                           can improve the inversion. *default* is None
    save_path              path to save the data file
    station_fn             full path to station file written by WSStation
    station_locations      numpy structured array for station locations keys:
                               * *station* --> station name
                               * *east*    --> relative eastern location in
                                               grid
                               * *north*   --> relative northern location in
                                               grid
                           if input a station file is written
    station_east           relative locations of station in east direction
    station_north          relative locations of station in north direction
    station_names          names of stations

    units                  [ 'mv' | 'else' ] units of Z, needs to be mv for
                           ws3dinv. *default* is 'mv'
    wl_out_fn              Winglink .out file which describes a 3D grid
    wl_site_fn             Wingling .sites file which gives station locations
    z_data                 impedance tensors of data with shape:
                           (n_station, n_periods, 2, 2)
    z_data_err             error of data impedance tensors with error map
                           applied, shape (n_stations, n_periods, 2, 2)
    z_err                  [ float | 'data' ]
                           'data' to set errors as data errors or
                           give a percent error to impedance tensor elements
                           *default* is .05 or 5%  if given as percent, ie. 5%
                           then it is converted to .05.
    z_err_floor            percent error floor, anything below this error will
                           be set to z_err_floor.  *default* is None
    z_err_map              [zxx, zxy, zyx, zyy] for n_z = 8
                           [zxy, zyx] for n_z = 4
                           Value in percent to multiply the error by, which
                           give the user power to down weight bad data, so
                           the resulting error will be z_err_map*z_err
    ====================== ====================================================

    ====================== ====================================================
    Methods                Description
    ====================== ====================================================
    build_data             builds the data from .edi files
    write_data_file        writes a data file from attribute data.  This way
                           you can read in a data file, change some parameters
                           and rewrite.
    read_data_file         reads in a ws3dinv data file
    ====================== ====================================================

    """

    def __init__(self, mt_dataframe=None, **kwargs):
        self.logger = logger
        self.dataframe = mt_dataframe
        self.save_path = Path()
        self.fn_basename = "WSDataFile.dat"
        self.units = "mv"
        self.ncol = 5
        self.ptol = 0.15
        self.z_error = 0.05
        self.z_error_floor = None
        self.z_error_map = [10, 1, 1, 10]
        self.n_z = 8
        self.period_list = None
        self.edi_list = None
        self.station_locations = None
        self.rotation_angle = None

        self.station_east = None
        self.station_north = None
        self.station_names = None

        self.z_data = None
        self.z_data_err = None

        self.wl_site_fn = None
        self.wl_out_fn = None

        self.data_fn = None
        self.station_fn = None

        self.data = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        # make sure the error given is a decimal percent
        if type(self.z_error) is not str and self.z_error > 1:
            self.z_error /= 100.0

        # make sure the error floor given is a decimal percent
        if self.z_error_floor is not None and self.z_error_floor > 1:
            self.z_error_floor /= 100.0

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

        self._mt_dataframe.dataframe.reset_index(drop=True, inplace=True)

    @property
    def period(self):
        if self.dataframe is not None:
            return np.sort(self.dataframe.period.unique())

    @property
    def data_filename(self):
        return self.save_path.joinpath(self.fn_basename)

    @data_filename.setter
    def data_filename(self, value):
        if value is not None:
            value = Path(value)
            if value.parent == Path("."):
                self.fn_basename = value.name
            else:
                self.save_path = value.parent
                self.fn_basename = value.name

    def get_n_stations(self):
        if self.dataframe is not None:
            return (
                self.dataframe.loc[
                    (self.dataframe.zxx != 0)
                    | (self.dataframe.zxy != 0)
                    | (self.dataframe.zyx != 0)
                    | (self.dataframe.zyy != 0),
                    "station",
                ]
                .unique()
                .size
            )

    def get_period_df(self, period):
        return (
            self.dataframe.loc[self.dataframe.period == period]
            .groupby("station")
            .first()
            .reset_index()
        )

    def write_data_file(self, **kwargs):
        """
        Writes a data file based on the attribute data

        Key Word Arguments:
        ---------------------
            **data_fn** : string
                          full path to data file name

            **save_path** : string
                            directory path to save data file, will be written
                            as save_path/data_basename
            **data_basename** : string
                                basename of data file to be saved as
                                save_path/data_basename
                                *default* is WSDataFile.dat

        .. note:: if any of the data attributes have been reset, be sure
                  to call build_data() before write_data_file.
        """
        # get units correctly
        zconv = 1
        if self.units == "mv":
            zconv = 1.0 / 796.0

        for key in ["data_fn", "save_path", "data_basename"]:
            try:
                setattr(self, key, kwargs[key])
            except KeyError:
                pass

        # -----Write data file--------------------------------------------------
        n_stations = self.get_n_stations()
        n_periods = self.period.size
        station_locations = self.dataframe.station_locations.copy()

        lines = []
        data_lines = []
        error_lines = []
        error_map_lines = []
        lines.append(f"{n_stations:d} {n_periods:d} {self.n_z:d}\n")

        # write N-S locations
        lines.append("Station_Location: N-S \n")
        for ii in range(n_stations / self.n_z + 1):
            for ll in range(self.n_z):
                index = ii * self.n_z + ll
                try:
                    lines.append(
                        f"{station_locations.model_north[index]:+.4e} "
                    )
                except IndexError:
                    pass
            lines.append("\n")

        # write E-W locations
        lines.append("Station_Location: E-W \n")
        for ii in range(n_stations / self.n_z + 1):
            for ll in range(self.n_z):
                index = ii * self.n_z + ll
                try:
                    lines.append(
                        f"{station_locations.model_east[index]:+.4e} "
                    )
                except IndexError:
                    pass
            lines.append("\n")

        # write impedance tensor components
        for ii, p1 in enumerate(self.period):
            pdf = self.get_period_df(p1)
            data_lines.append(f"DATA_Period: {p1:3.6f}\n")
            error_lines.append(f"ERROR_Period: {p1:3.6f}\n")
            error_map_lines.append(f"ERMAP_Period: {p1:3.6f}\n")
            for row in pdf.itertuples():
                data_lines.append(
                    f"{row.zxx.real * zconv:+.4e} "
                    f"{row.zxx.imag * zconv:+.4e} "
                    f"{row.zxy.real * zconv:+.4e} "
                    f"{row.zxy.imag * zconv:+.4e} "
                    f"{row.zyx.real * zconv:+.4e} "
                    f"{row.zyx.imag * zconv:+.4e} "
                    f"{row.zyy.real * zconv:+.4e} "
                    f"{row.zyy.imag * zconv:+.4e}\n"
                )
                error_lines.append(
                    f"{row.zxx.real * self.z_error:+.4e} "
                    f"{row.zxx.imag * self.z_error:+.4e} "
                    f"{row.zxy.real * self.z_error:+.4e} "
                    f"{row.zxy.imag * self.z_error:+.4e} "
                    f"{row.zyx.real * self.z_error:+.4e} "
                    f"{row.zyx.imag * self.z_error:+.4e} "
                    f"{row.zyy.real * self.z_error:+.4e} "
                    f"{row.zyy.imag * self.z_error:+.4e} "
                )

                error_map_lines.append(
                    f"{self.z_error_map[0]:+.4e} "
                    f"{self.z_error_map[0]:+.4e} "
                    f"{self.z_error_map[1]:+.4e} "
                    f"{self.z_error_map[1]:+.4e} "
                    f"{self.z_error_map[2]:+.4e} "
                    f"{self.z_error_map[2]:+.4e} "
                    f"{self.z_error_map[3]:+.4e} "
                    f"{self.z_error_map[3]:+.4e} "
                )

        with open(self.data_filename, "w") as fid:
            fid.write("".join(lines, data_lines, error_lines, error_map_lines))

        self.logger.info(f"Wrote WS3DINV file to {self.data_filename}")
        return self.data_filename

    def read_data_file(self, data_filename):
        """
        read in data file

        Arguments:
        -----------
            **data_fn** : string
                          full path to data file
            **wl_sites_fn** : string
                              full path to sites file output by winglink.
                              This is to match the station name with station
                              number.
            **station_fn** : string
                             full path to station location file written by
                             WSStation

        Fills Attributes:
        ------------------
            **data** : structure np.ndarray
                      fills the attribute WSData.data with values

            **period_list** : np.ndarray()
                             fills the period list with values.
        """

        if self.units == "mv":
            zconv = 796.0
        else:
            zconv = 1

        self.data_filename = data_filename

        with open(self.data_filename) as fid:
            dlines = fid.readlines()

        # get size number of stations, number of frequencies,
        # number of Z components
        n_stations, n_periods, nz = np.array(
            dlines[0].strip().split(), dtype="int"
        )
        nsstart = 2

        self.n_z = nz
        # make a structured array to keep things in for convenience
        z_shape = (n_periods, 2, 2)
        t_shape = (n_periods, 1, 2)
        data_dtype = [
            ("station", "|S10"),
            ("east", float),
            ("north", float),
            ("z_data", (complex, z_shape)),
            ("z_data_err", (complex, z_shape)),
            ("z_err_map", (complex, z_shape)),
            ("tipper_data", (complex, t_shape)),
            ("tipper_data_err", (complex, t_shape)),
            ("tipper_err_map", (complex, t_shape)),
        ]
        self.data = np.zeros(n_stations, dtype=data_dtype)

        findlist = []
        for ii, dline in enumerate(dlines[1:50], 1):
            if dline.find("Station_Location: N-S") == 0:
                findlist.append(ii)
            elif dline.find("Station_Location: E-W") == 0:
                findlist.append(ii)
            elif dline.find("DATA_Period:") == 0:
                findlist.append(ii)

        ncol = len(dlines[nsstart].strip().split())

        # get site names if entered a sites file
        self.data["station"] = np.arange(n_stations)

        # get N-S locations
        for ii, dline in enumerate(dlines[findlist[0] + 1 : findlist[1]], 0):
            dline = dline.strip().split()
            for jj in range(ncol):
                try:
                    self.data["north"][ii * ncol + jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break

        # get E-W locations
        for ii, dline in enumerate(dlines[findlist[1] + 1 : findlist[2]], 0):
            dline = dline.strip().split()
            for jj in range(self.n_z):
                try:
                    self.data["east"][ii * ncol + jj] = float(dline[jj])
                except IndexError:
                    pass
                except ValueError:
                    break
        # make some empty array to put stuff into
        self.period_list = np.zeros(n_periods)

        # get data
        per = 0
        error_find = False
        errmap_find = False
        data_count = 0
        for ii, dl in enumerate(dlines[findlist[2] :]):
            if dl.lower().find("period") > 0:
                st = 0

                if dl.lower().find("data") == 0:
                    dkey = "z_data"
                    self.period_list[per] = float(dl.strip().split()[1])

                elif dl.lower().find("error") == 0:
                    dkey = "z_data_err"
                    if not error_find:
                        error_find = True
                        per = 0

                elif dl.lower().find("ermap") == 0:
                    dkey = "z_err_map"
                    if not errmap_find:
                        errmap_find = True
                        per = 0

                # print '-'*20+dkey+'-'*20
                per += 1
                # print(dl)
            elif not dl.startswith('#'):
                
                if dkey == "z_err_map":
                    zline = np.array(dl.strip().split(), dtype=float)
                    if len(zline) >= 8:
                        self.data[st][dkey][per - 1, :] = np.array(
                            [
                                [
                                    zline[0] - 1j * zline[1],
                                    zline[2] - 1j * zline[3],
                                ],
                                [
                                    zline[4] - 1j * zline[5],
                                    zline[6] - 1j * zline[7],
                                ],
                            ]
                        )
                    # append tipper data
                    if ((len(zline) == 12) or (len(zline) == 4)):
                        add_idx = len(zline) - 4
                        self.data[st]['tipper_err_map'][per - 1, :] = np.array(
                            [
                                [
                                    zline[0+add_idx] - 1j * zline[1+add_idx],
                                    zline[2+add_idx] - 1j * zline[3+add_idx],
                                ]
                            ]
                        )                      
                    
                elif dkey == 'z_data_err':
                    zline = np.array(dl.strip().split(), dtype=float) * zconv
                    if len(zline) >= 8:
                        self.data[st][dkey][per - 1, :] = np.array(
                            [
                                [
                                    zline[0] + 1j * zline[1],
                                    zline[2] + 1j * zline[3],
                                ],
                                [
                                    zline[4] + 1j * zline[5],
                                    zline[6] + 1j * zline[7],
                                ],
                            ]
                        )
                    # append tipper data
                    if ((len(zline) == 12) or (len(zline) == 4)):
                        add_idx = len(zline) - 4
                        self.data[st]['tipper_data_err'][per - 1, :] = np.array(
                            [
                                [
                                    zline[0+add_idx] - 1j * zline[1+add_idx],
                                    zline[2+add_idx] - 1j * zline[3+add_idx],
                                ]
                            ]
                        )  

                else:
                    
                    zline = np.array(dl.strip().split(), dtype=float) * zconv
                    # print(st)
                    if len(zline) >= 8:
                        self.data[st][dkey][per - 1, :] = np.array(
                            [
                                [
                                    zline[0] - 1j * zline[1],
                                    zline[2] - 1j * zline[3],
                                ],
                                [
                                    zline[4] - 1j * zline[5],
                                    zline[6] - 1j * zline[7],
                                ],
                            ]
                        )
                    # append tipper data
                    if ((len(zline) == 12) or (len(zline) == 4)):
                        add_idx = len(zline) - 4
                        self.data[st]['tipper_data'][per - 1, :] = np.array(
                            [
                                [
                                    zline[0+add_idx] - 1j * zline[1+add_idx],
                                    zline[2+add_idx] - 1j * zline[3+add_idx],
                                ]
                            ]
                        )                        
                data_count += len(zline)
                if data_count == self.n_z:
                    st += 1
                    data_count = 0

        self.station_east = self.data["east"]
        self.station_north = self.data["north"]
        self.station_names = self.data["station"]
        self.z_data = self.data["z_data"]
        # need to be careful when multiplying complex numbers
        self.z_data_err = (
            self.data["z_data_err"].real * self.data["z_err_map"].real
            + 1j * self.data["z_data_err"].imag * self.data["z_err_map"].imag
        )

        # make station_locations structure array
        self.station_locations = np.zeros(
            len(self.station_east),
            dtype=[
                ("station", "|S10"),
                ("east", float),
                ("north", float),
                ("east_c", float),
                ("north_c", float),
            ],
        )
        self.station_locations["east"] = self.data["east"]
        self.station_locations["north"] = self.data["north"]
        self.station_locations["station"] = self.data["station"]
