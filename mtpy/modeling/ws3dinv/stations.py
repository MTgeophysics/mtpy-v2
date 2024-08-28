# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:18:17 2023

@author: jpeacock
"""
# ==============================================================================

from pathlib import Path
import numpy as np
from loguru import logger

import mtpy.modeling.winglink as wl

from pyevtk.hl import pointsToVTK


# ==============================================================================
class WSStation:
    """Read and write a station file where the locations are relative to the
    3D mesh.

    ==================== ======================================================
    Attributes           Description
    ==================== ======================================================
    east                 array of relative locations in east direction
    elev                 array of elevations for each station
    names                array of station names
    north                array of relative locations in north direction
    station_fn           full path to station file
    save_path            path to save file to
    ==================== ======================================================

    ==================== ======================================================
    Methods              Description
    ==================== ======================================================
    read_station_file    reads in a station file
    write_station_file   writes a station file
    write_vtk_file       writes a vtk points file for station locations
    ==================== ======================================================
    """

    def __init__(self, station_fn=None, **kwargs):
        self.logger = logger
        self.east = None
        self.north = None
        self.elev = None
        self.names = None
        self.save_path = Path()
        self.fn_basename = "WS_Station_Locations.txt"

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def station_filename(self):
        """Station filename."""
        return self.save_path.joinpath(self.fn_basename)

    @station_filename.setter
    def station_filename(self, value):
        """Station filename."""
        if value is not None:
            value = Path(value)
            if value.parent == Path("."):
                self.fn_basename = value.name
            else:
                self.save_path = value.parent
                self.fn_basename = value.name

    def write_station_file(
        self,
        east=None,
        north=None,
        station_list=None,
        save_path=None,
        elev=None,
    ):
        """Write a station file to go with the data file.

        the locations are on a relative grid where (0, 0, 0) is the
        center of the grid.  Also, the stations are assumed to be in the center
        of the cell.

        Arguments::
                **east** : np.ndarray(n_stations)
                           relative station locations in east direction

                **north** : np.ndarray(n_stations)
                           relative station locations in north direction

                **elev** : np.ndarray(n_stations)
                           relative station locations in vertical direction

                **station_list** : list or np.ndarray(n_stations)
                                   name of stations

                **save_path** : string
                                directory or full path to save station file to
                                if a directory  the file will be saved as
                                save_path/WS_Station_Locations.txt
                                if save_path is none the current working directory
                                is used as save_path

        Outputs::
                **station_fn** : full path to station file
        """
        if east is not None:
            self.east = east
        if north is not None:
            self.north = north
        if station_list is not None:
            self.names = station_list
        if elev is not None:
            self.elev = elev
        else:
            if self.north is not None:
                self.elev = np.zeros_like(self.north)

        with open(self.station_filename, "w") as sfid:
            sfid.write(
                f"{'station':<14}{'east':^14}{'north':^14}{'elev':^14}\n"
            )
            for ee, nn, zz, ss in zip(
                self.east, self.north, self.elev, self.names
            ):
                ee = f"{ee:+.4e}"
                nn = f"{nn:+.4e}"
                zz = f"{zz:+.4e}"
                sfid.write(f"{ss:<14}{ee:^14}{nn:^14}{zz:^14}\n")

        self.logger.info(f"Wrote station locations to {self.station_filename}")
        return self.station_filename

    def read_station_file(self, station_filename):
        """Read in station file written by write_station_file.

        Arguments::
                **station_fn** : string
                                 full path to station file

        Outputs::
                **east** : np.ndarray(n_stations)
                           relative station locations in east direction

                **north** : np.ndarray(n_stations)
                           relative station locations in north direction
                **elev** : np.ndarray(n_stations)
                           relative station locations in vertical direction

                **station_list** : list or np.ndarray(n_stations)
                                   name of stations
        """
        self.station_filename = station_filename

        self.station_locations = np.loadtxt(
            self.station_fn,
            skiprows=1,
            dtype=[
                ("station", "|U10"),
                ("east_c", float),
                ("north_c", float),
                ("elev", float),
            ],
        )

        self.east = self.station_locations["east_c"]
        self.north = self.station_locations["north_c"]
        self.names = self.station_locations["station"]
        self.elev = self.station_locations["elev"]

    def write_vtk_file(self, save_path, vtk_basename="VTKStations"):
        """Write a vtk file to plot stations.

        Arguments::
                **save_path** : string
                                directory to save file to.  Will save as
                                save_path/vtk_basename

                **vtk_basename** : string
                                   base file name for vtk file, extension is
                                   automatically added.
        """
        save_path = Path(save_path)
        if save_path.is_dir():
            save_fn = save_path.joinpath(vtk_basename)

        if self.elev is None:
            self.elev = np.zeros_like(self.north)

        pointsToVTK(
            save_fn,
            self.north,
            self.east,
            self.elev,
            data={"value": np.ones_like(self.north)},
        )

        return save_fn

    def from_wl_write_station_file(self, sites_file, out_file, ncol=5):
        """Write a ws station file from the outputs of winglink.

        Arguments::
                **sites_fn** : string
                               full path to sites file output from winglink

                **out_fn** : string
                             full path to .out file output from winglink

                **ncol** : int
                           number of columns the data is in
                           *default* is 5
        """

        wl_east, wl_north, wl_station_list = wl.get_station_locations(
            sites_file, out_file, ncol=ncol
        )
        self.write_station_file(
            east=wl_east, north=wl_north, station_list=wl_station_list
        )
