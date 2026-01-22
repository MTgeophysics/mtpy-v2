# -*- coding: utf-8 -*-
"""
Might think about adding declination

Created on Mon Oct  3 15:04:12 2022

@author: jpeacock
"""

import json

# =============================================================================
# Imports
# =============================================================================
from copy import deepcopy
from typing import Any

import numpy as np
from loguru import logger
from mt_metadata.timeseries import Run, Survey
from mt_metadata.transfer_functions.io.tools import get_nm_elev
from mt_metadata.transfer_functions.tf import Station
from pyproj import CRS

from mtpy.utils.gis_tools import project_point


# =============================================================================


class MTLocation:
    """
    Location for a MT site or point measurement.

    Parameters
    ----------
    survey_metadata : Survey, optional
        Survey metadata object, by default None
    **kwargs : dict
        Additional keyword arguments for location attributes

    Attributes
    ----------
    latitude : float
        Latitude in decimal degrees
    longitude : float
        Longitude in decimal degrees
    elevation : float
        Elevation in meters
    east : float
        Easting coordinate in meters
    north : float
        Northing coordinate in meters
    datum_crs : CRS
        Datum coordinate reference system
    utm_crs : CRS
        UTM coordinate reference system
    model_east : float
        Model easting coordinate in meters
    model_north : float
        Model northing coordinate in meters
    model_elevation : float
        Model elevation in meters
    profile_offset : float
        Distance along profile in meters

    """

    def __init__(self, survey_metadata: Survey | None = None, **kwargs: Any) -> None:
        self.logger = logger
        if survey_metadata is None:
            self._survey_metadata = self._initiate_metadata()
        else:
            self._survey_metadata = self._validate_metadata(survey_metadata)

        self._east = 0
        self._north = 0
        self._datum_crs = CRS.from_epsg(4326)
        self._utm_crs = None
        self._geoid_crs = None
        self.model_east = 0
        self.model_north = 0
        self.model_elevation = 0
        self.profile_offset = 0

        self._key_attrs = [
            "latitude",
            "longitude",
            "elevation",
            "east",
            "north",
            "model_east",
            "model_north",
            "model_elevation",
            "datum_crs",
            "utm_crs",
            "datum_epsg",
            "utm_epsg",
            "profile_offset",
        ]

        for key, value in kwargs.items():
            if key in self._key_attrs:
                setattr(self, key, value)

        if self.east != 0 and self.north != None:
            if self.utm_crs is None:
                raise ValueError("Need to input UTM CRS if only setting east and north")

    def _initiate_metadata(self) -> Survey:
        """
        Initiate metadata with default Survey object.

        Returns
        -------
        Survey
            Initialized survey metadata object with one station and one run

        """
        survey_metadata = Survey(id=0)
        survey_metadata.add_station(Station(id=0))
        survey_metadata.stations[0].add_run(Run(id=0))

        return survey_metadata

    def _validate_metadata(self, survey_metadata: Any) -> Survey:
        """
        Validate and ensure metadata has required structure.

        Parameters
        ----------
        survey_metadata : Survey
            Survey metadata object to validate

        Returns
        -------
        Survey
            Validated survey metadata object

        Raises
        ------
        TypeError
            If survey_metadata is not a Survey object

        """
        if not isinstance(survey_metadata, Survey):
            raise TypeError(
                "Input metadata must be type "
                "mt_metadata.transfer_functions.tf.Survey, "
                f"not {type(survey_metadata)}."
            )
        if len(survey_metadata.stations) < 1:
            survey_metadata.add_station(Station(id=0))

        if len(survey_metadata.stations[0].runs) < 1:
            survey_metadata.stations[0].add_run(Run(id=0))

        return survey_metadata

    def __str__(self) -> str:
        """
        String representation of MTLocation.

        Returns
        -------
        str
            Formatted string with location information

        """
        lines = ["MT Location: ", "-" * 20]
        lines.append(f"  Latitude (deg):   {self.latitude:.6f}")
        lines.append(f"  Longitude (deg):  {self.longitude:.6f}")
        lines.append(f"  Elevation (m):    {self.elevation:.4f}")
        lines.append(f"  Datum crs:        {self.datum_crs}")
        lines.append("")
        lines.append(f"  Easting (m):      {self.east:.3f}")
        lines.append(f"  Northing (m):     {self.north:.3f}")
        lines.append(f"  UTM crs:          {self.utm_crs}")
        lines.append("")
        lines.append(f"  Model Easting (m):      {self.model_east:.3f}")
        lines.append(f"  Model Northing (m):     {self.model_north:.3f}")
        lines.append(f"  Model Elevation (m):    {self.model_elevation:.3f}")
        lines.append(f"  Profile Offset (m):     {self.profile_offset:.3f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Representation of MTLocation.

        Returns
        -------
        str
            Formatted string with location information

        """
        return self.__str__()

    def __eq__(self, other: "MTLocation") -> bool:
        """
        Compare two MTLocation objects for equality.

        Parameters
        ----------
        other : MTLocation
            Another MTLocation object to compare with

        Returns
        -------
        bool
            True if locations are equal, False otherwise

        Raises
        ------
        TypeError
            If other is not an MTLocation object

        """

        if not isinstance(other, MTLocation):
            raise TypeError(f"Can not compare MTLocation with {type(other)}")

        for key in self._key_attrs:
            og_value = getattr(self, key)
            other_value = getattr(other, key)

            if isinstance(og_value, float):
                if not np.isclose(og_value, other_value):
                    self.logger.info(f"{key} not equal {og_value} != {other_value}")
                    return False
            else:
                if not og_value == other_value:
                    self.logger.info(f"{key} not equal {og_value} != {other_value}")
                    return False
        return True

    def copy(self) -> "MTLocation":
        """
        Create a deep copy of the MTLocation object.

        Returns
        -------
        MTLocation
            Deep copy of the current MTLocation object

        """
        copied = type(self)()
        copied._survey_metadata = self._survey_metadata.copy()
        # not sure why this is needed, survey metadata copies fine, but here
        # it does not.
        if len(copied._survey_metadata.stations) == 0:
            copied._survey_metadata.add_station(self._survey_metadata.stations[0])
        for key in self._key_attrs:
            setattr(copied, key, deepcopy(getattr(self, key)))

        return copied

    @property
    def datum_crs(self) -> CRS | None:
        """
        Datum coordinate reference system.

        Returns
        -------
        CRS or None
            Datum CRS object

        """
        if self._datum_crs is not None:
            return self._datum_crs

    @property
    def datum_name(self) -> str | None:
        """
        Name of the datum coordinate reference system.

        Returns
        -------
        str or None
            Datum CRS name

        """
        if self._datum_crs is not None:
            return self._datum_crs.name

    @property
    def datum_epsg(self) -> int | None:
        """
        EPSG code of the datum coordinate reference system.

        Returns
        -------
        int or None
            Datum EPSG code

        """
        if self._datum_crs is not None:
            return self._datum_crs.to_epsg()

    @datum_epsg.setter
    def datum_epsg(self, value: int | str | None) -> None:
        """
        Set datum EPSG code.

        Parameters
        ----------
        value : int, str, or None
            EPSG code for datum CRS

        """
        if value not in ["", None, "None"]:
            self.datum_crs = value

    @datum_crs.setter
    def datum_crs(self, value: CRS | int | str | None) -> None:
        """
        Set datum coordinate reference system.

        Parameters
        ----------
        value : CRS, int, str, or None
            Datum CRS object, EPSG code, or CRS string

        Notes
        -----
        When datum CRS is changed, coordinates are reprojected accordingly

        """
        if value in [None, "None", "none", "null", ""]:
            return

        new_crs = CRS.from_user_input(value)

        if new_crs != self._datum_crs:
            if (
                self._datum_crs is not None
                and self.latitude != 0
                and self.longitude != 0
            ):
                (
                    self._survey_metadata.stations[0].location.longitude,
                    self._survey_metadata.stations[0].location.latitude,
                ) = project_point(
                    self.longitude, self.latitude, self._datum_crs, new_crs
                )

                self._east, self._north = project_point(
                    self.longitude, self.latitude, new_crs, self.utm_crs
                )

            elif (
                self.datum_crs is not None
                and self.east != 0
                and self.north != 0
                and self.latitude == 0
                and self.longitude == 0
            ):
                (
                    self._survey_metadata.stations[0].location.longitude,
                    self._survey_metadata.stations[0].location.latitude,
                ) = project_point(
                    self.east,
                    self.north,
                    self.utm_crs,
                    new_crs,
                )
            self._datum_crs = new_crs

    @property
    def utm_crs(self) -> CRS | None:
        """
        UTM coordinate reference system.

        Returns
        -------
        CRS or None
            UTM CRS object

        """
        if self._utm_crs is not None:
            return self._utm_crs

    @property
    def utm_name(self) -> str | None:
        """
        Name of the UTM coordinate reference system.

        Returns
        -------
        str or None
            UTM CRS name

        """
        if self._utm_crs is not None:
            return self._utm_crs.name

    @property
    def utm_epsg(self) -> int | None:
        """
        EPSG code of the UTM coordinate reference system.

        Returns
        -------
        int or None
            UTM EPSG code

        """
        if self._utm_crs is not None:
            return self._utm_crs.to_epsg()

    @utm_epsg.setter
    def utm_epsg(self, value: int | str | None) -> None:
        """
        Set UTM EPSG code.

        Parameters
        ----------
        value : int, str, or None
            EPSG code for UTM CRS

        """
        if value not in ["", None, "None"]:
            self.utm_crs = value

    @property
    def utm_zone(self) -> str | None:
        """
        UTM zone string.

        Returns
        -------
        str or None
            UTM zone identifier

        """
        if self._utm_crs is not None:
            return self._utm_crs.utm_zone

    @utm_crs.setter
    def utm_crs(self, value: CRS | int | str | None) -> None:
        """
        Set UTM coordinate reference system.

        Parameters
        ----------
        value : CRS, int, str, or None
            UTM CRS object, EPSG code, or CRS string

        Notes
        -----
        When UTM CRS is changed, coordinates are reprojected accordingly

        """
        if value in [None, "None", "none", "null", ""]:
            return

        new_crs = CRS.from_user_input(value)
        if value != self._utm_crs:
            # reproject easting, northing to new zone
            if self._utm_crs is not None and self.east != 0 and self.north != 0:
                self._east, self._north = project_point(
                    self.east, self.north, self._utm_crs, new_crs
                )

            if self.datum_crs is not None and self.east != 0 and self.north != 0:
                # reproject lat and lon base on new UTM datum
                (
                    self._survey_metadata.stations[0].location.longitude,
                    self._survey_metadata.stations[0].location.latitude,
                ) = project_point(
                    self.east,
                    self.north,
                    new_crs,
                    self.datum_crs,
                )

            # if east and north == 0 and lat and lon != 0 project to utm
            elif (
                self.datum_crs is not None
                and self.east == 0
                and self.north == 0
                and self.latitude != 0
                and self.longitude != 0
            ):
                self._east, self._north = project_point(
                    self.longitude,
                    self.latitude,
                    self.datum_crs,
                    new_crs,
                )

            self._utm_crs = new_crs

    @property
    def east(self) -> float:
        """
        Easting coordinate in meters.

        Returns
        -------
        float
            Easting coordinate

        """
        return self._east

    @east.setter
    def east(self, value: float) -> None:
        """
        Set easting coordinate.

        Parameters
        ----------
        value : float
            Easting coordinate in meters

        Notes
        -----
        Updates latitude/longitude if datum and UTM CRS are set

        """
        self._east = value
        if self.datum_crs is not None and self.utm_crs is not None and self._north != 0:
            (
                self._survey_metadata.stations[0].location.longitude,
                self._survey_metadata.stations[0].location.latitude,
            ) = project_point(self._east, self._north, self.utm_crs, self.datum_crs)

    @property
    def north(self) -> float:
        """
        Northing coordinate in meters.

        Returns
        -------
        float
            Northing coordinate

        """
        return self._north

    @north.setter
    def north(self, value: float) -> None:
        """
        Set northing coordinate.

        Parameters
        ----------
        value : float
            Northing coordinate in meters

        Notes
        -----
        Updates latitude/longitude if datum and UTM CRS are set

        """
        self._north = value
        if self.datum_crs is not None and self.utm_crs is not None and self._east != 0:
            (
                self._survey_metadata.stations[0].location.longitude,
                self._survey_metadata.stations[0].location.latitude,
            ) = project_point(self._east, self._north, self.utm_crs, self.datum_crs)

    @property
    def latitude(self) -> float:
        """
        Latitude in decimal degrees.

        Returns
        -------
        float
            Latitude coordinate

        """
        return self._survey_metadata.stations[0].location.latitude

    @latitude.setter
    def latitude(self, lat: float) -> None:
        """
        Set latitude coordinate.

        Parameters
        ----------
        lat : float
            Latitude in decimal degrees

        Notes
        -----
        Updates easting/northing if datum and UTM CRS are set

        """
        self._survey_metadata.stations[0].location.latitude = lat
        if (
            self.utm_crs is not None
            and self.datum_crs is not None
            and self._survey_metadata.stations[0].location.longitude != 0
        ):
            self._east, self._north = project_point(
                self._survey_metadata.stations[0].location.longitude,
                self._survey_metadata.stations[0].location.latitude,
                self.datum_crs,
                self.utm_crs,
            )

    @property
    def longitude(self) -> float:
        """
        Longitude in decimal degrees.

        Returns
        -------
        float
            Longitude coordinate

        """
        return self._survey_metadata.stations[0].location.longitude

    @longitude.setter
    def longitude(self, lon: float) -> None:
        """
        Set longitude coordinate.

        Parameters
        ----------
        lon : float
            Longitude in decimal degrees

        Notes
        -----
        Updates easting/northing if datum and UTM CRS are set

        """
        self._survey_metadata.stations[0].location.longitude = lon
        if (
            self.utm_crs is not None
            and self.datum_crs is not None
            and self._survey_metadata.stations[0].location.latitude != 0
        ):
            self._east, self._north = project_point(
                self._survey_metadata.stations[0].location.longitude,
                self._survey_metadata.stations[0].location.latitude,
                self.datum_crs,
                self.utm_crs,
            )

    @property
    def elevation(self) -> float:
        """
        Elevation in meters.

        Returns
        -------
        float
            Elevation above datum

        """
        return self._survey_metadata.stations[0].location.elevation

    @elevation.setter
    def elevation(self, elev: float) -> None:
        """
        Set elevation.

        Parameters
        ----------
        elev : float
            Elevation in meters

        """
        self._survey_metadata.stations[0].location.elevation = elev

    @property
    def model_east(self) -> float:
        """
        Model easting coordinate in meters.

        Returns
        -------
        float
            Model easting relative to model center

        """
        return self._model_east

    @model_east.setter
    def model_east(self, value: float) -> None:
        """
        Set model easting coordinate.

        Parameters
        ----------
        value : float
            Model easting in meters

        Raises
        ------
        ValueError
            If value cannot be converted to float

        """
        try:
            self._model_east = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Input should be a float not type {type(value)}")

    @property
    def model_north(self) -> float:
        """
        Model northing coordinate in meters.

        Returns
        -------
        float
            Model northing relative to model center

        """
        return self._model_north

    @model_north.setter
    def model_north(self, value: float) -> None:
        """
        Set model northing coordinate.

        Parameters
        ----------
        value : float
            Model northing in meters

        Raises
        ------
        ValueError
            If value cannot be converted to float

        """
        try:
            self._model_north = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Input should be a float not type {type(value)}")

    @property
    def model_elevation(self) -> float:
        """
        Model elevation in meters.

        Returns
        -------
        float
            Model elevation relative to model center

        """
        return self._model_elevation

    @model_elevation.setter
    def model_elevation(self, value: float) -> None:
        """
        Set model elevation.

        Parameters
        ----------
        value : float
            Model elevation in meters

        Raises
        ------
        ValueError
            If value cannot be converted to float

        """
        try:
            self._model_elevation = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Input should be a float not type {type(value)}")

    def compute_model_location(self, center_location: "MTLocation") -> None:
        """
        Compute model coordinates relative to model center.

        Parameters
        ----------
        center_location : MTLocation
            Center location of the model

        Notes
        -----
        Sets model_east, model_north, and model_elevation relative to center

        """

        self.model_east = self.east - center_location.model_east
        self.model_north = self.north - center_location.model_north
        self.model_elevation = self.elevation - center_location.model_elevation

    def project_onto_profile_line(
        self, profile_slope: float, profile_intersection: float
    ) -> None:
        """
        Project station location onto a profile line.

        Parameters
        ----------
        profile_slope : float
            Slope of the profile line
        profile_intersection : float
            Y-intercept of the profile line

        Raises
        ------
        ValueError
            If utm_crs is None

        Notes
        -----
        Sets the profile_offset attribute with distance along profile line

        """

        if self.utm_crs is None:
            raise ValueError("utm_crs is None, cannot project onto profile line.")

        profile_vector = np.array([1, profile_slope], dtype=float)
        profile_vector /= np.linalg.norm(profile_vector)

        station_vector = np.array([self.east, self.north - profile_intersection])

        self.profile_offset = np.linalg.norm(
            np.dot(profile_vector, station_vector) * profile_vector
        )

    def get_elevation_from_national_map(self) -> None:
        """
        Get elevation from DEM data of the US National Map.

        Notes
        -----
        Pulls data from the USGS National Map DEM. Plan to extend this to
        global coverage. Updates the elevation attribute if successful.

        """

        elev = get_nm_elev(self.latitude, self.longitude)
        if elev != 0:
            self.elevation = elev
        else:
            self.logger.warning("Could not get elevation data, not setting elevation")

    def to_json(self, filename: str) -> None:
        """
        Write location information to a JSON file.

        Parameters
        ----------
        filename : str
            Path to output JSON file

        """

        js_dict = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "north": self.north,
            "east": self.east,
            "model_east": self.model_east,
            "model_north": self.model_north,
            "model_elevation": self.model_elevation,
            "datum_crs": self.datum_crs.to_json_dict(),
            "utm_crs": self.utm_crs.to_json_dict(),
        }
        with open(filename, "w") as fid:
            json.dump(js_dict, fid)

    def from_json(self, filename: str) -> None:
        """
        Read location information from a JSON file.

        Parameters
        ----------
        filename : str
            Path to input JSON file

        """

        with open(filename, "r") as fid:
            js_dict = json.load(fid)

        for key, value in js_dict.items():
            if key in ["datum_crs", "utm_crs"]:
                if isinstance(value, dict):
                    setattr(self, key, CRS.from_json_dict(value))
                else:
                    setattr(self, key, CRS.from_user_input(value))
            else:
                setattr(self, key, value)
