# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:58:56 2022

@author: jpeacock
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mth5.helpers import validate_name

from mtpy.core import COORDINATE_REFERENCE_FRAME_OPTIONS, MTDataFrame
from mtpy.core.transfer_function import IMPEDANCE_UNITS
from mtpy.gis.shapefile_creator import ShapefileCreator
from mtpy.imaging import (
    PlotMultipleResponses,
    PlotPenetrationDepthMap,
    PlotPhaseTensorMaps,
    PlotPhaseTensorPseudoSection,
    PlotResidualPTMaps,
    PlotResPhaseMaps,
    PlotResPhasePseudoSection,
    PlotStations,
    PlotStrike,
)
from mtpy.modeling.errors import ModelErrors
from mtpy.modeling.modem import Data
from mtpy.modeling.occam2d import Occam2DData
from mtpy.modeling.simpeg.data_2d import Simpeg2DData
from mtpy.modeling.simpeg.data_3d import Simpeg3DData

from .mt import MT
from .mt_stations import MTStations


# =============================================================================


class MTData(OrderedDict, MTStations):
    """
    Collection of MT objects as an OrderedDict.

    Keys are formatted as `survey_id.station_id`. Has all functionality of an
    OrderedDict - can iterate over `.keys()`, `.values()` or `.items()`.
    Values are MT objects.

    Parameters
    ----------
    mt_list : list of MT, optional
        List of MT objects to initialize with, by default None
    **kwargs : dict
        Additional keyword arguments including utm_epsg, datum_epsg

    Attributes
    ----------
    z_model_error : ModelErrors
        Impedance model error configuration
    t_model_error : ModelErrors
        Tipper model error configuration
    data_rotation_angle : float
        Data rotation angle in degrees
    coordinate_reference_frame : str
        Coordinate reference frame ('ned' or 'enu')
    impedance_units : str
        Impedance units ('mt' or 'ohm')

    Notes
    -----
    Inherits from :class:`mtpy.core.MTStations` to handle geographic locations.
    Not yet optimized for speed - works fine for smaller surveys but can be
    slow for large datasets.

    """

    def __init__(self, mt_list: list[MT] | None = None, **kwargs: Any) -> None:
        self._coordinate_reference_frame_options = COORDINATE_REFERENCE_FRAME_OPTIONS

        self.z_model_error = ModelErrors(
            error_value=5,
            error_type="geometric_mean",
            floor=True,
            mode="impedance",
        )
        self.t_model_error = ModelErrors(
            error_value=0.02,
            error_type="absolute",
            floor=True,
            mode="tipper",
        )
        self.data_rotation_angle = 0
        self.coordinate_reference_frame = "ned"
        self._impedance_unit_factors = IMPEDANCE_UNITS
        self.impedance_units = "mt"

        self.model_parameters = {}

        self._copy_attrs = [
            "z_model_error",
            "t_model_error",
            "utm_crs",
            "datum_crs",
            "_center_lat",
            "_center_lon",
            "_center_elev",
            "shift_east",
            "shift_north",
            "rotation_angle",
            "data_rotation_angle",
            "model_parameters",
            "impedance_units",
        ]

        if mt_list is not None:
            for mt_obj in mt_list:
                self.add_station(mt_obj, compute_relative_location=False)

        MTStations.__init__(
            self,
            kwargs.pop("utm_epsg", None),
            datum_epsg=kwargs.pop("datum_epsg", None),
            **kwargs,
        )

    def _validate_item(self, mt_obj: MT | str | Path) -> MT:
        """
        Validate that input is an MT object or convert it.

        Parameters
        ----------
        mt_obj : MT, str, or Path
            MT object or path to MT data file

        Returns
        -------
        MT
            Validated MT object

        Raises
        ------
        TypeError
            If input is not MT object, string, or Path

        Notes
        -----
        If input is a string or Path, assumes it's a file path and reads it

        """
        if isinstance(mt_obj, (str, Path)):
            m = MT()
            m.read(mt_obj)
            return m

        elif not isinstance(mt_obj, MT):
            raise TypeError(
                f"entry must be a mtpy.core.MT object not type({type(mt_obj)})"
            )
        return mt_obj

    def __eq__(self, other: "MTData") -> bool:
        """
        Test equality with another MTData object.

        Parameters
        ----------
        other : MTData
            Another MTData object to compare with

        Returns
        -------
        bool
            True if equal, False otherwise

        Raises
        ------
        TypeError
            If other is not an MTData object

        """

        if not isinstance(other, MTData):
            raise TypeError(f"Can not compare MTData to {type(other)}.")

        for attr in self._copy_attrs:
            value_og = getattr(self, attr)
            value_other = getattr(other, attr)

            if value_og != value_other:
                self.logger.info(f"Attribute {attr}: {value_og} != {value_other}")
                return False
        fail = False
        if len(self) == len(other):
            for key in self.keys():
                mt1 = self[key]
                try:
                    mt2 = other[key]
                    if mt1 != mt2:
                        self.logger.info(f"Station {key} is not equal.")
                        fail = True
                except KeyError:
                    self.logger.info(f"Could not find {key} in other.")
                    fail = True
            if fail:
                return False
        else:
            self.logger.info(
                f"Length of MTData not the same {len(self)} != {len(other)}"
            )
            return False
        return True

    def __deepcopy__(self, memo: dict) -> "MTData":
        """
        Create a deep copy of MTData object.

        Parameters
        ----------
        memo : dict
            Memoization dictionary for deepcopy

        Returns
        -------
        MTData
            Deep copy of the original MTData object

        Notes
        -----
        Logger is skipped during copying to avoid duplication issues

        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__init__()
        memo[id(self)] = result
        for key in self._copy_attrs:
            value = getattr(self, key)
            setattr(result, key, deepcopy(value, memo))

        for mt_obj in self.values():
            result.add_station(mt_obj.copy(), compute_relative_location=False)

        return result

    def copy(self) -> "MTData":
        """
        Create a deep copy of MTData object.

        Returns
        -------
        MTData
            Deep copy of the original MTData object

        """
        copied = deepcopy(self)
        copied.logger = self.logger
        return copied

    def clone_empty(self) -> "MTData":
        """
        Create a copy of MTData excluding all MT objects.

        Returns
        -------
        MTData
            Empty MTData object with copied attributes but no stations

        """

        md = MTData()
        for attr in self._copy_attrs:
            setattr(md, attr, deepcopy(getattr(self, attr)))

        return md

    @property
    def coordinate_reference_frame(self) -> str:
        """
        Coordinate reference frame.

        Returns
        -------
        str
            Reference frame identifier ('NED' or 'ENU')

        """
        return self._coordinate_reference_frame_options[
            self._coordinate_reference_frame
        ].upper()

    @coordinate_reference_frame.setter
    def coordinate_reference_frame(self, value: str) -> None:
        """
        Set coordinate reference frame.

        Parameters
        ----------
        value : str
            Reference frame identifier. Options:

            - 'NED': x=North, y=East, z=+down
            - 'ENU': x=East, y=North, z=+up

        Raises
        ------
        ValueError
            If value is not a recognized reference frame

        Notes
        -----
        Updates coordinate reference frame for all MT objects in collection

        """

        if value.lower() not in self._coordinate_reference_frame_options:
            raise ValueError(
                f"{value} is not understood as a reference frame. "
                f"Options are {self._coordinate_reference_frame_options}"
            )
        if value in ["ned"] or "+" in value:
            value = "+"
        elif value in ["enu"] or "-" in value:
            value = "-"
            self.logger.warning(
                "MTpy-v2 is assumes a NED coordinate system where x=North, "
                "y=East, z=+down. By changing to ENU there maybe some "
                "incorrect values for angles and derivative products of the "
                "impedance tensor."
            )

        for mt_obj in self.values():
            mt_obj.coordinate_reference_frame = value

        self._coordinate_reference_frame = value

    @property
    def impedance_units(self) -> str:
        """
        Impedance units.

        Returns
        -------
        str
            Impedance units ('mt' or 'ohm')

        """
        return self._impedance_units

    @impedance_units.setter
    def impedance_units(self, value: str) -> None:
        """
        Set impedance units.

        Parameters
        ----------
        value : str
            Impedance units. Options: 'mt' [mV/km/nT] or 'ohm' [Ohms]

        Raises
        ------
        TypeError
            If value is not a string
        ValueError
            If value is not 'mt' or 'ohm'

        Notes
        -----
        Updates impedance units for all MT objects in collection

        """
        if not isinstance(value, str):
            raise TypeError("Units input must be a string.")
        if value.lower() not in self._impedance_unit_factors.keys():
            raise ValueError(f"{value} is not an acceptable unit for impedance.")

        self._impedance_units = value

        if self.mt_list is not None:
            for mt_obj in self.values():
                mt_obj.impedance_units = self._impedance_units

    @property
    def mt_list(self) -> list[MT]:
        """
        List of all MT objects.

        Returns
        -------
        list of MT
            List of all MT objects in the collection

        """
        return self.values()

    @mt_list.setter
    def mt_list(self, value: list[MT] | None) -> None:
        """
        Set MT list (not implemented).

        Parameters
        ----------
        value : list of MT or None
            List of MT objects

        Notes
        -----
        Not implemented - mainly here for inheritance from MTStations

        """
        if value is None:
            return
        if len(self.values()) != 0:
            self.logger.warning("mt_list cannot be set.")

    @property
    def survey_ids(self) -> list[str]:
        """
        Unique survey IDs in the collection.

        Returns
        -------
        list of str
            List of unique survey IDs

        """
        return list(set([key.split(".")[0] for key in self.keys()]))

    def get_survey(self, survey_id: str) -> "MTData":
        """
        Get all MT objects belonging to a specific survey.

        Parameters
        ----------
        survey_id : str
            Survey identifier

        Returns
        -------
        MTData
            New MTData object containing only stations from the specified survey

        """

        survey_list = [mt_obj for key, mt_obj in self.items() if survey_id in key]
        md = self.clone_empty()
        md.add_station(survey_list)
        return md

    def add_station(
        self,
        mt_object: MT | list[MT],
        survey: str | None = None,
        compute_relative_location: bool = True,
        interpolate_periods: np.ndarray | None = None,
        compute_model_error: bool = False,
    ) -> None:
        """
        Add MT object(s) to the collection.

        Parameters
        ----------
        mt_object : MT or list of MT
            MT object(s) to add
        survey : str, optional
            Survey name to assign, by default None
        compute_relative_location : bool, optional
            Whether to compute relative locations after adding.
            Set to False when adding many stations in a loop for efficiency,
            by default True
        interpolate_periods : np.ndarray, optional
            Periods to interpolate onto, by default None
        compute_model_error : bool, optional
            Whether to compute model errors, by default False

        Notes
        -----
        For efficiency when adding multiple stations, set
        compute_relative_location=False and call compute_relative_locations()
        after all stations are added

        """

        if not isinstance(mt_object, (list, tuple)):
            mt_object = [mt_object]

        for m in mt_object:
            m = self._validate_item(m)
            try:
                if self.utm_crs is not None:
                    m.utm_crs = self.utm_crs
            except AttributeError:
                pass
            if survey is not None:
                m.survey = survey

            if interpolate_periods is not None:
                if not isinstance(interpolate_periods, np.ndarray):
                    interpolate_periods = np.array(interpolate_periods)

                m = m.interpolate(interpolate_periods, bounds_error=False)

            if compute_model_error:
                m.compute_model_z_errors(
                    error_value=self.z_model_error.error_value,
                    error_type=self.z_model_error.error_type,
                    floor=self.z_model_error.floor,
                )
                m.compute_model_t_errors(
                    error_value=self.t_model_error.error_value,
                    error_type=self.t_model_error.error_type,
                    floor=self.t_model_error.floor,
                )

            self.__setitem__(f"{validate_name(m.survey)}.{m.station}", m)

        if compute_relative_location:
            self.compute_relative_locations()

    def add_tf(self, tf: MT | list[MT], **kwargs: Any) -> None:
        """
        Add transfer function (MT object).

        Parameters
        ----------
        tf : MT or list of MT
            Transfer function object(s) to add
        **kwargs : dict
            Additional keyword arguments passed to add_station

        See Also
        --------
        add_station : Main method for adding MT objects

        """
        self.add_station(tf, **kwargs)

    def remove_station(
        self, station_id: str | list[str], survey_id: str | None = None
    ) -> None:
        """
        Remove station(s) from the collection.

        Parameters
        ----------
        station_id : str or list of str
            Station identifier(s) to remove
        survey_id : str, optional
            Survey identifier, by default None

        """
        if not isinstance(station_id, (list, tuple)):
            station_id = [station_id]
        for st_id in station_id:
            key = self._get_station_key(st_id, survey_id)
            if key in self.keys():
                del self[key]

    def _get_station_key(self, station_id: str, survey_id: str | None) -> str:
        """
        Get station key from station and survey IDs.

        Parameters
        ----------
        station_id : str
            Station identifier
        survey_id : str or None
            Survey identifier

        Returns
        -------
        str
            Station key in format 'survey_id.station_id'

        Raises
        ------
        KeyError
            If station key cannot be found

        """

        if station_id is not None:
            if survey_id is not None:
                return f"{validate_name(survey_id)}.{station_id}"
            else:
                for key in self.keys():
                    if station_id in key:
                        if key.split(".")[1] == station_id:
                            return key
        raise KeyError(
            f"Could not find station_id = {station_id}, survey_id = {survey_id}"
        )

    def get_periods(self) -> np.ndarray:
        """
        Get all unique periods across all stations.

        Returns
        -------
        np.ndarray
            Sorted array of unique periods in seconds

        """

        df = self.to_dataframe()
        periods = df.period.unique()
        periods.sort()
        return periods

    def get_station(
        self,
        station_id: str | None = None,
        survey_id: str | None = None,
        station_key: str | None = None,
    ) -> MT:
        """
        Get an MT object from the collection.

        Parameters
        ----------
        station_id : str, optional
            Station identifier, by default None
        survey_id : str, optional
            Survey identifier, by default None
        station_key : str, optional
            Full station key as 'survey_id.station_id', by default None

        Returns
        -------
        MT
            MT object for the requested station

        Raises
        ------
        KeyError
            If station cannot be found

        Notes
        -----
        If station_key is None, tries to find key from station_id and survey_id

        """
        if station_key is not None:
            station_key = station_key
        else:
            station_key = self._get_station_key(station_id, validate_name(survey_id))

        try:
            return self[station_key]
        except KeyError:
            raise KeyError(f"Could not find {station_key} in MTData.")

    def apply_bounding_box(
        self, lon_min: float, lon_max: float, lat_min: float, lat_max: float
    ) -> "MTData":
        """
        Get subset of stations within a bounding box.

        Parameters
        ----------
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude

        Returns
        -------
        MTData
            New MTData object containing only stations within the bounding box

        """

        bb_df = self.station_locations.loc[
            (self.station_locations.longitude >= lon_min)
            & (self.station_locations.longitude <= lon_max)
            & (self.station_locations.latitude >= lat_min)
            & (self.station_locations.latitude <= lat_max)
        ]

        station_keys = [
            f"{survey}.{station}"
            for survey, station in zip(bb_df.survey, bb_df.station)
        ]

        return self.get_subset(station_keys)

    def get_subset(self, station_list: list[str]) -> "MTData":
        """
        Get subset of stations from a list.

        Parameters
        ----------
        station_list : list of str
            List of station identifiers or keys. Keys should be in format
            'survey_id.station_id'

        Returns
        -------
        MTData
            New MTData object containing only the requested stations

        """
        mt_data = self.clone_empty()
        for station in station_list:
            if station.count(".") > 0:
                mt_data.add_station(
                    self.get_station(station_key=station),
                    compute_relative_location=False,
                )
            else:
                mt_data.add_station(
                    self.get_station(station_id=station),
                    compute_relative_location=False,
                )

        return mt_data

    @property
    def n_stations(self) -> int:
        """
        Number of stations in the collection.

        Returns
        -------
        int
            Number of MT objects in the collection

        """

        if self.mt_list is not None:
            return len(self.mt_list)

    def to_dataframe(
        self,
        utm_crs: Any | None = None,
        cols: list[str] | None = None,
        impedance_units: str = "mt",
    ) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.

        Parameters
        ----------
        utm_crs : CRS, int, str, optional
            UTM coordinate reference system, by default None
        cols : list of str, optional
            Columns to include in dataframe, by default None (all columns)
        impedance_units : str, optional
            Impedance units ('mt' [mV/km/nT] or 'ohm' [Ohms]), by default 'mt'

        Returns
        -------
        pd.DataFrame
            DataFrame containing MT data from all stations

        """

        df_list = [
            mt_obj.to_dataframe(
                utm_crs=utm_crs, cols=cols, impedance_units=impedance_units
            ).dataframe
            for mt_obj in self.values()
        ]

        df = pd.concat(df_list)
        df.reset_index(drop=True, inplace=True)
        return df

    def to_mt_dataframe(
        self, utm_crs: Any | None = None, impedance_units: str = "mt"
    ) -> MTDataFrame:
        """
        Create an MTDataFrame object.

        Parameters
        ----------
        utm_crs : CRS, int, str, optional
            UTM coordinate reference system, by default None
        impedance_units : str, optional
            Impedance units ('mt' [mV/km/nT] or 'ohm' [Ohms]), by default 'mt'

        Returns
        -------
        MTDataFrame
            MTDataFrame object containing all station data

        """

        return MTDataFrame(
            self.to_dataframe(utm_crs=utm_crs, impedance_units=impedance_units)
        )

    def from_dataframe(self, df: pd.DataFrame, impedance_units: str = "mt") -> None:
        """
        Create MT objects from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing MT data
        impedance_units : str, optional
            Impedance units ('mt' [mV/km/nT] or 'ohm' [Ohms]), by default 'mt'

        """

        for station in df.station.unique():
            sdf = df.loc[df.station == station]
            mt_object = MT(period=sdf.period.unique())
            mt_object.from_dataframe(sdf, impedance_units=impedance_units)
            self.add_station(mt_object, compute_relative_location=False)

    def from_mt_dataframe(
        self, mt_df: MTDataFrame, impedance_units: str = "mt"
    ) -> None:
        """
        Create MT objects from an MTDataFrame.

        Parameters
        ----------
        mt_df : MTDataFrame
            MTDataFrame containing MT data
        impedance_units : str, optional
            Impedance units ('mt' [mV/km/nT] or 'ohm' [Ohms]), by default 'mt'

        """

        self.from_dataframe(mt_df.dataframe, impedance_units=impedance_units)

    def to_geo_df(
        self, model_locations: bool = False, data_type: str = "station_locations"
    ) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame for GIS operations.

        Parameters
        ----------
        model_locations : bool, optional
            If True, returns points in model coordinates, by default False
        data_type : str, optional
            Type of data to include. Options:

            - 'station_locations' or 'stations': Station locations only
            - 'phase_tensor' or 'pt': Phase tensor data
            - 'tipper' or 't': Tipper data
            - 'both' or 'shapefiles': Both phase tensor and tipper

            By default 'station_locations'

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with requested data in requested coordinates

        Raises
        ------
        ValueError
            If data_type is not supported

        """

        if data_type in ["station_locations", "stations"]:
            df = self.station_locations
        elif data_type in ["phase_tensor", "pt"]:
            df = self.to_mt_dataframe().phase_tensor
        elif data_type in ["tipper", "t"]:
            df = self.to_mt_dataframe().tipper
        elif data_type in ["both", "shapefiles"]:
            df = self.to_mt_dataframe().for_shapefiles
        else:
            raise ValueError(f"Option for 'data_type' {data_type} is unsupported.")
        if model_locations:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.model_east, df.model_north),
                crs=None,
            )
        else:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs=self.datum_crs,
            )

        return gdf

    def interpolate(
        self,
        new_periods: np.ndarray,
        f_type: str = "period",
        inplace: bool = True,
        bounds_error: bool = True,
        **kwargs: Any,
    ) -> "MTData" | None:
        """
        Interpolate all MT objects onto common period range.

        Parameters
        ----------
        new_periods : np.ndarray
            Target periods for interpolation
        f_type : str, optional
            Frequency type ('frequency' or 'period'), by default 'period'
        inplace : bool, optional
            If True, modifies collection in place, by default True
        bounds_error : bool, optional
            If True, only interpolates within data bounds, by default True
        **kwargs : dict
            Additional interpolation parameters

        Returns
        -------
        MTData or None
            New MTData object if inplace=False, None otherwise

        """

        if not inplace:
            mt_data = self.clone_empty()

        for mt_obj in self.values():
            if bounds_error:
                interp_periods = new_periods[
                    np.where(
                        (new_periods <= mt_obj.period.max())
                        & (new_periods >= mt_obj.period.min())
                    )
                ]
            else:
                interp_periods = new_periods

            new_mt_obj = mt_obj.interpolate(
                interp_periods,
                f_type=f_type,
                bounds_error=bounds_error,
                **kwargs,
            )

            if inplace:
                self.update(
                    {
                        f"{new_mt_obj.survey_metadata.id}.{new_mt_obj.station}": new_mt_obj
                    }
                )

            else:
                mt_data.add_station(new_mt_obj, compute_relative_location=False)

        if not inplace:
            return mt_data

    def rotate(self, rotation_angle: float, inplace: bool = True) -> "MTData" | None:
        """
        Rotate all MT data by a given angle.

        Parameters
        ----------
        rotation_angle : float
            Rotation angle in degrees, positive clockwise with north=0, east=90
        inplace : bool, optional
            If True, modifies collection in place, by default True

        Returns
        -------
        MTData or None
            New MTData object if inplace=False, None otherwise

        """
        if not inplace:
            mt_data = self.clone_empty()
            mt_data.data_rotation_angle = rotation_angle
        else:
            self.data_rotation_angle = rotation_angle

        for mt_obj in self.values():
            if not inplace:
                rot_mt_obj = mt_obj.rotate(rotation_angle, inplace=False)
                mt_data.add_station(rot_mt_obj, compute_relative_location=False)
            else:
                mt_obj.rotate(rotation_angle)

        if not inplace:
            return mt_data

    def get_profile(
        self, x1: float, y1: float, x2: float, y2: float, radius: float
    ) -> "MTData":
        """
        Get stations along a profile line.

        Parameters
        ----------
        x1 : float
            X-coordinate (longitude or easting) of profile start
        y1 : float
            Y-coordinate (latitude or northing) of profile start
        x2 : float
            X-coordinate (longitude or easting) of profile end
        y2 : float
            Y-coordinate (latitude or northing) of profile end
        radius : float
            Search radius in meters

        Returns
        -------
        MTData
            New MTData object containing only stations within radius of profile line

        Notes
        -----
        Calculation is done in UTM, therefore a UTM CRS must be set

        """

        profile_stations = self._extract_profile(x1, y1, x2, y2, radius)

        mt_data = self.clone_empty()
        for mt_obj in profile_stations:
            mt_data.add_station(mt_obj, compute_relative_location=False)

        return mt_data

    def compute_model_errors(
        self,
        z_error_value: float | None = None,
        z_error_type: str | None = None,
        z_floor: bool | None = None,
        t_error_value: float | None = None,
        t_error_type: str | None = None,
        t_floor: bool | None = None,
    ) -> None:
        """
        Compute model errors for all MT objects.

        Parameters
        ----------
        z_error_value : float, optional
            Error value for impedance, by default None
        z_error_type : str, optional
            Error type for impedance, by default None
        z_floor : bool, optional
            Apply floor to impedance errors, by default None
        t_error_value : float, optional
            Error value for tipper, by default None
        t_error_type : str, optional
            Error type for tipper, by default None
        t_floor : bool, optional
            Apply floor to tipper errors, by default None

        """

        if z_error_value is not None:
            self.z_model_error.error_value = z_error_value
        if z_error_type is not None:
            self.z_model_error.error_type = z_error_type
        if z_floor is not None:
            self.z_model_error.floor = z_floor

        if t_error_value is not None:
            self.t_model_error.error_value = t_error_value
        if t_error_type is not None:
            self.t_model_error.error_type = t_error_type
        if t_floor is not None:
            self.t_model_error.floor = t_floor

        for mt_obj in self.values():
            mt_obj.compute_model_z_errors(**self.z_model_error.error_parameters)
            mt_obj.compute_model_t_errors(**self.t_model_error.error_parameters)

    def get_nearby_stations(
        self, station_key: str, radius: float, radius_units: str = "m"
    ) -> list[str]:
        """
        Find stations near a given station.

        Parameters
        ----------
        station_key : str
            Station key in format 'survey_id.station_id'
        radius : float
            Search radius
        radius_units : str, optional
            Units for radius ('m' or 'deg'), by default 'm'

        Returns
        -------
        list of str
            List of station keys within the radius

        Raises
        ------
        ValueError
            If meters requested but no UTM CRS is set

        """
        # get the local station
        local_station = self.get_station(station_key=station_key)

        sdf = self.station_locations.copy()
        if radius_units in ["m", "meters", "metres"]:
            if self.utm_crs is None:
                raise ValueError(
                    "Cannot estimate distances in meters without a UTM CRS. Set 'utm_crs' first."
                )
            sdf["radius"] = np.sqrt(
                (local_station.east - sdf.east) ** 2 + (local_station.north - sdf.north)
            )
        elif radius_units in ["deg", "degrees"]:
            sdf["radius"] = np.sqrt(
                (local_station.longitude - sdf.longitude) ** 2
                + (local_station.latitude - sdf.latitude)
            )

        return [
            f"{row.survey}.{row.station}"
            for row in sdf.loc[(sdf.radius <= radius) & (sdf.radius > 0)].itertuples()
        ]

    def estimate_spatial_static_shift(
        self,
        station_key: str,
        radius: float,
        period_min: float,
        period_max: float,
        radius_units: str = "m",
        shift_tolerance: float = 0.15,
    ) -> tuple[float, float]:
        """
        Estimate static shift using nearby stations.

        Parameters
        ----------
        station_key : str
            Station key in format 'survey_id.station_id'
        radius : float
            Search radius for nearby stations
        period_min : float
            Minimum period for resistivity calculation
        period_max : float
            Maximum period for resistivity calculation
        radius_units : str, optional
            Units for radius ('m' or 'deg'), by default 'm'
        shift_tolerance : float, optional
            Tolerance for accepting shift, by default 0.15

        Returns
        -------
        tuple of float
            Static shift factors (sx, sy) for xy and yx components

        """
        md = self.get_subset(
            self.get_nearby_stations(station_key, radius, radius_units)
        )
        if len(md) == 0:
            self.logger.warning(
                f"Could not find any nearby stations for {station_key}."
            )
            return 1.0, 1.0

        local_site = self.get_station(station_key=station_key)

        interp_periods = local_site.period[
            np.where(
                (local_site.period >= period_min) & (local_site.period <= period_max)
            )
        ]

        local_site = local_site.interpolate(interp_periods)
        md.interpolate(interp_periods)

        df = md.to_dataframe()

        sx = np.nanmedian(df.res_xy) / np.nanmedian(local_site.Z.res_xy)
        sy = np.nanmedian(df.res_yx) / np.nanmedian(local_site.Z.res_yx)

        # check to see if the estimated static shift is within given tolerance
        if 1 - shift_tolerance < sx and sx < 1 + shift_tolerance:
            sx = 1.0
        # check to see if the estimated static shift is within given tolerance
        if 1 - shift_tolerance < sy and sy < 1 + shift_tolerance:
            sy = 1.0

        return sx, sy

    def estimate_starting_rho(self) -> None:
        """
        Estimate starting resistivity from all data.

        Notes
        -----
        Creates a plot showing mean and median apparent resistivity values
        per period across all stations

        """

        entries = []
        for mt_obj in self.values():
            for period, res_det in zip(mt_obj.period, mt_obj.Z.res_det):
                entries.append({"period": period, "res_det": res_det})

        res_df = pd.DataFrame(entries)

        mean_rho = res_df.groupby("period").mean()
        median_rho = res_df.groupby("period").median()

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        (l1,) = ax.loglog(mean_rho.index, mean_rho.res_det, lw=2, color=(0.75, 0.25, 0))
        (l2,) = ax.loglog(
            median_rho.index, median_rho.res_det, lw=2, color=(0, 0.25, 0.75)
        )

        ax.loglog(
            mean_rho.index,
            np.repeat(mean_rho.res_det.mean(), mean_rho.shape[0]),
            ls="--",
            lw=2,
            color=(0.75, 0.25, 0),
        )
        ax.loglog(
            median_rho.index,
            np.repeat(median_rho.res_det.median(), median_rho.shape[0]),
            ls="--",
            lw=2,
            color=(0, 0.25, 0.75),
        )

        ax.set_xlabel("Period (s)", fontdict={"size": 12, "weight": "bold"})
        ax.set_ylabel("Resistivity (Ohm-m)", fontdict={"size": 12, "weight": "bold"})

        ax.legend(
            [l1, l2],
            [
                f"Mean = {mean_rho.res_det.mean():.1f}",
                f"Median = {median_rho.res_det.median():.1f}",
            ],
            loc="upper left",
        )
        ax.grid(which="both", ls="--", color=(0.75, 0.75, 0.75))
        ax.set_xlim((res_df.period.min(), res_df.period.max()))

        plt.show()

    def to_modem_data(
        self, data_filename: str | Path | None = None, **kwargs: Any
    ) -> Data:
        """
        Create ModEM data file (deprecated).

        Parameters
        ----------
        data_filename : str, Path, optional
            Output filename, by default None
        **kwargs : dict
            Additional ModEM parameters

        Returns
        -------
        Data
            ModEM Data object

        """
        self.logger.warning(
            "'to_modem_data' will be deprecated in future versions, use 'to_modem'"
        )
        return self.to_modem(data_filename=data_filename, **kwargs)

    def to_modem(self, data_filename: str | Path | None = None, **kwargs: Any) -> Data:
        """
        Create ModEM data file.

        Parameters
        ----------
        data_filename : str, Path, optional
            Output filename for ModEM data file, by default None
        **kwargs : dict
            Additional ModEM parameters

        Returns
        -------
        Data
            ModEM Data object

        Raises
        ------
        ValueError
            If UTM CRS is not set

        """

        modem_kwargs = dict(self.model_parameters)
        modem_kwargs.update(kwargs)

        if np.all(self.station_locations.model_east == 0):
            if self.utm_crs is None:
                raise ValueError(
                    "Need to input data UTM EPSG or CRS to compute relative "
                    "station locations"
                )
            self.compute_relative_locations()

        modem_data = Data(
            dataframe=self.to_dataframe(),
            center_point=self.center_point,
            **modem_kwargs,
        )
        modem_data.z_model_error = self.z_model_error
        modem_data.t_model_error = self.t_model_error
        if data_filename is not None:
            modem_data.write_data_file(file_name=data_filename)

        return modem_data

    def from_modem_data(
        self, data_filename: str | Path, survey: str = "data", **kwargs: Any
    ) -> None:
        """
        Read ModEM data file (deprecated).

        Parameters
        ----------
        data_filename : str or Path
            Path to ModEM data file
        survey : str, optional
            Survey name to assign, by default 'data'
        **kwargs : dict
            Additional parameters

        """

        self.logger.warning(
            "'from_modem_data' will be deprecated in future versions, use 'from_modem'"
        )

        self.from_modem(data_filename, survey=survey, **kwargs)

    def from_modem(
        self, data_filename: str | Path, survey: str = "data", **kwargs: Any
    ) -> None:
        """
        Read ModEM data file.

        Parameters
        ----------
        data_filename : str or Path
            Path to ModEM data file
        survey : str, optional
            Survey name to assign to stations, by default 'data'
        **kwargs : dict
            Additional parameters passed to ModEM Data object

        """
        modem_data = Data(**kwargs)
        mdf = modem_data.read_data_file(data_filename)

        # set survey name to something useful
        mdf.dataframe["survey"] = survey

        self.from_dataframe(mdf.dataframe)
        self.z_model_error = ModelErrors(
            mode="impedance", **modem_data.z_model_error.error_parameters
        )
        self.t_model_error = ModelErrors(
            mode="tipper", **modem_data.t_model_error.error_parameters
        )
        self.data_rotation_angle = modem_data.rotation_angle
        self._center_lat = modem_data.center_point.latitude
        self._center_lon = modem_data.center_point.longitude
        self._center_elev = modem_data.center_point.elevation
        self.utm_epsg = modem_data.center_point.utm_epsg

        self.model_parameters = dict(
            [
                (key, value)
                for key, value in modem_data.model_parameters.items()
                if "." not in key
            ]
        )

    def from_occam2d_data(
        self, data_filename: str | Path, file_type: str = "data", **kwargs: Any
    ) -> None:
        """
        Read Occam2D data file (deprecated).

        Parameters
        ----------
        data_filename : str or Path
            Path to Occam2D data file
        file_type : str, optional
            File type ('data' or 'response'), by default 'data'
        **kwargs : dict
            Additional parameters

        """
        self.logger.warning(
            "'from_occam2d_data' will be deprecated in future versions, use 'from_occam2d'"
        )

        self.from_occam2d(data_filename, file_type="data", **kwargs)

    def from_occam2d(
        self, data_filename: str | Path, file_type: str = "data", **kwargs: Any
    ) -> None:
        """
        Read Occam2D data file.

        Parameters
        ----------
        data_filename : str or Path
            Path to Occam2D data file
        file_type : str, optional
            Type of file ('data' or 'response'/'model'), by default 'data'
        **kwargs : dict
            Additional parameters passed to Occam2DData

        Examples
        --------
        Read data file and plot:

        >>> from mtpy import MTData
        >>> md = MTData()
        >>> md.from_occam2d("/path/to/data/file.dat")
        >>> plot_stations = md.plot_stations(model_locations=True)

        Read response file:

        >>> md.from_occam2d("/path/to/response/file.dat")

        """

        occam2d_data = Occam2DData(**kwargs)
        occam2d_data.read_data_file(data_filename)
        if file_type in ["data"]:
            occam2d_data.dataframe["survey"] = "data"
        elif file_type in ["response", "model"]:
            occam2d_data.dataframe["survey"] = "model"

        self.from_dataframe(occam2d_data.dataframe)

    def to_occam2d_data(
        self, data_filename: str | Path | None = None, **kwargs: Any
    ) -> Occam2DData:
        """
        Write Occam2D data file (deprecated).

        Parameters
        ----------
        data_filename : str, Path, optional
            Output filename, by default None
        **kwargs : dict
            Additional Occam2D parameters

        Returns
        -------
        Occam2DData
            Occam2D Data object

        """
        self.logger.warning(
            "'to_occam2d_data' will be deprecated in future versions, use 'to_occam2d'"
        )

        self.to_occam2d(data_filename=data_filename, **kwargs)

    def to_occam2d(
        self, data_filename: str | Path | None = None, **kwargs: Any
    ) -> Occam2DData:
        """
        Write Occam2D data file.

        Parameters
        ----------
        data_filename : str, Path, optional
            Output filename, by default None
        **kwargs : dict
            Additional Occam2D parameters

        Returns
        -------
        Occam2DData
            Occam2D Data object

        """

        occam2d_data = Occam2DData(**kwargs)
        occam2d_data.dataframe = self.to_dataframe()
        if occam2d_data.profile_origin is None:
            occam2d_data.profile_origin = (
                self.center_point.east,
                self.center_point.north,
            )

        if data_filename is not None:
            occam2d_data.write_data_file(data_filename)
        return occam2d_data

    def add_white_noise(self, value: float, inplace: bool = True) -> "MTData" | None:
        """
        Add white noise to the data.

        Parameters
        ----------
        value : float
            Noise level as percentage (converted to decimal if > 1)
        inplace : bool, optional
            If True, modifies data in place, by default True

        Returns
        -------
        MTData or None
            New MTData object if inplace=False, None otherwise

        Notes
        -----
        Useful for synthetic tests

        """
        if value > 1:
            value = value / 100.0

        if not inplace:
            mt_list = []

        for station, mt_obj in self.items():
            if inplace:
                mt_obj.add_white_noise(value)

            else:
                mt_list.append(mt_obj.add_white_noise(value, inplace=False))

        if not inplace:
            return_data = self.clone_empty()
            return_data.add_station(mt_list)
            return return_data

    def to_simpeg_2d(self, **kwargs: Any) -> Simpeg2DData:
        """
        Create Simpeg 2D data object.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters including:

            - include_elevation : bool
            - invert_te : bool
            - invert_tm : bool

        Returns
        -------
        Simpeg2DData
            Simpeg 2D data object

        Notes
        -----
        All information is derived from the dataframe. The user should
        create the profile, interpolate, and estimate model errors from
        the MTData object first

        """

        return Simpeg2DData(self.to_dataframe(impedance_units="ohm"), **kwargs)

    def to_simpeg_3d(self, **kwargs: Any) -> Simpeg3DData:
        """
        Create Simpeg 3D data object.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters including:

            - include_elevation : bool
            - geographic_coordinates : bool
            - invert_z_xx : bool
            - invert_z_xy : bool
            - invert_z_yx : bool
            - invert_z_yy : bool
            - invert_t_zx : bool
            - invert_t_zy : bool
            - invert_types : list

        Returns
        -------
        Simpeg3DData
            Simpeg 3D data object

        Notes
        -----
        All information is derived from the dataframe. The user should
        interpolate and estimate model errors from the MTData object first

        """

        return Simpeg3DData(self.to_dataframe(impedance_units="ohm"), **kwargs)

    def plot_mt_response(
        self,
        station_key: str | list[str] | None = None,
        station_id: str | list[str] | None = None,
        survey_id: str | list[str] | None = None,
        **kwargs: Any,
    ) -> PlotMultipleResponses | Any:
        """
        Plot MT response for one or more stations.

        Parameters
        ----------
        station_key : str, list of str, optional
            Station key(s), by default None
        station_id : str, list of str, optional
            Station ID(s), by default None
        survey_id : str, list of str, optional
            Survey ID(s), by default None
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotMultipleResponses or plot object
            Plot object for the MT response(s)

        """

        if isinstance(station_key, (list, tuple)):
            mt_data = MTData()
            for sk in station_key:
                mt_data.add_station(
                    self.get_station(station_key=sk),
                    compute_relative_location=False,
                )
            return PlotMultipleResponses(mt_data, **kwargs)

        elif isinstance(station_id, (list, tuple)):
            mt_data = MTData()
            if isinstance(survey_id, (list, tuple)):
                if len(survey_id) != len(station_key):
                    raise ValueError("Number of survey must match number of stations")
            elif isinstance(survey_id, (str, type(None))):
                survey_id = [survey_id] * len(station_id)
            for survey, station in zip(survey_id, station_id):
                mt_data.add_station(
                    self.get_station(station_id=station, survey_id=survey),
                    compute_relative_location=False,
                )
            return PlotMultipleResponses(mt_data, **kwargs)

        else:
            mt_object = self.get_station(
                station_id=station_id,
                survey_id=survey_id,
                station_key=station_key,
            )
            return mt_object.plot_mt_response(**kwargs)

    def plot_stations(
        self,
        map_epsg: int = 4326,
        bounding_box: tuple[float, float, float, float] | None = None,
        model_locations: bool = False,
        **kwargs: Any,
    ) -> PlotStations:
        """
        Plot station locations on a map.

        Parameters
        ----------
        map_epsg : int, optional
            EPSG code for map projection, by default 4326
        bounding_box : tuple of float, optional
            Bounding box, by default None
        model_locations : bool, optional
            Use model coordinates, by default False
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotStations
            Station plot object

        """

        gdf = self.to_geo_df(model_locations=model_locations)
        if model_locations:
            kwargs["plot_cx"] = False
        return PlotStations(gdf, **kwargs)

    def plot_strike(self, **kwargs: Any) -> PlotStrike:
        """
        Plot strike angle.

        Parameters
        ----------
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotStrike
            Strike plot object

        """

        return PlotStrike(self, **kwargs)

    def plot_phase_tensor(
        self,
        station_key: str | None = None,
        station_id: str | None = None,
        survey_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot phase tensor elements for a station.

        Parameters
        ----------
        station_key : str, optional
            Station key, by default None
        station_id : str, optional
            Station ID, by default None
        survey_id : str, optional
            Survey ID, by default None
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        plot object
            Phase tensor plot object

        """

        mt_object = self.get_station(
            station_id=station_id,
            survey_id=survey_id,
            station_key=station_key,
        )
        return mt_object.plot_phase_tensor(**kwargs)

    def plot_phase_tensor_map(self, **kwargs: Any) -> PlotPhaseTensorMaps:
        """
        Plot phase tensor maps.

        Parameters
        ----------
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotPhaseTensorMaps
            Phase tensor map plot object

        """

        return PlotPhaseTensorMaps(mt_data=self, **kwargs)

    def plot_tipper_map(self, **kwargs: Any) -> PlotPhaseTensorMaps:
        """
        Plot tipper (induction vector) maps.

        Parameters
        ----------
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotPhaseTensorMaps
            Tipper map plot object

        """
        kwargs["plot_pt"] = False
        kwargs["plot_tipper"] = "yri"
        return PlotPhaseTensorMaps(mt_data=self, **kwargs)

    def plot_phase_tensor_pseudosection(
        self, mt_data: "MTData" | None = None, **kwargs: Any
    ) -> PlotPhaseTensorPseudoSection:
        """
        Plot phase tensor pseudosection.

        Parameters
        ----------
        mt_data : MTData, optional
            MT data object, by default None (uses self)
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotPhaseTensorPseudoSection
            Pseudosection plot object

        """

        return PlotPhaseTensorPseudoSection(mt_data=self, **kwargs)

    def plot_penetration_depth_1d(
        self,
        station_key: str | None = None,
        station_id: str | None = None,
        survey_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot 1D penetration depth.

        Parameters
        ----------
        station_key : str, optional
            Station key, by default None
        station_id : str, optional
            Station ID, by default None
        survey_id : str, optional
            Survey ID, by default None
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        plot object
            Penetration depth plot object

        Notes
        -----
        Based on Niblett-Bostick transformation

        """

        mt_object = self.get_station(
            station_id=station_id,
            survey_id=survey_id,
            station_key=station_key,
        )

        return mt_object.plot_depth_of_penetration(**kwargs)

    def plot_penetration_depth_map(self, **kwargs: Any) -> PlotPenetrationDepthMap:
        """
        Plot penetration depth in map view.

        Parameters
        ----------
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotPenetrationDepthMap
            Penetration depth map plot object

        """

        return PlotPenetrationDepthMap(self, **kwargs)

    def plot_resistivity_phase_maps(self, **kwargs: Any) -> PlotResPhaseMaps:
        """
        Plot apparent resistivity and/or phase maps.

        Parameters
        ----------
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotResPhaseMaps
            Resistivity/phase map plot object

        """

        return PlotResPhaseMaps(self, **kwargs)

    def plot_resistivity_phase_pseudosections(
        self, **kwargs: Any
    ) -> PlotResPhasePseudoSection:
        """
        Plot resistivity and phase pseudosections.

        Parameters
        ----------
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotResPhasePseudoSection
            Pseudosection plot object

        """

        return PlotResPhasePseudoSection(self, **kwargs)

    def plot_residual_phase_tensor_maps(
        self, survey_01: str, survey_02: str, **kwargs: Any
    ) -> PlotResidualPTMaps:
        """
        Plot residual phase tensor maps.

        Parameters
        ----------
        survey_01 : str
            First survey ID
        survey_02 : str
            Second survey ID
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        PlotResidualPTMaps
            Residual phase tensor map plot object

        """

        survey_data_01 = self.get_survey(survey_01)
        survey_data_02 = self.get_survey(survey_02)

        return PlotResidualPTMaps(survey_data_01, survey_data_02, **kwargs)

    def to_shp_pt_tipper(
        self,
        save_dir: str | Path,
        output_crs: Any | None = None,
        utm: bool = False,
        pt: bool = True,
        tipper: bool = True,
        periods: np.ndarray | None = None,
        period_tol: float | None = None,
        ellipse_size: float | None = None,
        arrow_size: float | None = None,
    ) -> dict[str, list[str]]:
        """
        Write phase tensor and tipper shape files.

        Parameters
        ----------
        save_dir : str or Path
            Directory to save shape files
        output_crs : CRS, int, str, optional
            Output coordinate reference system, by default None
        utm : bool, optional
            Use UTM coordinates, by default False
        pt : bool, optional
            Create phase tensor shapefiles, by default True
        tipper : bool, optional
            Create tipper shapefiles, by default True
        periods : np.ndarray, optional
            Periods to plot, by default None (all periods)
        period_tol : float, optional
            Tolerance for period matching, by default None
        ellipse_size : float, optional
            Size of phase tensor ellipses, by default None (auto)
        arrow_size : float, optional
            Size of tipper arrows, by default None (auto)

        Returns
        -------
        dict
            Dictionary of shapefile paths

        Notes
        -----
        If you have mixed periods, interpolate onto common periods first

        """

        sc = ShapefileCreator(self.to_mt_dataframe(), output_crs, save_dir=save_dir)
        sc.utm = utm
        if ellipse_size is None and pt:
            sc.ellipse_size = sc.estimate_ellipse_size()
        else:
            sc.ellipse_size = ellipse_size
        if arrow_size is None and tipper:
            sc.arrow_size = sc.estimate_arrow_size()
        else:
            sc.arrow_size = arrow_size

        return sc.make_shp_files(
            pt=pt, tipper=tipper, periods=periods, period_tol=period_tol
        )
