"""
==================
ModEM
==================

# Generate files for ModEM

# revised by JP 2017
# revised by AK 2017 to bring across functionality from ak branch

"""

from copy import deepcopy

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyevtk.hl import pointsToVTK
from pyproj import CRS
from scipy import stats

from mtpy.core.mt_location import MTLocation


# =============================================================================


class MTStations:
    """
    Object to deal with station location and geographic projection.

    Geographic projections are done using pyproj.CRS objects.

    Parameters
    ----------
    utm_epsg : int or str
        UTM EPSG code for projection
    datum_epsg : int or str, optional
        Datum EPSG code, by default None
    **kwargs : dict
        Additional keyword arguments

    Attributes
    ----------
    station_locations : geopandas.GeoDataFrame
        GeoDataFrame containing station metadata and coordinates
    utm_crs : pyproj.CRS
        UTM coordinate reference system
    datum_crs : pyproj.CRS
        Datum coordinate reference system
    rotation_angle : float
        Rotation angle in degrees
    shift_east : float
        Shift in east direction
    shift_north : float
        Shift in north direction

    Notes
    -----
    Takes in a list of :class:`mtpy.core.mt.MT` objects which inherit
    :class:`mtpy.core.mt_location.MTLocation` objects, which deal with
    transformation of point data using pyproj.

    """

    def __init__(
        self,
        utm_epsg: int | str | None,
        datum_epsg: int | str | None = None,
        station_locations: pd.DataFrame | gpd.GeoDataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        self.logger = logger

        self.dtype = dict(
            [
                ("survey", "U50"),
                ("station", "U50"),
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
            ]
        )

        self._datum_crs = CRS.from_epsg(4326)
        self._utm_crs = None
        self._center_lat = None
        self._center_lon = None
        self._center_elev = 0.0
        self.shift_east = 0
        self.shift_north = 0
        self.rotation_angle = 0.0
        self._station_locations = None
        self.utm_epsg = utm_epsg
        self.datum_epsg = datum_epsg

        mt_list = kwargs.pop("mt_list", None)

        for key in list(kwargs.keys()):
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        if station_locations is not None:
            if mt_list is not None:
                raise ValueError(
                    "Provide either mt_list or station_locations, not both"
                )
            self._set_station_locations(station_locations)
        elif mt_list is not None:
            self._set_station_locations(self._mt_list_to_dataframe(mt_list))
            if len(self._station_locations) > 0:
                self.compute_relative_locations()
                self.station_locations

    def _mt_list_to_dataframe(self, mt_list: list[Any]) -> pd.DataFrame:
        """Convert a sequence of MT-like objects into a station dataframe."""

        if len(mt_list) == 0:
            return pd.DataFrame(columns=list(self.dtype.keys()))

        entries = []
        for mt_obj in mt_list:
            entries.append(
                {
                    "survey": str(getattr(mt_obj, "survey", "")),
                    "station": str(getattr(mt_obj, "station", "")),
                    "latitude": float(getattr(mt_obj, "latitude", 0.0) or 0.0),
                    "longitude": float(getattr(mt_obj, "longitude", 0.0) or 0.0),
                    "elevation": float(getattr(mt_obj, "elevation", 0.0) or 0.0),
                    "datum_epsg": str(getattr(mt_obj, "datum_epsg", "") or ""),
                    "east": float(getattr(mt_obj, "east", 0.0) or 0.0),
                    "north": float(getattr(mt_obj, "north", 0.0) or 0.0),
                    "utm_epsg": str(getattr(mt_obj, "utm_epsg", "") or ""),
                    "model_east": float(getattr(mt_obj, "model_east", 0.0) or 0.0),
                    "model_north": float(getattr(mt_obj, "model_north", 0.0) or 0.0),
                    "model_elevation": float(
                        getattr(mt_obj, "model_elevation", 0.0) or 0.0
                    ),
                    "profile_offset": float(
                        getattr(mt_obj, "profile_offset", 0.0) or 0.0
                    ),
                }
            )

        return pd.DataFrame(entries, columns=list(self.dtype.keys()))

    def _set_station_locations(self, station_locations: pd.DataFrame) -> None:
        """Store a normalized station-locations dataframe for dataframe-backed use."""
        if station_locations is None:
            self._station_locations = None
            return
        if not isinstance(station_locations, pd.DataFrame):
            raise TypeError("station_locations must be a pandas.DataFrame")

        station_df = station_locations.copy()
        expected_columns = list(self.dtype.keys())
        for column in expected_columns:
            if column not in station_df.columns:
                if column in ["survey", "station", "datum_epsg", "utm_epsg"]:
                    station_df[column] = ""
                else:
                    station_df[column] = 0.0

        station_df = station_df.loc[:, expected_columns]
        station_df = station_df.reset_index(drop=True)

        text_columns = ["survey", "station", "datum_epsg", "utm_epsg"]
        for column in text_columns:
            station_df[column] = station_df[column].fillna("").astype(str)

        numeric_columns = [
            "latitude",
            "longitude",
            "elevation",
            "east",
            "north",
            "model_east",
            "model_north",
            "model_elevation",
            "profile_offset",
        ]
        for column in numeric_columns:
            station_df[column] = pd.to_numeric(
                station_df[column], errors="coerce"
            ).fillna(0.0)

        validated_datum = self._validate_epsg(station_df, key="datum")
        if validated_datum is not None:
            self.datum_epsg = validated_datum
            station_df.loc[:, "datum_epsg"] = str(validated_datum)

        validated_utm = self._validate_epsg(station_df, key="utm")
        if validated_utm is not None:
            self.utm_epsg = validated_utm
            station_df.loc[:, "utm_epsg"] = str(validated_utm)

        self._station_locations = self._to_geodataframe(station_df)

    def _to_geodataframe(self, station_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert a station table into a GeoDataFrame with a consistent CRS."""
        if isinstance(station_df, gpd.GeoDataFrame):
            gdf = station_df.copy()
        else:
            gdf = gpd.GeoDataFrame(
                station_df.copy(),
                geometry=gpd.points_from_xy(
                    station_df.longitude,
                    station_df.latitude,
                ),
                crs=self.datum_crs,
            )

        if gdf.crs is None and self.datum_crs is not None:
            gdf = gdf.set_crs(self.datum_crs)
        elif gdf.crs is not None and self.datum_crs is not None:
            gdf = gdf.to_crs(self.datum_crs)

        return gdf

    def __str__(self) -> str:
        """
        String representation of MTStations.

        Returns
        -------
        str
            Formatted string with station locations and center point

        """
        if self.station_locations is None:
            return ""
        elif len(self) == 0:
            return ""

        fmt_dict = dict(
            [
                ("survey", "<8"),
                ("station", "<8"),
                ("latitude", "<10.4f"),
                ("longitude", "<10.4f"),
                ("elevation", "<8.2f"),
                ("model_east", "<13.2f"),
                ("model_north", "<13.2f"),
                ("model_elevation", "<13.2f"),
                ("profile_offset", "<13.2f"),
                ("east", "<12.2f"),
                ("north", "<12.2f"),
                ("utm_epsg", "<6"),
                ("datum_epsg", "<6"),
            ]
        )
        lines = ["  ".join([n for n in self.station_locations.columns])]
        lines.append("-" * 72)
        for row in self.station_locations.itertuples():
            l = []
            for key in self.station_locations.columns:
                if key not in fmt_dict:
                    l.append(f"{getattr(row, key)}")
                else:
                    l.append(f"{getattr(row, key):{fmt_dict[key]}}")
            lines.append("".join(l))

        lines.append("\nModel Center:")
        l = []
        for n in [
            "latitude",
            "longitude",
            "elevation",
            "east",
            "north",
            "utm_epsg",
        ]:
            l.append(f"{getattr(self.center_point, n):{fmt_dict[n]}}")
        lines.append("".join(l))

        lines.append("\nMean Values:")
        l = []
        for n in ["latitude", "longitude", "elevation", "east", "north"]:
            l.append(f"{self.station_locations[n].mean():{fmt_dict[n]}}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Representation of MTStations.

        Returns
        -------
        str
            String representation

        """
        return self.__str__()

    def __eq__(self, other: "MTStations") -> bool:
        """
        Check equality with another MTStations object.

        Parameters
        ----------
        other : MTStations
            Another MTStations object to compare

        Returns
        -------
        bool
            True if equal, False otherwise

        Raises
        ------
        TypeError
            If other is not an MTStations object

        """
        if not isinstance(other, MTStations):
            raise TypeError(f"Can not compare {type(other)} to MTStations")

        if not self.station_locations.equals(other.station_locations):
            return False

        if not self.center_point == other.center_point:
            return False

        return True

    def __len__(self) -> int:
        """
        Return number of stations.

        Returns
        -------
        int
            Number of MT stations in the list

        """
        if self._station_locations is not None:
            return len(self._station_locations)
        return 0

    def copy(self) -> "MTStations":
        """
        Create a deep copy of the MTStations object.

        Returns
        -------
        MTStations
            Deep copy of MTStation object

        Notes
        -----
        At the moment this is very slow because it is making a lot
        of deep copies. Use sparingly.

        """

        if self._station_locations is not None:
            copied = MTStations(None, station_locations=self._station_locations)
        else:
            copied = MTStations(None, station_locations=None)
        for key in [
            "utm_crs",
            "datum_crs",
            "_center_lat",
            "_center_lon",
            "_center_elev",
            "shift_east",
            "shift_north",
            "rotation_angle",
        ]:
            setattr(copied, key, deepcopy(getattr(self, key)))

        return copied

    @property
    def model_epsg(self) -> int | None:
        """
        Model EPSG code.

        Returns
        -------
        int or None
            Model EPSG number from the model_crs object

        """
        return self.utm_epsg

    @model_epsg.setter
    def model_epsg(self, value: int | str) -> None:
        """
        Set model EPSG code.

        Parameters
        ----------
        value : int or str
            EPSG number for the model

        """
        self.utm_epsg = value

    @property
    def utm_crs(self) -> CRS | None:
        """
        UTM coordinate reference system.

        Returns
        -------
        pyproj.CRS or None
            UTM CRS object

        """
        if self._utm_crs is not None:
            return self._utm_crs

    @property
    def utm_name(self) -> str | None:
        """
        UTM coordinate reference system name.

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
        UTM EPSG code.

        Returns
        -------
        int or None
            UTM EPSG number

        """
        if self._utm_crs is not None:
            return self._utm_crs.to_epsg()

    @utm_epsg.setter
    def utm_epsg(self, value: int | str) -> None:
        """
        Set UTM EPSG code.

        Parameters
        ----------
        value : int or str
            EPSG number

        """
        self.utm_crs = value

    @property
    def utm_zone(self) -> str | None:
        """
        UTM zone.

        Returns
        -------
        str or None
            UTM Zone number

        """
        if self._utm_crs is not None:
            return self._utm_crs.utm_zone

    @utm_crs.setter
    def utm_crs(self, value: CRS | int | str) -> None:
        """
        Set UTM coordinate reference system.

        Parameters
        ----------
        value : pyproj.CRS, int, or str
            UTM CRS object, EPSG number, or proj4 string

        """
        if value in [None, "None", "none", "null"]:
            return

        self._utm_crs = CRS.from_user_input(value)
        if self._station_locations is not None:
            self._station_locations.loc[:, "utm_epsg"] = str(self.utm_epsg)

    @property
    def datum_crs(self) -> CRS | None:
        """
        Datum coordinate reference system.

        Returns
        -------
        pyproj.CRS or None
            Datum CRS object

        """
        if self._datum_crs is not None:
            return self._datum_crs

    @property
    def datum_name(self) -> str | None:
        """
        Datum coordinate reference system name.

        Returns
        -------
        str or None
            Datum well known name

        """
        if self._datum_crs is not None:
            return self._datum_crs.name

    @property
    def datum_epsg(self) -> int | None:
        """
        Datum EPSG code.

        Returns
        -------
        int or None
            Datum EPSG number

        """
        if self._datum_crs is not None:
            return self._datum_crs.to_epsg()

    @datum_epsg.setter
    def datum_epsg(self, value: int | str) -> None:
        """
        Set Datum EPSG code.

        Parameters
        ----------
        value : int or str
            Datum EPSG number

        """
        self.datum_crs = value

    @datum_crs.setter
    def datum_crs(self, value: CRS | int | str) -> None:
        """
        Set the datum coordinate reference system.

        Parameters
        ----------
        value : pyproj.CRS, int, or str
            Datum CRS object, EPSG number, or proj4 string

        """
        if value in [None, "None", "none", "null"]:
            return

        self._datum_crs = CRS.from_user_input(value)
        if self._station_locations is not None:
            gdf = self._station_locations
            if gdf.crs is None:
                gdf = gdf.set_crs(self._datum_crs)
            else:
                gdf = gdf.to_crs(self._datum_crs)
            gdf.loc[:, "latitude"] = gdf.geometry.y
            gdf.loc[:, "longitude"] = gdf.geometry.x
            gdf.loc[:, "datum_epsg"] = str(self.datum_epsg)
            self._station_locations = gdf

    @property
    def station_locations(self) -> pd.DataFrame | None:
        """
        Station locations dataframe.

        Returns
        -------
        pandas.DataFrame or None
            Dataframe of station location information

        """

        if self._station_locations is not None:
            return self._station_locations.copy()
        return None

    def _validate_epsg(self, df: pd.DataFrame, key: str = "datum") -> int | None:
        """
        Validate and consolidate EPSG numbers.

        Make sure that there is only one EPSG number for each of the Datum
        and UTM. If there are more than one, use the median value or the
        first in a unique list of EPSG numbers.

        Parameters
        ----------
        df : pandas.DataFrame
            Station_location dataframe
        key : str, optional
            Type of EPSG to validate ("datum" or "utm"), by default "datum"

        Returns
        -------
        int or None
            EPSG number

        """

        key = f"{key}_epsg"
        values = df[key].astype(str).str.strip()
        values = values[~values.isin(["", "None", "none", "NONE", "null"])]
        if values.empty:
            return None

        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        if numeric_values.empty:
            return None

        if numeric_values.nunique() > 1:
            epsg = numeric_values.median()
            self.logger.warning(
                f"Found more than one {key} number, using median EPSG number {epsg}"
            )
            return int(epsg)

        if getattr(self, key) is None:
            return int(numeric_values.iloc[0])

        return None

    def compute_relative_locations(self) -> None:
        """
        Calculate model station locations relative to the center point in meters.

        Uses `mtpy.core.MTLocation.compute_model_location` to calculate the
        relative distance.

        Notes
        -----
        Computes in place.

        """

        station_df = self.station_locations
        if station_df is None or station_df.empty:
            return

        center = self.center_point
        station_df.loc[:, "model_east"] = (
            station_df.east.to_numpy(dtype=float) - center.east
        )
        station_df.loc[:, "model_north"] = (
            station_df.north.to_numpy(dtype=float) - center.north
        )
        station_df.loc[:, "model_elevation"] = (
            station_df.elevation.to_numpy(dtype=float) - center.elevation
        )
        self._station_locations = self._to_geodataframe(station_df)

    # make center point a get property, can't set it.
    @property
    def center_point(self) -> MTLocation:
        """
        Calculate the center point from the given station locations.

        If _center attributes are set, that is returned as the center point.

        Otherwise, looks for non-zero locations in E-N first, then Lat/Lon and
        estimates the center point as (max - min) / 2...

        Returns
        -------
        mtpy.core.MTLocation
            Center point location object

        """

        center_location = MTLocation()
        if self._center_lat is not None and self._center_lon is not None:
            self.logger.debug("assigning center from user set values")
            center_location.latitude = self._center_lat
            center_location.longitude = self._center_lon
            center_location.elevation = self._center_elev
            center_location.utm_epsg = self.utm_epsg
            center_location.model_east = center_location.east
            center_location.model_north = center_location.north
            center_location.model_elevation = self._center_elev

            return center_location

        else:
            center_location.datum_epsg = self.datum_epsg
            center_location.utm_epsg = self.utm_epsg
            st_df = self.station_locations.copy()

            st_en = st_df.loc[(st_df.east != 0) & (st_df.north != 0)]
            if st_en.empty:
                st_ll = st_df.loc[(st_df.latitude != 0) & (st_df.longitude != 0)]
                if st_ll.empty:
                    raise ValueError("Station locations are all 0 cannot find center.")

                else:
                    self.logger.debug("locating center from latitude and longitude")
                    center_location.latitude = (
                        st_ll.latitude.max() + st_ll.latitude.min()
                    ) / 2
                    center_location.longitude = (
                        st_ll.longitude.max() + st_ll.longitude.min()
                    ) / 2

            else:
                self.logger.debug("locating center from UTM grid")
                center_location.east = (st_en.east.max() + st_en.east.min()) / 2
                center_location.north = (st_en.north.max() + st_en.north.min()) / 2

            center_location.model_east = center_location.east
            center_location.model_north = center_location.north
            center_location.model_elevation = self._center_elev

        return center_location

    def rotate_stations(self, rotation_angle: float) -> None:
        """
        Rotate stations model positions only.

        Assumes N is 0 and east is 90.

        Parameters
        ----------
        rotation_angle : float
            Rotation angle in degrees assuming N=0, E=90. Positive clockwise

        Notes
        -----
        Computes in place and rotates according to already set rotation angle.
        Therefore if the station locations have already been rotated, the
        function will rotate the already rotated stations. For example, if you
        rotate the stations 15 degrees, then again by 20 degrees, the resulting
        station locations will be 35 degrees rotated from the original locations.

        """

        cos_ang = np.cos(np.deg2rad(rotation_angle))
        sin_ang = np.sin(np.deg2rad(rotation_angle))
        rot_matrix = np.array([[cos_ang, sin_ang], [-sin_ang, cos_ang]])

        station_df = self.station_locations
        if station_df is None or station_df.empty:
            return

        coords = station_df.loc[:, ["model_east", "model_north"]].to_numpy(dtype=float)
        rotated = coords @ np.array([[cos_ang, -sin_ang], [sin_ang, cos_ang]])
        station_df.loc[:, "model_east"] = rotated[:, 0]
        station_df.loc[:, "model_north"] = rotated[:, 1]
        self._station_locations = self._to_geodataframe(station_df)

        self.rotation_angle += rotation_angle

        self.logger.info(
            f"Rotated stations by {rotation_angle:.1f} deg clockwise from N. "
            f"Total rotation = {self.rotation_angle:.1f} deg."
        )

    def center_stations(self, model_obj: Any) -> None:
        """
        Center station locations to the middle of cells.

        Useful for topography as it reduces edge effects of stations close to
        cell edges. Recalculates rel_east, rel_north to center of model cell.

        Parameters
        ----------
        model_obj : mtpy.modeling.modem.Model or mtpy.modeling.Structured
            Model object

        """

        station_df = self.station_locations
        if station_df is None or station_df.empty:
            return

        e_index = (
            np.searchsorted(
                model_obj.grid_east,
                station_df.model_east.to_numpy(dtype=float),
                side="right",
            )
            - 1
        )
        n_index = (
            np.searchsorted(
                model_obj.grid_north,
                station_df.model_north.to_numpy(dtype=float),
                side="right",
            )
            - 1
        )
        e_index = np.clip(e_index, 0, model_obj.grid_east.size - 2)
        n_index = np.clip(n_index, 0, model_obj.grid_north.size - 2)

        station_df.loc[:, "model_east"] = (
            model_obj.grid_east[e_index] + model_obj.grid_east[e_index + 1]
        ) / 2
        station_df.loc[:, "model_north"] = (
            model_obj.grid_north[n_index] + model_obj.grid_north[n_index + 1]
        ) / 2
        self._station_locations = self._to_geodataframe(station_df)

    def project_stations_on_topography(
        self,
        model_object: Any,
        air_resistivity: float = 1e12,
        sea_resistivity: float = 0.3,
        ocean_bottom: bool = False,
    ) -> None:
        """
        Project stations on topography of a given model.

        Parameters
        ----------
        model_object : mtpy.modeling.modem.Model
            Model object
        air_resistivity : float, optional
            Resistivity value of air cells in the model, by default 1e12
        sea_resistivity : float, optional
            Resistivity of sea water, by default 0.3
        ocean_bottom : bool, optional
            If True places stations at bottom of sea cells, by default False

        """

        station_df = self.station_locations
        if station_df is None or station_df.empty:
            return

        model_elevations = station_df.model_elevation.to_numpy(dtype=float).copy()

        # find index of each station on grid
        for ii, (sx, sy) in enumerate(
            zip(
                station_df.model_east.to_numpy(dtype=float),
                station_df.model_north.to_numpy(dtype=float),
            )
        ):
            # indices of stations on model grid
            sxi = np.where(
                (sx <= model_object.grid_east[1:]) & (sx > model_object.grid_east[:-1])
            )[0][0]

            syi = np.where(
                (sy <= model_object.grid_north[1:])
                & (sy > model_object.grid_north[:-1])
            )[0][0]

            # first, check if there are any air cells
            if np.any(model_object.res_model[syi, sxi] > 0.95 * air_resistivity):
                szi = np.amin(
                    np.where(
                        (model_object.res_model[syi, sxi] < 0.95 * air_resistivity)
                    )[0]
                )
            # otherwise place station at the top of the model
            else:
                szi = 0

            # JP: estimate ocean bottom stations if requested
            if ocean_bottom:
                if np.any(model_object.res_model[syi, sxi] <= sea_resistivity):
                    szi = np.amax(
                        np.where((model_object.res_model[syi, sxi] <= sea_resistivity))[
                            0
                        ]
                    )

            # get relevant grid point elevation
            topoval = model_object.grid_z[szi]

            # update elevation in station locations and data array, +1 m as
            # data elevation needs to be below the topography (as advised by Naser)
            model_elevations[ii] = topoval + 0.001

        station_df.loc[:, "model_elevation"] = model_elevations
        self._station_locations = self._to_geodataframe(station_df)

        # BM: After applying topography, center point of grid becomes
        #  highest point of surface model.
        self._center_elev = model_object.grid_z[0]

    def to_geopd(self) -> gpd.GeoDataFrame:
        """
        Create a geopandas dataframe.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with points from latitude and longitude

        """

        station_df = self.station_locations
        if station_df is None:
            return gpd.GeoDataFrame()
        return self._to_geodataframe(station_df)

    def to_shp(self, shp_fn: str | Path) -> str | Path:
        """
        Write a shapefile of the station locations.

        Uses geopandas which only takes in epsg numbers.

        Parameters
        ----------
        shp_fn : str or Path
            Full path to new shapefile

        Returns
        -------
        str or Path
            Path to the created shapefile

        """
        sdf = self.to_geopd()

        sdf.to_file(shp_fn)
        return shp_fn

    def to_csv(self, csv_fn: str | Path, geometry: bool = False) -> None:
        """
        Write a CSV file of the station locations.

        Parameters
        ----------
        csv_fn : str or Path
            Full path to new CSV file
        geometry : bool, optional
            Whether to include geometry column, by default False

        """
        sdf = self.to_geopd()
        use_columns = list(sdf.columns)
        if not geometry:
            use_columns.remove("geometry")
        sdf.to_csv(csv_fn, index=False, columns=use_columns)

    def to_vtk(
        self,
        vtk_fn: str | Path | None = None,
        vtk_save_path: str | Path | None = None,
        vtk_fn_basename: str = "ModEM_stations",
        geographic: bool = False,
        shift_east: float = 0,
        shift_north: float = 0,
        shift_elev: float = 0,
        units: str = "km",
        coordinate_system: str = "nez+",
    ) -> Path:
        """
        Write a VTK file for plotting in 3D like Paraview.

        Parameters
        ----------
        vtk_fn : str or Path, optional
            Full path to VTK file to be written, by default None
        vtk_save_path : str or Path, optional
            Directory to save VTK file to, by default None
        vtk_fn_basename : str, optional
            Filename basename of VTK file, note that .vtr extension is added,
            by default "ModEM_stations"
        geographic : bool, optional
            Use geographic coordinates, by default False
        shift_east : float, optional
            Shift in east direction, by default 0
        shift_north : float, optional
            Shift in north direction, by default 0
        shift_elev : float, optional
            Shift in elevation, by default 0
        units : str, optional
            Units for coordinates ("km", "m", or "ft"), by default "km"
        coordinate_system : str, optional
            Coordinate system ("nez+" or "enz-"), by default "nez+"

        Returns
        -------
        Path
            Full path to VTK file

        Raises
        ------
        ValueError
            If vtk_save_path is None when vtk_fn is None

        """

        if isinstance(units, str):
            if units.lower() == "km":
                scale = 1.0 / 1000.00
            elif units.lower() == "m":
                scale = 1.0
            elif units.lower() == "ft":
                scale = 3.2808
        elif isinstance(units, (int, float)):
            scale = units

        if vtk_fn is None:
            if vtk_save_path is None:
                raise ValueError("Need to input vtk_save_path")
            vtk_fn = Path(vtk_save_path, vtk_fn_basename)
        else:
            vtk_fn = Path(vtk_fn)

        if vtk_fn.suffix != "":
            vtk_fn = vtk_fn.parent.joinpath(vtk_fn.stem)

        sdf = self.station_locations.copy()

        if geographic:
            if "+" in coordinate_system:
                vtk_y = (sdf.north + shift_north) * scale
                vtk_x = (sdf.east + shift_east) * scale
                vtk_z = -1 * (sdf.elevation + shift_elev) * scale
                extra = -1 * (sdf.elevation + shift_elev) * scale
            elif "-" in coordinate_system:
                vtk_y = (sdf.north + shift_north) * scale
                vtk_x = (sdf.east + shift_east) * scale
                vtk_z = (sdf.elevation + shift_elev) * scale
                extra = (sdf.elevation + shift_elev) * scale
        else:
            if coordinate_system == "nez+":
                vtk_y = (sdf.model_north + shift_north) * scale
                vtk_x = (sdf.model_east + shift_east) * scale
                vtk_z = (sdf.model_elevation + shift_elev) * scale
                extra = (sdf.model_elevation + shift_elev) * scale
            elif coordinate_system == "enz-":
                vtk_x = (sdf.model_north + shift_north) * scale
                vtk_y = (sdf.model_east + shift_east) * scale
                vtk_z = -1 * (sdf.model_elevation + shift_elev) * scale
                extra = -1 * (sdf.model_elevation + shift_elev) * scale

        # write file
        pointsToVTK(
            vtk_fn.as_posix(),
            vtk_x.to_numpy(),
            vtk_y.to_numpy(),
            vtk_z.to_numpy(),
            data={"elevation": extra.to_numpy()},
        )

        self.logger.info(f"Wrote station VTK file to {vtk_fn}.vtu")
        return vtk_fn

    def generate_profile(
        self, units: str = "deg"
    ) -> tuple[float, float, float, float, dict[str, float]]:
        """
        Estimate a profile from the data.

        Parameters
        ----------
        units : str, optional
            Units for coordinates ("deg" or "m"), by default "deg"

        Returns
        -------
        x1 : float
            First x coordinate of profile line
        y1 : float
            First y coordinate of profile line
        x2 : float
            Second x coordinate of profile line
        y2 : float
            Second y coordinate of profile line
        profile_line : dict
            Dictionary with "slope" and "intercept" keys defining the profile line

        Raises
        ------
        ValueError
            If units is "m" but no UTM CRS is set

        """

        if units == "deg":
            x = self.station_locations.longitude
            y = self.station_locations.latitude

        elif units == "m":
            if self.utm_crs is not None:
                x = self.station_locations.east
                y = self.station_locations.north
            else:
                raise ValueError("Must input a UTM CRS or EPSG")

        # check regression for 2 profile orientations:
        # horizontal (N=N(E)) or vertical(E=E(N))
        # use the one with the lower standard deviation
        profile1 = stats.linregress(x, y)
        profile2 = stats.linregress(y, x)
        # if the profile is rather E=E(N), the parameters have to converted
        # into N=N(E) form:
        if profile2.stderr < profile1.stderr:
            profile_line = {
                "slope": 1.0 / profile2.slope,
                "intercept": -profile2.intercept / profile2.slope,
            }
        else:
            profile_line = {
                "slope": profile1.slope,
                "intercept": profile1.intercept,
            }

        # if the profile is closer to E-W, use minimum x to get profile ends,
        # otherwise use minimum y
        if -1 <= profile_line["slope"] <= 1:
            sx = np.array([x.min(), x.max()])
            sy = np.array([y[x.idxmin()], y[x.idxmax()]])
        else:
            sy = np.array([y.min(), y.max()])
            sx = np.array([x[y.idxmin()], x[y.idxmax()]])

        # get line through point perpendicular to profile
        m2 = -1.0 / profile_line["slope"]
        # two intercepts associated with each end point
        c2 = sy - m2 * sx
        # get point where the lines intercept the profile line
        x1, x2 = (c2 - profile_line["intercept"]) / (profile_line["slope"] - m2)
        # compute y points
        y1, y2 = profile_line["slope"] * np.array([x1, x2]) + profile_line["intercept"]

        # # to get x values of end points, need to project first station onto the line
        # # get min and max x values
        # sx = np.array([x.min(),x.max()])
        # # get line through x1 perpendicular to profile
        # m2 = -1./profile_line['slope']
        # # two intercepts associated with each end point
        # c2 = (profile_line['slope'] - m2)*sx  + profile_line['intercept']
        # # get point where the two lines intercept
        # x1,x2 = (c2 - profile_line['intercept'])/(profile_line['slope'] - m2)
        # # compute y points
        # y1, y2 = profile_line["slope"] * np.array([x1,x2]) + profile_line["intercept"]

        # else:
        # # to get x values of end points, need to project first station onto the line
        # # get min and max y values
        # sy = np.array([y.min(),y.max()])
        # # get line through y1 perpendicular to profile
        # m2 = -1./profile_line['slope']
        # # two intercepts associated with each end point
        # c2 = sy - (m2/profile_line['slope'])*(sy - profile_line['intercept'])
        # # get point where the two lines intercept
        # y1,y2 = ((profile_line['intercept']/profile_line['slope']) - c2/m2)/\
        #     (1./profile_line['slope'] - 1./m2)
        # # compute x points
        # x1,x2 = (np.array([y1,y2]) -  profile_line["intercept"])/profile_line["slope"]

        # x1 = x.min()
        # x2 = x.max()
        # y1 = profile_line["slope"] * x1 + profile_line["intercept"]
        # y2 = profile_line["slope"] * x2 + profile_line["intercept"]

        return x1, y1, x2, y2, profile_line

    def generate_profile_from_strike(
        self, strike: float, units: str = "deg"
    ) -> tuple[float, float, float, float, dict[str, float]]:
        """
        Estimate a profile line from a given geoelectric strike.

        Parameters
        ----------
        strike : float
            Geoelectric strike angle in degrees
        units : str, optional
            Units for coordinates ("deg" or "m"), by default "deg"

        Returns
        -------
        x1 : float
            First x coordinate of profile line
        y1 : float
            First y coordinate of profile line
        x2 : float
            Second x coordinate of profile line
        y2 : float
            Second y coordinate of profile line
        profile_line : dict
            Dictionary with "slope" and "intercept" keys defining the profile line

        Raises
        ------
        ValueError
            If units is "m" but no UTM CRS is set

        """

        if units == "deg":
            x = self.station_locations.longitude
            y = self.station_locations.latitude

        elif units == "m":
            if self.utm_crs is not None:
                x = self.station_locations.east
                y = self.station_locations.north
            else:
                raise ValueError("Must input a UTM CRS or EPSG")

        profile_line = {"slope": np.arctan(np.deg2rad(90 - strike))}
        profile_line["intercept"] = y.min() - profile_line["slope"] * x.min()

        x1 = x.min()
        x2 = x.max()
        y1 = profile_line["slope"] * x1 + profile_line["intercept"]
        y2 = profile_line["slope"] * x2 + profile_line["intercept"]

        return x1, y1, x2, y2, profile_line

    def _extract_profile(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float | None,
    ) -> gpd.GeoDataFrame:
        """
        Extract stations along a profile line that lie within the given radius.

        Parameters
        ----------
        x1 : float
            First x coordinate of profile line
        y1 : float
            First y coordinate of profile line
        x2 : float
            Second x coordinate of profile line
        y2 : float
            Second y coordinate of profile line
        radius : float or None
            Distance threshold from the profile line to include stations.
            If None, uses a very large value (1e12)

        Returns
        -------
        list
            List of MT objects sorted by profile offset along the line

        Raises
        ------
        ValueError
            If coordinates are in degrees but no UTM CRS is set

        """

        if np.abs(x2 - x1) < 100:
            if self.utm_crs is None:
                raise ValueError("Must input UTM CRS or EPSG.")
            point_1 = MTLocation(longitude=x1, latitude=y1, utm_crs=self.utm_crs)
            point_2 = MTLocation(longitude=x2, latitude=y2, utm_crs=self.utm_crs)
            x1 = point_1.east
            y1 = point_1.north
            x2 = point_2.east
            y2 = point_2.north

        if radius is None:
            radius = 1e12

        def distance(x, y):
            """Distance function."""
            return np.abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.sqrt(
                (x2 - x1) ** 2 + (y2 - y1) ** 2
            )

        slope = (y2 - y1) / (x2 - x1)
        intersection = y1 - slope * x1

        station_df = self.station_locations
        if station_df is None or station_df.empty:
            return gpd.GeoDataFrame(columns=list(self.dtype.keys()) + ["geometry"])

        profile_df = station_df.copy()
        profile_df["_distance"] = distance(
            profile_df.east.to_numpy(dtype=float),
            profile_df.north.to_numpy(dtype=float),
        )
        profile_df = profile_df.loc[profile_df["_distance"] <= radius].copy()

        if profile_df.empty:
            return self._to_geodataframe(profile_df.drop(columns=["_distance"]))

        profile_vector = np.array([1.0, slope], dtype=float)
        profile_vector /= np.linalg.norm(profile_vector)
        station_vectors = np.column_stack(
            [
                profile_df.east.to_numpy(dtype=float),
                profile_df.north.to_numpy(dtype=float) - intersection,
            ]
        )
        scalar_projection = station_vectors @ profile_vector
        offsets = np.abs(scalar_projection)

        offsets -= offsets.min()
        profile_df["profile_offset"] = offsets
        profile_df.sort_values("profile_offset", inplace=True)
        profile_df.drop(columns=["_distance"], inplace=True)
        profile_df.reset_index(drop=True, inplace=True)

        return self._to_geodataframe(profile_df)
