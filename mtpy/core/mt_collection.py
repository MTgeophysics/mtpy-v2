# -*- coding: utf-8 -*-
"""
Collection of MT stations

Created on Mon Jan 11 15:36:38 2021

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from mth5.mth5 import MTH5

from mtpy import MT
from mtpy.core.mt_data import MTData
from mtpy.imaging import (
    PlotMultipleResponses,
    PlotPenetrationDepth1D,
    PlotPenetrationDepthMap,
    PlotPhaseTensorMaps,
    PlotPhaseTensorPseudoSection,
    PlotResidualPTMaps,
    PlotResidualPTPseudoSection,
    PlotResPhaseMaps,
    PlotResPhasePseudoSection,
    PlotStations,
    PlotStrike,
)


# =============================================================================
#
# =============================================================================


class MTCollection:
    """
    Collection of transfer functions.

    The main working variable is `MTCollection.dataframe` which is a property
    that returns either the `master dataframe` that contains all the TF's in
    the MTH5 file, or the `working_dataframe` which is a dataframe that has
    been queried in some way.

    Parameters
    ----------
    working_directory : str, Path, optional
        Working directory path, by default None

    Attributes
    ----------
    mth5_basename : str
        Base name for MTH5 file
    working_dataframe : pd.DataFrame or None
        Subset dataframe for queries
    mth5_collection : MTH5
        MTH5 object for data storage

    Examples
    --------
    >>> mc = MTCollection()
    >>> mc.open_collection(filename="path/to/example/mth5.h5")
    >>> mc.working_dataframe = mc.master_dataframe.iloc[0:5]

    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self._cwd = Path().cwd()
        self.mth5_basename = "mt_collection"
        self.working_directory = working_directory
        self.working_dataframe = None

        self.mth5_collection = MTH5()

        self._added = False

        self.logger = logger

    def __str__(self) -> str:
        """
        String representation.

        Returns
        -------
        str
            String representation of the collection

        """
        lines = [f"Working Directory: {self.working_directory}"]
        lines.append(f"MTH5 file:         {self.mth5_filename}")
        if self.mth5_collection.h5_is_read():
            lines.append(f"\tNumber of Transfer Functions: {len(self.dataframe)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Representation.

        Returns
        -------
        str
            String representation

        """
        return self.__str__()

    def __enter__(self) -> "MTCollection":
        """
        Context manager entry.

        Returns
        -------
        MTCollection
            Self

        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Context manager exit.

        Parameters
        ----------
        exc_type : type
            Exception type
        exc_val : Exception
            Exception value
        exc_tb : traceback
            Exception traceback

        Returns
        -------
        bool
            False to propagate exceptions

        """
        self.close_collection()
        return False

    @property
    def working_directory(self) -> Path:
        """
        Working directory path.

        Returns
        -------
        Path
            Current working directory

        """
        return self._cwd

    @working_directory.setter
    def working_directory(self, value: str | Path | None) -> None:
        """
        Set working directory.

        Parameters
        ----------
        value : str, Path, optional
            Directory path

        Raises
        ------
        IOError
            If directory does not exist

        """
        if value is None:
            return
        value = Path(value)
        if not value.exists():
            raise IOError(f"could not find directory {value}")
        self._cwd = value

    @property
    def mth5_filename(self) -> Path:
        """
        MTH5 filename path.

        Returns
        -------
        Path
            Full path to MTH5 file

        """
        if self.mth5_basename.find(".h5") > 0:
            return self.working_directory.joinpath(f"{self.mth5_basename}")
        else:
            return self.working_directory.joinpath(f"{self.mth5_basename}.h5")

    @mth5_filename.setter
    def mth5_filename(self, value: str | Path) -> None:
        """
        Set MTH5 filename.

        Parameters
        ----------
        value : str or Path
            Path to MTH5 file

        """
        value = Path(value)
        self.working_directory = value.parent
        self.mth5_basename = value.stem

    @property
    def master_dataframe(self) -> pd.DataFrame | None:
        """
        Full summary of all transfer functions.

        Returns
        -------
        pd.DataFrame or None
            DataFrame with all transfer functions in the MTH5 file,
            or None if MTH5 is not open

        Notes
        -----
        Automatically updated when TFs are added

        """

        if self.mth5_collection.h5_is_read():
            return self.mth5_collection.tf_summary.to_dataframe()
        return None

    @property
    def dataframe(self) -> pd.DataFrame | None:
        """
        Working or master dataframe.

        Returns
        -------
        pd.DataFrame or None
            Returns working_dataframe if set, otherwise master_dataframe

        """
        if self.working_dataframe is None:
            return self.master_dataframe
        elif isinstance(self.working_dataframe, pd.DataFrame):
            if self.working_dataframe.empty:
                return None
            else:
                return self.working_dataframe
        return self.working_dataframe

    def has_data(self) -> bool:
        """
        Check if collection has data.

        Returns
        -------
        bool
            True if master_dataframe is not None

        """
        if self.master_dataframe is not None:
            return True
        return False

    @staticmethod
    def make_file_list(
        mt_path: str | Path | list[str | Path] | None,
        file_types: list[str] | None = None,
    ) -> list[Path]:
        """
        Get list of MT files from a given path.

        Parameters
        ----------
        mt_path : str, Path, list, optional
            Path(s) to MT transfer function files
        file_types : list of str, optional
            File extensions to search for, by default ['edi']

        Returns
        -------
        list of Path
            List of file paths

        Raises
        ------
        IOError
            If path does not exist
        TypeError
            If mt_path is not str, Path, or list

        """
        if file_types is None:
            file_types = ["edi"]

        def check_path(mt_path):
            """Check path."""
            if mt_path is None:
                return None
            else:
                mt_path = Path(mt_path)
                if not mt_path.exists():
                    msg = f"{mt_path} does not exists"
                    raise IOError(msg)
                return mt_path

        if isinstance(mt_path, (str, Path)):
            mt_path = [check_path(mt_path)]
        elif isinstance(mt_path, list):
            mt_path = [check_path(path) for path in mt_path]
        else:
            raise TypeError(f"Not sure what to do with {type(mt_path)}")
        fn_list = []
        for path in mt_path:
            for ext in file_types:
                fn_list += list(path.glob(f"*.{ext}"))
        return fn_list

    def open_collection(
        self,
        filename: str | Path | None = None,
        basename: str | None = None,
        working_directory: str | Path | None = None,
        mode: str = "a",
        **kwargs: Any,
    ) -> None:
        """
        Initialize an MTH5 collection.

        Parameters
        ----------
        filename : str, Path, optional
            Full path to MTH5 file, by default None
        basename : str, optional
            Base name for MTH5 file, by default None
        working_directory : str, Path, optional
            Working directory path, by default None
        mode : str, optional
            File open mode, by default 'a'
        **kwargs : dict
            Additional parameters

        """
        if filename is not None:
            self.mth5_filename = filename

        if basename is not None:
            self.mth5_basename = basename

        if working_directory is not None:
            self.working_directory = working_directory

        self.mth5_collection.open_mth5(self.mth5_filename, mode, **kwargs)

    def close_collection(self) -> None:
        """
        Close MTH5 file.

        """
        self.mth5_collection.close_mth5()

    def add_tf(
        self,
        transfer_function: MT | str | Path | list | MTData,
        new_survey: str | None = None,
        tf_id_extra: str | None = None,
    ) -> None:
        """
        Add transfer function(s) to the collection.

        Parameters
        ----------
        transfer_function : MT, str, Path, list, MTData
            Transfer function object(s), file path(s), or MTData object
        new_survey : str, optional
            New survey name, by default None
        tf_id_extra : str, optional
            Additional text to append to 'tf_id', by default None

        Notes
        -----
        For efficiency, create list first then add using add_tf([list])

        """
        if isinstance(transfer_function, MTData):
            self.from_mt_data(
                transfer_function,
                new_survey=new_survey,
                tf_id_extra=tf_id_extra,
            )
            return
        elif not isinstance(transfer_function, (list, tuple, np.ndarray)):
            transfer_function = [transfer_function]
            self.logger.warning(
                "If you are adding multiple transfer functions, suggest making "
                "a list of transfer functions first then adding the list using "
                "mt_collection.add_tf([list_of_tfs]). "
                "Otherwise adding transfer functions one by one will be slow."
            )

        surveys = []
        for item in transfer_function:
            if isinstance(item, MT):
                survey_id = self._from_mt_object(
                    item,
                    new_survey=new_survey,
                    tf_id_extra=tf_id_extra,
                    update_metadata=False,
                )
                if survey_id not in surveys:
                    surveys.append(survey_id)
            elif isinstance(item, (str, Path)):
                survey_id = self._from_file(
                    item,
                    new_survey=new_survey,
                    tf_id_extra=tf_id_extra,
                    update_metadata=False,
                )
                if survey_id not in surveys:
                    surveys.append(survey_id)
            else:
                raise TypeError(f"Not sure want to do with {type(item)}.")

        if self.mth5_collection.file_version in ["0.1.0"]:
            self.mth5_collection.survey_group.update_metadata()
        else:
            for survey_id in surveys:
                survey_group = self.mth5_collection.get_survey(survey_id)
                survey_group.update_metadata()
        self.mth5_collection.tf_summary.summarize()

    def get_tf(self, tf_id: str, survey: str | None = None) -> MT:
        """
        Get transfer function from collection.

        Parameters
        ----------
        tf_id : str
            Transfer function identifier
        survey : str, optional
            Survey name, by default None

        Returns
        -------
        MT
            MT object for the requested transfer function

        Raises
        ------
        ValueError
            If tf_id cannot be found

        """

        if survey is None:
            try:
                find_df = self.master_dataframe.loc[
                    self.master_dataframe.tf_id == tf_id
                ]
                find_df = find_df.iloc[0]
                self.logger.warning(
                    f"Found multiple transfer functions with ID {tf_id}. "
                    "Suggest setting survey, otherwise returning the "
                    f"TF from survey {find_df.survey}."
                )
            except IndexError:
                raise ValueError(f"Could not find {tf_id} in collection.")
        else:
            try:
                find_df = self.master_dataframe.loc[
                    (self.master_dataframe.tf_id == tf_id)
                    & (self.master_dataframe.survey == survey)
                ]
                find_df = find_df.iloc[0]
            except IndexError:
                raise ValueError(f"Could not find {survey}.{tf_id} in collection.")

        ref = find_df.hdf5_reference

        mt_object = MT()
        tf_object = self.mth5_collection.from_reference(ref)

        mt_object.__dict__.update(tf_object.__dict__)
        mt_object.station_metadata.update_time_period()
        mt_object.survey_metadata.update_time_period()

        return mt_object

    def _from_file(
        self,
        filename: str | Path,
        new_survey: str | None = None,
        tf_id_extra: str | None = None,
        update_metadata: bool = True,
    ) -> str:
        """
        Add transfer function from file.

        Parameters
        ----------
        filename : str or Path
            Path to transfer function file
        new_survey : str, optional
            New survey name, by default None
        tf_id_extra : str, optional
            Additional text to append to 'tf_id', by default None
        update_metadata : bool, optional
            Whether to update metadata, by default True

        Returns
        -------
        str
            Survey ID

        Raises
        ------
        ValueError
            If MTH5 is not writable
        TypeError
            If filename is not str or Path

        """

        if not self.mth5_collection.h5_is_write():
            raise ValueError("Must initiate an MTH5 file first.")
        if not isinstance(filename, (str, Path)):
            raise TypeError(f"filename must be a string or Path not {type(filename)}")
        mt_object = MT(filename)
        mt_object.read()

        return self._from_mt_object(
            mt_object,
            new_survey=new_survey,
            tf_id_extra=tf_id_extra,
            update_metadata=update_metadata,
        )

    def _from_mt_object(
        self,
        mt_object: MT,
        new_survey: str | None = None,
        tf_id_extra: str | None = None,
        update_metadata: bool = True,
    ) -> str:
        """
        Add transfer function from MT object.

        Parameters
        ----------
        mt_object : MT
            MT object to add
        new_survey : str, optional
            New survey name, by default None
        tf_id_extra : str, optional
            Additional text to append to 'tf_id', by default None
        update_metadata : bool, optional
            Whether to update metadata, by default True

        Returns
        -------
        str
            Survey ID

        """
        if new_survey is not None:
            mt_object.survey = new_survey
        if tf_id_extra is not None:
            mt_object.tf_id = f"{mt_object.tf_id}_{tf_id_extra}"
        if mt_object.survey_metadata.id in [None, "", "0"]:
            mt_object.survey_metadata.id = "unknown_survey"
        tf_group = self.mth5_collection.add_transfer_function(
            mt_object, update_metadata=update_metadata
        )
        try:
            mt_object.survey = tf_group.hdf5_group.parent.parent.parent.parent.attrs[
                "id"
            ]
        except KeyError:
            self.logger.warning("could not access survey.id attribute in H5.")
        self.logger.info(f"added {mt_object.survey}.{mt_object.station}")
        return mt_object.survey

    def to_mt_data(
        self,
        bounding_box: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> MTData:
        """
        Get transfer functions as MTData object.

        Parameters
        ----------
        bounding_box : tuple of float, optional
            Bounding box (lon_min, lon_max, lat_min, lat_max), by default None
        **kwargs : dict
            Additional parameters passed to MTData

        Returns
        -------
        MTData
            MTData object containing transfer functions

        """

        if bounding_box is not None:
            self.apply_bbox(*bounding_box)

        mt_data = MTData(**kwargs)

        for row in self.dataframe.itertuples():
            tf = self.get_tf(row.tf_id, survey=row.survey)

            mt_data.add_station(tf, compute_relative_location=False)

        # compute locations at the end
        mt_data.compute_relative_locations()

        return mt_data

    def from_mt_data(
        self,
        mt_data: MTData,
        new_survey: str | None = None,
        tf_id_extra: str | None = None,
    ) -> None:
        """
        Add data from MTData object to collection.

        Parameters
        ----------
        mt_data : MTData
            MTData object containing transfer functions
        new_survey : str, optional
            New survey name, by default None
        tf_id_extra : str, optional
            Additional text to append to 'tf_id', by default None

        Raises
        ------
        IOError
            If MTH5 is not writable

        Notes
        -----
        Use 'new_survey' to create a new survey. Use 'tf_id_extra' to add
        text onto 'tf_id' (useful for edited/manipulated data)

        """
        if self.mth5_collection.h5_is_write():
            self.add_tf(
                list(mt_data.values()),
                new_survey=new_survey,
                tf_id_extra=tf_id_extra,
            )

        else:
            raise IOError("MTH5 is not writeable, use 'open_mth5()'")

    def check_for_duplicates(
        self, locate: str = "location", sig_figs: int = 6
    ) -> pd.DataFrame | None:
        """
        Check for duplicate station locations.

        Parameters
        ----------
        locate : str, optional
            Column(s) to check for duplicates, by default 'location'
        sig_figs : int, optional
            Significant figures for rounding, by default 6

        Returns
        -------
        pd.DataFrame or None
            DataFrame of duplicates, or None if no data

        Raises
        ------
        ValueError
            If locate column not found

        """
        if self.has_data():
            if locate == "location":
                self.dataframe.latitude = np.round(self.dataframe.latitude, sig_figs)
                self.dataframe.longitude = np.round(self.dataframe.longitude, sig_figs)

                query = ["latitude", "longitude"]
            elif locate not in self.dataframe.columns:
                raise ValueError(f"Not sure what to do with {locate}.")
            else:
                query = [locate]
            return self.dataframe[self.dataframe.duplicated(query)]
        return None

    def apply_bbox(
        self, lon_min: float, lon_max: float, lat_min: float, lat_max: float
    ) -> None:
        """
        Set working dataframe to only stations within bounding box.

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

        """

        if self.has_data():
            msg = (
                "Applying bounding box: "
                f"lon_min = {lon_min:.6g}, "
                f"lon_max = {lon_max:.6g}, "
                f"lat_min = {lat_min:.6g}, "
                f"lat_max = {lat_max:.6g}"
            )
            self.logger.debug(msg)

            self.working_dataframe = self.master_dataframe.loc[
                (self.master_dataframe.longitude >= lon_min)
                & (self.master_dataframe.longitude <= lon_max)
                & (self.master_dataframe.latitude >= lat_min)
                & (self.master_dataframe.latitude <= lat_max)
            ]

    def to_geo_df(
        self,
        bounding_box: tuple[float, float, float, float] | None = None,
        epsg: int = 4326,
    ) -> gpd.GeoDataFrame:
        """
        Create a geopandas GeoDataFrame for GIS manipulation.

        Parameters
        ----------
        bounding_box : tuple of float, optional
            Bounding box as (lon_min, lon_max, lat_min, lat_max), by default None
        epsg : int, optional
            EPSG code for coordinate system, by default 4326

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with station locations and metadata

        """
        coordinate_system = f"epsg:{epsg}"
        if bounding_box is not None:
            self.apply_bbox(*bounding_box)

        df = self.dataframe

        gdf = gpd.GeoDataFrame(
            df[
                df.columns[
                    ~df.columns.isin(["hdf5_reference", "station_hdf5_reference"])
                ]
            ],
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=coordinate_system,
        )

        return gdf

    def to_shp(
        self,
        filename: str | Path,
        bounding_box: tuple[float, float, float, float] | None = None,
        epsg: int = 4326,
    ) -> gpd.GeoDataFrame | None:
        """
        Create a shapefile of station locations.

        Parameters
        ----------
        filename : str or Path
            Filename to save the shapefile to
        bounding_box : tuple of float, optional
            Bounding box as (lon_min, lon_max, lat_min, lat_max), by default None
        epsg : int, optional
            EPSG code for coordinate system, by default 4326

        Returns
        -------
        gpd.GeoDataFrame or None
            GeoDataFrame of stations, or None if no data

        """

        if self.has_data():
            gdf = self.to_geo_df(bounding_box=bounding_box, epsg=epsg)
            gdf.to_file(self.working_directory.joinpath(filename))

            return gdf
        return None

    def average_stations(
        self,
        cell_size_m: float,
        bounding_box: tuple[float, float, float, float] | None = None,
        count: int = 1,
        n_periods: int = 48,
        new_file: bool = True,
    ) -> None:
        """
        Average nearby stations to reduce data density.

        Parameters
        ----------
        cell_size_m : float
            Size of square in meters to search for nearby stations
        bounding_box : tuple of float, optional
            Bounding box as (lon_min, lon_max, lat_min, lat_max), by default None
        count : int, optional
            Starting count for averaged station names, by default 1
        n_periods : int, optional
            Number of periods for interpolation, by default 48
        new_file : bool, optional
            Whether to write averaged stations to EDI files, by default True

        """

        # cell size in degrees (bit of a hack for now)
        r = cell_size_m / 111000.0

        if bounding_box:
            self.apply_bbox(*bounding_box)
            df = self.dataframe
        else:
            df = self.master_dataframe
        new_fn_list = []
        for ee in np.arange(df.longitude.min() - r / 2, df.longitude.max() + r, r):
            for nn in np.arange(df.latitude.min() - r / 2, df.latitude.max() + r, r):
                bbox = (ee, ee + r, nn, nn + r)
                self.apply_bbox(*bbox)
                if self.dataframe is None:
                    continue

                if len(self.dataframe) > 1:
                    m_list = [
                        self.get_tf(row.tf_id, row.survey)
                        for row in self.dataframe.itertuples()
                    ]
                    # interpolate onto a similar period range
                    f_list = []
                    for m in m_list:
                        f_list += m.period.tolist()
                    f = np.unique(np.array(f_list))
                    f = np.logspace(np.log10(f.min()), np.log10(f.max()), n_periods)

                    m_list_interp = []
                    for m in m_list:
                        m_list_interp.append(m.interpolate(f, bounds_error=False))
                    avg_z = np.array(
                        [m.impedance.data for m in m_list_interp if m.has_impedance()]
                    )

                    avg_z_err = np.array(
                        [
                            m.impedance_error.data
                            for m in m_list_interp
                            if m.has_impedance()
                        ]
                    )
                    avg_t = np.array(
                        [m.tipper.data for m in m_list_interp if m.has_tipper()]
                    )
                    avg_t_err = np.array(
                        [m.tipper_error.data for m in m_list_interp if m.has_tipper()]
                    )

                    avg_z[np.where(avg_z == 0 + 0j)] = np.nan + 1j * np.nan
                    avg_z_err[np.where(avg_z_err == 0)] = np.nan
                    avg_t[np.where(avg_t == 0 + 0j)] = np.nan + 1j * np.nan
                    avg_t_err[np.where(avg_t_err == 0 + 0j)] = np.nan

                    avg_z = np.nanmean(avg_z, axis=0)
                    avg_z_err = np.nanmean(avg_z_err, axis=0)
                    avg_t = np.nanmean(avg_t, axis=0)
                    avg_t_err = np.nanmean(avg_t_err, axis=0)

                    mt_avg = MT()
                    mt_avg.frequency = f
                    mt_avg.impedance = avg_z
                    mt_avg.impedance_error = avg_z_err

                    mt_avg.tipper = avg_t
                    mt_avg.tipper_error = avg_t_err

                    mt_avg.latitude = np.mean(np.array([m.latitude for m in m_list]))
                    mt_avg.longitude = np.mean(np.array([m.longitude for m in m_list]))
                    mt_avg.elevation = np.mean(np.array([m.elevation for m in m_list]))
                    mt_avg.station = f"AVG{count:03}"
                    mt_avg.station_metadata.comments = (
                        "avgeraged_stations = " + ",".join([m.station for m in m_list])
                    )
                    mt_avg.survey_metadata.id = "averaged"
                    self.add_tf(mt_avg)

                    try:
                        if new_file:
                            edi_obj = mt_avg.write(save_dir=self.working_directory)
                            self.logger.info(f"wrote average file {edi_obj.fn}")
                        new_fn_list.append(edi_obj.fn)
                        count += 1
                    except Exception as error:
                        self.logger.exception("Failed to average files %s", error)
                else:
                    continue

    def plot_mt_response(
        self,
        tf_id: str | list | tuple | np.ndarray | pd.Series,
        survey: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot MT response for one or more stations.

        Parameters
        ----------
        tf_id : str or array-like
            Transfer function ID(s) to plot. Can be single ID string or array of IDs.
            If 2D array, each row is [tf_id, survey].
        survey : str, optional
            Survey name, by default None
        **kwargs : Any
            Additional keyword arguments passed to plotting function

        Returns
        -------
        Any
            Plot object from MT.plot_mt_response or MTData.plot

        """
        mt_data = MTData()
        if isinstance(tf_id, str):
            mt_object = self.get_tf(tf_id, survey=survey)
            return mt_object.plot_mt_response(**kwargs)
        elif isinstance(tf_id, (list, tuple, np.ndarray, pd.Series)):
            tf_request = np.array(tf_id)
            if len(tf_request.shape) > 1:
                for row in tf_request:
                    mt_data.add_station(self.get_tf(row[0], survey=row[1]))

            else:
                for row in tf_request:
                    mt_data.add_station(self.get_tf(row, survey=survey))
            return PlotMultipleResponses(mt_data, **kwargs)

        elif isinstance(tf_id, pd.DataFrame):
            for row in tf_id.itertuples():
                mt_data.add_station(self.get_tf(row.tf_id, survey=row.survey))
            return PlotMultipleResponses(mt_data, **kwargs)

    def plot_stations(
        self,
        map_epsg: int = 4326,
        bounding_box: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot station locations on a map.

        Parameters
        ----------
        map_epsg : int, optional
            EPSG code for map projection, by default 4326
        bounding_box : tuple of float, optional
            Bounding box as (lon_min, lon_max, lat_min, lat_max), by default None
        **kwargs : Any
            Additional keyword arguments passed to PlotStations

        Returns
        -------
        Any
            PlotStations object

        """
        if self.dataframe is not None:
            gdf = self.to_geo_df(epsg=map_epsg, bounding_box=bounding_box)
            return PlotStations(gdf, **kwargs)

    def plot_strike(self, mt_data: MTData | None = None, **kwargs: Any) -> Any:
        """
        Plot strike angle.

        Parameters
        ----------
        mt_data : MTData, optional
            MTData object, by default None (uses collection data)
        **kwargs : Any
            Additional keyword arguments passed to PlotStrike

        Returns
        -------
        Any
            PlotStrike object

        See Also
        --------
        mtpy.imaging.PlotStrike

        """
        if mt_data is None:
            mt_data = self.to_mt_data()
        return PlotStrike(mt_data, **kwargs)

    def plot_phase_tensor(
        self, tf_id: str, survey: str | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot phase tensor elements for a station.

        Parameters
        ----------
        tf_id : str
            Transfer function ID
        survey : str, optional
            Survey name, by default None
        **kwargs : Any
            Additional keyword arguments passed to MT.plot_phase_tensor

        Returns
        -------
        Any
            Plot object from MT.plot_phase_tensor

        """

        tf_obj = self.get_tf(tf_id, survey=survey)
        return tf_obj.plot_phase_tensor(**kwargs)

    def plot_phase_tensor_map(
        self, mt_data: MTData | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot phase tensor maps for transfer functions.

        Parameters
        ----------
        mt_data : MTData, optional
            MTData object, by default None (uses collection data)
        **kwargs : Any
            Additional keyword arguments passed to PlotPhaseTensorMaps

        Returns
        -------
        Any
            PlotPhaseTensorMaps object

        See Also
        --------
        mtpy.imaging.PlotPhaseTensorMaps

        """

        if mt_data is None:
            mt_data = self.to_mt_data()

        return PlotPhaseTensorMaps(mt_data=mt_data, **kwargs)

    def plot_phase_tensor_pseudosection(
        self, mt_data: MTData | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot pseudosection of phase tensor ellipses and induction vectors.

        Parameters
        ----------
        mt_data : MTData, optional
            MTData object, by default None (uses collection data)
        **kwargs : Any
            Additional keyword arguments passed to PlotPhaseTensorPseudoSection

        Returns
        -------
        Any
            PlotPhaseTensorPseudoSection object

        See Also
        --------
        mtpy.imaging.PlotPhaseTensorPseudosection

        """

        if mt_data is None:
            mt_data = self.to_mt_data()

        return PlotPhaseTensorPseudoSection(mt_data=mt_data, **kwargs)

    def plot_residual_phase_tensor(
        self,
        mt_data_01: MTData,
        mt_data_02: MTData,
        plot_type: str = "map",
        **kwargs: Any,
    ) -> Any:
        """
        Plot residual phase tensor between two datasets.

        Parameters
        ----------
        mt_data_01 : MTData
            First MTData object
        mt_data_02 : MTData
            Second MTData object
        plot_type : str, optional
            Type of plot: 'map' or 'pseudosection'/'ps', by default 'map'
        **kwargs : Any
            Additional keyword arguments passed to plotting function

        Returns
        -------
        Any
            PlotResidualPTMaps or PlotResidualPTPseudoSection object

        """

        if plot_type in ["map"]:
            return PlotResidualPTMaps(mt_data_01, mt_data_02, **kwargs)

        if plot_type in ["pseudosection", "ps"]:
            return PlotResidualPTPseudoSection(mt_data_01, mt_data_02, **kwargs)

    def plot_penetration_depth_1d(
        self, tf_id: str, survey: str | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot 1D penetration depth using Niblett-Bostick transformation.

        Data is rotated to estimated strike prior to estimation, and strike
        angles are interpreted for data points that are 3D.

        Parameters
        ----------
        tf_id : str
            Transfer function ID
        survey : str, optional
            Survey name, by default None
        **kwargs : Any
            Additional keyword arguments passed to PlotPenetrationDepth1D

        Returns
        -------
        Any
            PlotPenetrationDepth1D object

        See Also
        --------
        mtpy.analysis.niblettbostick.calculate_depth_of_investigation

        """

        tf_object = self.get_tf(tf_id, survey=survey)

        return PlotPenetrationDepth1D(tf_object, **kwargs)

    def plot_penetration_depth_map(
        self, mt_data: MTData | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot penetration depth in map view for a single period.

        Parameters
        ----------
        mt_data : MTData, optional
            MTData object, by default None (uses collection data)
        **kwargs : Any
            Additional keyword arguments passed to PlotPenetrationDepthMap

        Returns
        -------
        Any
            PlotPenetrationDepthMap object

        See Also
        --------
        mtpy.imaging.PlotPenetrationDepthMap

        """

        if mt_data is None:
            mt_data = self.to_mt_data()

        return PlotPenetrationDepthMap(mt_data, **kwargs)

    def plot_resistivity_phase_maps(
        self, mt_data: MTData | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot apparent resistivity and impedance phase maps.

        Parameters
        ----------
        mt_data : MTData, optional
            MTData object, by default None (uses collection data)
        **kwargs : Any
            Additional keyword arguments passed to PlotResPhaseMaps

        Returns
        -------
        Any
            PlotResPhaseMaps object

        See Also
        --------
        mtpy.imaging.PlotResPhaseMaps

        """

        if mt_data is None:
            mt_data = self.to_mt_data()

        return PlotResPhaseMaps(mt_data, **kwargs)

    def plot_resistivity_phase_pseudosections(
        self, mt_data: MTData | None = None, **kwargs: Any
    ) -> Any:
        """
        Plot resistivity and phase pseudosections along a profile.

        Parameters
        ----------
        mt_data : MTData, optional
            MTData object, by default None (uses collection data)
        **kwargs : Any
            Additional keyword arguments passed to PlotResPhasePseudoSection

        Returns
        -------
        Any
            PlotResPhasePseudoSection object

        """

        if mt_data is None:
            mt_data = self.to_mt_data()

        return PlotResPhasePseudoSection(mt_data, **kwargs)
