# -*- coding: utf-8 -*-
"""
.. module:: MT
   :synopsis: The main container for MT response functions.

.. moduleauthor:: Jared Peacock <jpeacock@usgs.gov>
"""

from __future__ import annotations

from copy import deepcopy

# =============================================================================
from pathlib import Path
from typing import Any

import numpy as np
from mt_metadata.transfer_functions.core import TF

from mtpy.core import COORDINATE_REFERENCE_FRAME_OPTIONS, Tipper, Z
from mtpy.core.mt_dataframe import MTDataFrame
from mtpy.core.mt_location import MTLocation
from mtpy.core.transfer_function import IMPEDANCE_UNITS
from mtpy.imaging import PlotMTResponse, PlotPenetrationDepth1D, PlotPhaseTensor
from mtpy.modeling.errors import ModelErrors
from mtpy.modeling.occam1d import Occam1DData
from mtpy.modeling.simpeg.recipes.inversion_1d import Simpeg1D
from mtpy.utils.estimate_tf_quality_factor import EMTFStats


# =============================================================================
class MT(TF, MTLocation):
    """
    Main container for MT response functions.

    Impedance and Tipper element nomenclature is E/H where the first
    letter represents the output channels and the second letter represents
    the input channels. For example, for an input of Hx and an output of Ey,
    the impedance tensor element is Zyx.

    Parameters
    ----------
    fn : str or Path, optional
        Filename to read data from
    impedance_units : str, optional
        Units for impedance, by default "mt"
    **kwargs : dict
        Additional keyword arguments including period, frequency, impedance,
        impedance_error, tipper, tipper_error, and transfer function data

    Attributes
    ----------
    coordinate_reference_frame : str
        Reference frame of the transfer function. Default is NED:
        - x = North
        - y = East
        - z = + Down

        Alternative is ENU:
        - x = East
        - y = North
        - z = + Up

    Notes
    -----
    Coordinate reference frame options for NED:
        - "+"
        - "z+"
        - "nez+"
        - "ned"
        - "exp(+ i\\omega t)"
        - "exp(+i\\omega t)"
        - None

    Coordinate reference frame options for ENU:
        - "-"
        - "z-"
        - "enz-"
        - "enu"
        - "exp(- i\\omega t)"
        - "exp(-i\\omega t)"

    """

    def __init__(
        self,
        fn: str | Path | None = None,
        impedance_units: str = "mt",
        **kwargs: Any,
    ) -> None:
        tf_kwargs = {}
        for key in [
            "period",
            "frequency",
            "impedance",
            "impedance_error",
            "impedance_model_error",
            "tipper",
            "tipper_error",
            "tipper_model_error",
            "transfer_function",
            "transfer_function_error",
            "transfer_function_model_error",
        ]:
            try:
                tf_kwargs[key] = kwargs.pop(key)
            except KeyError:
                pass

        TF.__init__(self, **tf_kwargs)
        MTLocation.__init__(self, survey_metadata=self._survey_metadata)

        self.fn = fn

        self._Z = Z()
        self._Tipper = Tipper()
        self._rotation_angle = 0

        self.save_dir = Path.cwd()

        self._coordinate_reference_frame_options = COORDINATE_REFERENCE_FRAME_OPTIONS

        self.coordinate_reference_frame = (
            self.station_metadata.transfer_function.sign_convention
        )

        self._impedance_unit_factors = IMPEDANCE_UNITS
        self.impedance_units = impedance_units

        for key, value in kwargs.items():
            setattr(self, key, value)

    def clone_empty(self) -> "MT":
        """
        Copy metadata but not the transfer function estimates.

        Returns
        -------
        MT
            New MT object with copied metadata but no transfer function data

        """

        new_mt_obj = MT()
        new_mt_obj.survey_metadata.update(self.survey_metadata)
        new_mt_obj.station_metadata.update(self.station_metadata)
        new_mt_obj.station_metadata.runs = self.station_metadata.runs
        new_mt_obj._datum_crs = self._datum_crs
        new_mt_obj._utm_crs = self._utm_crs
        new_mt_obj._east = self._east
        new_mt_obj._north = self._north
        new_mt_obj.model_east = self.model_east
        new_mt_obj.model_north = self.model_north
        new_mt_obj.model_elevation = self.model_elevation
        new_mt_obj._rotation_angle = self._rotation_angle
        new_mt_obj.profile_offset = self.profile_offset
        new_mt_obj.impedance_units = self.impedance_units

        return new_mt_obj

    def __deepcopy__(self, memo: dict) -> "MT":
        """
        Deepcopy function.

        Parameters
        ----------
        memo : dict
            Memoization dictionary for deepcopy

        Returns
        -------
        MT
            Deep copy of the MT object

        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["logger"]:
                continue

            setattr(result, k, deepcopy(v, memo))
        result.logger = self.logger
        return result

    def copy(self) -> "MT":
        """
        Copy function.

        Returns
        -------
        MT
            Copy of the MT object

        """
        return deepcopy(self)

    @property
    def coordinate_reference_frame(self) -> str:
        """
        Coordinate reference frame of the transfer function.

        Default is NED:
         - x = North
         - y = East
         - z = + down

        Returns
        -------
        str
            Coordinate reference frame identifier (NED or ENU)

        """

        return self._coordinate_reference_frame_options[
            self.station_metadata.transfer_function.sign_convention
        ].upper()

    @coordinate_reference_frame.setter
    def coordinate_reference_frame(self, value: str | None) -> None:
        """
        Set coordinate reference frame.

        Parameters
        ----------
        value : str, optional
            Reference frame identifier. Options are NED or ENU.

            NED:
             - x = North
             - y = East
             - z = + down

            ENU:
             - x = East
             - y = North
             - z = + up

        Raises
        ------
        ValueError
            If value is not a recognized reference frame option

        """

        if value is None:
            value = "+"
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

        self.station_metadata.transfer_function.sign_convention = value

    @property
    def impedance_units(self) -> str:
        """
        Impedance units.

        Returns
        -------
        str
            Current impedance units ("mt" or "ohm")

        """
        return self._impedance_units

    @impedance_units.setter
    def impedance_units(self, value: str) -> None:
        """
        Set impedance units.

        Parameters
        ----------
        value : str
            Impedance units, options are "mt" or "ohm"

        Raises
        ------
        TypeError
            If value is not a string
        ValueError
            If value is not an acceptable unit for impedance

        """
        if not isinstance(value, str):
            raise TypeError("Units input must be a string.")
        if value.lower() not in self._impedance_unit_factors.keys():
            raise ValueError(f"{value} is not an acceptable unit for impedance.")

        self._impedance_units = value

    @property
    def rotation_angle(self) -> float | np.ndarray:
        """
        Rotation angle in degrees from north.

        Returns
        -------
        float or numpy.ndarray
            Rotation angle(s) in degrees from north in the coordinate reference frame

        """
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, theta_r: float | np.ndarray) -> None:
        """
        Set rotation angle and rotate Z and Tipper.

        Rotation angle in degrees assuming North is 0 measuring clockwise
        positive to East as 90.

        Parameters
        ----------
        theta_r : float or numpy.ndarray
            Rotation angle in degrees

        Notes
        -----
        Upon setting, rotates Z and Tipper data
        TODO: figure this out with xarray

        """

        self.rotate(theta_r)
        self._rotation_angle += theta_r

    def rotate(self, theta_r: float | np.ndarray, inplace: bool = True) -> "MT" | None:
        """
        Rotate the data in degrees.

        Rotation assumes North is 0 measuring clockwise positive to East as 90.

        Parameters
        ----------
        theta_r : float or numpy.ndarray
            Rotation angle to rotate by in degrees
        inplace : bool, optional
            Rotate all transfer function in place, by default True

        Returns
        -------
        MT or None
            If inplace is False, returns a new MT object. Otherwise returns None

        """

        if self.has_impedance():
            new_z = self.Z.rotate(
                theta_r,
                inplace=False,
                coordinate_reference_frame=self.coordinate_reference_frame,
            )
        if self.has_tipper():
            new_t = self.Tipper.rotate(
                theta_r,
                inplace=False,
                coordinate_reference_frame=self.coordinate_reference_frame,
            )

        if inplace:
            if self.has_impedance():
                self.Z = new_z
            if self.has_tipper():
                self.Tipper = new_t

            self._rotation_angle += theta_r

            if isinstance(self._rotation_angle, (float, int)):
                self.logger.info(
                    f"Rotated transfer function by: {self._rotation_angle:.3f} "
                    "degrees clockwise in reference frame "
                    f"{self.coordinate_reference_frame}."
                )
            else:
                self.logger.info(
                    f"Rotated transfer function by: {self._rotation_angle.mean():.3f} "
                    "degrees clockwise in reference frame "
                    f"{self.coordinate_reference_frame}."
                )
        else:
            new_m = self.clone_empty()
            if self.has_impedance():
                new_m.Z = new_z
            if self.has_tipper():
                new_m.Tipper = new_t
            new_m._rotation_angle += theta_r
            return new_m

    @property
    def Z(self) -> Z:
        """
        Z object to hold impedance tensor.

        Returns
        -------
        mtpy.core.z.Z
            Z object containing impedance tensor data

        """

        if self.has_impedance():
            z_object = Z(
                z=self.impedance.to_numpy(),
                z_error=self.impedance_error.to_numpy(),
                frequency=self.frequency,
                z_model_error=self.impedance_model_error.to_numpy(),
            )
            z_object.units = self.impedance_units
            return z_object
        return Z()

    @Z.setter
    def Z(self, z_object: Z) -> None:
        """
        Set Z object.

        Recalculates phase tensor and invariants, which shouldn't change except
        for strike angle.

        Parameters
        ----------
        z_object : mtpy.core.z.Z
            Z object containing impedance data

        Notes
        -----
        Be sure to have appropriate units set. If a z object is given the
        underlying data is in mt units, even if the units are set to ohm.

        """
        self.impedance_units = z_object.units
        if not isinstance(z_object.frequency, type(None)):
            if self.frequency.size != z_object.frequency.size:
                self.frequency = z_object.frequency

            elif not (self.frequency == z_object.frequency).all():
                self.frequency = z_object.frequency
        # set underlying data to units of mt
        self.impedance = z_object._dataset.transfer_function.values
        self.impedance_error = z_object._dataset.transfer_function_error.values
        self.impedance_model_error = (
            z_object._dataset.transfer_function_model_error.values
        )

    @property
    def Tipper(self) -> Tipper:
        """
        Tipper object to hold tipper information.

        Returns
        -------
        mtpy.core.Tipper
            Tipper object containing tipper data

        """

        if self.has_tipper():
            return Tipper(
                tipper=self.tipper.to_numpy(),
                tipper_error=self.tipper_error.to_numpy(),
                frequency=self.frequency,
                tipper_model_error=self.tipper_model_error.to_numpy(),
            )

    @Tipper.setter
    def Tipper(self, t_object: Tipper | None) -> None:
        """
        Set tipper object.

        Recalculates tipper angle and magnitude.

        Parameters
        ----------
        t_object : mtpy.core.Tipper or None
            Tipper object containing tipper data

        """

        if t_object is None:
            return

        if not isinstance(t_object.frequency, type(None)):
            if not (self.frequency == t_object.frequency).all():
                self.frequency = t_object.frequency
        self.tipper = t_object.tipper
        self.tipper_error = t_object.tipper_error
        self.tipper_model_error = t_object.tipper_model_error

    @property
    def pt(self):
        """
        PhaseTensor object.

        Returns
        -------
        mtpy.analysis.pt.PhaseTensor
            Phase tensor object

        """
        return self.Z.phase_tensor

    @property
    def ex_metadata(self) -> Any:
        """
        EX channel metadata.

        Returns
        -------
        Electric metadata object
            Metadata for EX channel

        """
        return self.station_metadata.runs[0].ex

    @ex_metadata.setter
    def ex_metadata(self, value: Any) -> None:
        """
        Set EX channel metadata.

        Parameters
        ----------
        value : Electric metadata object
            Metadata for EX channel

        """
        self.station_metadata.runs[0].ex = value

    @property
    def ey_metadata(self) -> Any:
        """
        EY channel metadata.

        Returns
        -------
        Electric metadata object
            Metadata for EY channel

        """
        return self.station_metadata.runs[0].ey

    @ey_metadata.setter
    def ey_metadata(self, value: Any) -> None:
        """
        Set EY channel metadata.

        Parameters
        ----------
        value : Electric metadata object
            Metadata for EY channel

        """
        self.station_metadata.runs[0].ey = value

    @property
    def hx_metadata(self) -> Any:
        """
        HX channel metadata.

        Returns
        -------
        Magnetic metadata object
            Metadata for HX channel

        """
        return self.station_metadata.runs[0].hx

    @hx_metadata.setter
    def hx_metadata(self, value: Any) -> None:
        """
        Set HX channel metadata.

        Parameters
        ----------
        value : Magnetic metadata object
            Metadata for HX channel

        """
        self.station_metadata.runs[0].hx = value

    @property
    def hy_metadata(self) -> Any:
        """
        HY channel metadata.

        Returns
        -------
        Magnetic metadata object
            Metadata for HY channel

        """
        return self.station_metadata.runs[0].hy

    @hy_metadata.setter
    def hy_metadata(self, value: Any) -> None:
        """
        Set HY channel metadata.

        Parameters
        ----------
        value : Magnetic metadata object
            Metadata for HY channel

        """
        self.station_metadata.runs[0].hy = value

    @property
    def hz_metadata(self) -> Any:
        """
        HZ channel metadata.

        Returns
        -------
        Magnetic metadata object
            Metadata for HZ channel

        """
        return self.station_metadata.runs[0].hz

    @hz_metadata.setter
    def hz_metadata(self, value: Any) -> None:
        """
        Set HZ channel metadata.

        Parameters
        ----------
        value : Magnetic metadata object
            Metadata for HZ channel

        """
        self.station_metadata.runs[0].hz = value

    @property
    def rrhx_metadata(self) -> Any:
        """
        Remote reference HX channel metadata.

        Returns
        -------
        Magnetic metadata object
            Metadata for remote reference HX channel

        """
        return self.station_metadata.runs[0].rrhx

    @property
    def rrhy_metadata(self) -> Any:
        """
        Remote reference HY channel metadata.

        Returns
        -------
        Magnetic metadata object
            Metadata for remote reference HY channel

        """
        return self.station_metadata.runs[0].rrhy

    def estimate_tf_quality(
        self,
        weights: dict[str, float] = {
            "bad": 0.35,
            "corr": 0.2,
            "diff": 0.2,
            "std": 0.2,
            "fit": 0.05,
        },
        round_qf: bool = False,
    ) -> float:
        """
        Estimate transfer function quality factor.

        Quality factor ranges from 0-5, with 5 being the best.

        Parameters
        ----------
        weights : dict, optional
            Dictionary of weight factors for different quality metrics.
            Keys: "bad", "corr", "diff", "std", "fit"
            Default: {"bad": 0.35, "corr": 0.2, "diff": 0.2, "std": 0.2, "fit": 0.05}
        round_qf : bool, optional
            Whether to round the quality factor, by default False

        Returns
        -------
        float
            Quality factor value between 0 and 5

        """
        if self.has_impedance():
            if self.has_tipper():
                tf_stats = EMTFStats(self.Z, self.Tipper)
            else:
                tf_stats = EMTFStats(self.Z, None)
        else:
            tf_stats = EMTFStats(None, self.Tipper)

        return tf_stats.estimate_quality_factor(weights, round_qf=round_qf)

    def remove_distortion(
        self,
        n_frequencies: int | None = None,
        comp: str = "det",
        only_2d: bool = False,
        inplace: bool = False,
    ) -> "MT" | None:
        """
        Remove distortion following Bibby et al. [2005].

        Parameters
        ----------
        n_frequencies : int, optional
            Number of frequencies to look for distortion from the highest frequency,
            by default None
        comp : str, optional
            Component to use, by default "det"
        only_2d : bool, optional
            Whether to only consider 2D distortion, by default False
        inplace : bool, optional
            Whether to modify in place, by default False

        Returns
        -------
        MT or None
            If inplace is False, returns new MT object with distortion removed.
            Otherwise returns None

        """
        if inplace:
            self.Z = self.Z.remove_distortion(
                n_frequencies=n_frequencies,
                comp=comp,
                only_2d=only_2d,
                inplace=False,
            )
        else:
            new_mt = self.clone_empty()
            new_mt.Z = self.Z.remove_distortion(
                n_frequencies=n_frequencies,
                comp=comp,
                only_2d=only_2d,
                inplace=False,
            )
            new_mt.Tipper = self.Tipper
            return new_mt

    def remove_static_shift(
        self, ss_x: float = 1.0, ss_y: float = 1.0, inplace: bool = False
    ) -> "MT" | None:
        """
        Remove static shift from the apparent resistivity.

        Assumes the original observed tensor Z is built by a static shift S
        and an unperturbated "correct" Z0:
            Z = S * Z0

        Therefore the correct Z will be:
            Z0 = S^(-1) * Z

        Parameters
        ----------
        ss_x : float, optional
            Correction factor for x component, by default 1.0
        ss_y : float, optional
            Correction factor for y component, by default 1.0
        inplace : bool, optional
            Whether to modify in place, by default False

        Returns
        -------
        MT or None
            If inplace is False, returns new Z object with static shift removed.
            Otherwise returns None

        """

        if inplace:
            self.Z = self.Z.remove_ss(
                reduce_res_factor_x=ss_x,
                reduce_res_factor_y=ss_y,
                inplace=False,
            )

        else:
            new_mt = self.clone_empty()
            new_mt.Z = self.Z.remove_ss(
                reduce_res_factor_x=ss_x,
                reduce_res_factor_y=ss_y,
                inplace=inplace,
            )
            new_mt.Tipper = self.Tipper
            return new_mt

    def interpolate(
        self,
        new_period: np.ndarray,
        method: str = "slinear",
        bounds_error: bool = True,
        f_type: str = "period",
        **kwargs: Any,
    ) -> "MT":
        """
        Interpolate the impedance tensor onto different frequencies.

        Parameters
        ----------
        new_period : numpy.ndarray
            1-d array of frequencies to interpolate onto. Must be within the bounds
            of the existing frequency range, anything outside will raise an error
        method : str, optional
            Interpolation method, by default "slinear"
        bounds_error : bool, optional
            Check if input frequencies are within original frequencies, by default True
        f_type : str, optional
            Frequency type, can be "frequency" or "period", by default "period"
        **kwargs : dict
            Additional keyword arguments for interpolation

        Returns
        -------
        MT
            New MT object with interpolated values

        Raises
        ------
        ValueError
            If f_type is not 'frequency' or 'period'
            If input frequencies are out of bounds

        """

        if f_type not in ["frequency", "freq", "period", "per"]:
            raise ValueError(
                "f_type must be either 'frequency' or 'period' not {f_type}"
            )

        # make sure the input is a numpy array
        if not isinstance(new_period, np.ndarray):
            new_period = np.array(new_period)

        if f_type in ["frequency", "freq"]:
            new_period = 1.0 / new_period

        # check the bounds of the new frequency array
        if bounds_error:
            if self.period.min() > new_period.min():
                raise ValueError(
                    f"New period minimum of {new_period.min():.5g} "
                    "is smaller than old period minimum of "
                    f"{self.period.min():.5g}.  The new period range "
                    "needs to be within the bounds of the old one."
                )
            if self.period.max() < new_period.max():
                raise ValueError(
                    f"New period maximum of {new_period.max():.5g} "
                    "is smaller than old frequency maximum of "
                    f"{self.period.max():.5g}.  The new period range "
                    "needs to be within the bounds of the old one."
                )

        new_m = self.clone_empty()
        theta_r = 0
        if isinstance(self._rotation_angle, (int, float)):
            if self._rotation_angle != 0:
                theta_r = float(self._rotation_angle)
        elif isinstance(self._rotation_angle, np.ndarray):
            if self._rotation_angle.mean() != 0:
                theta_r = float(self._rotation_angle.mean())
                self.logger.warning(
                    f"Station {self.station}: Using mean rotation angle of {theta_r:.2f} degrees."
                )
        new_m._rotation_angle = np.repeat(theta_r, len(new_period))
        if self.has_impedance():
            new_m.Z = self.Z.interpolate(new_period, method=method, **kwargs)
            if new_m.has_impedance():
                if np.all(np.isnan(new_m.Z.z)):
                    self.logger.warning(
                        f"Station {self.station}: Interpolated Z values are all NaN, "
                        "consider an alternative interpolation method. "
                        "See scipy.interpolate.interp1d for more information."
                    )
        if self.has_tipper():
            new_m.Tipper = self.Tipper.interpolate(new_period, method=method, **kwargs)
            if new_m.has_tipper():
                if np.all(np.isnan(new_m.Tipper.tipper)):
                    self.logger.warning(
                        f"Station {self.station}: Interpolated T values are all NaN, "
                        "consider an alternative interpolation method. "
                        "See scipy.interpolate.interp1d for more information."
                    )

        return new_m

    def plot_mt_response(self, **kwargs: Any) -> PlotMTResponse:
        """
        Create a PlotResponse object for plotting MT response.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for plotting

        Returns
        -------
        PlotMTResponse
            Plot response object

        Examples
        --------
        >>> mt_obj = MT(edi_file)
        >>> pr = mt_obj.plot_mt_response()
        >>> # For more info on plot_mt_response
        >>> help(pr)

        """

        plot_obj = PlotMTResponse(
            z_object=self.Z,
            t_object=self.Tipper,
            pt_obj=self.pt,
            station=self.station,
            **kwargs,
        )

        return plot_obj

    def plot_phase_tensor(self, **kwargs: Any) -> PlotPhaseTensor:
        """
        Plot phase tensor.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for plotting

        Returns
        -------
        PlotPhaseTensor
            Phase tensor plot object

        """
        kwargs["ellipse_size"] = 0.5
        return PlotPhaseTensor(self.pt, station=self.station, **kwargs)

    def plot_depth_of_penetration(self, **kwargs: Any) -> PlotPenetrationDepth1D:
        """
        Plot depth of penetration estimated from Niblett-Bostick estimation.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for plotting

        Returns
        -------
        PlotPenetrationDepth1D
            Penetration depth plot object

        """

        return PlotPenetrationDepth1D(self, **kwargs)

    def to_dataframe(
        self,
        utm_crs: Any = None,
        cols: list[str] | None = None,
        impedance_units: str = "mt",
    ) -> MTDataFrame:
        """
        Create a dataframe from the transfer function.

        For use with plotting and modeling.

        Parameters
        ----------
        utm_crs : str, int, or pyproj.CRS, optional
            UTM zone to project station to. Could be a name, pyproj.CRS,
            EPSG number, or anything that pyproj.CRS can intake, by default None
        cols : list of str, optional
            Column names to include, by default None
        impedance_units : str, optional
            Impedance units, "mt" [mV/km/nT] or "ohm" [Ohms], by default "mt"

        Returns
        -------
        MTDataFrame
            DataFrame containing MT data

        """
        if utm_crs is not None:
            self.utm_crs = utm_crs

        n_entries = self.period.size
        mt_df = MTDataFrame(n_entries=n_entries)

        mt_df.survey = self.survey
        mt_df.station = self.station
        mt_df.latitude = self.latitude
        mt_df.longitude = self.longitude
        mt_df.elevation = self.elevation
        mt_df.datum_epsg = self.datum_epsg
        mt_df.east = self.east
        mt_df.north = self.north
        mt_df.utm_epsg = self.utm_epsg
        mt_df.model_east = self.model_east
        mt_df.model_north = self.model_north
        mt_df.model_elevation = self.model_elevation
        mt_df.profile_offset = self.profile_offset

        mt_df.dataframe.loc[:, "period"] = self.period
        if self.has_impedance():
            z_object = self.Z
            z_object.units = impedance_units
            mt_df.from_z_object(z_object)
        if self.has_tipper():
            mt_df.from_t_object(self.Tipper)

        return mt_df

    def from_dataframe(
        self, mt_df: MTDataFrame | Any, impedance_units: str = "mt"
    ) -> None:
        """
        Fill transfer function attributes from a dataframe for a single station.

        Parameters
        ----------
        mt_df : MTDataFrame or DataFrame-like
            Dataframe containing MT data for a single station
        impedance_units : str, optional
            Impedance units, "mt" [mV/km/nT] or "ohm" [Ohms], by default "mt"

        Raises
        ------
        TypeError
            If input dataframe is not an MTDataFrame or cannot be converted
        ValueError
            If dataframe is invalid

        """

        if not isinstance(mt_df, MTDataFrame):
            try:
                mt_df = MTDataFrame(mt_df)
            except TypeError:
                raise TypeError(
                    f"Input dataframe must be an MTDataFrame not {type(mt_df)}"
                )
            except ValueError as error:
                raise ValueError(error)

        for key in [
            "survey",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "east",
            "north",
            "utm_epsg",
            "model_north",
            "model_east",
            "model_elevation",
            "profile_offset",
        ]:
            try:
                setattr(self, key, getattr(mt_df, key))
            except KeyError:
                continue

        self.tf_id = self.station

        z_obj = mt_df.to_z_object()
        z_obj.units = impedance_units
        self.Z = z_obj
        self.Tipper = mt_df.to_t_object()

    def compute_model_z_errors(
        self,
        error_value: float = 5,
        error_type: str = "geometric_mean",
        floor: bool = True,
    ) -> None:
        """
        Compute model errors based on the error type.

        Parameters
        ----------
        error_value : float, optional
            Error value/multiplier, by default 5
        error_type : str, optional
            Type of error calculation, by default "geometric_mean"
            Options:
                - "egbert" or "geometric_mean": error_value * sqrt(Zxy * Zyx)
                - "arithmetic_mean" or "mean_od": error_value * (Zxy + Zyx) / 2
                - "off_diagonals": zxx_error == zxy_error, zyx_error == zyy_error
                - "median": error_value * median(z)
                - "eigen": error_value * mean(eigen(z))
                - "percent": error_value * z
                - "absolute": error_value
        floor : bool, optional
            Whether to apply a floor to errors, by default True

        Notes
        -----
        Sets the impedance_model_error attribute with computed errors.

        """

        if not self.has_impedance():
            self.logger.warning(
                "MT Object contains no impedance data, cannot comput errors"
            )
            return

        z_model_error = ModelErrors(
            data=self.impedance,
            measurement_error=self.impedance_error,
            error_value=error_value,
            error_type=error_type,
            floor=floor,
            mode="impedance",
        )

        err = z_model_error.compute_error()

        if len(err.shape) == 1:
            z_error = np.zeros_like(self.impedance, dtype=float)
            z_error[:, 0, 0] = err
            z_error[:, 0, 1] = err
            z_error[:, 1, 0] = err
            z_error[:, 1, 1] = err

        else:
            z_error = err

        self.impedance_model_error = z_error

    def compute_model_t_errors(
        self,
        error_value: float = 0.02,
        error_type: str = "absolute",
        floor: bool = False,
    ) -> None:
        """
        Compute model errors for tipper based on the error type.

        Parameters
        ----------
        error_value : float, optional
            Error value/multiplier, by default 0.02
        error_type : str, optional
            Type of error calculation, by default "absolute"
            Options:
                - "percent": error_value * t
                - "absolute": error_value
        floor : bool, optional
            Whether to apply a floor to errors, by default False

        Notes
        -----
        Sets the tipper_model_error attribute with computed errors.

        """
        if not self.has_tipper():
            self.logger.warning(
                f"MT object for {self.station} contains no Tipper, cannot "
                "compute model errors"
            )
            return

        t_model_error = ModelErrors(
            data=self.tipper,
            measurement_error=self.tipper_error,
            error_value=error_value,
            error_type=error_type,
            floor=floor,
            mode="tipper",
        )

        err = t_model_error.compute_error()

        if len(err.shape) == 1:
            t_error = np.zeros_like(self.tipper, dtype=float)
            t_error[:, 0, 0] = err
            t_error[:, 0, 1] = err

        else:
            t_error = err

        self.tipper_model_error = t_error

    def add_model_error(
        self,
        comp: str | list[str] = [],
        z_value: float = 5,
        t_value: float = 0.05,
        periods: tuple[float, float] | None = None,
    ) -> None:
        """
        Add error to station's components for given period range.

        Parameters
        ----------
        comp : str or list of str, optional
            List of components to add data to. Valid components are:
            "zxx", "zxy", "zyx", "zyy", "tzx", "tzy", by default []
        z_value : float, optional
            Multiplier for impedance error, by default 5
        t_value : float, optional
            Multiplier for tipper error, by default 0.05
        periods : tuple of float, optional
            (min_period, max_period) to apply errors to, by default None (all periods)

        Raises
        ------
        ValueError
            If periods tuple doesn't contain exactly 2 values

        """
        c_dict = {
            "zxx": (0, 0),
            "zxy": (0, 1),
            "zyx": (1, 0),
            "zyy": (1, 1),
            "tzx": (0, 0),
            "tzy": (0, 1),
        }

        if isinstance(comp, str):
            comp = [comp]
        if periods is not None:
            if len(periods) != 2:
                msg = "Must enter a minimum and maximum period value"
                self.logger.error(msg)
                raise ValueError(msg)
            p_min = np.where(self.period >= min(periods))[0][0]
            p_max = np.where(self.period <= max(periods))[0][-1]
        else:
            p_min = 0
            p_max = len(self.period) - 1

        if self.has_impedance():
            z_model_error = self.impedance_model_error.copy().data
            for cc in [c for c in comp if c.startswith("z")]:
                try:
                    ii, jj = c_dict[cc]
                except KeyError:
                    msg = f"Component {cc} is not a valid component, skipping"
                    self.logger.warning(msg)
                    continue
                z_model_error[p_min:p_max, ii, jj] *= z_value
            self.impedance_model_error = z_model_error

        if self.has_tipper():
            t_model_error = self.tipper_model_error.copy().data
            for cc in [c for c in comp if c.startswith("t")]:
                try:
                    ii, jj = c_dict[cc]
                except KeyError:
                    msg = f"Component {cc} is not a valid component, skipping"
                    self.logger.warning(msg)
                    continue
                    t_model_error[p_min:p_max, ii, jj] += t_value

            self.tipper_model_error = t_model_error

    def find_flipped_phase(self) -> dict[str, bool]:
        """
        Identify if off-diagonal components are flipped from traditional quadrants.

        xy should be in the 1st quadrant (0-90 deg) and yx should be in the
        3rd quadrant (-180 to -90 deg).

        Returns
        -------
        dict
            Dictionary of components with bool values. True indicates flipped phase
            Keys: "zxy", "zyx"

        """

        flip_dict = {"zxy": False, "zyx": False}

        if self.Z.phase_xy.mean() < 0:
            flip_dict["zxy"] = True

        if self.Z.phase_yx.mean() > -90:
            flip_dict["zyx"] = True

        return flip_dict

    def flip_phase(
        self,
        zxx: bool = False,
        zxy: bool = False,
        zyx: bool = False,
        zyy: bool = False,
        tzx: bool = False,
        tzy: bool = False,
        inplace: bool = False,
    ) -> "MT" | None:
        """
        Flip the phase of components in case they're plotting in the wrong quadrant.

        Parameters
        ----------
        zxx : bool, optional
            Flip Z_xx phase, by default False
        zxy : bool, optional
            Flip Z_xy phase, by default False
        zyx : bool, optional
            Flip Z_yx phase, by default False
        zyy : bool, optional
            Flip Z_yy phase, by default False
        tzx : bool, optional
            Flip T_zx phase, by default False
        tzy : bool, optional
            Flip T_zy phase, by default False
        inplace : bool, optional
            Whether to modify in place, by default False

        Returns
        -------
        MT or None
            If inplace is False, returns new MT object with flipped phases.
            Otherwise returns None

        Notes
        -----
        Only flips the transfer function elements as the error is agnostic to sign.

        """

        c_dict = {
            "zxx": zxx,
            "zxy": zxy,
            "zyx": zyx,
            "zyy": zyy,
            "tzx": tzx,
            "tzy": tzy,
        }

        # Only need to flip the transfer function elements cause the error
        # is agnostic to sign.
        if inplace:
            for ckey, cbool in c_dict.items():
                if cbool:
                    self._transfer_function.transfer_function.loc[
                        getattr(self, f"index_{ckey}")
                    ] *= -1
        else:
            mt_obj = self.copy()
            for ckey, cbool in c_dict.items():
                if cbool:
                    mt_obj._transfer_function.transfer_function.loc[
                        getattr(self, f"index_{ckey}")
                    ] *= -1
            return mt_obj

    def remove_component(
        self,
        zxx: bool = False,
        zxy: bool = False,
        zyy: bool = False,
        zyx: bool = False,
        tzx: bool = False,
        tzy: bool = False,
        inplace: bool = False,
    ) -> "MT" | None:
        """
        Remove a component by setting it to NaN.

        Parameters
        ----------
        zxx : bool, optional
            Remove Z_xx, by default False
        zxy : bool, optional
            Remove Z_xy, by default False
        zyy : bool, optional
            Remove Z_yy, by default False
        zyx : bool, optional
            Remove Z_yx, by default False
        tzx : bool, optional
            Remove T_zx, by default False
        tzy : bool, optional
            Remove T_zy, by default False
        inplace : bool, optional
            Whether to modify in place, by default False

        Returns
        -------
        MT or None
            If inplace is False, returns new MT object with components removed.
            Otherwise returns None

        """
        c_dict = {
            "zxx": zxx,
            "zxy": zxy,
            "zyx": zyx,
            "zyy": zyy,
            "tzx": tzx,
            "tzy": tzy,
        }

        # set to nan
        if inplace:
            for ckey, cbool in c_dict.items():
                if cbool:
                    self._transfer_function.transfer_function.loc[
                        getattr(self, f"index_{ckey}")
                    ] = (np.nan + 1j * np.nan)
                    self._transfer_function.transfer_function_error.loc[
                        getattr(self, f"index_{ckey}")
                    ] = 0
                    self._transfer_function.transfer_function_model_error.loc[
                        getattr(self, f"index_{ckey}")
                    ] = 0
        else:
            mt_obj = self.copy()
            for ckey, cbool in c_dict.items():
                if cbool:
                    mt_obj._transfer_function.transfer_function.loc[
                        getattr(self, f"index_{ckey}")
                    ] = (np.nan + 1j * np.nan)
                    mt_obj._transfer_function.transfer_function_error.loc[
                        getattr(self, f"index_{ckey}")
                    ] = 0
                    mt_obj._transfer_function.transfer_function_model_error.loc[
                        getattr(self, f"index_{ckey}")
                    ] = 0
            return mt_obj

    def add_white_noise(self, value: float, inplace: bool = True) -> "MT" | None:
        """
        Add white noise to the data.

        Useful for synthetic tests.

        Parameters
        ----------
        value : float
            Noise level as a fraction (0-1) or percentage (0-100)
        inplace : bool, optional
            Whether to modify in place, by default True

        Returns
        -------
        MT or None
            If inplace is False, returns new MT object with added noise.
            Otherwise returns None

        """
        if value > 1:
            value = value / 100.0

        if not inplace:
            new_mt_obj = self.clone_empty()

        tf_shape = self._transfer_function.transfer_function.shape
        noise_real = 1 + np.random.random(tf_shape) * value * (-1) ** (
            np.random.randint(0, 3, tf_shape)
        )
        noise_imag = 1 + np.random.random(tf_shape) * value * (-1) ** (
            np.random.randint(0, 3, tf_shape)
        )

        if inplace:
            self._transfer_function[
                "transfer_function"
            ] = self._transfer_function.transfer_function.real * (noise_real) + (
                1j * self._transfer_function.transfer_function.imag * noise_imag
            )

            self._transfer_function["transfer_function_error"] = (
                self._transfer_function.transfer_function_error + value
            )

        else:
            new_mt_obj._transfer_function = self._transfer_function.copy()
            new_mt_obj._transfer_function[
                "transfer_function"
            ] = self._transfer_function.transfer_function.real * (noise_real) + (
                1j * self._transfer_function.transfer_function.imag * noise_imag
            )

            self._transfer_function["transfer_function_error"] = (
                self._transfer_function.transfer_function_error + value
            )
            return new_mt_obj

    def edit_curve(self, method: str = "default", tolerance: float = 0.05) -> None:
        """
        Try to remove bad points in a scientific way.

        Parameters
        ----------
        method : str, optional
            Method for curve editing, by default "default"
        tolerance : float, optional
            Tolerance for detecting bad points, by default 0.05

        Notes
        -----
        This method is intended to bring up a GUI interface.

        """

        # bring up a gui of some sort.

    def to_occam1d(
        self, data_filename: str | Path | None = None, mode: str = "det"
    ) -> Occam1DData:
        """
        Write an Occam1D data file.

        Parameters
        ----------
        data_filename : str or Path, optional
            Path to write file. If None, returns Occam1DData object without writing,
            by default None
        mode : str, optional
            Mode for inversion. Options: "te", "tm", "det", "tez", "tmz", "detz",
            by default "det"

        Returns
        -------
        Occam1DData
            Occam1D data object

        """

        occam_data = Occam1DData(self.to_dataframe(), mode=mode)
        if data_filename is not None:
            occam_data.write_data_file(data_filename)

        return occam_data

    def to_simpeg_1d(self, mode: str = "det", **kwargs: Any) -> Simpeg1D:
        """
        Run a 1D inversion using SimPEG.

        Default uses smooth parameters.

        Parameters
        ----------
        mode : str, optional
            Mode for inversion, by default "det"
        **kwargs : dict
            Additional keyword arguments for inversion configuration:
            - p_s : Smoothness parameter (default: smooth inversion)
            - p_z : Depth weighting parameter
            - use_irls : Whether to use IRLS for sharp inversions

        Returns
        -------
        Simpeg1D
            SimPEG 1D inversion object with results

        Examples
        --------
        To run sharp inversion:

        >>> mt_object.to_simpeg_1d({"p_s": 2, "p_z": 0, "use_irls": True})

        To run sharp inversion and compact:

        >>> mt_object.to_simpeg_1d({"p_s": 0, "p_z": 0, "use_irls": True})

        """
        if not self.Z._has_tf_model_error():
            self.compute_model_z_errors()
            self.logger.info("Using default errors for impedance")
        simpeg_1d = Simpeg1D(self.to_dataframe(), mode=mode, **kwargs)
        simpeg_1d.run_fixed_layer_inversion(**kwargs)
        simpeg_1d.plot_model_fitting(fig_num=1)
        simpeg_1d.plot_response(fig_num=2, **kwargs)

        return simpeg_1d
