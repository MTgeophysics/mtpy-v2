# -*- coding: utf-8 -*-
"""
.. module:: MT
   :synopsis: The main container for MT response functions.

.. moduleauthor:: Jared Peacock <jpeacock@usgs.gov>
"""

# =============================================================================
from pathlib import Path
from copy import deepcopy

import numpy as np

from mt_metadata.transfer_functions.core import TF

from mtpy.core.transfer_function import IMPEDANCE_UNITS
from mtpy.core import (
    Z,
    Tipper,
    COORDINATE_REFERENCE_FRAME_OPTIONS,
)
from mtpy.core.mt_location import MTLocation
from mtpy.core.mt_dataframe import MTDataFrame
from mtpy.utils.estimate_tf_quality_factor import EMTFStats

from mtpy.imaging import (
    PlotMTResponse,
    PlotPhaseTensor,
    PlotPenetrationDepth1D,
)
from mtpy.modeling.errors import ModelErrors
from mtpy.modeling.occam1d import Occam1DData
from mtpy.modeling.simpeg.recipes.inversion_1d import Simpeg1D


# =============================================================================
class MT(TF, MTLocation):
    """Basic MT container to hold all information necessary for a MT station
    including the following parameters.

    Impedance and Tipper element nomenclature is E/H therefore the first
    letter represents the output channels and the second letter represents
    the input channels.

    For example for an input of Hx and an output of Ey the impedance tensor
    element is Zyx.

    Coordinate reference frame of the transfer function is by defualt is NED

     - x = North
     - y = East
     - z = + Down

    The other option is ENU

    - x = East
    - y = North
    - z = + Up

    Other input options for the NED are:

        - "+"
        - "z+"
        - "nez+"
        - "ned"
        - "exp(+ i\\omega t)"
        - "exp(+i\\omega t)"
        - None

    And for ENU:

        - "-"
        - "z-"
        - "enz-"
        - "enu"
        - "exp(- i\\omega t)"
        - "exp(-i\\omega t)"

    """

    def __init__(self, fn=None, impedance_units="mt", **kwargs):
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

        self._coordinate_reference_frame_options = (
            COORDINATE_REFERENCE_FRAME_OPTIONS
        )

        self.coordinate_reference_frame = (
            self.station_metadata.transfer_function.sign_convention
        )

        self._impedance_unit_factors = IMPEDANCE_UNITS
        self.impedance_units = impedance_units

        for key, value in kwargs.items():
            setattr(self, key, value)

    def clone_empty(self):
        """Copy metadata but not the transfer function estimates."""

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

    def __deepcopy__(self, memo):
        """Deepcopy function."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["logger"]:
                continue

            setattr(result, k, deepcopy(v, memo))
        result.logger = self.logger
        return result

    def copy(self):
        """Copy function."""
        return deepcopy(self)

    @property
    def coordinate_reference_frame(self):
        f"""Coordinate reference frame of the transfer function
        
        Deafualt is NED

         - x = North
         - y = East
         - z = + down
         
        Options are:
            
            {self._coordinate_reference_frame_options}
        
        """

        return self._coordinate_reference_frame_options[
            self.station_metadata.transfer_function.sign_convention
        ].upper()

    @coordinate_reference_frame.setter
    def coordinate_reference_frame(self, value):
        """set coordinate_reference_frame

        options are NED, ENU

        NED

         - x = North
         - y = East
         - z = + down

        ENU

         - x = East
         - y = North
         - z = + up
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
    def impedance_units(self):
        """impedance units"""
        return self._impedance_units

    @impedance_units.setter
    def impedance_units(self, value):
        """impedance units setter options are [ mt | ohm ]"""
        if not isinstance(value, str):
            raise TypeError("Units input must be a string.")
        if value.lower() not in self._impedance_unit_factors.keys():
            raise ValueError(
                f"{value} is not an acceptable unit for impedance."
            )

        self._impedance_units = value

    @property
    def rotation_angle(self):
        """Rotation angle in degrees from north. In the coordinate reference frame"""
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, theta_r):
        """Set rotation angle in degrees assuming North is 0 measuring clockwise
        positive to East as 90.

        upon setting rotates Z and Tipper

        TODO: figure this out with xarray
        """

        self.rotate(theta_r)
        self._rotation_angle += theta_r

    def rotate(self, theta_r, inplace=True):
        """Rotate the data in degrees assuming North is 0 measuring clockwise
        positive to East as 90.

        :param theta_r: rotation angle to rotate by in degrees.
        :type theta_r: float
        :param inplace: rotate all transfer function in place, defaults to True.
        :type inplace: bool, optional
        :return: if inplace is False, returns a new MT object.
        :rtype: MT object

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
    def Z(self):
        r"""Mtpy.core.z.Z object to hold impedance tensor."""

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
    def Z(self, z_object):
        """Set z_object

        recalculate phase tensor and invariants, which shouldn't change except
        for strike angle.

        Be sure to have appropriate units set
        """
        # if a z object is given the underlying data is in mt units, even
        # if the units are set to ohm.
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
    def Tipper(self):
        """Mtpy.core.z.Tipper object to hold tipper information."""

        if self.has_tipper():
            return Tipper(
                tipper=self.tipper.to_numpy(),
                tipper_error=self.tipper_error.to_numpy(),
                frequency=self.frequency,
                tipper_model_error=self.tipper_model_error.to_numpy(),
            )

    @Tipper.setter
    def Tipper(self, t_object):
        """Set tipper object

        recalculate tipper angle and magnitude.
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
        r"""Mtpy.analysis.pt.PhaseTensor object to hold phase tenso."""
        return self.Z.phase_tensor

    @property
    def ex_metadata(self):
        """EX metadata."""
        return self.station_metadata.runs[0].ex

    @ex_metadata.setter
    def ex_metadata(self, value):
        """Set EX metadata."""
        self.station_metadata.runs[0].ex = value

    @property
    def ey_metadata(self):
        """EY metadata."""
        return self.station_metadata.runs[0].ey

    @ey_metadata.setter
    def ey_metadata(self, value):
        """Set EY metadata."""
        self.station_metadata.runs[0].ey = value

    @property
    def hx_metadata(self):
        """HX metadata."""
        return self.station_metadata.runs[0].hx

    @hx_metadata.setter
    def hx_metadata(self, value):
        """Set hx metadata."""
        self.station_metadata.runs[0].hx = value

    @property
    def hy_metadata(self):
        """HY metadata."""
        return self.station_metadata.runs[0].hy

    @hy_metadata.setter
    def hy_metadata(self, value):
        """Set hy metadata."""
        self.station_metadata.runs[0].hy = value

    @property
    def hz_metadata(self):
        """HZ metadata."""
        return self.station_metadata.runs[0].hz

    @hz_metadata.setter
    def hz_metadata(self, value):
        """Set hz metadata."""
        self.station_metadata.runs[0].hz = value

    @property
    def rrhx_metadata(self):
        """RRHX metadata."""
        return self.station_metadata.runs[0].rrhx

    @property
    def rrhy_metadata(self):
        """RRHY metadata."""
        return self.station_metadata.runs[0].rrhy

    def estimate_tf_quality(
        self,
        weights={
            "bad": 0.35,
            "corr": 0.2,
            "diff": 0.2,
            "std": 0.2,
            "fit": 0.05,
        },
        round_qf=False,
    ):
        """Estimate tranfer function quality factor 0-5, 5 being the best.
        :param weights: DESCRIPTION, defaults to
            {
            "bad": 0.35,
            "corr": 0.2,
            "diff": 0.2,
            "std": 0.2,
            "fit": 0.05,
            }.
        :type weights: TYPE, optional
        :param: DESCRIPTION.
        :type: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
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
        self, n_frequencies=None, comp="det", only_2d=False, inplace=False
    ):
        """Remove distortion following Bibby et al. [2005].
        :param inplace:
            Defaults to False.
        :param only_2d:
            Defaults to False.
        :param comp:
            Defaults to "det".
        :param n_frequencies: Number of frequencies to look for distortion from the
            highest frequency, defaults to None.
        :type n_frequencies: int, optional
        :return s: Distortion matrix.
        :rtype s: np.ndarray(2, 2, dtype=real)
        :return s: Z with distortion removed.
        :rtype s: mtpy.core.z.Z
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

    def remove_static_shift(self, ss_x=1.0, ss_y=1.0, inplace=False):
        """Remove static shift from the apparent resistivity

        Assume the original observed tensor Z is built by a static shift S
        and an unperturbated "correct" Z0 :

             * Z = S * Z0

        therefore the correct Z will be :
            * Z0 = S^(-1) * Z.
        :param inplace:
            Defaults to False.
        :param ss_x: Correction factor for x component, defaults to 1.0.
        :type ss_x: float, optional
        :param ss_y: Correction factor for y component, defaults to 1.0.
        :type ss_y: float, optional
        :return s: New Z object with static shift removed.
        :rtype s: mtpy.core.z.Z
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
        new_period,
        method="slinear",
        bounds_error=True,
        f_type="period",
        **kwargs,
    ):
        """Interpolate the impedance tensor onto different frequencies.
        :param z_log_space:
            Defaults to False.
        :param new_period: A 1-d array of frequencies to interpolate on
            to.  Must be with in the bounds of the existing frequency range,
            anything outside and an error will occur.
        :type new_period: np.ndarray
        :param method: Method to interpolate by, defaults to "slinear".
        :type method: string, optional
        :param bounds_error: Check for if input frequencies are within the
            original frequencies, defaults to True.
        :type bounds_error: boolean, optional
        :param f_type: Frequency type can be [ 'frequency' | 'period' ], defaults to "period".
        :type f_type: string, defaults to 'period', optional
        :param **kwargs: Key word arguments for `interp`.
        :type **kwargs: dictionary
        :raises ValueError: If input frequencies are out of bounds.
        :return: New MT object with interpolated values.
        :rtype: :class:`mtpy.core.MT`
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
            new_m.Tipper = self.Tipper.interpolate(
                new_period, method=method, **kwargs
            )
            if new_m.has_tipper():
                if np.all(np.isnan(new_m.Tipper.tipper)):
                    self.logger.warning(
                        f"Station {self.station}: Interpolated T values are all NaN, "
                        "consider an alternative interpolation method. "
                        "See scipy.interpolate.interp1d for more information."
                    )

        return new_m

    def plot_mt_response(self, **kwargs):
        """Returns a mtpy.imaging.plotresponse.PlotResponse object

        :Plot Response: ::

            >>> mt_obj = mt.MT(edi_file)
            >>> pr = mt.plot_mt_response()
            >>> # if you need more info on plot_mt_response
            >>> help(pr).
        """

        plot_obj = PlotMTResponse(
            z_object=self.Z,
            t_object=self.Tipper,
            pt_obj=self.pt,
            station=self.station,
            **kwargs,
        )

        return plot_obj

    def plot_phase_tensor(self, **kwargs):
        """Plot phase tensor.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        kwargs["ellipse_size"] = 0.5
        return PlotPhaseTensor(self.pt, station=self.station, **kwargs)

    def plot_depth_of_penetration(self, **kwargs):
        """Plot Depth of Penetration estimated from Niblett-Bostick estimation.
        :param **kwargs: DESCRIPTION.
        :type **kwargs: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        return PlotPenetrationDepth1D(self, **kwargs)

    def to_dataframe(self, utm_crs=None, cols=None, impedance_units="mt"):
        """Create a dataframe from the transfer function for use with plotting
        and modeling.
        :param cols:
            Defaults to None.
        :param utm_crs:
            Defaults to None.
        :param eter utm_crs: The utm zone to project station to, could be a
            name, pyproj.CRS, EPSG number, or anything that pyproj.CRS can intake.
        :type eter utm_crs: string, int, :class:`pyproj.CRS`
        :param impedance_units: ["mt" [mV/km/nT] | "ohm" [Ohms] ]
        :type impedance_units: str
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

    def from_dataframe(self, mt_df, impedance_units="mt"):
        """Fill transfer function attributes from a dataframe for a single station.
        :param mt_df:
        :param df: DESCRIPTION.
        :type df: TYPE
        :param impedance_units: ["mt" [mV/km/nT] | "ohm" [Ohms] ]
        :type impedance_units: str
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
        self, error_value=5, error_type="geometric_mean", floor=True
    ):
        """Compute mode errors based on the error type

        ========================== ===========================================
        key                        definition
        ========================== ===========================================
        egbert                     error_value * sqrt(Zxy * Zyx)
        geometric_mean             error_value * sqrt(Zxy * Zyx)
        arithmetic_mean            error_value * (Zxy + Zyx) / 2
        mean_od                    error_value * (Zxy + Zyx) / 2
        off_diagonals              zxx_error == zxy_error, zyx_error == zyy_error
        median                     error_value * median(z)
        eigen                      error_value * mean(eigen(z))
        percent                    error_value * z
        absolute                   error_value
        ========================== ===========================================.
        :param error_value: DESCRIPTION, defaults to 5.
        :type error_value: TYPE, optional
        :param error_type: DESCRIPTION, defaults to "geometric_mean".
        :type error_type: TYPE, optional
        :param floor: DESCRIPTION, defaults to True.
        :type floor: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
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
        self, error_value=0.02, error_type="absolute", floor=False
    ):
        """Compute mode errors based on the error type

        ========================== ===========================================
        key                        definition
        ========================== ===========================================
        percent                    error_value * t
        absolute                   error_value
        ========================== ===========================================.
        :param error_value: DESCRIPTION02, defaults to 0.02.
        :type error_value: TYPE, optional
        :param error_type: DESCRIPTION, defaults to "absolute".
        :type error_type: TYPE, optional
        :param floor: DESCRIPTION, defaults to False.
        :type floor: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
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

    def add_model_error(self, comp=[], z_value=5, t_value=0.05, periods=None):
        """Add error to a station's components for given period range.
        :param periods:
            Defaults to None.
        :param t_value:
            Defaults to 0.05.
        :param z_value:
            Defaults to 5.
        :param station: Name of station(s) to add error to.
        :type station: string or list of strings
        :param comp: List of components to add data to, valid components are, defaults to [].
        :return: Data array with added errors.
        :rtype: np.ndarray
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

    def find_flipped_phase(self):
        """Identify if the off-diagonal components are flipped from traditional
        quadrants.  xy should be in the 1st quadarant (0-90 deg) and yx
        should be in the 3rd quadrant (-180 to -90 deg)
        :return: A dictionary of components with a bool for flipped or not
            if flipped return value is True.
        :rtype: dict
        """

        flip_dict = {"zxy": False, "zyx": False}

        if self.Z.phase_xy.mean() < 0:
            flip_dict["zxy"] = True

        if self.Z.phase_yx.mean() > -90:
            flip_dict["zyx"] = True

        return flip_dict

    def flip_phase(
        self,
        zxx=False,
        zxy=False,
        zyx=False,
        zyy=False,
        tzx=False,
        tzy=False,
        inplace=False,
    ):
        """Flip the phase of a station in case its plotting in the wrong quadrant.
        :param inplace:
            Defaults to False.
        :param tzy:
            Defaults to False.
        :param tzx:
            Defaults to False.
        :param station: Name(s) of station to flip phase.
        :type station: string or list of strings
        :param station: Station name or list of station names.
        :type station: string or list
        :param zxx: Z_xx, defaults to False.
        :type zxx: TYPE, optional
        :param zxy: Z_xy, defaults to False.
        :type zxy: TYPE, optional
        :param zyy: Z_yx, defaults to False.
        :type zyy: TYPE, optional
        :param zyx: Z_yy, defaults to False.
        :type zyx: TYPE, optional
        :param tx: T_zx, defaults to False.
        :type tx: TYPE, optional
        :param ty: T_zy, defaults to False.
        :type ty: TYPE, optional
        :return: New_data.
        :rtype: np.ndarray
        :return: New mt_dict with components removed.
        :rtype: dictionary
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
        zxx=False,
        zxy=False,
        zyy=False,
        zyx=False,
        tzx=False,
        tzy=False,
        inplace=False,
    ):
        """Remove a component for a given station(s).
        :param inplace:
            Defaults to False.
        :param tzy:
            Defaults to False.
        :param tzx:
            Defaults to False.
        :param station: Station name or list of station names.
        :type station: string or list
        :param zxx: Z_xx, defaults to False.
        :type zxx: TYPE, optional
        :param zxy: Z_xy, defaults to False.
        :type zxy: TYPE, optional
        :param zyy: Z_yx, defaults to False.
        :type zyy: TYPE, optional
        :param zyx: Z_yy, defaults to False.
        :type zyx: TYPE, optional
        :param tx: T_zx, defaults to False.
        :type tx: TYPE, optional
        :param ty: T_zy, defaults to False.
        :type ty: TYPE, optional
        :return: New data array with components removed.
        :rtype: np.ndarray
        :return: New mt_dict with components removed.
        :rtype: dictionary
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

    def add_white_noise(self, value, inplace=True):
        """Add white noise to the data, useful for synthetic tests.
        :param value: DESCRIPTION.
        :type value: TYPE
        :param inplace: DESCRIPTION, defaults to True.
        :type inplace: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
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
            ] = self._transfer_function.transfer_function.real * (
                noise_real
            ) + (
                1j * self._transfer_function.transfer_function.imag * noise_imag
            )

            self._transfer_function["transfer_function_error"] = (
                self._transfer_function.transfer_function_error + value
            )

        else:
            new_mt_obj._transfer_function = self._transfer_function.copy()
            new_mt_obj._transfer_function[
                "transfer_function"
            ] = self._transfer_function.transfer_function.real * (
                noise_real
            ) + (
                1j * self._transfer_function.transfer_function.imag * noise_imag
            )

            self._transfer_function["transfer_function_error"] = (
                self._transfer_function.transfer_function_error + value
            )
            return new_mt_obj

    def edit_curve(self, method="default", tolerance=0.05):
        """
        try to remove bad points in a scientific way.

        """

        # bring up a gui of some sort.

    def to_occam1d(self, data_filename=None, mode="det"):
        """Write an Occam1DData data file.
        :param data_filename: Path to write file, if None returns Occam1DData
            object, defaults to None.
        :type data_filename: string or Path, optional
        :param mode: [ 'te', 'tm', 'det', 'tez', 'tmz', 'detz'], defaults to "det".
        :type mode: string, optional
        :return: Occam1DData object.
        :rtype: :class:`mtpy.modeling.occam1d.Occam1DData`
        """

        occam_data = Occam1DData(self.to_dataframe(), mode=mode)
        if data_filename is not None:
            occam_data.write_data_file(data_filename)

        return occam_data

    def to_simpeg_1d(self, mode="det", **kwargs):
        """Helper method to run a 1D inversion using Simpeg

        default is smooth parameters

        :To run sharp inversion:

        >>> mt_object.to_simpeg_1d({"p_s": 2, "p_z": 0, "use_irls": True})

        :To run sharp inversion adn compact:

        >>> mt_object.to_simpeg_1d({"p_s": 0, "p_z": 0, "use_irls": True}).
        :param mode:
            Defaults to "det".
        :param **kwargs: DESCRIPTION.
        :type **kwargs: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if not self.Z._has_tf_model_error():
            self.compute_model_z_errors()
            self.logger.info("Using default errors for impedance")
        simpeg_1d = Simpeg1D(self.to_dataframe(), mode=mode, **kwargs)
        simpeg_1d.run_fixed_layer_inversion(**kwargs)
        simpeg_1d.plot_model_fitting(fig_num=1)
        simpeg_1d.plot_response(fig_num=2, **kwargs)

        return simpeg_1d
