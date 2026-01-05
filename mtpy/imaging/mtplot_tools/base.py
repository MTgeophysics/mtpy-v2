# -*- coding: utf-8 -*-
"""
Base classes for plotting classes

:author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import interpolate, stats

from .map_interpolation_tools import interpolate_to_map
from .plot_settings import PlotSettings
from .plotters import add_raster


# =============================================================================
# Base
# =============================================================================


class PlotBase(PlotSettings):
    """
    Base class for plotting objects.

    Provides core plotting functionality including figure management,
    saving, updating, and redrawing plots.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments passed to PlotSettings parent class

    Attributes
    ----------
    logger : loguru.Logger
        Logger instance for the class

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.logger = logger

        self._basename = self.__class__.__name__.lower()

    def __str__(self) -> str:
        """
        Return string representation of the plotting object.

        Returns
        -------
        str
            String describing the plotting class

        """

        return f"Plotting {self.__class__.__name__}"

    def __repr__(self) -> str:
        """
        Return repr representation of the plotting object.

        Returns
        -------
        str
            String describing the plotting class

        """
        return self.__str__()

    def _set_subplot_params(self) -> None:
        """
        Set matplotlib subplot parameters from instance attributes.

        Sets font size and subplot spacing parameters including bottom,
        top, left, right margins, and optional wspace/hspace.

        """
        # set some parameters of the figure and subplot spacing
        plt.rcParams["font.size"] = self.font_size
        plt.rcParams["figure.subplot.bottom"] = self.subplot_bottom
        plt.rcParams["figure.subplot.top"] = self.subplot_top
        plt.rcParams["figure.subplot.left"] = self.subplot_left
        plt.rcParams["figure.subplot.right"] = self.subplot_right

        if self.subplot_wspace is not None:
            plt.rcParams["figure.subplot.wspace"] = self.subplot_wspace
        if self.subplot_hspace is not None:
            plt.rcParams["figure.subplot.hspace"] = self.subplot_hspace

    def plot(self) -> None:
        """
        Create the plot.

        This method should be overridden by subclasses to implement
        specific plotting functionality.

        """

    def save_plot(
        self,
        save_fn: str | Path,
        file_format: str = "pdf",
        orientation: str = "portrait",
        fig_dpi: int | None = None,
        close_plot: bool = True,
    ) -> None:
        """
        Save the figure to a file.

        Parameters
        ----------
        save_fn : str | Path
            Full path to save figure to. Can be:
            - Directory path: file will be saved as save_fn/basename.file_format
            - Full path: file will be saved to the given path, format inferred
              from extension
        file_format : str, optional
            File format for saved figure (pdf, eps, jpg, png, svg),
            by default 'pdf'
        orientation : str, optional
            Page orientation ('landscape' or 'portrait'), by default 'portrait'
        fig_dpi : int | None, optional
            Resolution in dots-per-inch. If None, uses the figure's dpi,
            by default None
        close_plot : bool, optional
            Whether to close the plot after saving, by default True

        Examples
        --------
        >>> # Save plot as jpg
        >>> p1.save_plot(r'/home/MT/figures', file_format='jpg')

        """

        if fig_dpi is None:
            fig_dpi = self.fig_dpi
        save_fn = Path(save_fn)
        if not save_fn.is_dir():
            file_format = save_fn.suffix[1:]
        else:
            save_fn = save_fn.joinpath(f"{self._basename}.{file_format}")
        self.fig.savefig(
            save_fn, dpi=fig_dpi, format=file_format, orientation=orientation
        )

        if close_plot:
            plt.close(self.fig)
        else:
            pass
        self.fig_fn = save_fn
        self.logger.info(f"Saved figure to: {self.fig_fn}")

    def update_plot(self) -> None:
        """
        Update the plot after changing figure or axes properties.

        Uses matplotlib's canvas draw method to refresh the display
        after modifying figure or axes attributes.

        Examples
        --------
        >>> [ax.grid(True, which='major') for ax in [p1.axr, p1.axp]]
        >>> p1.update_plot()

        """

        self.fig.canvas.draw()

    def redraw_plot(self) -> None:
        """
        Recreate the plot after updating attributes.

        Closes the current figure and calls plot() to create a new one
        with updated attributes.

        Examples
        --------
        >>> # Change the color and marker of the xy components
        >>> p1.xy_color = (0.5, 0.5, 0.9)
        >>> p1.xy_marker = '*'
        >>> p1.redraw_plot()

        """

        plt.close(self.fig)
        self.plot()


class PlotBaseMaps(PlotBase):
    """
    Base object for plot classes that use map views.

    Includes methods for interpolation of data onto map grids.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments passed to PlotBase parent class and for setting
        interpolation parameters

    Attributes
    ----------
    cell_size : float
        Size of grid cells for interpolation, by default 0.002
    n_padding_cells : int
        Number of padding cells around data extent, by default 10
    interpolation_method : str
        Interpolation method ('delaunay', 'linear', 'nearest', 'cubic'),
        by default 'delaunay'
    interpolation_power : int
        Power parameter for inverse distance weighting, by default 5
    nearest_neighbors : int
        Number of nearest neighbors to use in interpolation, by default 7

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.cell_size = 0.002
        self.n_padding_cells = 10
        self.interpolation_method = "delaunay"
        self.interpolation_power = 5
        self.nearest_neighbors = 7

        for key, value in kwargs.items():
            setattr(self, key, value)

    def interpolate_to_map(self, plot_array, component: str):
        """
        Interpolate data points onto a 2D map grid.

        Parameters
        ----------
        plot_array : np.ndarray
            Array containing data to interpolate
        component : str
            Name of the component being interpolated

        Returns
        -------
        tuple
            Interpolated grid data and coordinates

        """

        return interpolate_to_map(
            plot_array,
            component,
            cell_size=self.cell_size,
            n_padding_cells=self.n_padding_cells,
            interpolation_method=self.interpolation_method,
            interpolation_power=self.interpolation_power,
            nearest_neighbors=self.nearest_neighbors,
        )

    @staticmethod
    def get_interp1d_functions_z(tf, interp_type: str = "slinear") -> dict | None:
        """
        Create 1D interpolation functions for impedance tensor components.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object containing impedance data
        interp_type : str, optional
            Type of interpolation ('linear', 'slinear', 'cubic'),
            by default 'slinear'

        Returns
        -------
        dict | None
            Dictionary containing interpolation functions for each impedance
            component (zxx, zxy, zyx, zyy) with 'real', 'imag', 'err', and
            'model_err' sub-keys. Returns None if no Z data available.

        """
        if tf.Z is None:
            return None

        # interpolate the impedance tensor
        zmap = {0: "x", 1: "y"}
        interp_dict = {}
        for ii in range(2):
            for jj in range(2):
                comp = f"z{zmap[ii]}{zmap[jj]}"
                interp_dict[comp] = {}
                # need to look out for zeros in the impedance
                # get the indicies of non-zero components
                nz_index = np.nonzero(tf.Z.z[:, ii, jj])

                if len(nz_index[0]) == 0:
                    continue
                # get the non-zero components
                z_real = tf.Z.z[nz_index, ii, jj].real
                z_imag = tf.Z.z[nz_index, ii, jj].imag

                # get the frequencies of non-zero components
                f = tf.Z.frequency[nz_index]

                # create a function that does 1d interpolation
                interp_dict[comp]["real"] = interpolate.interp1d(
                    f, z_real, kind=interp_type
                )
                interp_dict[comp]["imag"] = interpolate.interp1d(
                    f, z_imag, kind=interp_type
                )

                if tf.Z._has_tf_error():
                    z_error = tf.Z.z_error[nz_index, ii, jj]
                    interp_dict[comp]["err"] = interpolate.interp1d(
                        f, z_error, kind=interp_type
                    )
                else:
                    interp_dict[comp]["err"] = None
                if tf.Z._has_tf_model_error():
                    z_model_error = tf.Z.z_model_error[nz_index, ii, jj]
                    interp_dict[comp]["model_err"] = interpolate.interp1d(
                        f, z_model_error, kind=interp_type
                    )
                else:
                    interp_dict[comp]["model_err"] = None

        return interp_dict

    @staticmethod
    def get_interp1d_functions_t(tf, interp_type: str = "slinear") -> dict | None:
        """
        Create 1D interpolation functions for tipper components.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object containing tipper data
        interp_type : str, optional
            Type of interpolation ('linear', 'slinear', 'cubic'),
            by default 'slinear'

        Returns
        -------
        dict | None
            Dictionary containing interpolation functions for tipper
            components (tzx, tzy) with 'real', 'imag', 'err', and
            'model_err' sub-keys. Returns None if no Tipper data available.

        """
        if tf.Tipper is None:
            return None

        # interpolate the impedance tensor
        zmap = {0: "x", 1: "y"}
        interp_dict = {}
        for jj in range(2):
            comp = f"tz{zmap[jj]}"
            interp_dict[comp] = {}
            # need to look out for zeros in the impedance
            # get the indicies of non-zero components
            nz_index = np.nonzero(tf.Tipper.tipper[:, 0, jj])

            if len(nz_index[0]) == 0:
                continue
            # get the non-zero components
            t_real = tf.Tipper.tipper[nz_index, 0, jj].real
            t_imag = tf.Tipper.tipper[nz_index, 0, jj].imag

            # get the frequencies of non-zero components
            f = tf.Tipper.frequency[nz_index]

            # create a function that does 1d interpolation
            interp_dict[comp]["real"] = interpolate.interp1d(
                f, t_real, kind=interp_type
            )
            interp_dict[comp]["imag"] = interpolate.interp1d(
                f, t_imag, kind=interp_type
            )

            if tf.Tipper._has_tf_error():
                t_err = tf.Tipper.tipper_error[nz_index, 0, jj]
                interp_dict[comp]["err"] = interpolate.interp1d(
                    f, t_err, kind=interp_type
                )
            else:
                interp_dict[comp]["err"] = None

            if tf.Tipper._has_tf_model_error():
                t_model_err = tf.Tipper.tipper_model_error[nz_index, 0, jj]
                interp_dict[comp]["model_err"] = interpolate.interp1d(
                    f, t_model_err, kind=interp_type
                )
            else:
                interp_dict[comp]["model_err"] = None

        return interp_dict

    def _get_interpolated_z(self, tf) -> np.ndarray:
        """
        Get interpolated impedance tensor at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Complex impedance tensor array of shape (1, 2, 2) at the
            specified plot period

        """
        if not hasattr(tf, "z_interp_dict"):
            tf.z_interp_dict = self.get_interp1d_functions_z(tf)
        return np.nan_to_num(
            np.array(
                [
                    [
                        tf.z_interp_dict["zxx"]["real"](1 / self.plot_period)[0]
                        + 1j
                        * tf.z_interp_dict["zxx"]["imag"](1.0 / self.plot_period)[0],
                        tf.z_interp_dict["zxy"]["real"](1.0 / self.plot_period)[0]
                        + 1j
                        * tf.z_interp_dict["zxy"]["imag"](1.0 / self.plot_period)[0],
                    ],
                    [
                        tf.z_interp_dict["zyx"]["real"](1.0 / self.plot_period)[0]
                        + 1j
                        * tf.z_interp_dict["zyx"]["imag"](1.0 / self.plot_period)[0],
                        tf.z_interp_dict["zyy"]["real"](1.0 / self.plot_period)[0]
                        + 1j
                        * tf.z_interp_dict["zyy"]["imag"](1.0 / self.plot_period)[0],
                    ],
                ]
            )
        ).reshape((1, 2, 2))

    def _get_interpolated_z_error(self, tf) -> np.ndarray:
        """
        Get interpolated impedance tensor error at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Impedance tensor error array of shape (1, 2, 2) at the
            specified plot period. Returns zeros if no error data available.

        """
        if not hasattr(tf, "z_interp_dict"):
            tf.z_interp_dict = self.get_interp1d_functions_z(tf)
        if tf.z_interp_dict["zxy"]["err"] is not None:
            return np.nan_to_num(
                np.array(
                    [
                        [
                            tf.z_interp_dict["zxx"]["err"](1.0 / self.plot_period)[0],
                            tf.z_interp_dict["zxy"]["err"](1.0 / self.plot_period)[0],
                        ],
                        [
                            tf.z_interp_dict["zyx"]["err"](1.0 / self.plot_period)[0],
                            tf.z_interp_dict["zyy"]["err"](1.0 / self.plot_period)[0],
                        ],
                    ]
                )
            ).reshape((1, 2, 2))
        else:
            return np.zeros((1, 2, 2), dtype=float)

    def _get_interpolated_z_model_error(self, tf) -> np.ndarray:
        """
        Get interpolated impedance tensor model error at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Impedance tensor model error array of shape (1, 2, 2) at the
            specified plot period. Returns zeros if no model error data available.

        """
        if not hasattr(tf, "z_interp_dict"):
            tf.z_interp_dict = self.get_interp1d_functions_z(tf)
        if tf.z_interp_dict["zxy"]["model_err"] is not None:
            return np.nan_to_num(
                np.array(
                    [
                        [
                            tf.z_interp_dict["zxx"]["model_err"](
                                1.0 / self.plot_period
                            )[0],
                            tf.z_interp_dict["zxy"]["model_err"](
                                1.0 / self.plot_period
                            )[0],
                        ],
                        [
                            tf.z_interp_dict["zyx"]["model_err"](
                                1.0 / self.plot_period
                            )[0],
                            tf.z_interp_dict["zyy"]["model_err"](
                                1.0 / self.plot_period
                            )[0],
                        ],
                    ]
                )
            ).reshape((1, 2, 2))
        else:
            return np.zeros((1, 2, 2), dtype=float)

    def _get_interpolated_t(self, tf) -> np.ndarray:
        """
        Get interpolated tipper at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Complex tipper array of shape (1, 1, 2) at the specified
            plot period. Returns zeros if no tipper data available.

        """
        if not hasattr(tf, "t_interp_dict"):
            tf.t_interp_dict = self.get_interp1d_functions_t(tf)
        if not tf.has_tipper():
            return np.zeros((1, 1, 2), dtype=complex)
        return np.nan_to_num(
            np.array(
                [
                    [
                        [
                            tf.t_interp_dict["tzx"]["real"](1.0 / self.plot_period)[0]
                            + 1j
                            * tf.t_interp_dict["tzx"]["imag"](1.0 / self.plot_period)[
                                0
                            ],
                            tf.t_interp_dict["tzy"]["real"](1.0 / self.plot_period)[0]
                            + 1j
                            * tf.t_interp_dict["tzy"]["imag"](1.0 / self.plot_period)[
                                0
                            ],
                        ]
                    ]
                ]
            )
        ).reshape((1, 1, 2))

    def _get_interpolated_t_err(self, tf):
        """Get interpolated t err.
        :param tf: DESCRIPTION.
        :type tf: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if not hasattr(tf, "t_interp_dict"):
            tf.t_interp_dict = self.get_interp1d_functions_t(tf)

        if not tf.has_tipper():
            return np.zeros((1, 1, 2), dtype=float)
        if tf.Tipper._has_tf_error():
            return np.nan_to_num(
                np.array(
                    [
                        [
                            [
                                tf.t_interp_dict["tzx"]["err"](1.0 / self.plot_period)[
                                    0
                                ],
                                tf.t_interp_dict["tzy"]["err"](1.0 / self.plot_period)[
                                    0
                                ],
                            ]
                        ]
                    ]
                )
            ).reshape((1, 1, 2))
        else:
            return np.zeros((1, 1, 2), dtype=float)

    def _get_interpolated_t_model_err(self, tf) -> np.ndarray:
        """
        Get interpolated tipper model error at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Tipper model error array of shape (1, 1, 2) at the specified
            plot period. Returns zeros if no tipper model error data available.

        """
        if not hasattr(tf, "t_interp_dict"):
            tf.t_interp_dict = self.get_interp1d_functions_t(tf)

        if not tf.has_tipper():
            return np.zeros((1, 1, 2), dtype=float)
        if tf.Tipper._has_tf_error():
            return np.nan_to_num(
                np.array(
                    [
                        [
                            [
                                tf.t_interp_dict["tzx"]["model_err"](
                                    1.0 / self.plot_period
                                )[0],
                                tf.t_interp_dict["tzy"]["model_err"](
                                    1.0 / self.plot_period
                                )[0],
                            ]
                        ]
                    ]
                )
            ).reshape((1, 1, 2))
        else:
            return np.zeros((1, 1, 2), dtype=float)

    def add_raster(
        self, ax, raster_fn: str | Path, add_colorbar: bool = True, **kwargs
    ):
        """
        Add a raster image to a matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axis to add raster to
        raster_fn : str | Path
            Path to raster file (readable by rasterio)
        add_colorbar : bool, optional
            Whether to add a colorbar, by default True
        **kwargs : dict
            Additional keyword arguments passed to rasterio plotting

        Returns
        -------
        matplotlib image or collection
            The raster plot object

        """

        return add_raster(ax, raster_fn, add_colorbar=add_colorbar, **kwargs)


class PlotBaseProfile(PlotBase):
    """
    Base object for profile plots like pseudo sections.

    Provides functionality for creating profile views of MT data along
    a linear transect.

    Parameters
    ----------
    tf_list : list or MTCollection
        List of transfer function objects or MTCollection
    **kwargs : dict
        Additional keyword arguments for PlotBase parent class and
        profile settings

    Attributes
    ----------
    mt_data : list or MTCollection
        MT data to plot
    profile_vector : array-like | None
        Profile direction vector
    profile_angle : float | None
        Profile angle in degrees
    profile_line : tuple | None
        Profile line parameters (slope, intercept)
    profile_reverse : bool
        Whether to reverse profile direction, by default False
    x_stretch : float
        Horizontal stretching factor for profile, by default 5000
    y_stretch : float
        Vertical stretching factor for profile, by default 1000
    y_scale : str
        Y-axis scale type ('period' or 'frequency'), by default 'period'

    """

    def __init__(self, tf_list, **kwargs) -> None:
        super().__init__(**kwargs)

        self.mt_data = tf_list
        self.profile_vector = None
        self.profile_angle = None
        self.profile_line = None
        self.profile_reverse = False

        self.x_stretch = 5000
        self.y_stretch = 1000
        self.y_scale = "period"

        self._rotation_angle = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

    # ---need to rotate data on setting rotz
    @property
    def rotation_angle(self) -> float:
        """
        Get rotation angle for data.

        Returns
        -------
        float
            Rotation angle in degrees

        """
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value: float) -> None:
        """
        Set rotation angle for all transfer functions.

        Parameters
        ----------
        value : float
            Rotation angle in degrees to apply to all data

        """
        for tf in self.mt_data:
            tf.rotation_angle = value
        self._rotation_angle = value

    def _get_profile_line(
        self, x: np.ndarray | None = None, y: np.ndarray | None = None
    ) -> None:
        """
        Calculate profile line using linear regression through data points.

        Determines the best-fit line through station locations and projects
        all stations onto this profile line.

        Parameters
        ----------
        x : np.ndarray | None, optional
            X coordinates of stations. If None, uses longitude from mt_data,
            by default None
        y : np.ndarray | None, optional
            Y coordinates of stations. If None, uses latitude from mt_data,
            by default None

        Raises
        ------
        ValueError
            If only one of x or y is provided

        """

        if np.any(self.mt_data.station_locations.profile_offset != 0):
            return

        if x is None and y is None:
            x = np.zeros(self.mt_data.n_stations)
            y = np.zeros(self.mt_data.n_stations)

            for ii, tf in enumerate(self.mt_data.values()):
                x[ii] = tf.longitude
                y[ii] = tf.latitude

        elif x is None or y is None:
            raise ValueError("get_profile")

        # check regression for 2 profile orientations:
        # horizontal (N=N(E)) or vertical(E=E(N))
        # use the one with the lower standard deviation
        profile1 = stats.linregress(x, y)
        profile2 = stats.linregress(y, x)
        # if the profile is rather E=E(N), the parameters have to converted
        # into N=N(E) form:
        if profile2.stderr < profile1.stderr:
            self.profile_line = (
                1.0 / profile2.slope,
                -profile2.intercept / profile2.slope,
            )
        else:
            self.profile_line = profile1[:2]

        for mt_obj in self.mt_data.values():
            mt_obj.project_onto_profile_line(self.profile_line[0], self.profile_line[1])

    def _get_offset(self, tf) -> float:
        """
        Get approximate offset distance for a station along the profile.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with profile_offset attribute

        Returns
        -------
        float
            Scaled offset distance along profile, negative if profile_reverse
            is True

        """

        direction = 1
        if self.profile_reverse:
            direction = -1

        return direction * tf.profile_offset * self.x_stretch

    def _get_interpolated_z(self, tf) -> np.ndarray:
        """
        Get interpolated impedance tensor at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Complex impedance tensor array of shape (2, 2) at the
            specified plot period

        """
        if not hasattr(tf, "z_interp_dict"):
            tf.z_interp_dict = self.get_interp1d_functions_z(tf)
        return np.nan_to_num(
            np.array(
                [
                    [
                        tf.z_interp_dict["zxx"]["real"](1 / self.plot_period)
                        + 1j * tf.z_interp_dict["zxx"]["imag"](1.0 / self.plot_period),
                        tf.z_interp_dict["zxy"]["real"](1.0 / self.plot_period)
                        + 1j * tf.z_interp_dict["zxy"]["imag"](1.0 / self.plot_period),
                    ],
                    [
                        tf.z_interp_dict["zyx"]["real"](1.0 / self.plot_period)
                        + 1j * tf.z_interp_dict["zyx"]["imag"](1.0 / self.plot_period),
                        tf.z_interp_dict["zyy"]["real"](1.0 / self.plot_period)
                        + 1j * tf.z_interp_dict["zyy"]["imag"](1.0 / self.plot_period),
                    ],
                ]
            )
        )

    def _get_interpolated_z_error(self, tf) -> np.ndarray:
        """
        Get interpolated impedance tensor error at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Impedance tensor error array of shape (2, 2) at the
            specified plot period

        """
        if not hasattr(tf, "z_interp_dict"):
            tf.z_interp_dict = self.get_interp1d_functions_z(tf)
        if tf.z_interp_dict["zxy"]["err"] is not None:
            return np.nan_to_num(
                np.array(
                    [
                        [
                            tf.z_interp_dict["zxx"]["err"](1.0 / self.plot_period),
                            tf.z_interp_dict["zxy"]["err"](1.0 / self.plot_period),
                        ],
                        [
                            tf.z_interp_dict["zyx"]["err"](1.0 / self.plot_period),
                            tf.z_interp_dict["zyy"]["err"](1.0 / self.plot_period),
                        ],
                    ]
                )
            )

    def _get_interpolated_t(self, tf) -> np.ndarray:
        """
        Get interpolated tipper at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Complex tipper array of shape (1, 1, 2) at the specified
            plot period. Returns zeros if no tipper dictionary available.

        """

        if tf.t_interp_dict == {}:
            return np.zeros((1, 1, 2), dtype=complex)
        return np.nan_to_num(
            np.array(
                [
                    [
                        [
                            tf.t_interp_dict["tzx"]["real"](1.0 / self.plot_period)
                            + 1j
                            * tf.t_interp_dict["tzx"]["imag"](1.0 / self.plot_period),
                            tf.t_interp_dict["tzy"]["real"](1.0 / self.plot_period)
                            + 1j
                            * tf.t_interp_dict["tzy"]["imag"](1.0 / self.plot_period),
                        ]
                    ]
                ]
            )
        )

    def _get_interpolated_t_err(self, tf) -> np.ndarray:
        """
        Get interpolated tipper error at plot period.

        Parameters
        ----------
        tf : MT or Transfer Function object
            Transfer function object with interpolation functions

        Returns
        -------
        np.ndarray
            Tipper error array of shape (1, 1, 2) at the specified
            plot period. Returns zeros if no tipper dictionary available.

        """

        if tf.t_interp_dict == {}:
            return np.array((1, 1, 2), dtype=float)
        return np.nan_to_num(
            np.array(
                [
                    [
                        [
                            tf.t_interp_dict["tzx"]["err"](1.0 / self.plot_period),
                            tf.t_interp_dict["tzy"]["err"](1.0 / self.plot_period),
                        ]
                    ]
                ]
            )
        )
