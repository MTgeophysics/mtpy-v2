# -*- coding: utf-8 -*-
"""Bokeh map-plot base class with interpolation helpers.

Provides the map interpolation and transfer-function interpolation utilities
used by map-based plotting classes, while inheriting from the param-driven
``BokehPlotBase``.
"""

from __future__ import annotations

import numpy as np


try:
    import param
except ImportError:  # pragma: no cover
    raise ImportError(
        "param is required for Bokeh plot classes. Install with `pip install param`."
    )

from scipy import interpolate

from mtpy.imaging.mtplot_tools.map_interpolation_tools import interpolate_to_map

from .bokeh_plot_base import BokehPlotBase


class BokehPlotBaseMaps(BokehPlotBase):
    """Base object for Bokeh plot classes that use map views.

    Mirrors the interpolation behavior of ``PlotBaseMaps`` while keeping the
    class hierarchy fully in the Bokeh/param ecosystem.
    """

    cell_size = param.Number(default=0.002, bounds=(0, None), doc="Map grid cell size")
    n_padding_cells = param.Integer(
        default=10,
        bounds=(0, None),
        doc="Number of padding cells around map extent",
    )
    interpolation_method = param.ObjectSelector(
        default="delaunay",
        objects=["delaunay", "fancy", "triangulate", "nearest", "linear", "cubic"],
        doc="Map interpolation method",
    )
    interpolation_power = param.Number(
        default=5,
        bounds=(0, None),
        doc="IDW interpolation power for triangulation methods",
    )
    nearest_neighbors = param.Integer(
        default=7,
        bounds=(1, None),
        doc="Nearest neighbors for triangulation interpolation",
    )
    use_mt_data_preinterpolation = param.Boolean(
        default=True,
        doc="Cache interpolated mt_data per plot period",
    )

    def _iter_mt_objects(self):
        """Yield MT objects from supported container types."""

        if not hasattr(self, "mt_data"):
            raise AttributeError("BokehPlotBaseMaps subclasses must set self.mt_data")

        if hasattr(self.mt_data, "values"):
            yield from self.mt_data.values()
            return

        if hasattr(self.mt_data, "_iter_station_paths") and hasattr(
            self.mt_data, "get_station"
        ):
            if hasattr(self.mt_data, "compute"):
                self.mt_data.compute()
            for station_path in self.mt_data._iter_station_paths():
                yield self.mt_data.get_station(station_path, as_mt=True)
            return

        raise TypeError("mt_data must provide values() or MTData-style station access")

    def _get_mt_data_for_plot_period(self):
        """Return mt_data interpolated to the active plot period when supported."""

        if not hasattr(self, "mt_data"):
            raise AttributeError("BokehPlotBaseMaps subclasses must set self.mt_data")

        if not self.use_mt_data_preinterpolation:
            return self.mt_data

        if not (
            hasattr(self.mt_data, "interpolate")
            and hasattr(self.mt_data, "_iter_station_paths")
            and hasattr(self.mt_data, "get_station")
        ):
            return self.mt_data

        if not hasattr(self, "_interpolated_mt_data_cache"):
            self._interpolated_mt_data_cache = None
        if not hasattr(self, "_interpolated_mt_data_cache_period"):
            self._interpolated_mt_data_cache_period = None

        target_period = float(self.plot_period)
        if (
            self._interpolated_mt_data_cache is not None
            and self._interpolated_mt_data_cache_period is not None
            and np.isclose(self._interpolated_mt_data_cache_period, target_period)
        ):
            return self._interpolated_mt_data_cache

        try:
            interpolated = self.mt_data.interpolate(
                np.array([target_period], dtype=float),
                inplace=False,
                bounds_error=False,
            )
            if interpolated is not None:
                self._interpolated_mt_data_cache = interpolated
                self._interpolated_mt_data_cache_period = target_period
                return interpolated
        except Exception as error:
            self.logger.debug(
                "Falling back to per-station interpolation for plot period "
                f"{target_period}: {error}"
            )

        return self.mt_data

    def _get_mt_objects(self):
        """Return MT objects as a list for repeated calculations."""

        data_source = self._get_mt_data_for_plot_period()

        if hasattr(data_source, "values"):
            return list(data_source.values())

        if hasattr(data_source, "_iter_station_paths") and hasattr(
            data_source, "get_station"
        ):
            if hasattr(data_source, "compute"):
                data_source.compute()
            return [
                data_source.get_station(station_path, as_mt=True)
                for station_path in data_source._iter_station_paths()
            ]

        return list(self._iter_mt_objects())

    def interpolate_to_map(self, plot_array, component: str):
        """Interpolate data points onto a 2D map grid."""

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
        """Create 1D interpolation functions for impedance tensor components."""
        if tf.Z is None:
            return None

        zmap = {0: "x", 1: "y"}
        interp_dict = {}
        for ii in range(2):
            for jj in range(2):
                comp = f"z{zmap[ii]}{zmap[jj]}"
                interp_dict[comp] = {}
                nz_index = np.nonzero(tf.Z.z[:, ii, jj])

                if len(nz_index[0]) == 0:
                    continue

                z_real = tf.Z.z[nz_index, ii, jj].real
                z_imag = tf.Z.z[nz_index, ii, jj].imag
                f = tf.Z.frequency[nz_index]

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
        """Create 1D interpolation functions for tipper components."""
        if tf.Tipper is None:
            return None

        zmap = {0: "x", 1: "y"}
        interp_dict = {}
        for jj in range(2):
            comp = f"tz{zmap[jj]}"
            interp_dict[comp] = {}
            nz_index = np.nonzero(tf.Tipper.tipper[:, 0, jj])

            if len(nz_index[0]) == 0:
                continue

            t_real = tf.Tipper.tipper[nz_index, 0, jj].real
            t_imag = tf.Tipper.tipper[nz_index, 0, jj].imag
            f = tf.Tipper.frequency[nz_index]

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

    def _get_plot_period_index(self, tf, rtol: float = 1e-6) -> int | None:
        """Return index of the configured plot period if present in TF data."""
        period = getattr(tf, "period", None)
        if period is None:
            return None

        period_array = np.asarray(period, dtype=float)
        if period_array.size == 0:
            return None

        idx = np.where(np.isclose(period_array, float(self.plot_period), rtol=rtol))[0]
        if idx.size == 0:
            return None
        return int(idx[0])

    def _get_interpolated_z(self, tf) -> np.ndarray:
        """Get interpolated impedance tensor at plot period."""
        idx = self._get_plot_period_index(tf)
        if idx is not None and tf.Z is not None:
            try:
                return np.nan_to_num(np.asarray(tf.Z.z[idx : idx + 1], dtype=complex))
            except Exception:
                pass

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
        """Get interpolated impedance tensor error at plot period."""
        idx = self._get_plot_period_index(tf)
        if idx is not None and tf.Z is not None and tf.Z._has_tf_error():
            try:
                return np.nan_to_num(
                    np.asarray(tf.Z.z_error[idx : idx + 1], dtype=float)
                )
            except Exception:
                pass

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
        return np.zeros((1, 2, 2), dtype=float)

    def _get_interpolated_z_model_error(self, tf) -> np.ndarray:
        """Get interpolated impedance tensor model error at plot period."""
        idx = self._get_plot_period_index(tf)
        if idx is not None and tf.Z is not None and tf.Z._has_tf_model_error():
            try:
                return np.nan_to_num(
                    np.asarray(tf.Z.z_model_error[idx : idx + 1], dtype=float)
                )
            except Exception:
                pass

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
        return np.zeros((1, 2, 2), dtype=float)

    def _get_interpolated_t(self, tf) -> np.ndarray:
        """Get interpolated tipper at plot period."""
        idx = self._get_plot_period_index(tf)
        if idx is not None and tf.has_tipper() and tf.Tipper is not None:
            try:
                return np.nan_to_num(
                    np.asarray(tf.Tipper.tipper[idx : idx + 1], dtype=complex)
                )
            except Exception:
                pass

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

    def _get_interpolated_t_err(self, tf) -> np.ndarray:
        """Get interpolated tipper error at plot period."""
        idx = self._get_plot_period_index(tf)
        if idx is not None and tf.has_tipper() and tf.Tipper._has_tf_error():
            try:
                return np.nan_to_num(
                    np.asarray(tf.Tipper.tipper_error[idx : idx + 1], dtype=float)
                )
            except Exception:
                pass

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
        return np.zeros((1, 1, 2), dtype=float)

    def _get_interpolated_t_model_err(self, tf) -> np.ndarray:
        """Get interpolated tipper model error at plot period."""
        idx = self._get_plot_period_index(tf)
        if idx is not None and tf.has_tipper() and tf.Tipper._has_tf_model_error():
            try:
                return np.nan_to_num(
                    np.asarray(tf.Tipper.tipper_model_error[idx : idx + 1], dtype=float)
                )
            except Exception:
                pass

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
        return np.zeros((1, 1, 2), dtype=float)
