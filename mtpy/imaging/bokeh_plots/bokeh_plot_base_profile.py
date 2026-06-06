# -*- coding: utf-8 -*-
"""Bokeh profile-plot base class.

Provides profile-line helpers shared by Bokeh pseudosection-style plotting
classes while staying in the Bokeh/param hierarchy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


try:
    import param
except ImportError:  # pragma: no cover
    raise ImportError(
        "param is required for Bokeh plot classes. Install with `pip install param`."
    )

from .bokeh_plot_base import BokehPlotBase


class BokehPlotBaseProfile(BokehPlotBase):
    """Base object for Bokeh profile plots like pseudosections."""

    profile_vector = param.Parameter(default=None, doc="Profile direction vector")
    profile_angle = param.Parameter(default=None, doc="Profile angle in degrees")
    profile_line = param.Parameter(default=None, doc="Profile line (slope, intercept)")
    profile_reverse = param.Boolean(default=False, doc="Reverse profile direction")

    x_stretch = param.Number(default=5000, doc="Horizontal profile scale factor")
    y_stretch = param.Number(default=1000, doc="Vertical profile scale factor")
    y_scale = param.ObjectSelector(
        default="period",
        objects=["period", "frequency"],
        doc="Y-axis scale for profile plots",
    )

    def __init__(self, mt_data, **kwargs):
        param_names = set(type(self).param)
        param_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
        other_kwargs = {k: v for k, v in kwargs.items() if k not in param_names}

        super().__init__(**param_kwargs)
        self.mt_data = mt_data
        self._rotation_angle = 0

        for key, value in other_kwargs.items():
            setattr(self, key, value)

    @property
    def rotation_angle(self) -> float:
        """Get rotation angle for profile data."""

        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value: float) -> None:
        """Set rotation angle for all transfer functions in the profile data."""

        if hasattr(self.mt_data, "rotate") and hasattr(self.mt_data, "get_station"):
            self.mt_data.rotate(value, inplace=True)
        else:
            for tf in self._iter_mt_objects():
                tf.rotation_angle = value
        self._rotation_angle = value

    def _iter_mt_objects(self):
        """Yield MT objects from supported container types."""

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

    def _get_mt_objects(self):
        """Return MT objects as a list for repeated profile operations."""

        return list(self._iter_mt_objects())

    def _sync_mt_data_profile_offsets(
        self,
        x_column: str,
        y_column: str,
    ) -> dict[tuple[str, str], float]:
        """Compute and persist profile offsets for MTData-backed station attrs."""

        if not (
            hasattr(self.mt_data, "station_locations")
            and hasattr(self.mt_data, "_iter_station_paths")
            and hasattr(self.mt_data, "get_station")
        ):
            return {}

        station_df = self.mt_data.station_locations
        if station_df is None or station_df.empty:
            return {}
        if (
            x_column not in station_df.columns
            or y_column not in station_df.columns
            or "survey" not in station_df.columns
            or "station" not in station_df.columns
        ):
            return {}

        x_values = station_df[x_column].to_numpy(dtype=float)
        y_values = station_df[y_column].to_numpy(dtype=float)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite):
            return {}

        profile_vector = np.array([1.0, self.profile_line[0]], dtype=float)
        profile_vector /= np.linalg.norm(profile_vector)
        station_vectors = np.column_stack(
            [x_values[finite], y_values[finite] - self.profile_line[1]]
        )
        offsets = np.abs(station_vectors @ profile_vector)
        offsets -= offsets.min()

        key_to_path: dict[tuple[str, str], str] = {}
        for station_path in self.mt_data._iter_station_paths():
            attrs = self.mt_data.get_station(station_path).attrs
            key = (str(attrs.get("survey", "")), str(attrs.get("station", "")))
            key_to_path[key] = station_path

        offset_lookup: dict[tuple[str, str], float] = {}
        profile_df = station_df.loc[finite, ["survey", "station"]].copy()
        profile_df.loc[:, "profile_offset"] = offsets

        for row in profile_df.itertuples(index=False):
            key = (str(getattr(row, "survey", "")), str(getattr(row, "station", "")))
            offset_value = float(getattr(row, "profile_offset", 0.0))
            offset_lookup[key] = offset_value

            station_path = key_to_path.get(key)
            if station_path is None:
                continue
            self.mt_data.get_station(station_path).attrs[
                "profile_offset"
            ] = offset_value

        return offset_lookup

    def _get_profile_line(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> None:
        """Calculate profile line using linear regression through station locations."""

        station_locations = getattr(self.mt_data, "station_locations", None)
        if (
            station_locations is not None
            and hasattr(station_locations, "columns")
            and "profile_offset" in station_locations.columns
        ):
            offsets = np.nan_to_num(
                station_locations["profile_offset"].to_numpy(dtype=float),
                nan=0.0,
            )
            if offsets.size > 0 and np.any(offsets != 0):
                return

        mt_objects = None
        coordinate_columns: tuple[str, str] | None = None

        if x is None and y is None:
            if (
                station_locations is not None
                and getattr(station_locations, "empty", True) is False
            ):
                for x_col, y_col in [("longitude", "latitude"), ("east", "north")]:
                    if (
                        x_col not in station_locations.columns
                        or y_col not in station_locations.columns
                    ):
                        continue
                    x_values = station_locations[x_col].to_numpy(dtype=float)
                    y_values = station_locations[y_col].to_numpy(dtype=float)
                    finite = np.isfinite(x_values) & np.isfinite(y_values)
                    if np.count_nonzero(finite) < 2:
                        continue
                    x = x_values[finite]
                    y = y_values[finite]
                    coordinate_columns = (x_col, y_col)
                    break

            if x is None or y is None:
                mt_objects = self._get_mt_objects()
                x = np.zeros(len(mt_objects))
                y = np.zeros(len(mt_objects))

                for ii, tf in enumerate(mt_objects):
                    x[ii] = tf.longitude
                    y[ii] = tf.latitude

        elif x is None or y is None:
            raise ValueError("get_profile")

        if x.size < 2 or y.size < 2:
            return

        profile1 = stats.linregress(x, y)
        profile2 = stats.linregress(y, x)
        if profile2.stderr < profile1.stderr:
            self.profile_line = (
                1.0 / profile2.slope,
                -profile2.intercept / profile2.slope,
            )
        else:
            self.profile_line = profile1[:2]

        offset_lookup: dict[tuple[str, str], float] = {}
        if coordinate_columns is not None:
            offset_lookup = self._sync_mt_data_profile_offsets(*coordinate_columns)

        if mt_objects is None:
            mt_objects = self._get_mt_objects()

        if offset_lookup:
            for mt_obj in mt_objects:
                key = (
                    str(getattr(mt_obj, "survey", "")),
                    str(getattr(mt_obj, "station", "")),
                )
                if key in offset_lookup:
                    mt_obj.profile_offset = offset_lookup[key]
                else:
                    mt_obj.project_onto_profile_line(
                        self.profile_line[0],
                        self.profile_line[1],
                    )
            return

        for mt_obj in mt_objects:
            mt_obj.project_onto_profile_line(self.profile_line[0], self.profile_line[1])

    def _get_offset(self, tf) -> float:
        """Get scaled offset distance for a station along the profile."""

        direction = -1 if self.profile_reverse else 1

        offset_value = None
        station_locations = getattr(self.mt_data, "station_locations", None)
        if (
            station_locations is not None
            and hasattr(station_locations, "itertuples")
            and hasattr(station_locations, "columns")
            and "profile_offset" in station_locations.columns
            and "station" in station_locations.columns
        ):
            tf_station = str(getattr(tf, "station", ""))
            tf_survey = str(getattr(tf, "survey", ""))

            for row in station_locations.itertuples(index=False):
                row_station = str(getattr(row, "station", ""))
                if row_station != tf_station:
                    continue

                row_survey = str(getattr(row, "survey", ""))
                if tf_survey and row_survey != tf_survey:
                    continue

                row_offset = getattr(row, "profile_offset", None)
                if row_offset is None or not np.isfinite(row_offset):
                    continue

                offset_value = float(row_offset)
                break

        if offset_value is None:
            offset_value = float(getattr(tf, "profile_offset", 0.0) or 0.0)

        return direction * offset_value * self.x_stretch
