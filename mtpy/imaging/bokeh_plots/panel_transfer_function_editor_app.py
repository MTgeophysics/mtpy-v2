"""Panel app for interactive transfer-function editing.

This module provides a Bokeh-backed Panel application for editing apparent
resistivity, phase, and tipper data from an MT station before downstream
modeling.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.events import DoubleTap, Tap
from bokeh.layouts import column as bk_column
from bokeh.models import BoxSelectTool, ColumnDataSource, Whisker
from bokeh.plotting import figure

try:
    from mtpy.modeling.simpeg.recipes.inversion_1d import Simpeg1D as _Simpeg1D
except ImportError:  # pragma: no cover
    _Simpeg1D = None  # type: ignore[assignment,misc]

_MODE_MAP: dict[str, str] = {"xy": "te", "yx": "tm", "det": "det"}


@dataclass(frozen=True)
class _FrameBundle:
    """Container for the scalar series displayed by one editor view."""

    frame: pd.DataFrame
    title: str


class TransferFunctionSeriesEditor(param.Parameterized):
    """Interactive point editor for a single transfer-function series.

    The editor mirrors the InteractivePointEditor prototype but is structured
    for Panel reuse. It supports tap selection, box selection, point deletion,
    and error inflation. The current working series is cached per
    component/view key so a user can switch between components without losing
    edits in the current session.
    """

    def __init__(self, title: str, y_axis_type: str, color: str, **params: Any) -> None:
        super().__init__(**params)

        self.title = title
        self.y_axis_type = y_axis_type
        self._default_color = color
        self._default_marker = "circle"
        self._frame_builder: Callable[[str | None, str | None], _FrameBundle] | None = (
            None
        )
        self._working_frames: dict[tuple[str | None, str | None], pd.DataFrame] = {}
        self._original_frames: dict[tuple[str | None, str | None], pd.DataFrame] = {}
        self._active_key: tuple[str | None, str | None] | None = None
        self._marked_for_deletion: set[int] = set()

        self._component_widget = pn.widgets.Select(
            name="Component",
            options=[],
            value=None,
            width=140,
        )
        self._component_widget.param.watch(self._on_component_changed, "value")

        self._view_widget = pn.widgets.Select(
            name="View",
            options=[],
            value=None,
            width=120,
            visible=False,
        )
        self._view_widget.param.watch(self._on_view_changed, "value")

        self._mode_widget = pn.widgets.RadioButtonGroup(
            name="Mode",
            options=["Delete points", "Increase error"],
            value="Delete points",
            button_type="default",
        )

        self._error_factor_widget = pn.widgets.FloatInput(
            name="Error increase factor",
            value=0.05,
            step=0.01,
            width=170,
        )

        self._threshold_widget = pn.widgets.FloatInput(
            name="Selection threshold",
            value=0.05,
            step=0.01,
            width=150,
        )

        self._apply_button = pn.widgets.Button(
            name="Apply edits",
            button_type="warning",
            width=130,
        )
        self._apply_button.on_click(self.apply_edits)

        self._revert_button = pn.widgets.Button(
            name="Revert",
            button_type="success",
            width=100,
        )
        self._revert_button.on_click(self.revert_changes)

        self._status = pn.pane.Markdown(
            "_No data loaded yet._",
            styles={"color": "#555"},
        )

        self._source = ColumnDataSource(
            data={
                "period": [],
                "value": [],
                "error": [],
                "value_upper": [],
                "value_lower": [],
                "use": [],
                "marker": [],
                "color": [],
            }
        )

        self._plot = figure(
            title=title,
            x_axis_type="log",
            y_axis_type=y_axis_type,
            height=360,
            width=920,
            tools="pan,wheel_zoom,box_zoom,reset,save,tap,box_select",
            active_scroll="wheel_zoom",
        )
        self._plot.scatter(
            "period",
            "value",
            source=self._source,
            size=8,
            marker="marker",
            color="color",
            line_color="color",
        )
        self._whisker = Whisker(
            source=self._source,
            base="period",
            upper="value_upper",
            lower="value_lower",
            line_color="#325ea8",
        )
        self._plot.add_layout(self._whisker)
        self._plot.add_tools(BoxSelectTool(dimensions="both"))
        self._plot.on_event(Tap, self._on_tap)
        self._plot.on_event(DoubleTap, self.clear_selection)
        self._source.selected.on_change("indices", self._on_box_select)

        self.layout = pn.Column(
            pn.Row(
                self._component_widget,
                self._view_widget,
                self._mode_widget,
                self._error_factor_widget,
                self._threshold_widget,
                align="start",
            ),
            pn.Row(self._apply_button, self._revert_button, align="start"),
            self._status,
            self._plot,
        )

    def set_frame_builder(
        self,
        builder: Callable[[str | None, str | None], _FrameBundle],
        component_options: list[str],
        view_options: list[str] | None = None,
        initial_component: str | None = None,
        initial_view: str | None = None,
    ) -> None:
        """Attach a data builder and configure the component/view controls."""

        self._frame_builder = builder
        self._component_widget.options = component_options
        self._component_widget.value = (
            initial_component if initial_component in component_options else None
        )
        if self._component_widget.value is None and component_options:
            self._component_widget.value = component_options[0]

        view_options = view_options or []
        self._view_widget.options = view_options
        if view_options:
            self._view_widget.visible = True
            self._view_widget.value = (
                initial_view if initial_view in view_options else None
            )
            if self._view_widget.value is None:
                self._view_widget.value = view_options[0]
        else:
            self._view_widget.visible = False
            self._view_widget.value = None

        self._reload_active_frame(force=True)

    def load_frame(self, frame: pd.DataFrame, title: str | None = None) -> None:
        """Load a prebuilt frame directly into the editor."""

        self._frame_builder = None
        key = (self._component_widget.value, self._view_widget.value)
        self._original_frames[key] = frame.copy().reset_index(drop=True)
        self._working_frames[key] = frame.copy().reset_index(drop=True)
        if title is not None:
            self.title = title
            self._plot.title.text = title
        self._active_key = key
        self._marked_for_deletion.clear()
        self._refresh_source()
        self._status.object = f"Loaded {len(frame)} point(s)."
        self._status.styles = {"color": "#1a6600"}

    def _current_key(self) -> tuple[str | None, str | None]:
        return (self._component_widget.value, self._view_widget.value)

    def _reload_active_frame(self, force: bool = False) -> None:
        key = self._current_key()
        if not force and key == self._active_key:
            return

        if self._active_key is not None and self._active_key in self._working_frames:
            self._working_frames[self._active_key] = self._working_frame.copy()

        self._active_key = key
        if key not in self._working_frames:
            if self._frame_builder is None:
                self._working_frames[key] = pd.DataFrame(
                    columns=["period", "value", "error", "use"]
                )
                self._original_frames[key] = self._working_frames[key].copy()
            else:
                bundle = self._frame_builder(key[0], key[1])
                self._working_frames[key] = bundle.frame.reset_index(drop=True).copy()
                self._original_frames[key] = bundle.frame.reset_index(drop=True).copy()
                self.title = bundle.title
                self._plot.title.text = bundle.title

        self._working_frame = self._working_frames[key].copy().reset_index(drop=True)
        self._marked_for_deletion.clear()
        self._refresh_source()
        self._status.object = f"Loaded {len(self._working_frame)} point(s)."
        self._status.styles = {"color": "#1a6600"}

    def _refresh_source(self) -> None:
        frame = getattr(self, "_working_frame", pd.DataFrame())
        if frame is None or frame.empty:
            self._source.data = {
                "period": [],
                "value": [],
                "error": [],
                "value_upper": [],
                "value_lower": [],
                "use": [],
                "marker": [],
                "color": [],
            }
            return

        periods = frame["period"].to_numpy(dtype=float)
        values = frame["value"].to_numpy(dtype=float)
        errors = frame["error"].to_numpy(dtype=float)
        use = frame["use"].to_numpy(dtype=bool)

        # Keep the source free of invalid points for log axes.
        valid = np.isfinite(periods) & np.isfinite(values) & np.isfinite(errors)
        valid &= periods > 0
        if self.y_axis_type == "log":
            valid &= values > 0
            valid &= values - errors > 0

        periods = periods[valid]
        values = values[valid]
        errors = errors[valid]
        use = use[valid]

        marked = np.zeros(len(values), dtype=bool)
        for idx in self._marked_for_deletion:
            if idx < len(marked):
                marked[idx] = True

        if self.y_axis_type == "log":
            value_upper = values + errors
            value_lower = np.maximum(values - errors, 1e-12)
        else:
            value_upper = values + errors
            value_lower = values - errors

        colors = np.where(
            marked, "#000000", np.where(use, self._default_color, "#808080")
        )
        markers = np.where(
            marked, "x", np.where(use, self._default_marker, self._default_marker)
        )

        self._source.data = {
            "period": list(periods),
            "value": list(values),
            "error": list(errors),
            "value_upper": list(value_upper),
            "value_lower": list(value_lower),
            "use": list(use),
            "marker": list(markers),
            "color": list(colors),
        }

        if len(values) == 0:
            self._status.object = "_No valid points available for this view._"
            self._status.styles = {"color": "#777"}

    def _nearest_index(self, event) -> int | None:
        if not self._source.data["period"]:
            return None

        click_x = float(event.x)
        click_y = float(event.y)
        x_vals = np.asarray(self._source.data["period"], dtype=float)
        y_vals = np.asarray(self._source.data["value"], dtype=float)

        if self._plot.x_axis_type == "log":
            click_x = np.log10(click_x)
            x_vals = np.log10(x_vals)
        if self._plot.y_axis_type == "log":
            click_y = np.log10(click_y)
            y_vals = np.log10(y_vals)

        dist_sq = (x_vals - click_x) ** 2 + (y_vals - click_y) ** 2
        idx = int(np.argmin(dist_sq))

        x_range = x_vals.max() - x_vals.min() if x_vals.size else 0.0
        y_range = y_vals.max() - y_vals.min() if y_vals.size else 0.0
        max_range = max(x_range, y_range)
        if max_range <= 0:
            return idx
        if np.sqrt(dist_sq[idx]) > self._threshold_widget.value * max_range:
            return None
        return idx

    def _on_tap(self, event) -> None:
        idx = self._nearest_index(event)
        if idx is None or self._working_frame.empty:
            return

        if self._mode_widget.value == "Delete points":
            self._marked_for_deletion.add(idx)
            self._refresh_source()
            return

        factor = float(self._error_factor_widget.value)
        self._working_frame.loc[idx, "error"] = float(
            self._working_frame.loc[idx, "error"]
        ) * (1.0 + factor)
        self._refresh_source()
        self._status.object = (
            f"Updated error for point {idx} by factor {1.0 + factor:.3f}."
        )
        self._status.styles = {"color": "#7a5200"}

    def _on_box_select(self, attr, old, new) -> None:
        if not new or self._working_frame.empty:
            return

        if self._mode_widget.value == "Delete points":
            self._marked_for_deletion.update(int(i) for i in new)
        else:
            factor = float(self._error_factor_widget.value)
            for idx in new:
                idx = int(idx)
                self._working_frame.loc[idx, "error"] = float(
                    self._working_frame.loc[idx, "error"]
                ) * (1.0 + factor)
            self._status.object = f"Updated {len(new)} selected point(s) error bars."
            self._status.styles = {"color": "#7a5200"}

        self._source.selected.indices = []
        self._refresh_source()

    def _on_component_changed(self, _event=None) -> None:
        self._reload_active_frame()

    def _on_view_changed(self, _event=None) -> None:
        self._reload_active_frame()

    def apply_edits(self, _event=None) -> None:
        if self._working_frame.empty:
            return

        if self._marked_for_deletion:
            for idx in sorted(self._marked_for_deletion):
                if idx in self._working_frame.index:
                    self._working_frame.loc[idx, "use"] = False
            self._marked_for_deletion.clear()

        self._working_frames[self._current_key()] = self._working_frame.copy()
        self._refresh_source()
        active_count = int(
            np.count_nonzero(self._working_frame["use"].to_numpy(dtype=bool))
        )
        self._status.object = f"Applied edits. {active_count} point(s) remain active."
        self._status.styles = {"color": "#1a6600"}

    def revert_changes(self, _event=None) -> None:
        key = self._current_key()
        if key not in self._original_frames:
            return

        self._working_frame = self._original_frames[key].copy().reset_index(drop=True)
        self._working_frames[key] = self._working_frame.copy()
        self._marked_for_deletion.clear()
        self._refresh_source()
        self._status.object = "Reverted to the originally loaded data for this view."
        self._status.styles = {"color": "#1a6600"}

    def clear_selection(self, event=None) -> None:
        """Clear Bokeh selection without changing the underlying data."""

        self._source.selected.indices = []

    @property
    def frame(self) -> pd.DataFrame:
        """Return the current working frame."""

        return self._working_frame.copy()

    def get_frame(
        self,
        component: str | None = None,
        view: str | None = None,
    ) -> pd.DataFrame:
        """Return a cached frame for a given component/view key."""

        key = (
            component if component is not None else self._component_widget.value,
            view if view is not None else self._view_widget.value,
        )
        frame = self._working_frames.get(key)
        if frame is None:
            return pd.DataFrame(columns=["period", "value", "error", "use"])
        return frame.copy()


class TransferFunctionEditorPanelApp(param.Parameterized):
    """Panel application for interactive transfer-function editing."""

    sizing_mode = param.Selector(
        default="stretch_width",
        objects=["stretch_width", "fixed", "stretch_both", "stretch_height"],
        doc="Panel sizing mode",
    )

    def __init__(self, mt_data=None, **params: Any) -> None:
        super().__init__(**params)

        self._mt_data = mt_data
        self._station_df_full: pd.DataFrame | None = None
        self._series_editors: dict[
            tuple[str, str, str | None], TransferFunctionSeriesEditor
        ] = {}
        self._active_res_editors: list[TransferFunctionSeriesEditor] = []
        self._active_phase_editors: list[TransferFunctionSeriesEditor] = []
        self._active_tipper_editors: list[TransferFunctionSeriesEditor] = []
        self._station_widget = pn.widgets.Select(
            name="Station",
            options=[],
            value=None,
            width=420,
        )
        self._load_button = pn.widgets.Button(
            name="Load Station",
            button_type="primary",
            width=150,
        )
        self._load_button.on_click(self._on_load_station_clicked)

        self._status = pn.pane.Markdown(
            "_Load a station to begin editing transfer-function data._",
            styles={"color": "#555"},
        )
        self._output = pn.pane.Markdown("", styles={"color": "#333"})

        self._edit_dimension_widget = pn.widgets.RadioButtonGroup(
            name="Edit dimensionality",
            options=["1D", "2D", "3D"],
            value="1D",
            button_type="default",
        )
        self._edit_mode_widget = pn.widgets.Select(
            name="Mode",
            options=["xy", "yx", "det"],
            value="xy",
            width=120,
        )
        self._edit_dimension_widget.param.watch(self._on_dimension_changed, "value")
        self._edit_mode_widget.param.watch(self._on_dimension_changed, "value")

        self._res_row_container = pn.Row(sizing_mode="stretch_width")
        self._phase_row_container = pn.Row(sizing_mode="stretch_width")
        self._tipper_row_container = pn.Row(sizing_mode="stretch_width")

        self._res_placeholder = pn.pane.Markdown(
            "_Load a station to display apparent resistivity editors._",
            styles={"color": "#777"},
        )
        self._phase_placeholder = pn.pane.Markdown(
            "_Load a station to display phase editors._",
            styles={"color": "#777"},
        )
        self._tipper_placeholder = pn.pane.Markdown(
            "_Load a station to display tipper editors._",
            styles={"color": "#777"},
        )
        self._res_row_container.objects = [self._res_placeholder]
        self._phase_row_container.objects = [self._phase_placeholder]
        self._tipper_row_container.objects = [self._tipper_placeholder]

        self._rotation_widget = pn.widgets.FloatInput(
            name="Rotate (deg)",
            value=0.0,
            step=0.5,
            width=120,
            disabled=True,
        )
        self._static_shift_widget = pn.widgets.FloatInput(
            name="Static shift factor",
            value=1.0,
            step=0.05,
            width=150,
            disabled=True,
        )
        self._interpolate_button = pn.widgets.Button(
            name="Interpolate",
            button_type="default",
            width=120,
            disabled=True,
        )
        self._swap_phase_button = pn.widgets.Button(
            name="Swap phase",
            button_type="default",
            width=120,
            disabled=True,
        )
        self._future_tools = pn.Accordion(
            (
                "Future tools",
                pn.Column(
                    pn.pane.Markdown(
                        "Rotate, interpolation, static shift, and phase swap are reserved here for later."
                    ),
                    pn.Row(
                        self._rotation_widget,
                        self._static_shift_widget,
                        self._interpolate_button,
                        self._swap_phase_button,
                        align="start",
                    ),
                ),
            )
        )

        # ── 1D Inversion widgets ──────────────────────────────────────────
        self._inv_mode_widget = pn.widgets.RadioButtonGroup(
            name="Mode",
            options=["xy", "yx", "det"],
            value="xy",
            button_type="default",
        )
        self._inv_n_layers_widget = pn.widgets.IntInput(
            name="N layers",
            value=50,
            step=1,
            start=2,
            end=400,
            width=110,
        )
        self._inv_dz_widget = pn.widgets.FloatInput(
            name="First layer thickness (m)",
            value=5.0,
            step=0.5,
            width=190,
        )
        self._inv_z_factor_widget = pn.widgets.FloatInput(
            name="Layer growth factor",
            value=1.2,
            step=0.05,
            width=160,
        )
        self._inv_rho_initial_widget = pn.widgets.FloatInput(
            name="Starting resistivity (Ω·m)",
            value=100.0,
            step=5.0,
            width=210,
        )
        self._inv_rho_reference_widget = pn.widgets.FloatInput(
            name="Reference resistivity (Ω·m)",
            value=100.0,
            step=5.0,
            width=210,
        )
        self._inv_max_iter_widget = pn.widgets.IntInput(
            name="maxIter",
            value=40,
            step=1,
            start=1,
            end=500,
            width=100,
        )
        self._inv_max_iter_cg_widget = pn.widgets.IntInput(
            name="maxIterCG",
            value=30,
            step=1,
            start=1,
            end=500,
            width=100,
        )
        self._inv_alpha_s_widget = pn.widgets.FloatInput(
            name="alpha_s",
            value=1e-10,
            step=1e-10,
            width=120,
        )
        self._inv_alpha_z_widget = pn.widgets.FloatInput(
            name="alpha_z",
            value=1.0,
            step=0.1,
            width=110,
        )
        self._inv_beta0_ratio_widget = pn.widgets.FloatInput(
            name="beta0_ratio",
            value=1.0,
            step=0.1,
            width=120,
        )
        self._inv_cooling_factor_widget = pn.widgets.FloatInput(
            name="coolingFactor",
            value=2.0,
            step=0.1,
            width=120,
        )
        self._inv_cooling_rate_widget = pn.widgets.IntInput(
            name="coolingRate",
            value=1,
            step=1,
            start=1,
            end=50,
            width=110,
        )
        self._inv_chi_factor_widget = pn.widgets.FloatInput(
            name="chi_factor",
            value=1.0,
            step=0.1,
            width=110,
        )
        self._inv_use_irls_widget = pn.widgets.Checkbox(name="use_irls", value=False)
        self._inv_p_s_widget = pn.widgets.FloatInput(
            name="p_s",
            value=2.0,
            step=0.1,
            width=80,
        )
        self._inv_p_z_widget = pn.widgets.FloatInput(
            name="p_z",
            value=2.0,
            step=0.1,
            width=80,
        )
        self._inv_run_button = pn.widgets.Button(
            name="Run 1D Inversion",
            button_type="success",
            width=160,
        )
        self._inv_run_button.on_click(self._on_run_inversion_clicked)
        self._inv_status = pn.pane.Markdown(
            "_Configure inversion and click Run._",
            styles={"color": "#555"},
        )
        self._inv_output = pn.pane.Markdown("", styles={"color": "#333"})
        self._inv_response_plot = pn.pane.Bokeh(
            sizing_mode="stretch_width", min_height=740
        )
        self._inv_model_plot = pn.pane.Bokeh(
            sizing_mode="stretch_width",
            min_height=740,
            max_width=220,
        )
        self._inv_simpeg: Any = None

        self._refresh_station_options()

    @staticmethod
    def _component_color(prefix: str, component: str, view: str | None = None) -> str:
        cmap = {
            ("res", "xx"): "#1f77b4",
            ("res", "xy"): "#2ca02c",
            ("res", "yx"): "#ff7f0e",
            ("res", "yy"): "#9467bd",
            ("res", "det"): "#325ea8",
            ("phase", "xx"): "#c23b22",
            ("phase", "xy"): "#b24b3f",
            ("phase", "yx"): "#8c2d26",
            ("phase", "yy"): "#d95f0e",
            ("phase", "det"): "#b24b3f",
            ("t", "zx", "real"): "#7b61ff",
            ("t", "zy", "real"): "#4c78a8",
            ("t", "zx", "imag"): "#9467bd",
            ("t", "zy", "imag"): "#2b8cbe",
        }
        if prefix == "t":
            return cmap.get((prefix, component, view), "#7b61ff")
        return cmap.get((prefix, component), "#325ea8")

    @staticmethod
    def _sanitize_station_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce known scalar transfer-function columns to numeric values."""

        clean_df = df.copy()
        numeric_cols = [
            "frequency",
            "res_xy",
            "res_xy_error",
            "res_yx",
            "res_yx_error",
            "res_xx",
            "res_xx_error",
            "res_yy",
            "res_yy_error",
            "res_det",
            "res_det_error",
            "phase_xy",
            "phase_xy_error",
            "phase_yx",
            "phase_yx_error",
            "phase_xx",
            "phase_xx_error",
            "phase_yy",
            "phase_yy_error",
            "phase_det",
            "phase_det_error",
            "t_zx_error",
            "t_zy_error",
            "t_mag_real",
            "t_mag_real_error",
            "t_mag_imag",
            "t_mag_imag_error",
            "t_angle_real",
            "t_angle_real_error",
            "t_angle_imag",
            "t_angle_imag_error",
        ]
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
        return clean_df

    def set_mt_data(self, mt_data) -> None:
        """Set the MTData reference and refresh the station picker."""

        self._mt_data = mt_data
        self._refresh_station_options()
        if mt_data is None:
            self._station_df_full = None
            self._status.object = "_Load MT data in the Data tab first._"
            self._status.styles = {"color": "#777"}

    def _refresh_station_options(self) -> None:
        if self._mt_data is None:
            self._station_widget.options = []
            self._station_widget.value = None
            return

        paths = list(self._mt_data._iter_station_paths())
        self._station_widget.options = paths
        self._station_widget.value = paths[0] if paths else None

    def _active_station_dataframe(self) -> pd.DataFrame:
        if self._station_df_full is None:
            return pd.DataFrame()
        return self._station_df_full.copy()

    @staticmethod
    def _available_components(df: pd.DataFrame, prefix: str) -> list[str]:
        components = []
        for component in (
            ["xx", "xy", "yx", "yy", "det"] if prefix != "t" else ["zx", "zy"]
        ):
            if prefix == "t":
                if (
                    f"t_{component}" in df.columns
                    or f"t_{component}_error" in df.columns
                ):
                    components.append(component)
            else:
                if f"{prefix}_{component}" in df.columns:
                    components.append(component)
        return components

    def _build_component_frame(
        self,
        df: pd.DataFrame,
        prefix: str,
        component: str | None,
        view: str | None = None,
    ) -> _FrameBundle:
        if df.empty or component is None:
            return _FrameBundle(
                pd.DataFrame(columns=["period", "value", "error", "use"]),
                f"{prefix.upper()}",
            )

        frequency = df.get("frequency")
        if frequency is None:
            return _FrameBundle(
                pd.DataFrame(columns=["period", "value", "error", "use"]),
                f"{prefix.upper()} {component}",
            )

        period = 1.0 / pd.to_numeric(frequency, errors="coerce")
        use = np.ones(len(df), dtype=bool)

        if prefix in {"res", "phase"}:
            value_col = f"{prefix}_{component}"
            error_col = f"{prefix}_{component}_error"
            if value_col not in df.columns:
                return _FrameBundle(
                    pd.DataFrame(columns=["period", "value", "error", "use"]),
                    f"{prefix.upper()} {component}",
                )
            value = pd.to_numeric(df[value_col], errors="coerce")
            error = pd.to_numeric(
                df.get(error_col, pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            use &= np.isfinite(period.to_numpy(dtype=float))
            use &= np.isfinite(value.to_numpy(dtype=float))
            use &= np.isfinite(error.to_numpy(dtype=float))
            if prefix == "res":
                use &= value.to_numpy(dtype=float) > 0
                use &= (value.to_numpy(dtype=float) - error.to_numpy(dtype=float)) > 0
            frame = pd.DataFrame(
                {
                    "period": period,
                    "value": value,
                    "error": error,
                    "use": use,
                }
            )
            frame = frame.loc[use].reset_index(drop=True)
            title = f"{prefix.upper()} {component.upper()}"
            return _FrameBundle(frame, title)

        # Tipper uses a real/imaginary view of the complex t_zx / t_zy columns.
        value_col = f"t_{component}"
        error_col = f"t_{component}_error"
        if value_col not in df.columns:
            if f"t_mag_{view}" in df.columns:
                value_col = f"t_mag_{view}"
            else:
                return _FrameBundle(
                    pd.DataFrame(columns=["period", "value", "error", "use"]),
                    f"Tipper {component}",
                )

        values = np.asarray(
            df[value_col], dtype=complex if value_col.startswith("t_") else float
        )
        if view == "imag" and value_col.startswith("t_"):
            values = np.imag(values)
        elif view == "real" and value_col.startswith("t_"):
            values = np.real(values)
        else:
            values = np.asarray(values, dtype=float)

        error = pd.to_numeric(
            df.get(error_col, pd.Series(np.nan, index=df.index)), errors="coerce"
        )
        use &= np.isfinite(period.to_numpy(dtype=float))
        use &= np.isfinite(values.astype(float, copy=False))
        use &= np.isfinite(error.to_numpy(dtype=float))
        frame = pd.DataFrame(
            {
                "period": period,
                "value": values,
                "error": error,
                "use": use,
            }
        )
        frame = frame.loc[use].reset_index(drop=True)
        view_label = view or "real"
        title = f"Tipper {component.upper()} ({view_label})"
        return _FrameBundle(frame, title)

    def _configure_editor(
        self,
        editor: TransferFunctionSeriesEditor,
        prefix: str,
        components: list[str],
        views: list[str] | None = None,
    ) -> None:
        df = self._active_station_dataframe()

        def builder(component: str | None, view: str | None) -> _FrameBundle:
            return self._build_component_frame(df, prefix, component, view)

        if not components:
            editor.set_frame_builder(builder, [], views or [], None, None)
            return

        component = components[0]
        view = views[0] if views else None
        editor.set_frame_builder(builder, components, views, component, view)

    def _get_or_create_editor(
        self, prefix: str, component: str, view: str | None = None
    ) -> TransferFunctionSeriesEditor:
        key = (prefix, component, view)
        editor = self._series_editors.get(key)
        if editor is not None:
            return editor

        title = (
            f"Tipper {component.upper()} ({view})"
            if prefix == "t"
            else f"{prefix.upper()} {component.upper()}"
        )
        y_axis_type = "log" if prefix == "res" else "linear"
        color = self._component_color(prefix, component, view)
        editor = TransferFunctionSeriesEditor(
            title=title,
            y_axis_type=y_axis_type,
            color=color,
        )
        editor._component_widget.visible = False
        editor._view_widget.visible = False
        self._series_editors[key] = editor
        return editor

    def _selected_edit_components(
        self,
    ) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        """Resolve plotted components from the 1D/2D/3D selector widgets."""

        df = self._active_station_dataframe()
        res_available = self._available_components(df, "res")
        phase_available = self._available_components(df, "phase")
        tip_available = self._available_components(df, "t")

        dimensionality = str(self._edit_dimension_widget.value)
        mode = str(self._edit_mode_widget.value)

        if dimensionality == "1D":
            requested = [mode]
            tip_requested = [("zx", "real")]
        elif dimensionality == "2D":
            requested = ["det"] if mode == "det" else ["xy", "yx"]
            tip_requested = [("zx", "real"), ("zy", "real")]
        else:
            requested = ["xx", "xy", "yx", "yy"]
            tip_requested = [
                ("zx", "real"),
                ("zx", "imag"),
                ("zy", "real"),
                ("zy", "imag"),
            ]

        res_components = [c for c in requested if c in res_available]
        phase_components = [c for c in requested if c in phase_available]
        tip_specs = [spec for spec in tip_requested if spec[0] in tip_available]
        return res_components, phase_components, tip_specs

    def _configure_fixed_editor(
        self,
        editor: TransferFunctionSeriesEditor,
        prefix: str,
        component: str,
        view: str | None = None,
    ) -> None:
        def builder(_component: str | None, _view: str | None) -> _FrameBundle:
            return self._build_component_frame(
                self._active_station_dataframe(),
                prefix=prefix,
                component=component,
                view=view,
            )

        view_options = [view] if view is not None else None
        editor.set_frame_builder(
            builder,
            component_options=[component],
            view_options=view_options,
            initial_component=component,
            initial_view=view,
        )

    def _refresh_editor_rows(self) -> None:
        """Rebuild resistivity/phase/tipper rows for current 1D/2D/3D setting."""

        if self._station_df_full is None:
            self._res_row_container.objects = [self._res_placeholder]
            self._phase_row_container.objects = [self._phase_placeholder]
            self._tipper_row_container.objects = [self._tipper_placeholder]
            self._active_res_editors = []
            self._active_phase_editors = []
            self._active_tipper_editors = []
            return

        res_components, phase_components, tip_specs = self._selected_edit_components()

        self._active_res_editors = []
        self._active_phase_editors = []
        self._active_tipper_editors = []

        for component in res_components:
            editor = self._get_or_create_editor("res", component)
            self._configure_fixed_editor(editor, "res", component)
            self._active_res_editors.append(editor)

        for component in phase_components:
            editor = self._get_or_create_editor("phase", component)
            self._configure_fixed_editor(editor, "phase", component)
            self._active_phase_editors.append(editor)

        for component, view in tip_specs:
            editor = self._get_or_create_editor("t", component, view)
            self._configure_fixed_editor(editor, "t", component, view)
            self._active_tipper_editors.append(editor)

        self._res_row_container.objects = (
            [ed.layout for ed in self._active_res_editors]
            if self._active_res_editors
            else [
                pn.pane.Markdown("_No apparent resistivity data for this selection._")
            ]
        )
        self._phase_row_container.objects = (
            [ed.layout for ed in self._active_phase_editors]
            if self._active_phase_editors
            else [pn.pane.Markdown("_No phase data for this selection._")]
        )
        self._tipper_row_container.objects = (
            [ed.layout for ed in self._active_tipper_editors]
            if self._active_tipper_editors
            else [pn.pane.Markdown("_No tipper data for this selection._")]
        )

    def _on_dimension_changed(self, _event=None) -> None:
        dimensionality = str(self._edit_dimension_widget.value)
        if dimensionality == "1D":
            self._edit_mode_widget.options = ["xy", "yx", "det"]
            if self._edit_mode_widget.value not in self._edit_mode_widget.options:
                self._edit_mode_widget.value = "xy"
            self._edit_mode_widget.disabled = False
        elif dimensionality == "2D":
            self._edit_mode_widget.options = ["xy/yx", "det"]
            if self._edit_mode_widget.value not in self._edit_mode_widget.options:
                self._edit_mode_widget.value = "xy/yx"
            self._edit_mode_widget.disabled = False
        else:
            self._edit_mode_widget.options = ["full tensor"]
            self._edit_mode_widget.value = "full tensor"
            self._edit_mode_widget.disabled = True

        self._refresh_editor_rows()

    def _load_station_into_editors(self) -> None:
        self._refresh_editor_rows()

    def _get_cached_component_frame(
        self, prefix: str, component: str, view: str | None = None
    ) -> pd.DataFrame:
        editor = self._series_editors.get((prefix, component, view))
        if editor is None:
            return self._build_component_frame(
                self._active_station_dataframe(), prefix, component, view
            ).frame
        return editor.get_frame(component=component, view=view)

    def _on_load_station_clicked(self, _event=None) -> None:
        if self._mt_data is None:
            self._status.object = "⚠️ No MTData loaded."
            self._status.styles = {"color": "#7a5200"}
            return

        station_key = self._station_widget.value
        if not station_key:
            self._status.object = "⚠️ Select a station first."
            self._status.styles = {"color": "#7a5200"}
            return

        try:
            mt_obj = self._mt_data.get_station(station_key, as_mt=True)
            station_df = self._sanitize_station_dataframe(
                mt_obj.to_dataframe().dataframe
            )
            self._station_df_full = station_df
            self._load_station_into_editors()
            self._output.object = ""
            self._status.object = (
                f"✅ Loaded station `{station_key}` for transfer-function editing."
            )
            self._status.styles = {"color": "#1a6600"}
        except Exception as exc:
            self._status.object = (
                f"❌ Error loading station: `{type(exc).__name__}: {exc}`"
            )
            self._status.styles = {"color": "#b00020"}
            self._output.object = traceback.format_exc()

    def _build_inversion_dataframe(self) -> pd.DataFrame:
        """Build Simpeg1D input dataframe from edited resistivity/phase frames."""

        mode = str(self._inv_mode_widget.value)
        res_df = self._get_cached_component_frame("res", mode)
        phase_df = self._get_cached_component_frame("phase", mode)
        if res_df.empty or phase_df.empty:
            return pd.DataFrame(
                columns=[
                    "frequency",
                    "res",
                    "res_error",
                    "phase",
                    "phase_error",
                    "use",
                ]
            )

        merged = pd.merge(
            res_df.rename(
                columns={
                    "value": "res",
                    "error": "res_error",
                    "use": "use_res",
                }
            )[["period", "res", "res_error", "use_res"]],
            phase_df.rename(
                columns={
                    "value": "phase",
                    "error": "phase_error",
                    "use": "use_phase",
                }
            )[["period", "phase", "phase_error", "use_phase"]],
            on="period",
            how="inner",
        )
        if merged.empty:
            return pd.DataFrame(
                columns=[
                    "frequency",
                    "res",
                    "res_error",
                    "phase",
                    "phase_error",
                    "use",
                ]
            )

        merged["use"] = merged["use_res"].astype(bool) & merged["use_phase"].astype(
            bool
        )
        merged["frequency"] = 1.0 / pd.to_numeric(merged["period"], errors="coerce")
        inv_df = merged[
            ["frequency", "res", "res_error", "phase", "phase_error", "use"]
        ].copy()

        for col in ["frequency", "res", "res_error", "phase", "phase_error"]:
            inv_df[col] = pd.to_numeric(inv_df[col], errors="coerce")

        valid = np.isfinite(inv_df["frequency"]) & (inv_df["frequency"] > 0)
        valid &= np.isfinite(inv_df["res"]) & np.isfinite(inv_df["res_error"])
        valid &= np.isfinite(inv_df["phase"]) & np.isfinite(inv_df["phase_error"])
        inv_df = inv_df.loc[valid].reset_index(drop=True)
        return inv_df

    def _build_inversion_response_figure(self, run_df: pd.DataFrame):
        """Create observed/predicted response plot for the inversion tab."""

        if run_df is None or run_df.empty:
            return None

        plot_df = run_df.copy()
        plot_df["period"] = 1.0 / plot_df["frequency"].astype(float)
        plot_df["phase_plot"] = np.where(
            plot_df["phase"] < 0.0,
            plot_df["phase"] + 180.0,
            plot_df["phase"],
        )
        plot_df["res_upper"] = plot_df["res"] + plot_df["res_error"]
        plot_df["res_lower"] = np.clip(
            plot_df["res"] - plot_df["res_error"], 1e-12, None
        )
        plot_df["phase_upper"] = plot_df["phase_plot"] + plot_df["phase_error"]
        plot_df["phase_lower"] = plot_df["phase_plot"] - plot_df["phase_error"]

        source = ColumnDataSource(plot_df)

        res_fig = figure(
            title="Observed vs Predicted Apparent Resistivity",
            x_axis_type="log",
            y_axis_type="log",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            height=360,
            width=1000,
        )
        res_fig.scatter("period", "res", source=source, size=7, color="#325ea8")
        res_fig.add_layout(
            Whisker(
                source=source,
                base="period",
                upper="res_upper",
                lower="res_lower",
                line_color="#325ea8",
            )
        )
        res_fig.yaxis.axis_label = "Resistivity (ohm-m)"

        phase_fig = figure(
            title="Observed vs Predicted Phase",
            x_axis_type="log",
            x_range=res_fig.x_range,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            height=320,
            width=1000,
        )
        phase_fig.scatter(
            "period", "phase_plot", source=source, size=7, color="#b24b3f"
        )
        phase_fig.add_layout(
            Whisker(
                source=source,
                base="period",
                upper="phase_upper",
                lower="phase_lower",
                line_color="#b24b3f",
            )
        )
        phase_fig.yaxis.axis_label = "Phase (deg)"
        phase_fig.xaxis.axis_label = "Period (s)"

        if self._inv_simpeg is not None and self._inv_simpeg.output_dict:
            final_iter = sorted(self._inv_simpeg.output_dict.keys())[-1]
            dpred = self._inv_simpeg.output_dict[final_iter]["dpred"].reshape((-1, 2))
            periods = self._inv_simpeg.periods.to_numpy(dtype=float)
            phase_model = self._inv_simpeg._phase_for_plotting(dpred[:, 1])
            res_fig.line(periods, dpred[:, 0], color="#1f7a1f", line_width=2)
            phase_fig.line(periods, phase_model, color="#1f7a1f", line_width=2)

        return bk_column(res_fig, phase_fig, sizing_mode="stretch_width")

    def _build_inversion_model_figure(self):
        """Create recovered layered resistivity model figure."""

        if self._inv_simpeg is None or not self._inv_simpeg.output_dict:
            return None

        final_iter = sorted(self._inv_simpeg.output_dict.keys())[-1]
        model_m = self._inv_simpeg.output_dict[final_iter]["m"]
        rho = 1.0 / np.exp(model_m)
        z = self._inv_simpeg._plot_z

        x_step = np.repeat(rho, 2)
        y_step = np.repeat(z[:-1], 2)
        y_step = np.r_[y_step, z[-1]]
        x_step = np.r_[x_step, rho[-1]]

        fig = figure(
            title="Recovered 1D Resistivity Model",
            x_axis_type="log",
            y_axis_type="linear",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            height=700,
            width=220,
        )
        fig.line(x_step, y_step, line_width=2, color="#111111")
        fig.y_range.start = float(np.nanmax(z)) if np.size(z) > 0 else 1.0
        fig.y_range.end = 0.0
        fig.xaxis.axis_label = "Resistivity (ohm-m)"
        fig.yaxis.axis_label = "Depth (km)"
        return fig

    def _refresh_inversion_plots(self, run_df: pd.DataFrame) -> None:
        """Refresh response/model panes in the inversion section."""

        self._inv_response_plot.object = self._build_inversion_response_figure(run_df)
        self._inv_model_plot.object = self._build_inversion_model_figure()

    def _on_run_inversion_clicked(self, _event=None) -> None:
        """Run Simpeg1D inversion from edited resistivity/phase data."""

        if _Simpeg1D is None:
            self._inv_status.object = "❌ Simpeg1D import failed. Install optional modeling dependencies first."
            self._inv_status.styles = {"color": "#b00020"}
            return
        if self._station_df_full is None:
            self._inv_status.object = "⚠️ Load a station before running inversion."
            self._inv_status.styles = {"color": "#7a5200"}
            return

        try:
            inv_df = self._build_inversion_dataframe()
            run_df = inv_df.loc[inv_df["use"].astype(bool)].copy()
            if run_df.empty:
                raise ValueError(
                    "No active edited points for the selected mode. Apply/edit points first."
                )

            self._inv_status.object = "⏳ Running Simpeg1D inversion..."
            self._inv_status.styles = {"color": "#555"}
            mode = _MODE_MAP[str(self._inv_mode_widget.value)]

            sim = _Simpeg1D(
                mt_dataframe=self._station_df_full,
                mode=mode,
                resistivity_error=10.0,
                phase_error=2.5,
            )
            sim.n_layers = int(self._inv_n_layers_widget.value)
            sim.dz = float(self._inv_dz_widget.value)
            sim.z_factor = float(self._inv_z_factor_widget.value)
            sim.rho_initial = float(self._inv_rho_initial_widget.value)
            sim.rho_reference = float(self._inv_rho_reference_widget.value)
            sim._sub_df = run_df[
                ["frequency", "res", "res_error", "phase", "phase_error"]
            ].copy()

            sim.run_fixed_layer_inversion(
                maxIter=int(self._inv_max_iter_widget.value),
                maxIterCG=int(self._inv_max_iter_cg_widget.value),
                alpha_s=float(self._inv_alpha_s_widget.value),
                alpha_z=float(self._inv_alpha_z_widget.value),
                beta0_ratio=float(self._inv_beta0_ratio_widget.value),
                coolingFactor=float(self._inv_cooling_factor_widget.value),
                coolingRate=int(self._inv_cooling_rate_widget.value),
                chi_factor=float(self._inv_chi_factor_widget.value),
                use_irls=bool(self._inv_use_irls_widget.value),
                p_s=float(self._inv_p_s_widget.value),
                p_z=float(self._inv_p_z_widget.value),
            )

            self._inv_simpeg = sim
            self._refresh_inversion_plots(run_df)

            final_iter = sorted(sim.output_dict.keys())[-1]
            final_fit = sim.output_dict[final_iter].get("f", np.nan)
            phi_d = sim.output_dict[final_iter].get("phi_d", np.nan)
            phi_m = sim.output_dict[final_iter].get("phi_m", np.nan)
            beta = sim.output_dict[final_iter].get("beta", np.nan)
            self._inv_output.object = (
                f"**Inversion complete**  \n"
                f"Final iteration: `{final_iter}`  \n"
                f"Final misfit: `{final_fit:.4g}`  \n"
                f"phi_d: `{phi_d:.4g}`  \n"
                f"phi_m: `{phi_m:.4g}`  \n"
                f"beta: `{beta:.4g}`"
            )
            self._inv_status.object = "✅ Inversion complete."
            self._inv_status.styles = {"color": "#1a6600"}
        except Exception as exc:
            self._inv_output.object = ""
            self._inv_status.object = (
                f"❌ Inversion error: `{type(exc).__name__}: {exc}`  \n"
                f"{traceback.format_exc()}"
            )
            self._inv_status.styles = {"color": "#b00020"}

    @property
    def view(self):
        return pn.Column(
            pn.pane.Markdown("### Transfer-Function Editor"),
            pn.pane.Markdown(
                "_Edit apparent resistivity, phase, and tipper data before modeling._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(self._station_widget, self._load_button, align="end"),
            pn.Row(self._edit_dimension_widget, self._edit_mode_widget, align="end"),
            self._future_tools,
            pn.pane.Markdown("#### Apparent Resistivity"),
            self._res_row_container,
            pn.pane.Markdown("#### Phase"),
            self._phase_row_container,
            pn.pane.Markdown("#### Tipper"),
            self._tipper_row_container,
            pn.layout.Divider(),
            pn.pane.Markdown("### 1D Inversion"),
            pn.pane.Markdown(
                "_Run Simpeg1D directly from the edited transfer-function points._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(self._inv_mode_widget, self._inv_run_button, align="end"),
            pn.pane.Markdown("#### Model Parameters"),
            pn.Row(
                self._inv_n_layers_widget,
                self._inv_dz_widget,
                self._inv_z_factor_widget,
                self._inv_rho_initial_widget,
                self._inv_rho_reference_widget,
                align="start",
            ),
            pn.pane.Markdown("#### Inversion Parameters"),
            pn.Row(
                self._inv_max_iter_widget,
                self._inv_max_iter_cg_widget,
                self._inv_alpha_s_widget,
                self._inv_alpha_z_widget,
                self._inv_beta0_ratio_widget,
                self._inv_cooling_factor_widget,
                align="start",
            ),
            pn.Row(
                self._inv_cooling_rate_widget,
                self._inv_chi_factor_widget,
                self._inv_use_irls_widget,
                self._inv_p_s_widget,
                self._inv_p_z_widget,
                align="start",
            ),
            self._inv_status,
            self._inv_output,
            pn.Row(self._inv_model_plot, self._inv_response_plot, align="start"),
            self._status,
            self._output,
            sizing_mode=self.sizing_mode,
        )

    def panel(self):
        """Return the Panel application layout."""

        return self.view


if __name__ == "__main__":
    pn.extension()
    app = TransferFunctionEditorPanelApp()
    app.panel().servable()
