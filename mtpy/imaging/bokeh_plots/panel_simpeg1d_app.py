"""Panel app for interactive Simpeg1D setup, editing, and inversion."""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure

from mtpy.modeling.simpeg.recipes.inversion_1d import Simpeg1D


_MODE_MAP: dict[str, str] = {
    "xy": "te",
    "yx": "tm",
    "det": "det",
}


class Simpeg1DPanelApp(param.Parameterized):
    """Interactive Panel app for Simpeg1D inversion from an MTData station."""

    sizing_mode = param.Selector(
        default="stretch_width",
        objects=["stretch_width", "fixed", "stretch_both", "stretch_height"],
        doc="Panel sizing mode",
    )

    def __init__(self, mt_data=None, **params: Any) -> None:
        super().__init__(**params)

        self._mt_data = mt_data
        self._simpeg = None
        self._station_df_full = None
        self._data_df = pd.DataFrame(
            columns=["frequency", "res", "res_error", "phase", "phase_error", "use"]
        )
        self._source = ColumnDataSource(data={})

        self._station_widget = pn.widgets.Select(
            name="Station",
            options=[],
            value=None,
            width=420,
        )
        self._mode_widget = pn.widgets.RadioButtonGroup(
            name="Mode",
            options=["xy", "yx", "det"],
            value="det",
            button_type="default",
        )
        self._load_button = pn.widgets.Button(
            name="Load Station Data",
            button_type="primary",
            width=180,
        )
        self._load_button.on_click(self._on_load_station_clicked)

        self._res_model_error_widget = pn.widgets.FloatInput(
            name="Res model error (%)",
            value=10.0,
            step=0.5,
            width=170,
        )
        self._res_model_error_widget.param.watch(
            lambda *_: self._update_data_errors(), "value"
        )

        self._phase_model_error_widget = pn.widgets.FloatInput(
            name="Phase model error (deg)",
            value=2.5,
            step=0.1,
            width=170,
        )
        self._phase_model_error_widget.param.watch(
            lambda *_: self._update_data_errors(), "value"
        )

        self._selected_res_error_widget = pn.widgets.FloatInput(
            name="Selected res error add (%)",
            value=10.0,
            step=0.5,
            width=180,
        )
        self._selected_phase_error_widget = pn.widgets.FloatInput(
            name="Selected phase error add (deg)",
            value=2.5,
            step=0.1,
            width=190,
        )
        self._apply_error_button = pn.widgets.Button(
            name="Apply Errors to Selected",
            button_type="warning",
            width=220,
        )
        self._apply_error_button.on_click(self._on_apply_errors_clicked)

        self._delete_selected_button = pn.widgets.Button(
            name="Delete Selected Points",
            button_type="danger",
            width=180,
        )
        self._delete_selected_button.on_click(self._on_delete_selected_clicked)

        self._n_layers_widget = pn.widgets.IntInput(
            name="N layers",
            value=50,
            step=1,
            start=2,
            end=400,
            width=120,
        )
        self._dz_widget = pn.widgets.FloatInput(
            name="First layer thickness (m)",
            value=5.0,
            step=0.5,
            width=200,
        )
        self._z_factor_widget = pn.widgets.FloatInput(
            name="Layer growth factor",
            value=1.2,
            step=0.05,
            width=170,
        )
        self._rho_initial_widget = pn.widgets.FloatInput(
            name="Starting resistivity (ohm-m)",
            value=100.0,
            step=5.0,
            width=220,
        )
        self._rho_reference_widget = pn.widgets.FloatInput(
            name="Reference resistivity (ohm-m)",
            value=100.0,
            step=5.0,
            width=220,
        )

        self._max_iter_widget = pn.widgets.IntInput(
            name="maxIter",
            value=40,
            step=1,
            start=1,
            end=500,
            width=110,
        )
        self._max_iter_cg_widget = pn.widgets.IntInput(
            name="maxIterCG",
            value=30,
            step=1,
            start=1,
            end=500,
            width=110,
        )
        self._alpha_s_widget = pn.widgets.FloatInput(
            name="alpha_s",
            value=1e-10,
            step=1e-10,
            width=120,
        )
        self._alpha_z_widget = pn.widgets.FloatInput(
            name="alpha_z",
            value=1.0,
            step=0.1,
            width=120,
        )
        self._beta0_ratio_widget = pn.widgets.FloatInput(
            name="beta0_ratio",
            value=1.0,
            step=0.1,
            width=130,
        )
        self._cooling_factor_widget = pn.widgets.FloatInput(
            name="coolingFactor",
            value=2.0,
            step=0.1,
            width=130,
        )
        self._cooling_rate_widget = pn.widgets.IntInput(
            name="coolingRate",
            value=1,
            step=1,
            start=1,
            end=50,
            width=120,
        )
        self._chi_factor_widget = pn.widgets.FloatInput(
            name="chi_factor",
            value=1.0,
            step=0.1,
            width=120,
        )
        self._use_irls_widget = pn.widgets.Checkbox(name="use_irls", value=False)
        self._p_s_widget = pn.widgets.FloatInput(
            name="p_s",
            value=2.0,
            step=0.1,
            width=90,
        )
        self._p_z_widget = pn.widgets.FloatInput(
            name="p_z",
            value=2.0,
            step=0.1,
            width=90,
        )

        self._run_button = pn.widgets.Button(
            name="Run Inversion",
            button_type="success",
            width=150,
        )
        self._run_button.on_click(self._on_run_clicked)

        self._status = pn.pane.Markdown(
            "_Load a station to begin 1D modeling._",
            styles={"color": "#555"},
        )
        self._output = pn.pane.Markdown("", styles={"color": "#333"})

        self._data_table = pn.widgets.Tabulator(
            self._data_df,
            selectable="checkbox",
            pagination="local",
            page_size=15,
            show_index=False,
            sizing_mode="stretch_width",
            editors={
                "res_error": {"type": "number", "step": 0.01},
                "phase_error": {"type": "number", "step": 0.01},
                "use": {"type": "tickCross"},
            },
            hidden_columns=[],
        )
        self._data_table.param.watch(self._on_table_value_changed, "value")

        self._response_plot = pn.pane.Bokeh(sizing_mode="stretch_width", min_height=760)
        self._model_plot = pn.pane.Bokeh(
            sizing_mode="stretch_width", min_height=760, max_width=200
        )

        self._refresh_station_options()

    @staticmethod
    def _sanitize_station_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric transfer-function columns to float/NaN.

        Some station files provide missing error values as ``None`` objects.
        Converting these columns to numeric avoids downstream type comparisons
        like ``float < NoneType`` inside Simpeg1D preprocessing.
        """

        clean_df = df.copy()
        numeric_cols = [
            "frequency",
            "res_xy",
            "res_xy_error",
            "phase_xy",
            "phase_xy_error",
            "res_yx",
            "res_yx_error",
            "phase_yx",
            "phase_yx_error",
            "z_xx",
            "z_xx_error",
            "z_xy",
            "z_xy_error",
            "z_yx",
            "z_yx_error",
            "z_yy",
            "z_yy_error",
        ]
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        return clean_df

    def set_mt_data(self, mt_data) -> None:
        """Set MTData reference and refresh station picker options."""

        self._mt_data = mt_data
        self._refresh_station_options()
        if mt_data is None:
            self._simpeg = None
            self._data_df = pd.DataFrame(
                columns=[
                    "frequency",
                    "res",
                    "res_error",
                    "phase",
                    "phase_error",
                    "use",
                ]
            )
            self._data_table.value = self._data_df
            self._response_plot.object = None
            self._model_plot.object = None
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

    def _update_data_errors(self):
        if self._data_df is None or self._data_df.empty:
            return

        res_pct = float(self._res_model_error_widget.value)
        phase_abs = float(self._phase_model_error_widget.value)

        self._data_df["res_error"] = np.maximum(
            self._data_df["res_error"].astype(float),
            np.abs(self._data_df["res"].astype(float)) * res_pct / 100.0,
        )
        self._data_df["phase_error"] = np.maximum(
            self._data_df["phase_error"].astype(float),
            np.ones(len(self._data_df), dtype=float) * phase_abs,
        )

        self._refresh_plots()

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
            mt_df_full = self._sanitize_station_dataframe(
                mt_obj.to_dataframe().dataframe
            )

            sim_mode = _MODE_MAP[self._mode_widget.value]
            sim = Simpeg1D(
                mt_dataframe=mt_df_full,
                mode=sim_mode,
                resistivity_error=float(self._res_model_error_widget.value),
                phase_error=float(self._phase_model_error_widget.value),
            )
            self._simpeg = sim
            self._station_df_full = mt_df_full

            self._data_df = sim._sub_df.copy()
            self._data_df["use"] = True

            self._data_table.value = self._data_df
            self._refresh_plots()

            self._output.object = ""
            self._status.object = (
                f"✅ Loaded station `{station_key}` in mode `{self._mode_widget.value}`."
            )
            self._status.styles = {"color": "#1a6600"}
        except Exception as exc:
            self._status.object = (
                f"❌ Error loading station: `{type(exc).__name__}: {exc}`"
            )
            self._status.styles = {"color": "#b00020"}

    def _on_table_value_changed(self, _event=None) -> None:
        if self._data_table.value is None:
            return
        self._data_df = self._data_table.value.copy()
        self._refresh_plots()

    def _on_delete_selected_clicked(self, _event=None) -> None:
        if self._data_df is None or self._data_df.empty:
            return

        selected = (
            list(self._source.selected.indices) if self._source is not None else []
        )
        if not selected:
            selected = list(self._data_table.selection or [])
        if not selected:
            self._status.object = "⚠️ Select one or more points to delete."
            self._status.styles = {"color": "#7a5200"}

            return

        self._data_df = self._data_df.drop(self._data_df.index[selected]).reset_index(
            drop=True
        )
        self._data_table.value = self._data_df
        self._refresh_plots()
        self._status.object = f"✅ Deleted {len(selected)} point(s)."
        self._status.styles = {"color": "#1a6600"}

    def _on_apply_errors_clicked(self, _event=None) -> None:
        if self._data_df is None or self._data_df.empty:
            return

        selected = (
            list(self._source.selected.indices) if self._source is not None else []
        )
        if not selected:
            selected = list(self._data_table.selection or [])
        if not selected:
            self._status.object = "⚠️ Select one or more points first."
            self._status.styles = {"color": "#7a5200"}
            return

        res_pct = float(self._selected_res_error_widget.value)
        phase_abs = float(self._selected_phase_error_widget.value)

        selected_res = self._data_df.loc[selected, "res"].to_numpy(dtype=float)
        add_res_err = np.abs(selected_res) * res_pct / 100.0
        cur_res_err = self._data_df.loc[selected, "res_error"].to_numpy(dtype=float)

        self._data_df.loc[selected, "res_error"] = np.maximum(cur_res_err, add_res_err)
        self._data_df.loc[selected, "phase_error"] = np.maximum(
            self._data_df.loc[selected, "phase_error"].to_numpy(dtype=float),
            np.ones(len(selected), dtype=float) * phase_abs,
        )

        self._data_table.value = self._data_df
        self._refresh_plots()
        self._status.object = (
            f"✅ Updated error bars for {len(selected)} selected point(s)."
        )
        self._status.styles = {"color": "#1a6600"}

    def _build_response_figure(self):
        if self._data_df is None or self._data_df.empty:
            return None

        plot_df = self._data_df.copy()
        plot_df["period"] = 1.0 / plot_df["frequency"].astype(float)
        plot_df["phase_plot"] = np.where(
            plot_df["phase"] < 0, plot_df["phase"] + 180.0, plot_df["phase"]
        )
        plot_df["res_upper"] = plot_df["res"] + plot_df["res_error"]
        plot_df["res_lower"] = np.clip(
            plot_df["res"] - plot_df["res_error"], 1e-12, None
        )
        plot_df["phase_upper"] = plot_df["phase_plot"] + plot_df["phase_error"]
        plot_df["phase_lower"] = plot_df["phase_plot"] - plot_df["phase_error"]

        self._source = ColumnDataSource(plot_df)

        res_fig = figure(
            title="Apparent Resistivity",
            x_axis_type="log",
            y_axis_type="log",
            tools="pan,wheel_zoom,box_zoom,reset,save,tap,box_select,lasso_select",
            active_scroll="wheel_zoom",
            height=380,
            width=1000,
            # sizing_mode="stretch_width",
        )
        res_fig.scatter("period", "res", source=self._source, size=7, color="#325ea8")

        res_whisker = Whisker(
            source=self._source,
            base="period",
            upper="res_upper",
            lower="res_lower",
            line_color="#325ea8",
        )
        res_fig.add_layout(res_whisker)
        res_fig.yaxis.axis_label = "Resistivity (ohm-m)"

        phase_fig = figure(
            title="Phase",
            x_axis_type="log",
            tools="pan,wheel_zoom,box_zoom,reset,save,tap,box_select,lasso_select",
            active_scroll="wheel_zoom",
            x_range=res_fig.x_range,
            height=340,
            width=1000,
            # sizing_mode="stretch_width",
        )
        phase_fig.scatter(
            "period", "phase_plot", source=self._source, size=7, color="#b24b3f"
        )
        phase_whisker = Whisker(
            source=self._source,
            base="period",
            upper="phase_upper",
            lower="phase_lower",
            line_color="#b24b3f",
        )
        phase_fig.add_layout(phase_whisker)
        phase_fig.yaxis.axis_label = "Phase (deg)"
        phase_fig.xaxis.axis_label = "Period (s)"

        if self._simpeg is not None and self._simpeg.output_dict:
            final_iter = sorted(self._simpeg.output_dict.keys())[-1]
            dpred = self._simpeg.output_dict[final_iter]["dpred"].reshape((-1, 2))
            p = self._simpeg.periods.to_numpy(dtype=float)
            phase_model = self._simpeg._phase_for_plotting(dpred[:, 1])
            res_fig.line(p, dpred[:, 0], color="#1f7a1f", line_width=2)
            phase_fig.line(p, phase_model, color="#1f7a1f", line_width=2)

        return column(res_fig, phase_fig, sizing_mode="stretch_width")

    def _build_model_figure(self):
        if self._simpeg is None or not self._simpeg.output_dict:
            return None

        final_iter = sorted(self._simpeg.output_dict.keys())[-1]
        model_m = self._simpeg.output_dict[final_iter]["m"]
        rho = 1.0 / np.exp(model_m)
        z = self._simpeg._plot_z

        # Build a step-like curve from rho/z
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
            height=720,
            width=200,
            # sizing_mode="stretch_width",
        )
        fig.line(x_step, y_step, line_width=2, color="#111111")
        fig.y_range.start = 30.0
        fig.y_range.end = 0.0
        fig.xaxis.axis_label = "Resistivity (ohm-m)"
        fig.yaxis.axis_label = "Depth (km)"
        return fig

    def _refresh_plots(self) -> None:
        response_layout = self._build_response_figure()
        model_fig = self._build_model_figure()

        self._response_plot.object = response_layout
        self._model_plot.object = model_fig

    def _on_run_clicked(self, _event=None) -> None:
        if self._station_df_full is None:
            self._status.object = "⚠️ Load station data before running inversion."
            self._status.styles = {"color": "#7a5200"}
            return

        if self._data_df is None or self._data_df.empty:
            self._status.object = "⚠️ No data points available for inversion."
            self._status.styles = {"color": "#7a5200"}
            return

        try:
            self._status.object = "⏳ Running Simpeg1D inversion..."
            self._status.styles = {"color": "#555"}

            sim = Simpeg1D(
                mt_dataframe=self._station_df_full,
                mode=_MODE_MAP[self._mode_widget.value],
                resistivity_error=float(self._res_model_error_widget.value),
                phase_error=float(self._phase_model_error_widget.value),
            )
            sim.n_layers = int(self._n_layers_widget.value)
            sim.dz = float(self._dz_widget.value)
            sim.z_factor = float(self._z_factor_widget.value)
            sim.rho_initial = float(self._rho_initial_widget.value)
            sim.rho_reference = float(self._rho_reference_widget.value)

            run_df = self._data_df[self._data_df["use"]].copy()
            if run_df.empty:
                raise ValueError("All points are disabled. Enable at least one row.")

            sim._sub_df = run_df[
                ["frequency", "res", "res_error", "phase", "phase_error"]
            ].copy()

            sim.run_fixed_layer_inversion(
                maxIter=int(self._max_iter_widget.value),
                maxIterCG=int(self._max_iter_cg_widget.value),
                alpha_s=float(self._alpha_s_widget.value),
                alpha_z=float(self._alpha_z_widget.value),
                beta0_ratio=float(self._beta0_ratio_widget.value),
                coolingFactor=float(self._cooling_factor_widget.value),
                coolingRate=int(self._cooling_rate_widget.value),
                chi_factor=float(self._chi_factor_widget.value),
                use_irls=bool(self._use_irls_widget.value),
                p_s=float(self._p_s_widget.value),
                p_z=float(self._p_z_widget.value),
            )

            self._simpeg = sim
            self._refresh_plots()

            final_iter = sorted(sim.output_dict.keys())[-1]
            final_fit = sim.output_dict[final_iter].get("f", np.nan)
            phi_d = sim.output_dict[final_iter].get("phi_d", np.nan)
            phi_m = sim.output_dict[final_iter].get("phi_m", np.nan)
            beta = sim.output_dict[final_iter].get("beta", np.nan)
            self._output.object = (
                f"**Inversion complete**  \n"
                f"Final iteration: `{final_iter}`  \n"
                f"Target misfit: `{sim.n_layers:.4g}`  \n"
                f"Final misfit: `{final_fit:.4g}`  \n"
                f"phi_d: `{phi_d:.4g}`  \n"
                f"phi_m: `{phi_m:.4g}`  \n"
                f"beta: `{beta:.4g}`"
            )

            self._status.object = "✅ Inversion complete."
            self._status.styles = {"color": "#1a6600"}
        except Exception as exc:
            self._output.object = ""
            self._status.object = (
                f"❌ Inversion error: `{type(exc).__name__}: {exc}`  "
                f"\n{traceback.format_exc()}"
            )
            self._status.styles = {"color": "#b00020"}

    @property
    def view(self):
        return pn.Column(
            pn.pane.Markdown("### Simpeg1D Modeling"),
            pn.pane.Markdown(
                "_Pick a station, choose mode, configure inversion settings, and run._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(
                self._station_widget, self._mode_widget, self._load_button, align="end"
            ),
            pn.Row(
                self._res_model_error_widget,
                self._phase_model_error_widget,
                align="end",
            ),
            pn.Row(
                self._selected_res_error_widget,
                self._selected_phase_error_widget,
                align="end",
            ),
            pn.Row(
                self._apply_error_button,
                self._delete_selected_button,
                align="end",
            ),
            self._data_table,
            pn.layout.Divider(),
            pn.pane.Markdown("#### Model Parameters"),
            pn.Row(
                self._n_layers_widget,
                self._dz_widget,
                self._z_factor_widget,
                self._rho_initial_widget,
                self._rho_reference_widget,
                align="start",
            ),
            pn.pane.Markdown("#### Inversion Parameters"),
            pn.Row(
                self._max_iter_widget,
                self._max_iter_cg_widget,
                self._alpha_s_widget,
                self._alpha_z_widget,
                self._beta0_ratio_widget,
                self._cooling_factor_widget,
                align="start",
            ),
            pn.Row(
                self._cooling_rate_widget,
                self._chi_factor_widget,
                self._use_irls_widget,
                self._p_s_widget,
                self._p_z_widget,
                align="start",
            ),
            self._run_button,
            self._status,
            self._output,
            pn.layout.Divider(),
            pn.Row(
                self._model_plot,
                self._response_plot,
                align="start",
            ),
            sizing_mode=self.sizing_mode,
        )
