"""Panel tab for editing a single station's MT response.

The tab keeps one :class:`EditMTResponse` (and its Panel layout) alive and
swaps new station data into it with ``set_data`` instead of rebuilding the
object, widgets and callbacks on every station selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import panel as pn
import param

from mtpy.core.mt_data import MTData
from mtpy.imaging.bokeh_plots.edit_mt_response import EditMTResponse


class MTResponseEditorTab(param.Parameterized):
    """Panel tab for editing a single station's transfer function.

    Wraps :class:`~mtpy.imaging.bokeh_plots.edit_mt_response.EditMTResponse`,
    which only exposes an edit-mode layout (mask points, add model error,
    interpolate, static shift, rotation, and phase flips) for one station at
    a time, without plotting the phase tensor.
    """

    sizing_mode: str = param.Selector(
        default="stretch_width",
        objects=["stretch_width", "fixed", "stretch_both", "stretch_height"],
        doc="Panel sizing mode for this tab's layout.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._mt_data = None
        # Optional hook called with the station path after Save Edits succeeds.
        self.on_station_saved = None
        # Persistent editor and its Panel layout, reused across stations.
        self._plot_obj = None
        self._plot_panel = None

        self._station_widget = pn.widgets.Select(name="Station", options=[], width=320)
        self._plot_response_all_widget = pn.widgets.Checkbox(
            name="Plot Response",
            value=False,
            visible=False,
        )
        self._plot_response_all_widget.param.watch(self._on_plot_all_toggled, "value")
        self._plot_response_widget = pn.widgets.CheckBoxGroup(
            name="Plot Responses (same station name)",
            options={},
            value=[],
            width=160,
            visible=False,
        )
        self._station_widget.param.watch(self._on_station_selected, "value")
        self._load_button = pn.widgets.Button(
            name="Load Station",
            button_type="primary",
            width=160,
            disabled=True,
        )
        self._load_button.on_click(self._on_load_clicked)
        self._status = pn.pane.Markdown(
            "_Load data in the **Data** tab, select a station, then click "
            "**Load Station**._",
            styles={"color": "#555"},
        )
        self._display = pn.Column(sizing_mode=self.sizing_mode)
        # Global error type: applies to every plot created afterwards.
        self._error_type_widget = pn.widgets.RadioButtonGroup(
            name="Error Type",
            options=["data", "model"],
            value="data",
            button_type="success",
        )
        self._rms_pane = pn.pane.Markdown(
            "", styles={"color": "#333", "font-size": "0.9em"}
        )

    @staticmethod
    def _rms_text(mt_obj: Any, responses: list) -> str:
        """RMS of (data - response) / model error per component, per response.

        Mirrors modem_plot_response_gui: only periods shared with the response
        and non-zero response values are included.
        """

        def _rms(data, resp, err, d_period, r_period, shape_idx):
            idx = np.where(np.isin(d_period, r_period))[0]
            ridx = np.where(np.isin(r_period, d_period))[0]
            if len(idx) == 0:
                return None
            ii, jj = shape_idx
            d = data[idx, ii, jj]
            r = resp[ridx, ii, jj]
            with np.errstate(divide="ignore", invalid="ignore"):
                diff = (d - r) / err[idx, ii, jj]
            valid = (r != 0) & np.isfinite(diff)
            if not valid.any():
                return None
            return float(np.sqrt(np.mean(np.abs(diff[valid]) ** 2)))

        lines = []
        for label, z_resp, t_resp in responses:
            parts = []
            if mt_obj.has_impedance() and z_resp is not None:
                err = mt_obj.Z.z_model_error
                if err is None or not np.any(err):
                    err = mt_obj.Z.z_error
                for name, ij in (
                    ("Zxx", (0, 0)),
                    ("Zxy", (0, 1)),
                    ("Zyx", (1, 0)),
                    ("Zyy", (1, 1)),
                ):
                    val = _rms(
                        mt_obj.Z.z, z_resp.z, err, mt_obj.Z.period, z_resp.period, ij
                    )
                    if val is not None:
                        parts.append(f"{name}={val:.2f}")
            if mt_obj.has_tipper() and t_resp is not None:
                err = mt_obj.Tipper.tipper_model_error
                if err is None or not np.any(err):
                    err = mt_obj.Tipper.tipper_error
                for name, ij in (("Tx", (0, 0)), ("Ty", (0, 1))):
                    val = _rms(
                        mt_obj.Tipper.tipper,
                        t_resp.tipper,
                        err,
                        mt_obj.Tipper.period,
                        t_resp.period,
                        ij,
                    )
                    if val is not None:
                        parts.append(f"{name}={val:.2f}")
            if parts:
                lines.append(f"**RMS vs {label}:** " + ", ".join(parts))
        return "  \n".join(lines)

    @staticmethod
    def _split_station_path(station_key: str) -> tuple[str, str, str, str] | None:
        """Split ``/{surveys}/{survey}/{stations}/{station}`` into its 4 parts."""
        parts = station_key.strip("/").split("/")
        return tuple(parts) if len(parts) == 4 else None

    def _same_name_station_paths(self, station_key: str) -> dict[str, str]:
        """Return ``{"survey/station": path}`` for other stations sharing *station_key*'s name.

        Surveys are unrestricted; the selected station itself is excluded.
        """
        parsed = self._split_station_path(station_key) if station_key else None
        if parsed is None:
            return {}
        _surveys_node, _survey, _stations_node, station = parsed
        matches: dict[str, str] = {}
        for path in self._station_widget.options or []:
            if path == station_key:
                continue
            other = self._split_station_path(path)
            if other is not None and other[3] == station:
                matches[f"{other[1]}/{other[3]}"] = path
        return matches

    def _on_station_selected(self, event: param.parameterized.Event) -> None:
        """Offer same-name stations from other surveys as plottable responses."""
        self._update_response_visibility(event.new)
        if self._mt_data is not None and event.new:
            self._on_load_clicked(event)

    def _on_plot_all_toggled(self, event: param.parameterized.Event) -> None:
        """Select (or clear) every same-name station in the response selector."""
        options = self._plot_response_widget.options or {}
        self._plot_response_widget.value = list(options.values()) if event.new else []

    def _update_response_visibility(self, station_key: str | None) -> None:
        """Refresh the response selector for the selected station.

        When 'Plot Response' is checked it stays checked across stations and
        auto-selects every same-name station for the new selection.
        """
        matches = self._same_name_station_paths(station_key) if station_key else {}
        self._plot_response_widget.options = matches
        self._plot_response_widget.value = (
            list(matches.values()) if self._plot_response_all_widget.value else []
        )
        self._plot_response_widget.visible = bool(matches)
        self._plot_response_all_widget.visible = bool(matches)

    def _reset_plot(self) -> None:
        """Drop the persistent editor so the next load builds a fresh one."""
        self._plot_obj = None
        self._plot_panel = None
        self._display.objects = []

    def set_mt_data(self, mt_data: MTData | None) -> None:
        """Refresh the station picker for a newly loaded (or cleared) MTData."""
        self._mt_data = mt_data
        self._reset_plot()

        if mt_data is None:
            self._station_widget.options = []
            self._load_button.disabled = True
            self._plot_response_widget.options = {}
            self._plot_response_widget.value = []
            self._plot_response_widget.visible = False
            self._plot_response_all_widget.visible = False
            self._status.object = (
                "_Load data in the **Data** tab, select a station, then click "
                "**Load Station**._"
            )
            self._status.styles = {"color": "#555"}
            return

        paths = list(mt_data._iter_station_paths())
        self._station_widget.options = paths
        if paths:
            self._station_widget.value = paths[0]
        self._update_response_visibility(self._station_widget.value)
        self._load_button.disabled = not bool(paths)
        self._status.object = "_Select a station and click **Load Station**._"
        self._status.styles = {"color": "#555"}

    def _show_station(
        self,
        mt_obj: Any,
        responses: list,
        station_key: str,
        plot_model_error: bool,
    ) -> None:
        """Plot *mt_obj*, reusing the persistent editor when one exists."""

        def _save_callback(z, t, mt_obj=mt_obj, key=station_key):
            self._save_edits(key, mt_obj, z, t)

        if self._plot_obj is not None and self._plot_panel is not None:
            self._plot_obj.set_data(
                z_object=mt_obj.Z,
                t_object=mt_obj.Tipper,
                responses=responses,
                station=mt_obj.station,
                save_edits_callback=_save_callback,
                plot_model_error=plot_model_error,
            )
        else:
            self._plot_obj = EditMTResponse(
                z_object=mt_obj.Z,
                t_object=mt_obj.Tipper,
                responses=responses,
                station=mt_obj.station,
                save_edits_callback=_save_callback,
                plot_model_error=plot_model_error,
                show_plot=False,
            )
            self._plot_panel = self._plot_obj.panel(sizing_mode=self.sizing_mode)
        self._display.objects = [self._plot_panel]

    def _on_load_clicked(self, event: param.parameterized.Event) -> None:
        if self._mt_data is None:
            self._status.object = "⚠️ No data loaded."
            self._status.styles = {"color": "#7a5200"}
            return

        station_key = self._station_widget.value
        if not station_key:
            self._status.object = "⚠️ No station selected."
            self._status.styles = {"color": "#7a5200"}
            return

        self._load_button.disabled = True
        self._rms_pane.object = ""
        self._status.object = f"⏳ Loading **{station_key}**…"
        self._status.styles = {"color": "#555"}

        try:
            # The selected station is always the editable primary; any chosen
            # same-name stations (from any survey) are overlaid as responses.
            mt_obj = self._mt_data.get_station(station_key, as_mt=True)
            labels = {
                path: label
                for label, path in (self._plot_response_widget.options or {}).items()
            }
            responses = []
            if self._plot_response_widget.visible:
                for path in self._plot_response_widget.value or []:
                    response_obj = self._mt_data.get_station(path, as_mt=True)
                    responses.append(
                        (labels.get(path, path), response_obj.Z, response_obj.Tipper)
                    )

            try:
                self._show_station(
                    mt_obj,
                    responses,
                    station_key,
                    self._error_type_widget.value == "model",
                )
            except Exception:
                # A failed in-place swap can leave the editor half updated;
                # rebuild it from scratch before giving up.
                if self._plot_obj is None:
                    raise
                self._reset_plot()
                self._show_station(
                    mt_obj,
                    responses,
                    station_key,
                    self._error_type_widget.value == "model",
                )
            self._rms_pane.object = self._rms_text(mt_obj, responses)
            self._status.object = f"✅ Loaded **{station_key}**."
            self._status.styles = {"color": "#1a6600"}
        except Exception as exc:
            self._reset_plot()
            self._status.object = f"❌ Error: `{type(exc).__name__}: {exc}`"
            self._status.styles = {"color": "#b00020"}
        finally:
            self._load_button.disabled = False

    def _save_edits(
        self, station_key: str, mt_obj: Any, z_obj: Any, t_obj: Any
    ) -> None:
        """Replace *station_key* in the MTData with the edited Z/Tipper."""
        if self._mt_data is None:
            raise RuntimeError("No data loaded.")

        if z_obj is not None:
            if mt_obj.period.size != z_obj.period.size:
                # TF cannot resize in place; reset to the new period grid
                mt_obj._transfer_function = mt_obj._initialize_transfer_function(
                    periods=z_obj.period
                )
            mt_obj.Z = z_obj
        if t_obj is not None:
            mt_obj.Tipper = t_obj
        self._mt_data.add_station(mt_obj, overwrite=True)

        self._status.object = f"💾 Saved edits to **{station_key}**."
        self._status.styles = {"color": "#1a6600"}
        if self.on_station_saved is not None:
            self.on_station_saved(station_key)

    @property
    def view(self) -> pn.viewable.Viewable:
        """Return the complete Panel layout for this tab."""
        return pn.Column(
            pn.pane.Markdown("### MT Response Editor"),
            pn.pane.Markdown(
                "_Edit mode only: mask points, add model error, interpolate, "
                "static shift, rotate, and flip phase for a single station._",
                styles={"color": "#777", "font-size": "0.85em"},
            ),
            pn.Row(
                self._station_widget,
                pn.Spacer(width=10),
                pn.Column(self._plot_response_all_widget, self._plot_response_widget),
                pn.Spacer(width=5),
                pn.Column(self._load_button, self._error_type_widget),
                pn.Spacer(width=10),
                pn.Column(self._status, self._rms_pane),
                align="start",
            ),
            pn.layout.Divider(),
            self._display,
            sizing_mode=self.sizing_mode,
        )
