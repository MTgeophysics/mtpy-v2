import copy

import numpy as np
from bokeh.events import DoubleTap, Tap
from bokeh.layouts import column, row, Spacer
from bokeh.models import BoxSelectTool, Button, ColumnDataSource, NumericInput, Select
from bokeh.plotting import curdoc, figure


class InteractivePointEditor:
    """
    Interactive Bokeh widget for selecting points on a scatter plot,
    marking them for deletion, or increasing their vertical error bars.
    Supports both linear and log axes.
    """

    def __init__(
        self,
        x,
        y,
        err_y=None,
        default_color="navy",
        default_size=8,
        default_marker="circle",
        x_axis_type="linear",
        y_axis_type="linear",
        selection_threshold=0.05,
        error_increase_factor=0.05,
    ):
        n = len(x)

        if err_y is None:
            err_y = np.array(y) * 0.1  # default 10% error

        self.original_data = dict(
            x=list(x),
            y=list(y),
            err_y=list(err_y),
            use=[True] * n,
            size=[default_size] * n,
            marker=[default_marker] * n,
            color=[default_color] * n,
        )
        self.default_color = default_color
        self.default_size = default_size
        self.default_marker = default_marker
        self.x_axis_type = x_axis_type
        self.y_axis_type = y_axis_type
        self.selection_threshold = selection_threshold
        self.error_increase_factor = error_increase_factor

        self.source = ColumnDataSource(data=copy.deepcopy(self.original_data))
        self.marked_for_deletion = set()

        self.mode_select = Select(
            title="Click Mode:",
            value="Mark for Deletion",
            options=["Mark for Deletion", "Increase Error"],
        )

        self.error_factor_input = NumericInput(
            title="Error Increase Factor",
            value=self.error_increase_factor,
            mode="float",
            low=0.01,
            high=0.9,
            format="0.000",
        )

        self.threshold_input = NumericInput(
            title="Selection Threshold (relative)",
            value=self.selection_threshold,
            mode="float",
            low=0.001,
            high=1.0,
        )
        self.threshold_input.on_change("value", self._update_threshold)

        self.apply_button = Button(label="Apply Edits", button_type="danger")
        self.apply_button.on_click(self.apply_edits)

        self.revert_button = Button(label="Revert Changes", button_type="success")
        self.revert_button.on_click(self.revert_changes)

        self.plot = figure(
            title="Click points to edit, Apply to confirm deletions",
            tools="tap,box_select",
            width=700,
            height=500,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
        )

        self.plot.segment(
            x0="x",
            y0="y_err_bottom",
            x1="x",
            y1="y_err_top",
            source=self.source,
            color="color",
            line_width=1,
        )

        self.plot.scatter(
            "x", "y", size="size", color="color", marker="marker", source=self.source
        )
        self.plot.add_tools(BoxSelectTool(dimensions="both"))

        self._update_error_columns()

        self.plot.on_event(Tap, self.tap_callback)
        self.plot.on_event(DoubleTap, self.clear_selection)
        self.source.selected.on_change("indices", self.box_select_callback)

        self.layout = column(
            row(self.mode_select, self.error_factor_input, self.threshold_input),
            Spacer(height=10),
            self.plot,
            Spacer(height=10),
            row(self.apply_button, self.revert_button),
        )

    def _update_threshold(self, attr, old, new):
        self.selection_threshold = new

    def _update_error_columns(self):
        y = np.array(self.source.data["y"])
        err_y = np.array(
            self.source.data["err_y"]
        )  # fractional error, e.g., 0.2 for 20%

        if getattr(self, "y_axis_type", "linear") == "log":
            # Multiplicative symmetric error bars
            y_err_top = y * (1 + err_y)
            y_err_bottom = y / (1 + err_y)
        else:
            # Standard additive error bars
            y_err_top = y + err_y
            y_err_bottom = y - err_y

        # Ensure no invalid values for log scale
        if getattr(self, "y_axis_type", "linear") == "log":
            y_err_bottom = np.maximum(y_err_bottom, 1e-12)

        self.source.data["y_err_top"] = list(y_err_top)
        self.source.data["y_err_bottom"] = list(y_err_bottom)

    def _find_nearest_point(self, event):
        click_x = event.x
        click_y = event.y
        x_vals = np.array(self.source.data["x"])
        y_vals = np.array(self.source.data["y"])

        if self.x_axis_type == "log":
            click_x = np.log10(click_x)
            x_vals = np.log10(x_vals)
        if self.y_axis_type == "log":
            click_y = np.log10(click_y)
            y_vals = np.log10(y_vals)

        dist_sq = (x_vals - click_x) ** 2 + (y_vals - click_y) ** 2
        idx = int(np.argmin(dist_sq))

        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()
        max_range = max(x_range, y_range)
        if np.sqrt(dist_sq[idx]) > self.selection_threshold * max_range:
            return None
        return idx

    def tap_callback(self, event):
        idx = self._find_nearest_point(event)
        if idx is None:
            return

        if self.mode_select.value == "Mark for Deletion":
            self.marked_for_deletion.add(idx)
            self.source.data["marker"][idx] = "x"
            self.source.data["color"][idx] = "black"

        elif self.mode_select.value == "Increase Error":
            factor = self.error_factor_input.value
            self.source.data["err_y"][idx] *= 1 + factor
            self.source.data["color"][idx] = (
                self.default_color if self.source.data["use"][idx] else "gray"
            )
            self._update_error_columns()
            print(
                f"Updated error for point {idx}: new err_y = {self.source.data['err_y'][idx]}"
                f" (error factor {factor})"
            )

        self.source.trigger("data", self.source.data, self.source.data)

    def box_select_callback(self, attr, old, new):
        if not new:
            return

        if self.mode_select.value == "Mark for Deletion":
            for i in new:
                self.marked_for_deletion.add(i)
                self.source.data["marker"][i] = "x"
                self.source.data["color"][i] = "black"

        elif self.mode_select.value == "Increase Error":
            factor = self.error_factor_input.value
            for i in new:
                self.source.data["err_y"][i] *= 1 + factor
                self.source.data["color"][i] = (
                    self.default_color if self.source.data["use"][i] else "gray"
                )
            self._update_error_columns()

        self.source.selected.indices = []
        self.source.trigger("data", self.source.data, self.source.data)

    def apply_edits(self):
        for i in self.marked_for_deletion:
            self.source.data["use"][i] = False
            self.source.data["marker"][i] = self.default_marker
            self.source.data["color"][i] = "gray"
            self.source.data["size"][i] = self.default_size
        self.marked_for_deletion.clear()
        self.source.trigger("data", self.source.data, self.source.data)

    def revert_changes(self):
        self.source.data = copy.deepcopy(self.original_data)
        self._update_error_columns()
        self.marked_for_deletion.clear()

    def clear_selection(self, event=None):
        """
        Clear any current selection without reverting all changes.
        This is triggered by a double-click on the plot.
        """
        # Clear Bokeh's selection
        self.source.selected.indices = []

        # Reset any temporary orange color from "Increase Error" mode
        for i in range(len(self.source.data["color"])):
            if self.source.data["color"][i] == "orange":
                self.source.data["color"][i] = self.default_color

        # Trigger a data refresh
        self.source.trigger("data", self.source.data, self.source.data)


# --- Example usage with log-spaced test data ---
import numpy as np


# Create some example data
x_vals = np.logspace(-3, 3, 20)  # from 0.001 to 1000
y_vals = np.logspace(-3, 3, 20)  # same range
err_y_vals = y_vals * 0.1  # 10% vertical error

# Create the interactive editor instance
editor = InteractivePointEditor(
    x=x_vals,
    y=y_vals,
    err_y=err_y_vals,
    x_axis_type="log",  # try "linear" for normal axes
    y_axis_type="log",
)

# Add the editor layout to the Bokeh document
curdoc().add_root(editor.layout)
curdoc().title = "Interactive Point Editor"
