"""Bokeh implementation of MT strike angle rose diagram plotting.

This module provides a Bokeh translation of
mtpy.imaging.plot_strike.PlotStrike for use in Panel dashboards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mtpy.core import Tipper
from mtpy.imaging.mtplot_tools import PlotBase

try:
    from bokeh.io import show as bokeh_show
    from bokeh.layouts import column as bokeh_column
    from bokeh.layouts import gridplot
    from bokeh.layouts import row as bokeh_row
    from bokeh.models import ColumnDataSource, Label, Range1d
    from bokeh.plotting import figure
except ImportError:  # pragma: no cover - optional dependency
    bokeh_show = None
    bokeh_column = None
    bokeh_row = None
    gridplot = None
    ColumnDataSource = None
    Label = None
    Range1d = None
    figure = None


class PlotStrike(PlotBase):
    """Bokeh rose-diagram plotter for MT strike angles.

    Mimics :class:`mtpy.imaging.plot_strike.PlotStrike` but produces
    interactive Bokeh figures instead of matplotlib axes, making it
    suitable for Panel-based web applications.

    Parameters
    ----------
    mt_data : MTData or dict-like
        Container of MT objects.  Must support ``values()`` iteration or
        the MTData ``_iter_station_paths`` / ``get_station`` protocol.
    **kwargs
        Any attribute of the class can be overridden via keyword argument.

    Attributes
    ----------
    strike_df : pd.DataFrame or None
        Built by :meth:`make_strike_df`.  Columns:
        ``estimate``, ``period``, ``plot_strike``, ``measured_strike``.
    layout : Bokeh layout or None
        The last layout produced by :meth:`plot`.
    figures : dict
        Mapping of label → individual Bokeh figure objects produced
        by the last call to :meth:`plot`.
    """

    def __init__(self, mt_data, **kwargs):
        self._rotation_angle = 0
        self.mt_data = mt_data

        super().__init__(**kwargs)

        # ---- plot-control attributes (mirror matplotlib version) ----
        self.plot_range = "data"
        self.plot_orientation = "h"
        self.plot_type = 2

        self.period_tolerance = 0.05
        self.pt_error_floor = None
        self.fold = False
        self.bin_width = 5
        self.color = True
        self.color_inv = (0.7, 0, 0.2)
        self.color_pt = (0.2, 0, 0.7)
        self.color_tip = (0.2, 0.65, 0.2)
        self.ring_spacing = 10
        self.ring_limits = None
        self.plot_orthogonal = False
        self.plot_pt = True
        self.plot_tipper = True
        self.plot_invariant = True
        self.print_stats = False

        self.title_dict = {
            -5: "1e-5 - 1e-4 s",
            -4: "1e-4 - 1e-3 s",
            -3: "1e-3 - 1e-2 s",
            -2: "1e-2 - 1e-1 s",
            -1: "1e-1 - 1e0 s",
            0: "1e0 - 1e1 s",
            1: "1e1 - 1e2 s",
            2: "1e2 - 1e3 s",
            3: "1e3 - 1e4 s",
            4: "1e4 - 1e5 s",
            5: "1e5 - 1e6 s",
            6: "1e6 - 1e7 s",
        }

        self.strike_df = None
        self.layout = None
        self.fig = None
        self.figures = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.show_plot:
            self.plot()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rotation_angle(self):
        """Rotation angle applied to all MT data."""
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value):
        """Rotate all stations and rebuild the strike DataFrame."""
        if hasattr(self.mt_data, "rotate") and hasattr(self.mt_data, "get_station"):
            self.mt_data.rotate(value, inplace=True)
        else:
            for mt in self._iter_mt_objects():
                mt.rotation_angle = value
        self._rotation_angle = value
        self.make_strike_df()

    # ------------------------------------------------------------------
    # Data methods (identical logic to matplotlib PlotStrike)
    # ------------------------------------------------------------------

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

        raise TypeError(
            "mt_data must provide values() or MTData-style station access"
        )

    def make_strike_df(self):
        """Build a DataFrame of strike angles from all MT objects.

        .. note::
            Polar plots assume the azimuth is measured counter-clockwise
            from the positive x-axis (East).  All angles are converted
            via ``270 - angle`` so that North plots at the top.
        """
        entries = []

        for mt in self._iter_mt_objects():
            if mt.has_impedance():
                zinv = mt.Z.invariants
                zs = 270 - zinv.strike
                for period, plot_strike, strike in zip(mt.period, zs, zinv.strike):
                    entries.append(
                        {
                            "estimate": "invariant",
                            "period": period,
                            "plot_strike": plot_strike,
                            "measured_strike": strike,
                        }
                    )

                pt = mt.pt
                az = 270 - pt.azimuth
                az_err = pt.azimuth_error
                az[pt.phimax == 0] = np.nan
                if self.pt_error_floor:
                    az[np.where(az_err > self.pt_error_floor)] = 0.0
                for period, plot_strike, strike in zip(mt.period, az, pt.azimuth):
                    entries.append(
                        {
                            "estimate": "pt",
                            "period": period,
                            "plot_strike": plot_strike,
                            "measured_strike": strike,
                        }
                    )

            if mt.has_tipper():
                tip = mt.Tipper
                if isinstance(tip, Tipper):
                    if tip.tipper is None:
                        tip = Tipper(
                            np.zeros((len(mt.period), 1, 2), dtype="complex"),
                            frequency=[1],
                        )

                tipr = 270 - tip.angle_real
                tipr[np.where(abs(tipr) == 0.0)] = np.nan
                tipr[np.where(abs(tipr) == 90)] = np.nan
                tipr[np.where(abs(tipr) == 180.0)] = np.nan
                tipr[np.where(abs(tipr) == 270)] = np.nan

                tip.angle_real[np.where(abs(tipr) == 0.0)] = np.nan
                tip.angle_real[np.where(abs(tipr) == 90)] = np.nan
                tip.angle_real[np.where(abs(tipr) == 180.0)] = np.nan
                tip.angle_real[np.where(abs(tipr) == 270)] = np.nan

                for period, plot_strike, strike in zip(
                    mt.period, tipr, tip.angle_real
                ):
                    entries.append(
                        {
                            "estimate": "tipper",
                            "period": period,
                            "plot_strike": plot_strike,
                            "measured_strike": strike,
                        }
                    )

        self.strike_df = pd.DataFrame(entries)

    def get_mean(self, estimate_df):
        """Return mean strike angle (0-360°)."""
        return estimate_df.measured_strike.mean(skipna=True) % 360

    def get_median(self, estimate_df):
        """Return median strike angle (0-360°)."""
        return estimate_df.measured_strike.median(skipna=True) % 360

    def get_mode(self, estimate_df):
        """Return histogram-mode strike angle (0-360°)."""
        bins = np.linspace(-360, 360, 146)
        binned = pd.cut(estimate_df["measured_strike"], bins).value_counts()
        s_mode = binned.index[np.argmax(binned)].mid
        return s_mode % 360

    def get_estimate(self, estimate, period_range=None):
        """Slice the strike DataFrame by estimate type and optional period range.

        Parameters
        ----------
        estimate : str
            One of ``"invariant"``, ``"pt"``, ``"tipper"``.
        period_range : tuple[float, float] or None
            ``(period_min, period_max)`` in seconds.
        """
        if period_range is None:
            return self.strike_df.loc[self.strike_df.estimate == estimate]

        estimate_df = self.strike_df.loc[self.strike_df.estimate == estimate]
        return estimate_df.loc[
            (self.strike_df.period >= period_range[0])
            & (self.strike_df.period < period_range[1])
        ]

    def get_stats(self, estimate, period_range=None):
        """Return ``(median, mode, mean)`` strike statistics.

        Parameters
        ----------
        estimate : str
            One of ``"invariant"``, ``"pt"``, ``"tipper"``.
        period_range : tuple or None
            Optional period range filter.

        Returns
        -------
        tuple[float, float, float]
            ``(s_median, s_mode, s_mean)``
        """
        estimate_df = self.get_estimate(estimate, period_range)
        s_mean = self.get_mean(estimate_df)
        s_median = self.get_median(estimate_df)
        s_mode = self.get_mode(estimate_df)

        msg = f"Strike statistics for {estimate} "
        if period_range is None:
            msg += "in all periods "
        else:
            msg += (
                f"period range {period_range[0]:.3g} to "
                f"{period_range[1]:.3g} (s) "
            )
        msg += f"median={s_median:.1f} mode={s_mode:.1f} mean={s_mean:.1f}"
        self.logger.debug(msg)
        if self.print_stats:
            print(msg)

        return s_median, s_mode, s_mean

    def get_plot_array(self, estimate, period_range=None):
        """Return angles (degrees) suitable for a rose histogram.

        Applies fold and orthogonal mirroring as configured.
        """
        estimate_df = self.get_estimate(estimate, period_range)
        st_array = estimate_df.plot_strike.to_numpy().flatten()
        st_array = st_array[np.isfinite(st_array)] % 360
        plot_array = np.hstack([st_array, (st_array + 180) % 360])

        if self.plot_orthogonal:
            plot_array = np.hstack([plot_array, (plot_array + 90) % 360])
        if self.fold:
            plot_array %= 180

        return plot_array

    def _get_histogram_range(self):
        """Return ``(min, max)`` for histogram binning."""
        return (0, 180) if self.fold else (0, 360)

    def _get_bin_range(self):
        """Return array of log10(period) decade start values to iterate."""
        if self.plot_range == "data":
            return np.arange(
                np.floor(np.log10(self.strike_df.period.min())),
                np.ceil(np.log10(self.strike_df.period.max())),
                1,
            )
        return np.arange(np.floor(self.plot_range[0]), np.ceil(self.plot_range[1]), 1)

    def _get_n_subplots(self):
        """Return number of enabled estimate types."""
        return sum([self.plot_invariant, self.plot_pt, self.plot_tipper])

    # ------------------------------------------------------------------
    # Bokeh helpers
    # ------------------------------------------------------------------

    def _require_bokeh(self):
        if figure is None or ColumnDataSource is None or Range1d is None:
            raise ImportError(
                "Bokeh is required for PlotStrike (Bokeh edition). "
                "Install it with `pip install bokeh`."
            )

    @staticmethod
    def _rgb_to_hex(rgb):
        """Convert an (r, g, b) tuple with values in [0, 1] to a CSS hex string."""
        r, g, b = [int(np.clip(c, 0, 1) * 255) for c in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _compute_bar_colors(self, hist, estimate):
        """Return a list of CSS hex color strings for each histogram bar.

        Mirrors the matplotlib ``_set_bar_color`` gradient logic.
        """
        max_count = hist.max() if hist.max() > 0 else 1
        colors = []
        for count in hist:
            if estimate == "invariant":
                if self.color:
                    fc = float(count) / max_count * 0.8
                    colors.append(self._rgb_to_hex((0.75, 1 - fc, 0)))
                else:
                    colors.append(self._rgb_to_hex(self.color_inv))
            elif estimate == "pt":
                if self.color:
                    fc = float(count) / max_count * 0.8
                    colors.append(self._rgb_to_hex((1 - fc, 0, 1 - fc)))
                else:
                    colors.append(self._rgb_to_hex(self.color_pt))
            elif estimate == "tipper":
                if self.color:
                    fc = float(count) / max_count * 0.9
                    colors.append(
                        self._rgb_to_hex(
                            (
                                self.color_tip[0],
                                1 - fc,
                                self.color_tip[-1],
                            )
                        )
                    )
                else:
                    colors.append(self._rgb_to_hex(self.color_tip))
            else:
                colors.append("#888888")
        return colors

    def _get_max_count(self, estimate, period_range=None):
        """Return the radial extent to use for a single rose figure."""
        if self.ring_limits is not None:
            return self.ring_limits[1]
        hist_range = self._get_histogram_range()
        n_bins = int((hist_range[1] - hist_range[0]) / self.bin_width)
        plot_array = self.get_plot_array(estimate, period_range)
        valid = plot_array[np.nonzero(plot_array)]
        if len(valid) == 0:
            return 1
        hist, _ = np.histogram(valid, bins=n_bins, range=hist_range)
        return max(1, int(hist.max()))

    def _make_rose_figure(self, title, max_count):
        """Create an empty polar-style rose-diagram Bokeh figure.

        Draws concentric grid rings, radial spokes, and cardinal-direction
        labels.  Bar wedges are added separately by :meth:`_add_rose_bars`.

        Parameters
        ----------
        title : str
            Figure title.
        max_count : int
            Maximum histogram count; sets the figure's radial extent.

        Returns
        -------
        bokeh.plotting.figure
        """
        ring_max = max_count if max_count > 0 else 1
        padding = ring_max * 1.4

        fig = figure(
            title=title,
            width=320,
            height=320,
            x_range=Range1d(-padding, padding),
            y_range=Range1d(-padding, padding),
            tools="pan,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        fig.xaxis.visible = False
        fig.yaxis.visible = False
        fig.grid.visible = False

        theta = np.linspace(0, 2 * np.pi, 200)

        # Concentric grid rings at ring_spacing intervals
        ring_step = max(self.ring_spacing, 1)
        for r in range(ring_step, ring_max + ring_step, ring_step):
            fig.line(
                r * np.cos(theta),
                r * np.sin(theta),
                line_color="lightgray",
                line_width=0.5,
                line_alpha=0.8,
            )

        # Outer boundary
        fig.line(
            ring_max * np.cos(theta),
            ring_max * np.sin(theta),
            line_color="gray",
            line_width=1.0,
        )

        # Radial spokes every 30°
        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            fig.line(
                [0, ring_max * np.cos(rad)],
                [0, ring_max * np.sin(rad)],
                line_color="lightgray",
                line_width=0.5,
                line_alpha=0.6,
            )

        # Cardinal direction labels
        # Convention matches matplotlib polar: 0=East, 90=North, 180=West, 270=South
        label_r = ring_max * 1.25
        for deg, text in [(0, "E"), (90, "N"), (180, "W"), (270, "S")]:
            rad = np.deg2rad(deg)
            fig.add_layout(
                Label(
                    x=label_r * np.cos(rad),
                    y=label_r * np.sin(rad),
                    text=text,
                    text_align="center",
                    text_baseline="middle",
                    text_font_size="10px",
                    text_font_style="bold",
                )
            )

        return fig

    def _add_rose_bars(self, fig, plot_array, estimate):
        """Draw rose histogram bars on *fig* and return the histogram array.

        Parameters
        ----------
        fig : bokeh.plotting.figure
            Target figure (created by :meth:`_make_rose_figure`).
        plot_array : np.ndarray
            Angles in degrees for the histogram.
        estimate : str
            One of ``"invariant"``, ``"pt"``, ``"tipper"``.

        Returns
        -------
        np.ndarray
            Histogram count array (shape ``(n_bins,)``).
        """
        hist_range = self._get_histogram_range()
        n_bins = int((hist_range[1] - hist_range[0]) / self.bin_width)

        valid = plot_array[np.nonzero(plot_array)]
        hist, edges = np.histogram(valid, bins=n_bins, range=hist_range)

        if hist.max() == 0:
            return hist

        colors = self._compute_bar_colors(hist, estimate)
        starts = np.deg2rad(edges[:-1])
        ends = np.deg2rad(edges[1:])

        mask = hist > 0
        src = ColumnDataSource(
            dict(
                start_angle=starts[mask].tolist(),
                end_angle=ends[mask].tolist(),
                outer_radius=hist[mask].astype(float).tolist(),
                fill_color=np.array(colors)[mask].tolist(),
            )
        )

        fig.annular_wedge(
            x=0,
            y=0,
            inner_radius=0.0,
            outer_radius="outer_radius",
            start_angle="start_angle",
            end_angle="end_angle",
            fill_color="fill_color",
            line_color="black",
            line_width=0.5,
            source=src,
        )

        return hist

    def _add_mode_label(self, fig, estimate, max_count, period_range=None):
        """Add a text annotation with the modal strike angle.

        Parameters
        ----------
        fig : bokeh.plotting.figure
        estimate : str
        max_count : int
        period_range : tuple or None
        """
        _, st_mode, _ = self.get_stats(estimate, period_range)
        ring_max = max_count if max_count > 0 else 1

        if estimate == "invariant":
            bg_color = self._rgb_to_hex(self.color_inv)
        elif estimate == "pt":
            bg_color = self._rgb_to_hex(self.color_pt)
        else:
            bg_color = self._rgb_to_hex(self.color_tip)

        fig.add_layout(
            Label(
                x=0,
                y=-(ring_max * 1.15),
                text=f"Mode: {st_mode:.1f}\u00b0",
                text_align="center",
                text_baseline="top",
                text_font_size="10px",
                background_fill_color=bg_color,
                background_fill_alpha=0.25,
            )
        )

    # ------------------------------------------------------------------
    # Main plot methods
    # ------------------------------------------------------------------

    def _plot_all_periods(self, show=True):
        """Create one rose diagram per enabled estimate over all periods.

        This is ``plot_type=2``.

        Parameters
        ----------
        show : bool
            Call ``bokeh_show`` if ``self.show_plot`` is also True.

        Returns
        -------
        Bokeh layout
        """
        self._require_bokeh()

        enabled = [
            est
            for est, flag in [
                ("invariant", self.plot_invariant),
                ("pt", self.plot_pt),
                ("tipper", self.plot_tipper),
            ]
            if flag
        ]

        titles = {
            "invariant": "Strike (Z)",
            "pt": "PT Azimuth",
            "tipper": "Tipper Strike",
        }

        figs = []
        self.figures = {}

        for estimate in enabled:
            max_count = self._get_max_count(estimate)
            fig = self._make_rose_figure(titles[estimate], max_count)
            self._add_rose_bars(fig, self.get_plot_array(estimate), estimate)
            self._add_mode_label(fig, estimate, max_count)
            figs.append(fig)
            self.figures[estimate] = fig

        if "h" in self.plot_orientation:
            self.layout = bokeh_row(*figs)
        else:
            self.layout = bokeh_column(*figs)

        self.fig = self.layout

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def _plot_per_period(self, show=True):
        """Create rose diagrams for each period decade (``plot_type=1``).

        For horizontal orientation the grid has rows = estimate types and
        columns = period decades.  For vertical orientation the axes are
        transposed.

        Parameters
        ----------
        show : bool

        Returns
        -------
        Bokeh gridplot layout
        """
        self._require_bokeh()

        bin_range = self._get_bin_range()

        enabled = [
            est
            for est, flag in [
                ("invariant", self.plot_invariant),
                ("pt", self.plot_pt),
                ("tipper", self.plot_tipper),
            ]
            if flag
        ]

        titles = {
            "invariant": "Strike (Z)",
            "pt": "PT Azimuth",
            "tipper": "Tipper Strike",
        }

        # Build grid with rows = period bands, cols = estimates
        grid_by_period = []
        self.figures = {}

        for bb in bin_range:
            period_range = [10**bb, 10 ** (bb + 1)]
            period_label = self.title_dict.get(int(bb), f"10^{int(bb)} s")

            row_figs = []
            for estimate in enabled:
                max_count = self._get_max_count(estimate, period_range)
                title = f"{titles[estimate]}\n{period_label}"
                fig = self._make_rose_figure(title, max_count)
                self._add_rose_bars(
                    fig,
                    self.get_plot_array(estimate, period_range),
                    estimate,
                )
                self._add_mode_label(fig, estimate, max_count, period_range)
                row_figs.append(fig)
                key = f"{estimate}_{int(bb)}"
                self.figures[key] = fig

            grid_by_period.append(row_figs)

        # Horizontal: rows = estimates, cols = periods (transpose)
        if "h" in self.plot_orientation and grid_by_period:
            grid = [list(col) for col in zip(*grid_by_period)]
        else:
            grid = grid_by_period

        self.layout = gridplot(grid)
        self.fig = self.layout

        if show and self.show_plot and bokeh_show is not None:
            bokeh_show(self.layout)

        return self.layout

    def plot(self, show=True):
        """Build and return an interactive Bokeh rose-diagram layout.

        Populates :attr:`strike_df` if not already built, then dispatches
        to :meth:`_plot_all_periods` (``plot_type=2``) or
        :meth:`_plot_per_period` (``plot_type=1``).

        Parameters
        ----------
        show : bool
            If True *and* ``self.show_plot`` is True, display the layout
            in a browser via ``bokeh_show``.

        Returns
        -------
        Bokeh layout object
        """
        if self.strike_df is None:
            self.make_strike_df()

        if self.plot_type == 1:
            return self._plot_per_period(show=show)
        return self._plot_all_periods(show=show)
