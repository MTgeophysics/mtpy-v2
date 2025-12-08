# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:11:57 2024

@author: jpeacock
"""

import matplotlib.pyplot as plt

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from simpeg import data
from simpeg.electromagnetics import natural_source as nsem

from mtpy.imaging.mtplot_tools import plot_phase, plot_resistivity


# =============================================================================
class Simpeg2DData:
    """ """

    def __init__(self, dataframe, **kwargs):
        self.dataframe = dataframe

        # nez+ as keys then enz- as values
        self.component_map = {
            "te": {"simpeg": "yx", "z+": "xy"},
            "tm": {"simpeg": "xy", "z+": "yx"},
        }

        self.include_elevation = True
        self.invert_te = True
        self.invert_tm = True
        self.invert_zxy = False
        self.invert_zyx = False

        self.invert_impedance = False

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def invert_impedance(self):
        return self._invert_impedance

    @invert_impedance.setter
    def invert_impedance(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                f"invert_impedance must be a boolean, not type{type(value)}"
            )

        if value:
            self.invert_zxy = True
            self.invert_zyx = True
            self.invert_te = False
            self.invert_tm = False
            self._invert_impedance = True
        if not value:
            self.invert_zxy = False
            self.invert_zyx = False
            self.invert_te = True
            self.invert_tm = True
            self._invert_impedance = False

    @property
    def station_locations(self):
        """
        get station locations in utm geographic coordinates if True, otherwise
        will be in model coordinates.

        :param geographic: DESCRIPTION, defaults to True
        :type geographic: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        station_df = self.dataframe.groupby("station").nth(0)

        offsets = station_df.profile_offset
        if np.all(offsets == 0):
            raise ValueError("Need to calculate profile_offset.")

        if self.include_elevation:
            return np.c_[station_df.profile_offset, station_df.elevation]
        else:
            return np.c_[station_df.profile_offset, np.zeros(station_df.shape[0])]

    @property
    def frequencies(self):
        """
        frequencies from the data frame

        :return: DESCRIPTION
        :rtype: TYPE

        """

        # surveys sort from small to large.
        return np.sort(1.0 / self.dataframe.period.unique())

    @property
    def n_frequencies(self):
        return self.frequencies.size

    @property
    def n_stations(self):
        return self.dataframe.station.unique().size

    def _get_mode_sources(self, simpeg_mode):
        """
        get mode  objects

        :return: DESCRIPTION
        :rtype: TYPE

        """
        rx_locs = self.station_locations.copy()

        if not self.invert_impedance:
            rx_list = [
                nsem.receivers.Impedance(
                    rx_locs,
                    orientation=simpeg_mode,
                    component="apparent_resistivity",
                ),
                nsem.receivers.Impedance(
                    rx_locs, orientation=simpeg_mode, component="phase"
                ),
            ]

            src_list = [
                nsem.sources.Planewave(rx_list, frequency=f) for f in self.frequencies
            ]
        else:
            rx_list = [
                nsem.receivers.Impedance(
                    rx_locs,
                    orientation=simpeg_mode,
                    component="real",
                ),
                nsem.receivers.Impedance(
                    rx_locs,
                    orientation=simpeg_mode,
                    component="imag",
                ),
            ]

            src_list = [
                nsem.sources.Planewave(rx_list, frequency=f) for f in self.frequencies
            ]
        return nsem.Survey(src_list)

    @property
    def te_survey(self):
        """
        survey for TE mode (simpeg = "yx")

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._get_mode_sources(self.component_map["te"]["simpeg"])

    @property
    def tm_survey(self):
        """
        survey for TM mode (simpeg = "xy")

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._get_mode_sources(self.component_map["tm"]["simpeg"])

    def _get_data_observations(self, mode, impedance=False):
        """
        get data

        the output format needs to be [frequency 1 res, frequency 1 phase, ...]
        and frequency is in order of smallest to largest.

        Data needs to be ordered by station [te, tm](f)

        :param mode: [ 'te' | 'tm' ]
        :type simpeg_mode: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        comp = self.component_map[mode]["z+"]
        # there is probably a more efficient method here using pandas
        res = []
        phase = []
        for ff in self.frequencies:
            f_df = self.dataframe[self.dataframe.period == 1.0 / ff]
            if not self.invert_impedance:
                res.append(f_df[f"res_{comp}"])
                phase.append(f_df[f"phase_{comp}"])
            else:
                res.append(f_df[f"z_{comp}"].values.real)
                phase.append(f_df[f"z_{comp}"].values.imag)

        return np.hstack((res, phase)).flatten()

    @property
    def te_observations(self):
        """
        TE observations
        """
        return self._get_data_observations("te")

    @property
    def tm_observations(self):
        """
        TM observations
        """
        return self._get_data_observations("tm")

    def _get_data_errors(self, mode):
        """
        get data
        :param mode: [ 'te' | 'tm' ]
        :type simpeg_mode: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        comp = self.component_map[mode]["z+"]
        res = []
        phase = []
        # there is probably a more efficient method here using pandas
        for ff in np.sort(self.frequencies):
            f_df = self.dataframe[self.dataframe.period == 1.0 / ff]
            if not self.invert_impedance:
                res.append(f_df[f"res_{comp}_model_error"])
                phase.append(f_df[f"phase_{comp}_model_error"])
            else:
                res.append(f_df[f"z_{comp}_model_error"])
                phase.append(f_df[f"z_{comp}_model_error"])

        return np.hstack((res, phase)).flatten()

    @property
    def te_data_errors(self):
        return self._get_data_errors("te")

    @property
    def tm_data_errors(self):
        return self._get_data_errors("tm")

    @property
    def te_data(self):
        """
        simpeg Data object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return data.Data(
            self.te_survey,
            dobs=self.te_observations,
            standard_deviation=self.te_data_errors,
        )

    @property
    def tm_data(self):
        """
        simpeg Data object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return data.Data(
            self.tm_survey,
            dobs=self.tm_observations,
            standard_deviation=self.tm_data_errors,
        )

    def plot_response(self, **kwargs):
        """

        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        fig = plt.figure(kwargs.get("fig_num", 1))

        te_data = self.te_data.dobs.reshape((self.n_frequencies, 2, self.n_stations))
        te_data_errors = self.te_data.standard_deviation.reshape(
            (self.n_frequencies, 2, self.n_stations)
        )
        tm_data = self.tm_data.dobs.reshape((self.n_frequencies, 2, self.n_stations))
        tm_data_errors = self.tm_data.standard_deviation.reshape(
            (self.n_frequencies, 2, self.n_stations)
        )

        if not self.invert_impedance:
            ax_xy_res = fig.add_subplot(2, 2, 1)
            ax_yx_res = fig.add_subplot(2, 2, 2, sharex=ax_xy_res)
            ax_xy_phase = fig.add_subplot(2, 2, 3, sharex=ax_xy_res)
            ax_yx_phase = fig.add_subplot(2, 2, 4, sharex=ax_xy_res)
            for ii in range(self.n_stations):
                ax_xy_res.loglog(
                    1.0 / self.frequencies,
                    te_data[:, 0, ii],
                    color=(0.5, 0.5, ii / self.n_stations),
                )
                ax_xy_phase.semilogx(
                    1.0 / self.frequencies,
                    te_data[:, 1, ii],
                    color=(0.25, 0.25, ii / self.n_stations),
                )
                ax_yx_res.loglog(
                    1.0 / self.frequencies,
                    tm_data[:, 0, ii],
                    color=(0.5, ii / self.n_stations, 0.75),
                )
                ax_yx_phase.semilogx(
                    1.0 / self.frequencies,
                    tm_data[:, 1, ii],
                    color=(0.25, ii / self.n_stations, 0.75),
                )

            ax_xy_phase.set_xlabel("Period (s)")
            ax_yx_phase.set_xlabel("Period (s)")
            ax_xy_res.set_ylabel("Apparent Resistivity")
            ax_xy_phase.set_ylabel("Phase")

            ax_xy_res.set_title("TE")
            ax_yx_res.set_title("TM")
        else:
            ax_xy_res = fig.add_subplot(2, 2, 1)
            ax_yx_res = fig.add_subplot(2, 2, 2, sharex=ax_xy_res, sharey=ax_xy_res)
            ax_xy_phase = fig.add_subplot(
                2,
                2,
                3,
                sharex=ax_xy_res,
            )
            ax_yx_phase = fig.add_subplot(2, 2, 4, sharex=ax_xy_res, sharey=ax_xy_phase)
            for ii in range(self.n_stations):
                plot_resistivity(
                    ax_xy_res,
                    1.0 / self.frequencies,
                    te_data[:, 0, ii],
                    color=(0.5, 0.5, ii / self.n_stations),
                    label=self.dataframe.station.unique()[ii],
                    error=te_data_errors[:, 0, ii],
                )
                plot_phase(
                    ax_xy_phase,
                    1.0 / self.frequencies,
                    te_data[:, 1, ii],
                    color=(0.25, 0.25, ii / self.n_stations),
                    label=self.dataframe.station.unique()[ii],
                    error=te_data_errors[:, 1, ii],
                )
                plot_resistivity(
                    ax_yx_res,
                    1.0 / self.frequencies,
                    tm_data[:, 0, ii],
                    color=(0.5, ii / self.n_stations, 0.75),
                    label=self.dataframe.station.unique()[ii],
                    error=tm_data_errors[:, 0, ii],
                )
                plot_phase(
                    ax_yx_phase,
                    1.0 / self.frequencies,
                    tm_data[:, 1, ii],
                    color=(0.25, ii / self.n_stations, 0.75),
                    label=self.dataframe.station.unique()[ii],
                    error=tm_data_errors[:, 1, ii],
                )
                # ax_xy_res.loglog(
                #     1.0 / self.frequencies,
                #     np.abs(te_data[:, 0, ii]),
                #     color=(0.5, 0.5, ii / self.n_stations),
                # )
                # ax_xy_phase.loglog(
                #     1.0 / self.frequencies,
                #     np.abs(te_data[:, 1, ii]),
                #     color=(0.25, 0.25, ii / self.n_stations),
                # )
                # ax_yx_res.loglog(
                #     1.0 / self.frequencies,
                #     np.abs(tm_data[:, 0, ii]),
                #     color=(0.5, ii / self.n_stations, 0.75),
                # )
                # ax_yx_phase.loglog(
                #     1.0 / self.frequencies,
                #     np.abs(tm_data[:, 1, ii]),
                #     color=(0.25, ii / self.n_stations, 0.75),
                # )

            ax_xy_phase.set_xlabel("Period (s)")
            ax_yx_phase.set_xlabel("Period (s)")
            ax_xy_res.set_ylabel("Real Impedance [Ohms]")
            ax_xy_phase.set_ylabel("Imag Impedance [Ohms]")

            ax_xy_res.set_title("Zxy (TE)")
            ax_yx_res.set_title("Zyx (TM)")

        for ax in [ax_xy_res, ax_xy_phase, ax_yx_res, ax_yx_phase]:
            ax.grid(
                True,
                alpha=0.25,
                which="both",
                color=(0.25, 0.25, 0.25),
                lw=0.25,
            )

        plt.show()
