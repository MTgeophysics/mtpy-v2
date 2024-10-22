# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:11:57 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from simpeg.electromagnetics import natural_source as nsem
from simpeg import data

import matplotlib.pyplot as plt


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

        for key, value in kwargs.items():
            setattr(self, key, value)

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
            return np.c_[
                station_df.profile_offset, np.zeros(station_df.shape[0])
            ]

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
        rx_list = [
            nsem.receivers.PointNaturalSource(
                rx_locs,
                orientation=simpeg_mode,
                component="apparent_resistivity",
            ),
            nsem.receivers.PointNaturalSource(
                rx_locs, orientation=simpeg_mode, component="phase"
            ),
        ]

        src_list = [
            nsem.sources.Planewave(rx_list, frequency=f)
            for f in self.frequencies
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

    def _get_data_observations(self, mode):
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
            res.append(f_df[f"res_{comp}"])
            # flip into the appropriate coordinate system
            if comp in ["xy"]:
                phase.append(1 * f_df[f"phase_{comp}"] + 90)
            elif comp in ["yx"]:
                phase.append(1 * f_df[f"phase_{comp}"] - 270)

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
            res.append(f_df[f"res_{comp}_model_error"])
            phase.append(f_df[f"phase_{comp}_model_error"])

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
        ax_xy_res = fig.add_subplot(2, 2, 1)
        ax_yx_res = fig.add_subplot(2, 2, 2, sharex=ax_xy_res)
        ax_xy_phase = fig.add_subplot(2, 2, 3, sharex=ax_xy_res)
        ax_yx_phase = fig.add_subplot(2, 2, 4, sharex=ax_xy_res)

        te_data = self.te_data.dobs.reshape(
            (self.n_frequencies, 2, self.n_stations)
        )
        tm_data = self.tm_data.dobs.reshape(
            (self.n_frequencies, 2, self.n_stations)
        )
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

        for ax in [ax_xy_res, ax_xy_phase, ax_yx_res, ax_yx_phase]:
            ax.grid(
                True,
                alpha=0.25,
                which="both",
                color=(0.25, 0.25, 0.25),
                lw=0.25,
            )

        plt.show()
