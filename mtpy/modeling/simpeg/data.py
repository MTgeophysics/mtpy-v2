# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:11:57 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from SimPEG.electromagnetics import natural_source as nsem
from SimPEG.electromagnetics.static import utils as sutils
from SimPEG import (
    maps,
    utils,
    optimization,
    objective_function,
    inversion,
    inverse_problem,
    directives,
    data_misfit,
    regularization,
    data,
)


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
        return 1.0 / self.dataframe.period.unique()

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
    def survey_te(self):
        """
        survey for TE mode (simpeg = "yx")

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._get_mode_sources(self.component_map["te"]["simpeg"])

    @property
    def survey_tm(self):
        """
        survey for TM mode (simpeg = "xy")

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._get_mode_sources(self.component_map["te"]["simpeg"])

    def _get_data_observations(self, mode):
        """
        get data
        :param mode: [ 'te' | 'tm' ]
        :type simpeg_mode: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        mode = self.component_map[mode]["z+"]

        return np.hstack(
            (self.dataframe[f"res_{mode}"], self.dataframe[f"phase_{mode}"])
        )

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

        mode = self.component_map[mode]["z+"]

        return np.hstack(
            (
                self.dataframe[f"res_{mode}_model_error"],
                self.dataframe[f"phase_{mode}_model_error"],
            )
        )

    @property
    def te_data_errors(self):
        return self._get_data_errors("te")

    @property
    def tm_data_errors(self):
        return self._get_data_errors("tm")

    def _get_data(self):
        """
        We are assuming that the dataframe provide has a data point for each
        period.  This can be achieved by using

         >>> MTData.interpolate(new_period, bounds_error=False)

        This will place Nans where there are no data and can be handeled later.


        :return: DESCRIPTION
        :rtype: TYPE

        """
