# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:11:57 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================


# =============================================================================
class Simpeg2DData:
    """ """

    def __init__(self, dataframe, **kwargs):
        self.dataframe = dataframe

        # nez+ as keys then enz- as values
        self.component_map = {
            "xx": "yy",
            "xy": "yx",
            "yx": "xy",
            "yy": "xx",
            "zx": "zy",
            "zy": "zx",
        }

    def get_station_locations(self, utm=True):
        """
        get station locations in utm geographic coordinates if True, otherwise
        will be in model coordinates.

        :param geographic: DESCRIPTION, defaults to True
        :type geographic: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        station_df = self.dataframe.groupby("station")
        if utm:
            east = station_df.east
            north = station_df.north
        else:
            east = self.dataframe.model_east
            north = self.dataframe.model_north

    def from_dataframe():
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """
