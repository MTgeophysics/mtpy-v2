# =============================================================================
# Imports
# =============================================================================
import warnings
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from simpeg.electromagnetics import natural_source as nsem
from simpeg import (
    maps,
    optimization,
    inversion,
    inverse_problem,
    directives,
    data_misfit,
    regularization,
)
from pymatsolver import Pardiso

# from dask.distributed import Client, LocalCluster
from mtpy.modeling.simpeg.data_2d import Simpeg2DData
from mtpy.modeling.simpeg.make_2d_mesh import QuadTreeMesh

warnings.filterwarnings("ignore")


# =============================================================================

class Simpeg3DData:
    """ """

    def __init__(self, dataframe, **kwargs):
        self.dataframe = dataframe

        # nez+ as keys then enz- as values
        self.component_map = {
            "xx": {"simpeg": "yy", "z+": "xx"},
            "xy": {"simpeg": "yx", "z+": "xy"},
            "yx": {"simpeg": "xy", "z+": "yx"},
            "yy": {"simpeg": "xx", "z+": "yy"},
            "zx": {"simpeg": "zy", "z+": "zx"},
            "zy": {"simpeg": "zx", "z+": "zy"},
        }

        self.include_elevation = False
        self.topography = None # should be a geotiff or asc file
        self.geographic_coordinates = True

    @property
    def station_locations(self):
        """
        return just station locations in appropriate coordinates, default is
        geographic.
        """
        if self.geographic_coordinates:
            station_df = self.dataframe.groupby("station").nth(0)

            if self.include_elevation:
                return np.c_[
                    station_df.east,
                    station_df.north,
                    station_df.elevation
                    ]
            else:
                return np.c_[
                    station_df.east,
                    station_df.north,
                    np.zeros(station_df.elevation.size)
                    ]

    def frequecies(self):
        """unique frequencies from the dataframe
        """

        return 1./self.dataframe.period.unique()
    
    @property
    def n_frequencies(self):
        return self.frequencies.size

    @property
    def n_stations(self):
        return self.dataframe.station.unique().size
    
    def _get_mode_sources(self, orientation):
        """Get the source for each mode
        """
        source_real = nsem.receivers.PointNaturalSource(
            self.station_locations, orientation=orientation, component="real"
        )
        source_imag = nsem.receivers.PointNaturalSource(
            self.station_locations, orientation=orientation, component="imag"
        )

        return [source_real, source_imag]
        
    

    