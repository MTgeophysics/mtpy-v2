# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:17:41 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import unittest
from scipy.constants import mu_0
from discretize.tests import check_derivative
import discretize
import matplotlib.patheffects as pe
from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.static import utils as sutils
from simpeg import (
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
from discretize import TensorMesh
from pymatsolver import Pardiso
from scipy.spatial import cKDTree
from scipy.stats import norm

# from dask.distributed import Client, LocalCluster
import dill
from geoana.em.fdem import skin_depth
import discretize.utils as dis_utils
import warnings

warnings.filterwarnings("ignore")

from mtpy.modeling.simpeg.data import Simpeg2DData
from mtpy.modeling.simpeg.make_2d_mesh import QuadTreeMesh

# =============================================================================


class Simpe2D:

    def __init__(self, **kwargs):
        self.data = Simpeg2DData()
