# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:32:42 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import time

import numpy as np

from mtpy.modeling.occam1d import Occam1DData, Occam1DModel

# =============================================================================
class Occam1DStartup(object):
    """
    read and write input files for Occam1D

    ====================== ====================================================
    Attributes             Description
    ====================== ====================================================
    _ss                    string spacing
    _startup_fn            basename of startup file *default* is OccamStartup1D
    data_fn                full path to data file
    debug_level            debug level *default* is 1
    description            description of inversion for your self
                           *default* is 1D_Occam_Inv
    max_iter               maximum number of iterations *default* is 20
    model_fn               full path to model file
    rough_type             roughness type *default* is 1
    save_path              full path to save files to
    start_iter             first iteration number *default* is 0
    start_lagrange         starting lagrange number on log scale
                           *default* is 5
    start_misfit           starting misfit value *default* is 100
    start_rho              starting resistivity value (halfspace) in log scale
                           *default* is 100
    start_rough            starting roughness (ignored by Occam1D)
                           *default* is 1E7
    startup_fn             full path to startup file
    target_rms             target rms *default* is 1.0
    ====================== ====================================================
    """

    def __init__(self, data_fn=None, model_fn=None, **kwargs):
        self.data_fn = data_fn
        self.model_fn = model_fn

        if self.data_fn is not None:
            self.save_path = self.data_fn.parent
        elif self.model_fn is not None:
            self.save_path = self.model_fn.parent

        self.startup_fn = None
        self.rough_type = 1
        self.max_iter = 20
        self.target_rms = 1
        self.start_rho = 100
        self.description = "1D_Occam_Inv"
        self.start_lagrange = 5.0
        self.start_rough = 1.0e7
        self.debug_level = 1
        self.start_iter = 0
        self.start_misfit = 100
        self.min_max_bounds = None
        self.model_step = None
        self._startup_fn = "OccamStartup1D"
        self._ss = " " * 3

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def data_fn(self):
        return self._data_fn

    @data_fn.setter
    def data_fn(self, fn):
        if fn is not None:
            self._data_fn = Path(fn)
        else:
            self._data_fn = None

    @property
    def model_fn(self):
        return self._model_fn

    @model_fn.setter
    def model_fn(self, fn):
        if fn is not None:
            self._model_fn = Path(fn)
        else:
            self._model_fn = None

    def write_startup_file(self, save_path=None, **kwargs):
        """
        Make a 1D input file for Occam 1D

        Arguments:
        ---------
            **savepath** : full path to save input file to, if just path then
                           saved as savepath/input

            **model_fn** : full path to model file, if None then assumed to be in
                            savepath/model.mod

            **data_fn** : full path to data file, if None then assumed to be
                            in savepath/TE.dat or TM.dat

            **rough_type** : roughness type. *default* = 0

            **max_iter** : maximum number of iterations. *default* = 20

            **target_rms** : target rms value. *default* = 1.0

            **start_rho** : starting resistivity value on linear scale.
                            *default* = 100

            **description** : description of the inversion.

            **start_lagrange** : starting Lagrange multiplier for smoothness.
                           *default* = 5

            **start_rough** : starting roughness value. *default* = 1E7

            **debuglevel** : something to do with how Fortran debuggs the code
                             Almost always leave at *default* = 1

            **start_iter** : the starting iteration number, handy if the
                            starting model is from a previous run.
                            *default* = 0

            **start_misfit** : starting misfit value. *default* = 100

        Returns:
        --------
            **Occam1D.inputfn** : full path to input file.

        :Example: ::

            >>> old = occam.Occam1D()
            >>> old.make1DdataFile('MT01',edipath=r"/home/Line1",
            >>>                    savepath=r"/home/Occam1D/Line1/Inv1_TE",
            >>>                    mode='TE')
            >>> Wrote Data File: /home/Occam1D/Line1/Inv1_TE/MT01TE.dat
            >>>
            >>> old.make1DModelFile(savepath=r"/home/Occam1D/Line1/Inv1_TE",
            >>>                     nlayers=50,bottomlayer=10000,z1layer=50)
            >>> Wrote Model file: /home/Occam1D/Line1/Inv1_TE/Model1D
            >>>
            >>> old.make1DInputFile(rhostart=10,targetrms=1.5,maxiter=15)
            >>> Wrote Input File: /home/Occam1D/Line1/Inv1_TE/Input1D
        """

        if save_path is not None:
            self.save_path = save_path
        if not self.save_path.is_dir():
            self.save_path.mkdir()

        self.startup_fn = self.save_path.joinpath(self._startup_fn)

        # --> read data file
        if self.data_fn is None:
            raise IOError("Need to input data file name.")
        else:
            data = Occam1DData()
            data.read_data_file(self.data_fn)

        # --> read model file
        if self.model_fn is None:
            raise IOError("Need to input model file name.")
        else:
            model = Occam1DModel()
            model.read_model_file(self.model_fn)

            # --> get any keywords
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

            # --> write input file
        infid = open(self.startup_fn, "w")
        infid.write(f"{'Format:':<21}OCCAMITER_FLEX\n")
        infid.write(f"{'Description:':<21}{self.description}\n")
        infid.write(f"{'Model File:':<21}{self.model_fn.name}\n")
        infid.write(f"{'Data File:':<21}{self.data_fn.name}\n")
        infid.write(f"{'Date/Time:':<21}{time.ctime()}\n")
        infid.write(f"{'Max Iter:':<21}{self.max_iter}\n")
        infid.write(f"{'Target Misfit:':<21}{self.target_rms}\n")
        infid.write(f"{'Roughness Type:':<21}{self.rough_type}\n")
        if self.min_max_bounds == None:
            infid.write(f"{'!Model Bounds:':<21}min,max\n")
        else:
            infid.write(
                "{0:<21}{1},{2}\n".format(
                    "Model Bounds:",
                    self.min_max_bounds[0],
                    self.min_max_bounds[1],
                )
            )
        if self.model_step == None:
            infid.write(f"{'!Model Value Steps:':<21}stepsize\n")
        else:
            infid.write(f"{'Model Value Steps:':<21}{self.model_step}\n")
        infid.write(f"{'Debug Level:':<21}{self.debug_level}\n")
        infid.write(f"{'Iteration:':<21}{self.start_iter}\n")
        infid.write(f"{'Lagrange Value:':<21}{self.start_lagrange}\n")
        infid.write(f"{'Roughness Value:':<21}{self.start_rough}\n")
        infid.write(f"{'Misfit Value:':<21}{self.start_misfit}\n")
        infid.write(f"{'Misfit Reached:':<21}{0}\n")
        infid.write(f"{'Param Count:':<21}{model.num_params}\n")

        for ii in range(model.num_params):
            infid.write(f"{self._ss}{np.log10(self.start_rho):.2f}\n")

        infid.close()
        print(f"Wrote Input File: {self.startup_fn}")

    def read_startup_file(self, startup_fn):
        """
        reads in a 1D input file

        Arguments:
        ---------
            **inputfn** : full path to input file

        Returns:
        --------
            **Occam1D.indict** : dictionary with keys following the header and

                *'res'* : an array of resistivity values

        :Example: ::

            >>> old = occam.Occam1d()
            >>> old.savepath = r"/home/Occam1D/Line1/Inv1_TE"
            >>> old.read1DInputFile()
        """
        if startup_fn is not None:
            self.startup_fn = startup_fn

        if self.startup_fn is None:
            raise IOError("Need to input a startup file.")

        self._startup_fn = Path(self.startup_fn).name
        self.save_path = Path(self.startup_fn).parent

        with open(self.startup_fn, "r") as infid:
            ilines = infid.readlines()

        self.indict = {}
        res = []

        # split the keys and values from the header information
        for iline in ilines:
            if iline.find(":") >= 0:
                ikey = iline[0:20].strip()[:-1]
                ivalue = iline[20:].split("!")[0].strip()
                if ikey.find("!") == 0:
                    pass
                else:
                    setattr(self, ikey.lower().replace(" ", "_"), ivalue)
                self.indict[ikey[:-1]] = ivalue
            else:
                try:
                    res.append(float(iline.strip()))
                except ValueError:
                    pass

        # make the resistivity array ready for models to be input
        self.indict["res"] = np.zeros((len(res), 3))
        self.indict["res"][:, 0] = res
