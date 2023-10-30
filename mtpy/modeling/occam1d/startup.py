# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:32:42 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

# =============================================================================
class Startup(object):
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
            self.save_path = os.path.dirname(self.data_fn)
        elif self.model_fn is not None:
            self.save_path = os.path.dirname(self.model_fn)

        self.startup_fn = None
        self.rough_type = kwargs.pop("rough_type", 1)
        self.max_iter = kwargs.pop("max_iter", 20)
        self.target_rms = kwargs.pop("target_rms", 1)
        self.start_rho = kwargs.pop("start_rho", 100)
        self.description = kwargs.pop("description", "1D_Occam_Inv")
        self.start_lagrange = kwargs.pop("start_lagrange", 5.0)
        self.start_rough = kwargs.pop("start_rough", 1.0e7)
        self.debug_level = kwargs.pop("debug_level", 1)
        self.start_iter = kwargs.pop("start_iter", 0)
        self.start_misfit = kwargs.pop("start_misfit", 100)
        self.min_max_bounds = kwargs.pop("min_max_bounds", None)
        self.model_step = kwargs.pop("model_step", None)
        self._startup_fn = "OccamStartup1D"
        self._ss = " " * 3

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
        if os.path.isdir(self.save_path) == False:
            os.mkdir(self.save_path)

        self.startup_fn = os.path.join(self.save_path, self._startup_fn)

        # --> read data file
        if self.data_fn is None:
            raise IOError("Need to input data file name.")
        else:
            data = Data()
            data.read_data_file(self.data_fn)

        # --> read model file
        if self.model_fn is None:
            raise IOError("Need to input model file name.")
        else:
            model = Model()
            model.read_model_file(self.model_fn)

            # --> get any keywords
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

            # --> write input file
        infid = open(self.startup_fn, "w")
        infid.write("{0:<21}{1}\n".format("Format:", "OCCAMITER_FLEX"))
        infid.write("{0:<21}{1}\n".format("Description:", self.description))
        infid.write(
            "{0:<21}{1}\n".format(
                "Model File:", os.path.basename(self.model_fn)
            )
        )
        infid.write(
            "{0:<21}{1}\n".format("Data File:", os.path.basename(self.data_fn))
        )
        infid.write("{0:<21}{1}\n".format("Date/Time:", time.ctime()))
        infid.write("{0:<21}{1}\n".format("Max Iter:", self.max_iter))
        infid.write("{0:<21}{1}\n".format("Target Misfit:", self.target_rms))
        infid.write("{0:<21}{1}\n".format("Roughness Type:", self.rough_type))
        if self.min_max_bounds == None:
            infid.write("{0:<21}{1}\n".format("!Model Bounds:", "min,max"))
        else:
            infid.write(
                "{0:<21}{1},{2}\n".format(
                    "Model Bounds:",
                    self.min_max_bounds[0],
                    self.min_max_bounds[1],
                )
            )
        if self.model_step == None:
            infid.write(
                "{0:<21}{1}\n".format("!Model Value Steps:", "stepsize")
            )
        else:
            infid.write(
                "{0:<21}{1}\n".format("Model Value Steps:", self.model_step)
            )
        infid.write("{0:<21}{1}\n".format("Debug Level:", self.debug_level))
        infid.write("{0:<21}{1}\n".format("Iteration:", self.start_iter))
        infid.write(
            "{0:<21}{1}\n".format("Lagrange Value:", self.start_lagrange)
        )
        infid.write("{0:<21}{1}\n".format("Roughness Value:", self.start_rough))
        infid.write("{0:<21}{1}\n".format("Misfit Value:", self.start_misfit))
        infid.write("{0:<21}{1}\n".format("Misfit Reached:", 0))
        infid.write("{0:<21}{1}\n".format("Param Count:", model.num_params))

        for ii in range(model.num_params):
            infid.write(
                "{0}{1:.2f}\n".format(self._ss, np.log10(self.start_rho))
            )

        infid.close()
        print("Wrote Input File: {0}".format(self.startup_fn))

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

        self._startup_fn = os.path.basename(self.startup_fn)
        self.save_path = os.path.dirname(self.startup_fn)

        infid = open(self.startup_fn, "r")
        ilines = infid.readlines()
        infid.close()

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
