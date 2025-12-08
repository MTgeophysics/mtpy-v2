# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:29:22 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================


# =============================================================================
class Occam1DModel(object):
    """
    read and write the model file fo Occam1D

    All depth measurements are in meters.

    ======================== ==================================================
    Attributes               Description
    ======================== ==================================================
    _model_fn                basename for model file *default* is Model1D
    _ss                      string spacing in model file *default* is 3*' '
    _string_fmt              format of model layers *default* is '.0f'
    air_layer_height         height of air layer *default* is 10000
    bottom_layer             bottom of the model *default* is 50000
    itdict                   dictionary of values from iteration file
    iter_fn                  full path to iteration file
    model_depth              array of model depths
    model_fn                 full path to model file
    model_penalty            array of penalties for each model layer
    model_preference_penalty array of model preference penalties for each layer
    model_prefernce          array of preferences for each layer
    model_res                array of resistivities for each layer
    n_layers                 number of layers in the model
    num_params               number of parameters to invert for (n_layers+2)
    pad_z                    padding of model at depth *default* is 5 blocks
    save_path                path to save files
    target_depth             depth of target to investigate
    z1_layer                 depth of first layer *default* is 10
    ======================== ==================================================

    ======================== ==================================================
    Methods                  Description
    ======================== ==================================================
    write_model_file         write an Occam1D model file, where depth increases
                             on a logarithmic scale
    read_model_file          read an Occam1D model file
    read_iter_file           read an .iter file output by Occam1D
    ======================== ==================================================

    :Example: ::

        >>> #--> make a model file
        >>> m1 = occam1d.Model()
        >>> m1.write_model_file(save_path=r"/home/occam1d/mt01/TE")
    """

    def __init__(self, model_fn=None, **kwargs):
        self.model_fn = model_fn
        self.iter_fn = None

        self.n_layers = kwargs.pop("n_layers", 100)
        self.bottom_layer = kwargs.pop("bottom_layer", None)
        self.target_depth = kwargs.pop("target_depth", None)
        self.pad_z = kwargs.pop("pad_z", 5)
        self.z1_layer = kwargs.pop("z1_layer", 10)
        self.air_layer_height = kwargs.pop("zir_layer_height", 10000)
        self._set_layerdepth_defaults()

        self.save_path = kwargs.pop("save_path", None)
        if self.model_fn is not None and self.save_path is None:
            self.save_path = os.path.dirname(self.model_fn)

        self._ss = " " * 3
        self._string_fmt = ".0f"
        self._model_fn = "Model1D"
        self.model_res = None
        self.model_depth = None
        self.model_penalty = None
        self.model_prefernce = None
        self.model_preference_penalty = None
        self.num_params = None

    def _set_layerdepth_defaults(self, z1_threshold=3.0, bottomlayer_threshold=2.0):
        """
        set target depth, bottom layer and z1 layer, making sure all the layers
        are consistent with each other and will work in the inversion
        (e.g. check target depth is not deeper than bottom layer)
        """

        if self.target_depth is None:
            if self.bottom_layer is None:
                # if neither target_depth nor bottom_layer are set, set defaults
                self.target_depth = 10000.0
            else:
                self.target_depth = mtcc.roundsf(self.bottom_layer / 5.0, 1.0)

        if self.bottom_layer is None:
            self.bottom_layer = 5.0 * self.target_depth
        # if bottom layer less than a factor of 2 greater than target depth then adjust deeper
        elif float(self.bottom_layer) / self.target_depth < bottomlayer_threshold:
            self.bottom_layer = bottomlayer_threshold * self.target_depth
            print(
                "bottom layer not deep enough for target depth, set to {} m".format(
                    self.bottom_layer
                )
            )

        if self.z1_layer is None:
            self.z1_layer = mtcc.roundsf(self.target_depth / 1000.0, 0)
        elif self.target_depth / self.z1_layer < z1_threshold:
            self.z1_layer = self.target_depth / z1_threshold
            print(
                f"z1 layer not deep enough for target depth, set to {self.z1_layer} m"
            )

    def write_model_file(self, save_path=None, **kwargs):
        """
        Makes a 1D model file for Occam1D.

        Arguments:
        ----------

            **save_path** :path to save file to, if just path saved as
                          savepath\model.mod, if None defaults to dirpath

            **n_layers** : number of layers

            **bottom_layer** : depth of bottom layer in meters

            **target_depth** : depth to target under investigation

            **pad_z** : padding on bottom of model past target_depth

            **z1_layer** : depth of first layer in meters

            **air_layer_height** : height of air layers in meters

        Returns:
        --------

            **Occam1D.modelfn** = full path to model file

        ..Note: This needs to be redone.

        :Example: ::

            >>> old = occam.Occam1D()
            >>> old.make1DModelFile(savepath=r"/home/Occam1D/Line1/Inv1_TE",
            >>>                     nlayers=50,bottomlayer=10000,z1layer=50)
            >>> Wrote Model file: /home/Occam1D/Line1/Inv1_TE/Model1D
        """
        if save_path is not None:
            self.save_path = save_path
            if os.path.isdir == False:
                os.mkdir(self.save_path)

        self.model_fn = os.path.join(self.save_path, self._model_fn)

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

        if self.model_depth is None:
            # ---------create depth layers--------------------
            log_z = np.logspace(
                np.log10(self.z1_layer),
                np.log10(
                    self.target_depth
                    - np.logspace(
                        np.log10(self.z1_layer),
                        np.log10(self.target_depth),
                        num=self.n_layers,
                    )[-2]
                ),
                num=self.n_layers - self.pad_z,
            )
            ztarget = np.array([zz - zz % 10 ** np.floor(np.log10(zz)) for zz in log_z])
            log_zpad = np.logspace(
                np.log10(self.target_depth),
                np.log10(
                    self.bottom_layer
                    - np.logspace(
                        np.log10(self.target_depth),
                        np.log10(self.bottom_layer),
                        num=self.pad_z,
                    )[-2]
                ),
                num=self.pad_z,
            )
            zpadding = np.array(
                [zz - zz % 10 ** np.floor(np.log10(zz)) for zz in log_zpad]
            )
            z_nodes = np.append(ztarget, zpadding)
            self.model_depth = np.array(
                [z_nodes[: ii + 1].sum() for ii in range(z_nodes.shape[0])]
            )
        else:
            self.n_layers = len(self.model_depth)

        self.num_params = self.n_layers + 2
        # make the model file
        modfid = open(self.model_fn, "w")
        modfid.write("Format: Resistivity1DMod_1.0" + "\n")
        modfid.write(f"#LAYERS:    {self.num_params}\n")
        modfid.write("!Set free values to -1 or ? \n")
        modfid.write(
            "!penalize between 1 and 0,"
            + "0 allowing jump between layers and 1 smooth. \n"
        )
        modfid.write("!preference is the assumed resistivity on linear scale. \n")
        modfid.write("!pref_penalty needs to be put if preference is not 0 [0,1]. \n")
        modfid.write(
            "! {0}\n".format(
                self._ss.join(
                    [
                        "top_depth",
                        "resistivity",
                        "penalty",
                        "preference",
                        "pref_penalty",
                    ]
                )
            )
        )
        modfid.write(
            self._ss.join(
                [
                    str(-self.air_layer_height),
                    "1d12",
                    "0",
                    "0",
                    "0",
                    "!air layer",
                    "\n",
                ]
            )
        )
        modfid.write(
            self._ss.join(["0", "-1", "0", "0", "0", "!first ground layer", "\n"])
        )
        for ll in self.model_depth:
            modfid.write(
                self._ss.join(
                    [
                        f"{np.ceil(ll):{{1}}}",
                        "-1",
                        "1",
                        "0",
                        "0",
                        "\n",
                    ]
                )
            )

        modfid.close()

        print(f"Wrote Model file: {self.model_fn}")

    def read_model_file(self, model_fn=None):
        """

        will read in model 1D file

        Arguments:
        ----------
            **modelfn** : full path to model file

        Fills attributes:
        --------

            * model_depth' : depth of model in meters

            * model_res : value of resisitivity

            * model_penalty : penalty

            * model_preference : preference

            * model_penalty_preference : preference penalty

        :Example: ::

            >>> m1 = occam1d.Model()
            >>> m1.savepath = r"/home/Occam1D/Line1/Inv1_TE"
            >>> m1.read_model_file()
        """
        if model_fn is not None:
            self.model_fn = model_fn
        if self.model_fn is None:
            raise IOError("Need to input a model file")
        elif os.path.isfile(self.model_fn) == False:
            raise IOError(f"Could not find{self.model_fn}, check path")

        self._model_fn = os.path.basename(self.model_fn)
        self.save_path = os.path.dirname(self.model_fn)
        mfid = open(self.model_fn, "r")
        mlines = mfid.readlines()
        mfid.close()
        mdict = {}
        mdict["nparam"] = 0
        for key in ["depth", "res", "pen", "pref", "prefpen"]:
            mdict[key] = []

        for mm, mline in enumerate(mlines):
            if mline.find("!") == 0:
                pass
            elif mline.find(":") >= 0:
                mlst = mline.strip().split(":")
                mdict[mlst[0]] = mlst[1]
            else:
                mlst = mline.strip().split()
                mdict["depth"].append(float(mlst[0]))
                if mlst[1] == "?":
                    mdict["res"].append(-1)
                elif mlst[1] == "1d12":
                    mdict["res"].append(1.0e12)
                else:
                    try:
                        mdict["res"].append(float(mlst[1]))
                    except ValueError:
                        mdict["res"].append(-1)
                mdict["pen"].append(float(mlst[2]))
                mdict["pref"].append(float(mlst[3]))
                mdict["prefpen"].append(float(mlst[4]))
                if mlst[1] == "-1" or mlst[1] == "?":
                    mdict["nparam"] += 1

        # make everything an array
        for key in ["depth", "res", "pen", "pref", "prefpen"]:
            mdict[key] = np.array(mdict[key])

        # create an array with empty columns to put the TE and TM models into
        mres = np.zeros((len(mdict["res"]), 2))
        mres[:, 0] = mdict["res"]
        mdict["res"] = mres

        # make attributes
        self.model_res = mdict["res"]
        self.model_depth = mdict["depth"]
        self.model_penalty = mdict["pen"]
        self.model_prefernce = mdict["pref"]
        self.model_preference_penalty = mdict["prefpen"]
        self.num_params = mdict["nparam"]

    def read_iter_file(self, iter_fn=None, model_fn=None):
        """
        read an 1D iteration file

        Arguments:
        ----------
            **imode** : mode to read from

        Returns:
        --------
            **Occam1D.itdict** : dictionary with keys of the header:

            **model_res** : fills this array with the appropriate
                            values (0) for data, (1) for model

        :Example: ::

            >>> m1 = occam1d.Model()
            >>> m1.model_fn = r"/home/occam1d/mt01/TE/Model1D"
            >>> m1.read_iter_file(r"/home/Occam1D/Inv1_TE/M01TE_15.iter")

        """

        if iter_fn is not None:
            self.iter_fn = iter_fn

        if self.iter_fn is None:
            raise IOError("Need to input iteration file")

        if model_fn is not None:
            self.model_fn = model_fn
        if self.model_fn is None:
            raise IOError("Need to input a model file")
        else:
            self.read_model_file()

        freeparams = np.where(self.model_res == -1)[0]

        with open(self.iter_fn, "r") as ifid:
            ilines = ifid.readlines()

        self.itdict = {}
        model = []
        for ii, iline in enumerate(ilines):
            if iline.find(":") >= 0:
                ikey = iline[0:20].strip()
                ivalue = iline[20:].split("!")[0].strip()
                self.itdict[ikey[:-1]] = ivalue
            else:
                try:
                    ilst = iline.strip().split()
                    for kk in ilst:
                        model.append(float(kk))
                except ValueError:
                    pass

        # put the model values into the model dictionary into the res array
        # for easy manipulation and access.
        model = np.array(model)
        self.model_res[freeparams, 1] = model
