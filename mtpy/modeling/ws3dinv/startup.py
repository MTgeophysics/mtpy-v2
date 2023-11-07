# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:33:52 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from loguru import logger

# =============================================================================


class WSStartup:
    """
    read and write startup files

    :Example: ::

        >>> import mtpy.modeling.ws3dinv as ws
        >>> dfn = r"/home/MT/ws3dinv/Inv1/WSDataFile.dat"
        >>> ifn = r"/home/MT/ws3dinv/Inv1/init3d"
        >>> sws = ws.WSStartup(data_fn=dfn, initial_fn=ifn)


    =================== =======================================================
    Attributes          Description
    =================== =======================================================
    apriori_fn          full path to *a priori* model file
                        *default* is 'default'
    control_fn          full path to model index control file
                        *default* is 'default'
    data_fn             full path to data file
    error_tol           error tolerance level
                        *default* is 'default'
    initial_fn          full path to initial model file
    lagrange            starting lagrange multiplier
                        *default* is 'default'
    max_iter            max number of iterations
                        *default* is 10
    model_ls            model length scale
                        *default* is 5 0.3 0.3 0.3
    output_stem         output file name stem
                        *default* is 'ws3dinv'
    save_path           directory to save file to
    startup_fn          full path to startup file
    static_fn           full path to statics file
                        *default* is 'default'
    target_rms          target rms
                        *default* is 1.0
    =================== =======================================================

    """

    def __init__(self, data_fn=None, initial_fn=None, **kwargs):
        self.logger = logger
        self.data_fn = Path(data_fn)
        self.initial_fn = Path(initial_fn)

        self.output_stem = "ws3dinv"
        self.apriori_fn = "default"
        self.model_ls = [5, 0.3, 0.3, 0.3]
        self.target_rms = 1.0
        self.control_fn = "default"
        self.max_iter = 10
        self.error_tol = "default"
        self.static_fn = "default"
        self.lagrange = "default"
        self.save_path = Path()
        self.fn_basename = "startup"

        self._startup_keys = [
            "data_file",
            "output_file",
            "initial_model_file",
            "prior_model_file",
            "control_model_index",
            "target_rms",
            "max_no_iteration",
            "model_length_scale",
            "lagrange_info",
            "error_tol_level",
            "static_file",
        ]

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def startup_fn(self):
        return self.save_path.joinpath(self.fn_basename)

    @startup_fn.setter
    def startup_fn(self, value):
        if value is None:
            self.save_path = Path()

        else:
            value = Path(value)
            self.save_path = value.parent
            self.fn_basename = value.name

    def write_startup_file(self, save_path):
        """
        makes a startup file for WSINV3D.

        """
        if self.data_fn is None:
            raise IOError("Need to input data file name")

        if self.initial_fn is None:
            raise IOError("Need to input initial model file name")

        # create the output filename
        slines = []

        if self.startup_fn.parent == self.data_fn.parent:
            slines.append(f"{'DATA_FILE':<20}{self.data_fn.name}\n")
            if len(self.data_fn.name) > 70:
                self.logger.warning(
                    "Data file is too long, going to get an error at runtime"
                )
        else:
            slines.append(f"{'DATA_FILE':<20}{self.data_fn}\n")
            if len(self.data_fn) > 70:
                print(
                    "Data file is too long, going to get an error at runtime"
                )

        slines.append(f"{'OUTPUT_FILE':<20}{self.output_stem}\n")

        if self.startup_fn.parent == self.initial_fn.parent:
            slines.append(
                f"{'INITIAL_MODEL_FILE':<20}{self.initial_fn.name}\n"
            )
        else:
            slines.append(f"{'INITIAL_MODEL_FILE':<20}{self.initial_fn}\n")
        slines.append(f"{'PRIOR_MODEL_FILE':<20}{self.apriori_fn}\n")
        slines.append(f"{'CONTROL_MODEL_INDEX ':<20}{self.control_fn}\n")
        slines.append(f"{'TARGET_RMS':<20}{self.target_rms}\n")
        slines.append(f"{'MAX_NO_ITERATION':<20}{self.max_iter}\n")
        slines.append(
            "{0:<20}{1:.0f} {2:.1f} {3:.1f} {4:.1f}\n".format(
                "MODEL_LENGTH_SCALE",
                self.model_ls[0],
                self.model_ls[1],
                self.model_ls[2],
                self.model_ls[3],
            )
        )

        slines.append(f"{'LAGRANGE_INFO':<20}{self.lagrange} \n")
        slines.append(f"{'ERROR_TOL_LEVEL':<20}{self.error_tol} \n")
        slines.append(f"{'STATIC_FILE':<20}{self.static_fn} \n")

        with open(self.startup_fn, "w") as sfid:
            sfid.write("".join(slines))

        self.logger.info(f"Wrote startup file to: {self.startup_fn}")
        return self.startup_fn

    def read_startup_file(self, startup_fn):
        """
        read startup file fills attributes

        """
        self.startup_fn = startup_fn

        with open(self.startup_fn, "r") as sfid:
            slines = sfid.readlines()

        slines = [ss.strip().split()[1:] for ss in slines]

        self.data_fn = Path(slines[0][0].strip())
        if not self.data_fn.is_file():
            self.data_fn = self.save_path.joinpath(self.data_fn)
        self.output_stem = slines[1][0].strip()
        self.initial_fn = Path(slines[2][0].strip())
        if not self.initial_fn.is_file():
            self.initial_fn = self.save_path.joinpath(self.initial_fn)
        self.apriori_fn = slines[3][0].strip()
        self.control_fn = slines[4][0].strip()
        self.target_rms = float(slines[5][0].strip())
        self.max_iter = int(slines[6][0].strip())
        try:
            self.model_ls = [
                int(slines[7][0]),
                float(slines[7][1]),
                float(slines[7][2]),
                float(slines[7][3]),
            ]
        except ValueError:
            self.model_ls = slines[7][0]

        self.lagrange = slines[8][0].strip()
        self.error_tol = slines[9][0].strip()
        try:
            self.static_fn = slines[10][0].strip()
        except IndexError:
            self.logger.warning("Did not find static_fn")
