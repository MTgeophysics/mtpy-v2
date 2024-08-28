# -*- coding: utf-8 -*-
"""
==================
Occam1D
==================

    * Wrapper class to interact with Occam1D written by Kerry Keys at Scripps 
      adapted from the method of Constable et al., [1987].

    * This class only deals with the MT functionality of the Fortran code, so
      it can make the input files for computing the 1D MT response of an input
      model and or data.  It can also read the output and plot them in a
      useful way.

    * Note that when you run the inversion code, the convergence is quite
      quick, within the first few iterations, so have a look at the L2 cure
      to decide which iteration to plot, otherwise if you look at iterations
      long after convergence the models will be unreliable.



    * Key, K., 2009, 1D inversion of multicomponent, multi-frequency marine
      CSEM data: Methodology and synthetic studies for resolving thin
      resistive layers: Geophysics, 74, F9–F20.

    * The original paper describing the Occam's inversion approach is:

    * Constable, S. C., R. L. Parker, and C. G. Constable, 1987,
      Occam’s inversion –– A practical algorithm for generating smooth
      models from electromagnetic sounding data, Geophysics, 52 (03), 289–300.


    :Intended Use: ::

        >>> import mtpy.modeling.occam1d as occam1d
        >>> #--> make a data file
        >>> d1 = occam1d.Data()
        >>> d1.write_data_file(edi_file=r'/home/MT/mt01.edi', res_err=10, phase_err=2.5,
        >>> ...                save_path=r"/home/occam1d/mt01/TE", mode='TE')
        >>> #--> make a model file
        >>> m1 = occam1d.Model()
        >>> m1.write_model_file(save_path=d1.save_path, target_depth=15000)
        >>> #--> make a startup file
        >>> s1 = occam1d.Startup()
        >>> s1.data_fn = d1.data_fn
        >>> s1.model_fn = m1.model_fn
        >>> s1.save_path = m1.save_path
        >>> s1.write_startup_file()
        >>> #--> run occam1d from python
        >>> occam_path = r"/home/occam1d/Occam1D_executable"
        >>> occam1d.Run(s1.startup_fn, occam_path, mode='TE')
        >>> #--plot the L2 curve
        >>> l2 = occam1d.PlotL2(d1.save_path, m1.model_fn)
        >>> #--> see that iteration 7 is the optimum model to plot
        >>> p1 = occam1d.Plot1DResponse()
        >>> p1.data_te_fn = d1.data_fn
        >>> p1.model_fn = m1.model_fn
        >>> p1.iter_te_fn = r"/home/occam1d/mt01/TE/TE_7.iter"
        >>> p1.resp_te_fn = r"/home/occam1d/mt01/TE/TE_7.resp"
        >>> p1.plot()

@author: J. Peacock (Oct. 2013)
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import time
import subprocess
import string

import numpy as np

from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

import mtpy.core.mt as mt
import mtpy.utils.calculator as mtcc
import mtpy.analysis.geometry as mtg
import matplotlib.pyplot as plt
# =============================================================================


def parse_arguments(arguments):
    """Takes list of command line arguments obtained by passing in sys.argv
    reads these and returns a parser object

    author: Alison Kirkby (2016)
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Set up and run a set of isotropic occam1d model runs"
    )

    parser.add_argument(
        "edipath",
        help="folder containing edi files to use, full path or relative to working directory",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--program_location",
        help="path to the inversion program",
        type=str,
        default=r"/home/547/alk547/occam1d/OCCAM1DCSEM",
    )
    parser.add_argument(
        "-efr",
        "--resistivity_errorfloor",
        help="error floor in resistivity, percent",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-efp",
        "--phase_errorfloor",
        help="error floor in phase, degrees",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-efz",
        "--z_errorfloor",
        help="error floor in z, percent",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-wd",
        "--working_directory",
        help="working directory",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-m",
        "--modes",
        nargs="*",
        help="modes to run, any or all of TE, TM, det (determinant)",
        type=str,
        default=["TE"],
    )
    parser.add_argument(
        "-r",
        "--rotation_angle",
        help='angle to rotate the data by, in degrees or can define option "strike" to rotate to strike, or "file" to get rotation angle from file',
        type=str,
        default="0",
    )
    parser.add_argument(
        "-rfile",
        "--rotation_angle_file",
        help="file containing rotation angles, first column is station name (must match edis) second column is rotation angle",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-spr",
        "--strike_period_range",
        nargs=2,
        help="period range to use for calculation of strike if rotating to strike, two floats",
        type=float,
        default=[1e-3, 1e3],
    )
    parser.add_argument(
        "-sapp",
        "--strike_approx",
        help="approximate strike angle, the strike closest to this value is chosen",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "-q",
        "--remove_outofquadrant",
        help="whether or not to remove points outside of the first or third quadrant, True or False",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-itermax",
        "--iteration_max",
        help="maximum number of iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-rf",
        "--rms_factor",
        help="factor to multiply the minimum possible rms by to get the target rms for the second run",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "-rmsmin",
        "--rms_min",
        help="minimum target rms to assign, e.g. set a value of 1.0 to prevent overfitting data",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-nl",
        "--n_layers",
        help="number of layers in the inversion",
        type=int,
        default=80,
    )
    parser.add_argument(
        "-z1",
        "--z1_layer",
        help="thickness of z1 layer",
        type=float,
        default=10,
    )
    parser.add_argument(
        "-td",
        "--target_depth",
        help="target depth for the inversion in metres",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "-rho0",
        "--start_rho",
        help="starting resistivity value for the inversion",
        type=float,
        default=100,
    )
    parser.add_argument(
        "-s",
        "--master_savepath",
        help="master directory to save suite of runs into",
        default="inversion_suite",
    )

    args = parser.parse_args(arguments)
    args.working_directory = os.path.abspath(args.working_directory)
    if args.rotation_angle not in ["file", "strike"]:
        try:
            args.rotation_angle = float(args.rotation_angle)
        except:
            args.rotation_angle = 0.0

    return args


def update_inputs():
    """Update input parameters from command line

    author: Alison Kirkby (2016).
    """
    from sys import argv

    args = parse_arguments(argv[1:])
    cline_inputs = {}
    cline_keys = [i for i in dir(args) if i[0] != "_"]

    for key in cline_keys:
        cline_inputs[key] = getattr(args, key)

    return cline_inputs


def get_strike(mt_object, fmin, fmax, strike_approx=0):
    """Get the strike from the z array, choosing the strike angle that is closest
    to the azimuth of the PT ellipse (PT strike).

    if there is not strike available from the z array use the PT strike.
    """
    fselect = (mt_object.Z.freq > fmin) & (mt_object.Z.freq < fmax)

    # get median strike angles for frequencies needed (two strike angles due to 90 degree ambiguity)
    zstrike = mtg.strike_angle(z_object=mt_object.Z)[fselect]
    # put both strikes in the same quadrant for averaging
    zstrike = zstrike % 90
    zstrike = np.median(zstrike[np.isfinite(zstrike[:, 0])], axis=0)
    # add 90 to put one back in the other quadrant
    zstrike[1] += 90
    # choose closest value to approx_strike
    zstrike = zstrike[
        np.abs(zstrike - strike_approx)
        - np.amin(np.abs(zstrike - strike_approx))
        < 1e-3
    ]

    if len(zstrike) > 0:
        strike = zstrike[0]
    else:
        # if the data are 1d set strike to 90 degrees (i.e. no rotation)
        strike = 90.0

    return strike


def generate_inputfiles(**input_parameters):
    """Generate all the input files to run occam1d, return the path and the
    startup files to run.

    author: Alison Kirkby (2016)
    """
    edipath = op.join(
        input_parameters["working_directory"], input_parameters["edipath"]
    )
    edilist = [ff for ff in os.listdir(edipath) if ff.endswith(".edi")]

    wkdir_master = op.join(
        input_parameters["working_directory"],
        input_parameters["master_savepath"],
    )
    if not os.path.exists(wkdir_master):
        os.mkdir(wkdir_master)

    rundirs = {}

    for edifile in edilist:
        # read the edi file to get the station name
        eo = mt.MT(op.join(edipath, edifile))
        # print(input_parameters['rotation_angle'], input_parameters['working_directory'], input_parameters[
        #    'rotation_angle_file'])
        if input_parameters["rotation_angle"] == "strike":
            spr = input_parameters["strike_period_range"]
            fmax, fmin = [1.0 / np.amin(spr), 1.0 / np.amax(spr)]
            rotangle = (
                get_strike(
                    eo,
                    fmin,
                    fmax,
                    strike_approx=input_parameters["strike_approx"],
                )
                - 90.0
            ) % 180
        elif input_parameters["rotation_angle"] == "file":
            with open(
                op.join(
                    input_parameters["working_directory"],
                    input_parameters["rotation_angle_file"],
                )
            ) as f:
                line = f.readline().strip().split()

                while string.upper(line[0]) != string.upper(eo.station):
                    line = f.readline().strip().split()
                    if len(line) == 0:
                        line = ["", "0.0"]
                        break
            rotangle = float(line[1])
        else:
            rotangle = input_parameters["rotation_angle"]

        # create a working directory to store the inversion files in
        svpath = "station" + eo.station
        wd = op.join(wkdir_master, svpath)
        if not os.path.exists(wd):
            os.mkdir(wd)
        rundirs[svpath] = []

        # create the model file
        ocm = Model(
            n_layers=input_parameters["n_layers"],
            save_path=wd,
            target_depth=input_parameters["target_depth"],
            z1_layer=input_parameters["z1_layer"],
        )
        ocm.write_model_file()

        for mode in input_parameters["modes"]:
            # create a data file for each mode
            ocd = Data()
            ocd._data_fn = f"Occam1d_DataFile_rot{int(rotangle):03}"
            ocd.write_data_file(
                res_errorfloor=input_parameters["resistivity_errorfloor"],
                phase_errorfloor=input_parameters["phase_errorfloor"],
                z_errorfloor=input_parameters["z_errorfloor"],
                remove_outofquadrant=input_parameters["remove_outofquadrant"],
                mode=mode,
                edi_file=op.join(edipath, edifile),
                thetar=rotangle,
                save_path=wd,
            )

            ocs = Startup(
                data_fn=ocd.data_fn,
                model_fn=ocm.model_fn,
                start_rho=input_parameters["start_rho"],
            )
            startup_fn = "OccamStartup1D" + mode
            ocs.write_startup_file(
                save_path=wd,
                startup_fn=op.join(wd, startup_fn),
                max_iter=input_parameters["iteration_max"],
                target_rms=input_parameters["rms_min"]
                / input_parameters["rms_factor"],
            )
            rundirs[svpath].append(startup_fn)

    return wkdir_master, rundirs


def divide_inputs(work_to_do, size):
    """Divide list of inputs into chunks to send to each processor."""
    chunks = [[] for _ in range(size)]
    for i, d in enumerate(work_to_do):
        chunks[i % size].append(d)

    return chunks


def build_run():
    """Build input files and run a suite of models in series (pretty quick so won't bother parallelise)

    run Occam1d on each set of inputs.
    Occam is run twice. First to get the lowest possible misfit.
    we then set the target rms to a factor (default 1.05) times the minimum rms achieved
    and run to get the smoothest model.

    author: Alison Kirkby (2016).
    """
    # from mpi4py import MPI

    # get command line arguments as a dictionary
    input_parameters = update_inputs()

    # create the inputs and get the run directories
    master_wkdir, run_directories = generate_inputfiles(**input_parameters)

    # run Occam1d on each set of inputs.
    # Occam is run twice. First to get the lowest possible misfit.
    # we then set the target rms to a factor (default 1.05) times the minimum rms achieved
    # and run to get the smoothest model.
    for rundir in list(run_directories.keys()):
        wd = op.join(master_wkdir, rundir)
        os.chdir(wd)
        for startupfile in run_directories[rundir]:
            # define some parameters
            mode = startupfile[14:]
            iterstring = "RMSmin" + mode
            # run for minimum rms
            subprocess.call(
                [input_parameters["program_location"], startupfile, iterstring]
            )
            # read the iter file to get minimum rms
            iterfilelist = [
                ff
                for ff in os.listdir(wd)
                if (ff.startswith(iterstring) and ff.endswith(".iter"))
            ]
            # only run a second lot of inversions if the first produced outputs
            if len(iterfilelist) > 0:
                iterfile = max(iterfilelist)
                startup = Startup()
                startup.read_startup_file(op.join(wd, iterfile))
                # create a new startup file the same as the previous one but target rms is factor*minimum_rms
                target_rms = (
                    float(startup.misfit_value) * input_parameters["rms_factor"]
                )
                if target_rms < input_parameters["rms_min"]:
                    target_rms = input_parameters["rms_min"]
                startupnew = Startup(
                    data_fn=op.join(wd, startup.data_file),
                    model_fn=op.join(wd, startup.model_file),
                    max_iter=input_parameters["iteration_max"],
                    start_rho=input_parameters["start_rho"],
                    target_rms=target_rms,
                )
                startupnew.write_startup_file(
                    startup_fn=op.join(wd, startupfile), save_path=wd
                )
                # run occam again
                subprocess.call(
                    [
                        input_parameters["program_location"],
                        startupfile,
                        "Smooth" + mode,
                    ]
                )


if __name__ == "__main__":
    build_run()
