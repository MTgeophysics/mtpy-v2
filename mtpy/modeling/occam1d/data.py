# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:31:30 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mtpy.core import MTDataFrame
import mtpy.utils.calculator as mtcc

# =============================================================================
class Occam1DData(object):
    """
    reads and writes occam 1D data files

    ===================== =====================================================
    Attributes             Description
    ===================== =====================================================
    _data_fn              basename of data file *default* is Occam1DDataFile
    _header_line          header line for description of data columns
    _ss                   string spacing *default* is 6*' '
    _string_fmt           format of data *default* is '+.6e'
    data                  array of data
    data_fn               full path to data file
    freq                  frequency array of data
    mode                  mode to invert for [ 'TE' | 'TM' | 'det' ]
    phase_te              array of TE phase
    phase_tm              array of TM phase
    res_te                array of TE apparent resistivity
    res_tm                array of TM apparent resistivity
    resp_fn               full path to response file
    save_path             path to save files to
    ===================== =====================================================


    ===================== =====================================================
    Methods               Description
    ===================== =====================================================
    write_data_file       write an Occam1D data file
    read_data_file        read an Occam1D data file
    read_resp_file        read a .resp file output by Occam1D
    ===================== =====================================================

    :Example: ::

        >>> import mtpy.modeling.occam1d as occam1d
        >>> #--> make a data file for TE mode
        >>> d1 = occam1d.Data()
        >>> d1.write_data_file(edi_file=r'/home/MT/mt01.edi', res_err=10, phase_err=2.5,
        >>> ...                save_path=r"/home/occam1d/mt01/TE", mode='TE')

    """

    def __init__(self, mt_dataframe, **kwargs):
        self.logger = logger
        self.mt_dataframe = MTDataFrame(data=mt_dataframe)

        self._string_fmt = "+.6e"
        self._ss = 6 * " "
        self._acceptable_modes = ["te" "tm", "det", "detz", "tez", "tmz"]
        self._data_fn = "Occam1d_DataFile"
        self._header_line = "!{0}\n".format(
            "      ".join(["Type", "Freq#", "TX#", "Rx#", "Data", "Std_Error"])
        )
        self.mode = "det"
        self.data = None
        self.rotation_angle = 0

        self.data_1 = None
        self.data_1_error = None
        self.data_2 = None
        self.data_2_error = None

        self.save_path = Path().cwd()
        self.data_fn = self.save_path.joinpath(self._data_fn)

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

    def __str__(self):
        lines = ["Occam 1D Data:"]
        lines.append(f"\tMode:     {self.mode}")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    @property
    def mode_01(self):
        if self.mode == "te":
            return "RhoZxy"
        elif self.mode == "tm":
            return "RhoZyx"
        elif self.mode == "det":
            return "RhoZxy"
        elif self.mode == "detz":
            return "RealZxy"
        elif self.mode == "tez":
            return "RealZxy"
        elif self.mode == "tmz":
            return "RealZyx"

    @property
    def mode_02(self):
        if self.mode == "te":
            return "PhsZxy"
        elif self.mode == "tm":
            return "PhsZyx"
        elif self.mode == "det":
            return "PhsZxy"
        elif self.mode == "detz":
            return "ImagZxy"
        elif self.mode == "tez":
            return "ImagZxy"
        elif self.mode == "tmz":
            return "ImagZyx"

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):

        if mode not in self._acceptable_modes:
            raise ValueError(
                f"Mode {mode} not in accetable modes {self._acceptable_modes}"
            )
        self._mode = mode

    def _get_sub_dataframe(self):
        if self._mode == "te":
            sub_df = pd.DataFrame(
                {
                    "frequency": 1.0 / self.mt_dataframe.dataframe.period,
                    "data_1": self.mt_dataframe.dataframe.res_xy,
                    "data_1_error": self.mt_dataframe.dataframe.res_xy_model_error,
                    "data_2": self.mt_dataframe.dataframe.phase_xy,
                    "data_2_error": self.mt_dataframe.dataframe.phase_xy_model_error,
                }
            )

        elif self._mode == "tm":
            sub_df = pd.DataFrame(
                {
                    "frequency": 1.0 / self.mt_dataframe.dataframe.period,
                    "data_1": self.mt_dataframe.dataframe.res_yx,
                    "data_1_error": self.mt_dataframe.dataframe.res_yx_model_error,
                    "data_2": self.mt_dataframe.dataframe.phase_yx,
                    "data_2_error": self.mt_dataframe.dataframe.phase_yx_model_error,
                }
            )

        elif self._mode == "det":
            z_obj = self.mt_dataframe.to_z_object()

            sub_df = pd.DataFrame(
                {
                    "frequency": 1.0 / self.mt_dataframe.dataframe.period,
                    "data_1": z_obj.det.real,
                    "data_1_error": z_obj.det_model_error,
                    "data_2": z_obj.det.imag,
                    "data_2_error": z_obj.det_model_error,
                }
            )

        elif self._mode == "detz":
            z_obj = self.mt_dataframe.to_z_object()
            sub_df = pd.DataFrame(
                {
                    "frequency": 1.0 / self.mt_dataframe.dataframe.period,
                    "data_1": z_obj.det.real * np.pi * 4e-4,
                    "data_1_error": z_obj.det_model_error * np.pi * 4e-4,
                    "data_2": z_obj.det.imag * np.pi * 4e-4,
                    "data_2_error": z_obj.det_model_error * np.pi * 4e-4,
                }
            )

        elif self.mode == "tez":
            sub_df = pd.DataFrame(
                {
                    "frequency": 1.0 / self.mt_dataframe.dataframe.period,
                    "data_1": self.mt_dataframe.dataframe.zxy.real
                    * np.pi
                    * 4e-4,
                    "data_1_error": self.mt_dataframe.dataframe.zxy_model_error
                    * np.pi
                    * 4e-4,
                    "data_2": self.mt_dataframe.dataframe.zxy.imag
                    * np.pi
                    * 4e-4,
                    "data_2_error": self.mt_dataframe.dataframe.zxy_model_error
                    * np.pi
                    * 4e-4,
                }
            )

        elif self.mode == "tmz":
            sub_df = pd.DataFrame(
                {
                    "frequency": 1.0 / self.mt_dataframe.dataframe.period,
                    "data_1": self.mt_dataframe.dataframe.zyx.real
                    * np.pi
                    * 4e-4,
                    "data_1_error": self.mt_dataframe.dataframe.zyx_model_error
                    * np.pi
                    * 4e-4,
                    "data_2": self.mt_dataframe.dataframe.zyx.imag
                    * np.pi
                    * 4e-4,
                    "data_2_error": self.mt_dataframe.dataframe.zyx_model_error
                    * np.pi
                    * 4e-4,
                }
            )

        sub_df = sub_df.sort_values("frequency", ascending=False).reindex()

        return sub_df

    def write_data_file(
        self,
        filename,
        mode="det",
        remove_outofquadrant=False,
    ):
        """
        make1Ddatafile will write a data file for Occam1D

        Arguments:
        ---------
            **rp_tuple** : np.ndarray (freq, res, res_err, phase, phase_err)
                            with res, phase having shape (num_freq, 2, 2).

            **edi_file** : string
                          full path to edi file to be modeled.

            **save_path** : string
                           path to save the file, if None set to dirname of
                           station if edipath = None.  Otherwise set to
                           dirname of edipath.

            **thetar** : float
                         rotation angle to rotate Z. Clockwise positive and N=0
                         *default* = 0

            **mode** : [ 'te' | 'tm' | 'det']
                              mode to model can be (*default*='both'):
                                - 'te' for just TE mode (res/phase)
                                - 'tm' for just TM mode (res/phase)
                                - 'det' for the determinant of Z (converted to
                                        res/phase)
                              add 'z' to any of these options to model
                              impedance tensor values instead of res/phase


            **res_err** : float
                        errorbar for resistivity values.  Can be set to (
                        *default* = 'data'):

                        - 'data' for errorbars from the data
                        - percent number ex. 10 for ten percent

            **phase_err** : float
                          errorbar for phase values.  Can be set to (
                          *default* = 'data'):

                            - 'data' for errorbars from the data
                            - percent number ex. 10 for ten percent
            **res_errorfloor**: float
                                error floor for resistivity values
                                in percent
            **phase_errorfloor**: float
                                  error floor for phase in degrees
            **remove_outofquadrant**: True/False; option to remove the resistivity and
                                      phase values for points with phases out
                                      of the 1st/3rd quadrant (occam requires
                                      0 < phase < 90 degrees; phases in the 3rd
                                      quadrant are shifted to the first by
                                      adding 180 degrees)

        :Example: ::

            >>> import mtpy.modeling.occam1d as occam1d
            >>> #--> make a data file
            >>> d1 = occam1d.Data()
            >>> d1.write_data_file(edi_file=r'/home/MT/mt01.edi', res_err=10,
            >>> ...                phase_err=2.5, mode='TE',
            >>> ...                save_path=r"/home/occam1d/mt01/TE")
        """
        # be sure that the input mode is not case sensitive
        self.mode = mode.lower()

        sub_df = self._get_sub_dataframe()

        if remove_outofquadrant:
            self._remove_outofquadrant_phase()

        # --> write file
        # make sure the savepath exists, if not create it
        self.data_fn = Path(filename)

        # --> write file as a list of lines
        dlines = []

        dlines.append("Format:  EMData_1.1 \n")
        dlines.append(f"!mode:   {mode.upper()}\n")
        dlines.append(f"!rotation_angle = {self.rotation_angle:.2f}\n")

        # needs a transmitter to work so put in a dummy one
        dlines.append("# Transmitters: 1\n")
        dlines.append("0 0 0 0 0 \n")

        nf = sub_df.frequency.size
        # write frequencies
        dlines.append(f"# Frequencies:   {nf}\n")
        for ff in sub_df.frequency:
            dlines.append(f"   {ff:{self._string_fmt}}\n")

        # needs a receiver to work so put in a dummy one
        dlines.append("# Receivers: 1 \n")
        dlines.append("0 0 0 0 0 0 \n")

        # write data
        dlines.append(f"# Data:{self._ss}{2 * nf}\n")
        num_data_line = len(dlines)

        dlines.append(self._header_line)
        data_count = 0

        for row in sub_df.itertuples():
            # write lines
            dlines.append(
                self._ss.join(
                    [
                        self.mode_01,
                        str(row.Index + 1),
                        "0",
                        "1",
                        f"{row.data_1:{self._string_fmt}}",
                        f"{row.data_1_error:{self._string_fmt}}\n",
                    ]
                )
            )
            data_count += 1
            dlines.append(
                self._ss.join(
                    [
                        self.mode_02,
                        str(row.Index + 1),
                        "0",
                        "1",
                        f"{row.data_2:{self._string_fmt}}",
                        f"{row.data_2_error:{self._string_fmt}}\n",
                    ]
                )
            )
            data_count += 1

        # --> write file
        dlines[num_data_line - 1] = f"# Data:{self._ss}{data_count}\n"

        with open(self.data_fn, "w") as dfid:
            dfid.writelines(dlines)

        self.logger.info(f"Wrote Data File to : {self.data_fn}")

    def _remove_outofquadrant_phase(self, sub_df):
        """
        remove out of quadrant phase from data
        """
        # remove data points with phase out of quadrant
        if "z" in self.mode:
            sub_df.loc[
                (sub_df.data_1 / sub_df.data_2 > 0), ["data_1", "data_2"]
            ] = 0

        elif self.mode in ["det", "te", "tm"]:
            sub_df.loc[(sub_df.data_2 % 180 < 0), "data_2"] = 0

        return sub_df

    def _remove_zeros(self, sub_df):
        """
        remove zeros from the data frame

        :param sub_df: DESCRIPTION
        :type sub_df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        sub_df.loc[(sub_df != 0).any(axis=1)]
        return sub_df

    def read_data_file(self, data_fn):
        """
        reads a 1D data file

        Arguments:
        ----------
            **data_fn** : full path to data file

        Returns:
        --------
            **Occam1D.rpdict** : dictionary with keys:

                *'freq'* : an array of frequencies with length nf

                *'resxy'* : TE resistivity array with shape (nf,4) for (0) data,
                          (1) dataerr, (2) model, (3) modelerr

                *'resyx'* : TM resistivity array with shape (nf,4) for (0) data,
                          (1) dataerr, (2) model, (3) modelerr

                *'phasexy'* : TE phase array with shape (nf,4) for (0) data,
                            (1) dataerr, (2) model, (3) modelerr

                *'phaseyx'* : TM phase array with shape (nf,4) for (0) data,
                            (1) dataerr, (2) model, (3) modelerr

        :Example: ::

            >>> old = occam1d.Data()
            >>> old.data_fn = r"/home/Occam1D/Line1/Inv1_TE/MT01TE.dat"
            >>> old.read_data_file()
        """

        self.data_fn = Path(data_fn)
        if not self.data_fn.exists():
            raise IOError(f"Could not find {self.data_fn}, check path")

        self.save_path = self.data_fn.parent

        with open(self.data_fn, "r") as fid:
            dlines = fid.readlines()

        # make a dictionary of all the fields found so can put them into arrays
        finddict = {}
        for ii, dline in enumerate(dlines):
            if dline.find("#") <= 3:
                fkey = dline[2:].strip().split(":")[0]
                fvalue = ii
                finddict[fkey] = fvalue

        # get number of frequencies
        nfreq = int(
            dlines[finddict["Frequencies"]][2:].strip().split(":")[1].strip()
        )

        # frequency list
        freq = np.array(
            [
                float(ff)
                for ff in dlines[
                    finddict["Frequencies"] + 1 : finddict["Receivers"]
                ]
            ]
        )

        # data dictionary to put things into
        # check to see if there is alread one, if not make a new one
        data = {
            "frequency": freq,
            "zxy": np.zeros(nfreq, dtype=complex),
            "zyx": np.zeros(nfreq, dtype=complex),
            "res_xy": np.zeros(nfreq),
            "res_yx": np.zeros(nfreq),
            "phase_xy": np.zeros(nfreq),
            "phase_yx": np.zeros(nfreq),
            "zxy_model_error": np.zeros(nfreq),
            "zyx_model_error": np.zeros(nfreq),
            "res_xy_model_error": np.zeros(nfreq),
            "res_yx_model_error": np.zeros(nfreq),
            "phase_xy_model_error": np.zeros(nfreq),
            "phase_yx_model_error": np.zeros(nfreq),
        }

        # get data
        for dline in dlines[finddict["Data"] + 1 :]:
            if dline.find("!") == 0:
                pass
            else:
                dlst = dline.strip().split()
                dlst = [dd.strip() for dd in dlst]
                if len(dlst) > 4:
                    jj = int(dlst[1]) - 1
                    dvalue = float(dlst[4])
                    derr = float(dlst[5])
                    if dlst[0] in ["RhoZxy", "103"]:
                        self.mode = "te"
                        data["res_xy"][jj] = dvalue
                        data["res_xy_model_error"][jj] = derr
                    elif dlst[0] in ["PhsZxy", "104"]:
                        self.mode = "te"
                        data["phase_xy"][jj] = dvalue
                        data["phase_xy_model_error"][jj] = derr
                    elif dlst[0] in ["RhoZyx", "105"]:
                        self.mode = "tm"
                        data["res_yx"][jj] = dvalue
                        data["res_yx_model_error"][jj] = derr
                    elif dlst[0] in ["PhsZyx", "106"]:
                        self.mode = "TM"
                        data["phase_yx"][jj] = dvalue
                        data["phase_yx_model_error"][jj] = derr
                    elif dlst[0] in ["RealZxy", "113"]:
                        self.mode = "tez"
                        data["zxy"][jj] += dvalue / (np.pi * 4e-4)
                        data["zxy_model_error"][jj] = derr / (np.pi * 4e-4)
                    elif dlst[0] in ["ImagZxy", "114"]:
                        self.mode = "tez"
                        data["zxy"][jj] += 1j * dvalue / (np.pi * 4e-4)
                        data["zxy_model_error"][jj] = derr / (np.pi * 4e-4)
                    elif dlst[0] in ["RealZyx", "115"]:
                        self.mode = "tmz"
                        data["zyx"][jj] += dvalue / (np.pi * 4e-4)
                        data["zyx_model_error"][jj] = derr / (np.pi * 4e-4)
                    elif dlst[0] in ["ImagZyx", "116"]:
                        self.mode = "tmz"
                        data["zyx"][jj] += 1j * dvalue / (np.pi * 4e-4)
                        data["zyx_model_error"][jj] = derr / (np.pi * 4e-4)

        df = pd.DataFrame(data)
        self.mt_dataframe = MTDataFrame(data=df)

    def read_resp_file(self, resp_fn=None, data_fn=None):
        """
         read response file

         Arguments:
         ---------
             **resp_fn** : full path to response file

             **data_fn** : full path to data file

         Fills:
         --------

             *freq* : an array of frequencies with length nf

             *res_te* : TE resistivity array with shape (nf,4) for (0) data,
                       (1) dataerr, (2) model, (3) modelerr

             *res_tm* : TM resistivity array with shape (nf,4) for (0) data,
                       (1) dataerr, (2) model, (3) modelerr

             *phase_te* : TE phase array with shape (nf,4) for (0) data,
                         (1) dataerr, (2) model, (3) modelerr

             *phase_tm* : TM phase array with shape (nf,4) for (0) data,
                         (1) dataerr, (2) model, (3) modelerr

        :Example: ::
             >>> o1d = occam1d.Data()
             >>> o1d.data_fn = r"/home/occam1d/mt01/TE/Occam1D_DataFile_TE.dat"
             >>> o1d.read_resp_file(r"/home/occam1d/mt01/TE/TE_7.resp")

        """

        if resp_fn is not None:
            self.resp_fn = resp_fn
        if self.resp_fn is None:
            raise IOError("Need to input response file")

        if data_fn is not None:
            self.data_fn = data_fn
        if self.data_fn is None:
            raise IOError("Need to input data file")
        # --> read in data file
        self.read_data_file()

        # --> read response file
        dfid = open(self.resp_fn, "r")

        dlines = dfid.readlines()
        dfid.close()

        finddict = {}
        for ii, dline in enumerate(dlines):
            if dline.find("#") <= 3:
                fkey = dline[2:].strip().split(":")[0]
                fvalue = ii
                finddict[fkey] = fvalue

        for dline in dlines[finddict["Data"] + 1 :]:
            if dline.find("!") == 0:
                pass
            else:
                dlst = dline.strip().split()
                if len(dlst) > 4:
                    jj = int(dlst[1]) - 1
                    dvalue = float(dlst[4])
                    derr = float(dlst[5])
                    rvalue = float(dlst[6])
                    try:
                        rerr = float(dlst[7])
                    except ValueError:
                        rerr = 1000.0
                    if dlst[0] == "RhoZxy" or dlst[0] == "103":
                        self.res_te[0, jj] = dvalue
                        self.res_te[jj] = derr
                        self.res_te[2, jj] = rvalue
                        self.res_te[3, jj] = rerr
                    if dlst[0] == "PhsZxy" or dlst[0] == "104":
                        self.phase_te[0, jj] = dvalue
                        self.phase_te[jj] = derr
                        self.phase_te[2, jj] = rvalue
                        self.phase_te[3, jj] = rerr
                    if dlst[0] == "RhoZyx" or dlst[0] == "105":
                        self.res_tm[0, jj] = dvalue
                        self.res_tm[jj] = derr
                        self.res_tm[2, jj] = rvalue
                        self.res_tm[3, jj] = rerr
                    if dlst[0] == "PhsZyx" or dlst[0] == "106":
                        self.phase_tm[0, jj] = dvalue
                        self.phase_tm[jj] = derr
                        self.phase_tm[2, jj] = rvalue
                        self.phase_tm[3, jj] = rerr
                    if dlst[0] == "RealZxy" or dlst[0] == "113":
                        self.mode = "TEz"
                        self.data["zxy"][0, jj] = dvalue / (np.pi * 4e-4)
                        self.data["zxy"][jj] = derr / (np.pi * 4e-4)
                        self.data["zxy"][2, jj] = rvalue / (np.pi * 4e-4)
                        self.data["zxy"][3, jj] = rerr
                    if dlst[0] == "ImagZxy" or dlst[0] == "114":
                        self.mode = "TEz"
                        self.data["zxy"][0, jj] += 1j * dvalue / (np.pi * 4e-4)
                        self.data["zxy"][jj] = derr / (np.pi * 4e-4)
                        self.data["zxy"][2, jj] += 1j * rvalue / (np.pi * 4e-4)
                        self.data["zxy"][3, jj] = rerr
                    if dlst[0] == "RealZyx" or dlst[0] == "115":
                        self.mode = "TMz"
                        self.data["zyx"][0, jj] = dvalue / (np.pi * 4e-4)
                        self.data["zyx"][jj] = derr / (np.pi * 4e-4)
                        self.data["zyx"][2, jj] = rvalue / (np.pi * 4e-4)
                        self.data["zyx"][3, jj] = rerr
                    if dlst[0] == "ImagZyx" or dlst[0] == "116":
                        self.mode = "TMz"
                        self.data["zyx"][0, jj] += 1j * dvalue / (np.pi * 4e-4)
                        self.data["zyx"][jj] = derr / (np.pi * 4e-4)
                        self.data["zyx"][2, jj] += 1j * rvalue / (np.pi * 4e-4)
                        self.data["zyx"][3, jj] = rerr
        if "z" in self.mode:
            if "TE" in self.mode:
                pol = "xy"
            elif "TM" in self.mode:
                pol = "yx"
            for ii in [0, 2]:
                self.data["res" + pol][0 + ii] = (
                    0.2
                    * np.abs(self.data["z" + pol][0 + ii]) ** 2.0
                    / self.freq
                )
                self.data["phase" + pol][0 + ii] = np.rad2deg(
                    np.arctan(
                        self.data["z" + pol][0 + ii].imag
                        / self.data["z" + pol][0 + ii].real
                    )
                )

                self.data["res" + pol][1 + ii] = (
                    self.data["res" + pol][0 + ii]
                    * self.data["z" + pol][1 + ii].real
                    / np.abs(self.data["z" + pol][0 + ii])
                )

                for jjj in range(len(self.freq)):
                    self.data["phase" + pol][
                        1 + ii, jjj
                    ] = mtcc.z_error2r_phi_error(
                        self.data["z" + pol][0 + ii, jjj].real,
                        self.data["z" + pol][0 + ii, jjj].imag,
                        self.data["z" + pol][1 + ii, jjj].real,
                    )[
                        1
                    ]
            if pol == "xy":
                self.res_te = self.data["resxy"]
                self.phase_te = self.data["phasexy"]
            elif pol == "yx":
                self.res_tm = self.data["resyx"]
                self.phase_tm = self.data["phaseyx"]
