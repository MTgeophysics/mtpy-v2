# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:43:07 2019

Estimate Transfer Function Quality
    
    * based on simple statistics

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import os
import glob

import numpy as np
import pandas as pd
from scipy import interpolate

from loguru import logger


# =============================================================================
#
# =============================================================================
class EMTFStats:
    """
    Class to estimate data quality of EM transfer functions

    :param tf_dir: transfer function directory
    :type tf_dir: string

    :param stat_limits: criteria for statistics based on a 0-5 rating scale
    :type stat_limits: dictionary

    :Example: ::

        >>> from usgs_archive import estimate_tf_quality_factor as tfq
        >>> edi_dir = r"/home/edi_folders/survey_01"
        >>> q = EMTFStats()
        >>> stat_df = q.compute_statistics(edi_dir)
        >>> q_df = q.estimate_data_quality(stat_df=stat_df)
        >>> s_df = q.summarize_data_quality(q_df)
    """

    def __init__(self, z_object, t_object, **kwargs):
        self.z_object = z_object
        self.t_object = t_object
        self.stat_limits = {
            "std": {
                5: (0, 0.5),
                4: (0.5, 1.25),
                3: (1.25, 2.5),
                2: (2.5, 10.0),
                1: (10.0, 25.0),
                0: (25.0, 1e36),
            },
            "corr": {
                5: (0.975, 1.0),
                4: (0.9, 0.975),
                3: (0.75, 0.9),
                2: (0.5, 0.75),
                1: (0.25, 0.5),
                0: (-1.0, 0.25),
            },
            "diff": {
                5: (0.0, 0.5),
                4: (0.5, 1.0),
                3: (1.0, 2.0),
                2: (2.0, 5.0),
                1: (5.0, 10.0),
                0: (10.0, 1e36),
            },
            "fit": {
                5: (0, 5),
                4: (5, 15),
                3: (15, 50),
                2: (50, 100),
                1: (100, 200),
                0: (200, 1e36),
            },
            "bad": {
                5: (0, 2),
                4: (2, 4),
                3: (4, 10),
                2: (10, 15),
                1: (15, 20),
                0: (20, 1e36),
            },
        }

        self.z_dict = {(0, 0): "xx", (0, 1): "xy", (1, 0): "yx", (1, 1): "yy"}
        self.t_dict = {(0, 0): "x", (0, 1): "y"}
        self.types = (
            [
                f"{ll}_{ii}{jj}_{kk}"
                for ii in ["x", "y"]
                for jj in ["x", "y"]
                for kk in ["std", "corr", "diff", "fit"]
                for ll in ["res", "phase"]
            ]
            + [
                f"{ll}_{ii}_{kk}"
                for ii in ["x", "y"]
                for kk in ["std", "corr", "diff", "fit"]
                for ll in ["tipper"]
            ]
            + [
                f"bad_points_{ll}_{ii}{jj}"
                for ii in ["x", "y"]
                for jj in ["x", "y"]
                for ll in ["res", "phase"]
            ]
            + [f"bad_points_tipper_{ii}" for ii in ["x", "y"]]
        )

    def locate_bad_res_points(self, res):
        """
        try to locate bad points to remove
        """
        return self._locate_bad_points(res, 0, factor=np.cos(np.pi / 4))

    def locate_bad_phase_points(self, phase, test=5):
        """
        try to locate bad points to remove
        """
        return self._locate_bad_points(phase, test)

    def locate_bad_tipper_points(self, tipper, test=0.2):
        """
        try to locate bad points to remove
        """
        return self._locate_bad_points(tipper, test)

    def _locate_bad_points(self, array, test, factor=None):
        """
        locat bad points within an array


        :param array: DESCRIPTION
        :type array: TYPE
        :param test: DESCRIPTION
        :type test: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        ### estimate levearge points, or outliers
        ### estimate the median
        med = np.nanmedian(array)
        ### locate the point closest to the median
        tol = np.abs(np.nanmin(array - np.nanmedian(array)))
        m_index = np.where(
            (abs(array - med) >= tol * 0.95) & (abs(array - med) <= tol * 1.05)
        )[0][0]
        r_index = m_index + 1

        bad_points = []
        # go to the right
        # if factor is given
        if factor is None:
            while r_index < array.shape[0]:
                if abs(array[r_index] - array[r_index - 1]) > test:
                    bad_points.append(r_index)
                r_index += 1

            # go to the left
            l_index = m_index - 1
            while l_index > -1:
                if abs(array[l_index] - array[l_index - 1]) > test:
                    bad_points.append(l_index)
                l_index -= 1
        else:
            while r_index < array.shape[0]:
                if (
                    abs(array[r_index] - array[r_index - 1])
                    > factor * array[r_index]
                ):
                    bad_points.append(r_index)
                r_index += 1

            # go to the left
            l_index = m_index - 1
            while l_index > -1:
                if (
                    abs(array[l_index] - array[l_index - 1])
                    > factor * array[l_index]
                ):
                    bad_points.append(l_index)
                l_index -= 1

        return np.array(sorted(bad_points))

    def compute_statistics(self):
        """
        Compute statistics of the transfer functions in a given directory.

        Statistics are:

            * one-lag autocorrelation coefficient, estimator for smoothness
            * average of errors on components
            * fit to a least-squres smooth curve
            * normalized standard deviation of the first derivative, another
              smoothness estimator

        :param tf_dir: path to directory of transfer functions
        :type tf_dir: string

        :returns: data frame of all the statistics estimated
        :rtype: pandas.DataFrame

        .. note:: Writes a file to the tf_dir named tf_quality_statistics.csv
        """

        stat_array = np.zeros(
            1,
            dtype=[(key, float) for key in sorted(self.types)],
        )
        if self.z_object is not None:
            for ii in range(2):
                for jj in range(2):
                    flip = False
                    comp = self.z_dict[(ii, jj)]

                    ### locate bad points
                    bad_points_res = self.locate_bad_res_points(
                        self.z_object.resistivity[:, ii, jj]
                    )
                    stat_array[0][f"bad_points_res_{comp}"] = max(
                        [1, len(bad_points_res)]
                    )
                    bad_points_phase = self.locate_bad_phase_points(
                        self.z_object.phase[:, ii, jj]
                    )
                    stat_array[0][f"bad_points_phase_{comp}"] = max(
                        [1, len(bad_points_res)]
                    )
                    bad_points = np.unique(
                        np.append(bad_points_res, bad_points_phase)
                    )
                    ### need to get the data points that are within the reasonable range
                    ### and not 0
                    nz_index = np.nonzero(
                        self.z_object.resistivity[:, ii, jj]
                    )[0]
                    nz_mask = np.isin(nz_index, bad_points)
                    nz_index = np.delete(nz_index, nz_mask)

                    f = self.z_object.frequency[nz_index]
                    if len(f) < 2:
                        logger.warning(f"Could not compute stats for Z{comp}")
                        break
                    res = self.z_object.resistivity[nz_index, ii, jj]
                    if self.z_object.resistivity_error is not None:
                        res_error = self.z_object.resistivity_error[
                            nz_index, ii, jj
                        ]
                    else:
                        res_error = np.zeros_like(res)
                    phase = self.z_object.phase[nz_index, ii, jj]
                    if self.z_object.phase_error is not None:
                        phase_error = self.z_object.phase_error[
                            nz_index, ii, jj
                        ]
                    else:
                        phase_error = np.zeros_like(phase)

                    if f[0] > f[1]:
                        flip = True
                        f = f[::-1]
                        res = res[::-1]
                        res_error = res_error[::-1]
                        phase = phase[::-1]
                        phase_error = phase_error[::-1]

                    k = 7  # order of the fit
                    # knots, has to be at least to the bounds of f
                    if len(f) < k:
                        k = len(f) - 1

                    t = np.r_[
                        (f[0],) * (k + 1),
                        [min(1, f.mean())],
                        (f[-1],) * (k + 1),
                    ]

                    ### estimate a least squares fit
                    try:
                        ls_res = interpolate.make_lsq_spline(f, res, t, k)
                        ls_phase = interpolate.make_lsq_spline(f, phase, t, k)

                        ### compute a standard deviation between the ls fit and data
                        stat_array[0][f"res_{comp}_fit"] = (
                            res - ls_res(f)
                        ).std()
                        stat_array[0][f"phase_{comp}_fit"] = (
                            phase - ls_phase(f)
                        ).std()
                    except (ValueError, np.linalg.LinAlgError) as error:
                        stat_array[0][f"res_{comp}_fit"] = np.NaN
                        stat_array[0][f"phase_{comp}_fit"] = np.NaN
                        logger.error(f"Z{comp}: {error}")
                    ### taking median of the error is more robust
                    stat_array[0][f"res_{comp}_std"] = np.median(res_error)
                    stat_array[0][f"phase_{comp}_std"] = np.median(phase_error)

                    ### estimate smoothness
                    stat_array[0][f"res_{comp}_corr"] = np.corrcoef(
                        res[0:-1], res[1:]
                    )[0, 1]
                    stat_array[0][f"phase_{comp}_corr"] = np.corrcoef(
                        phase[0:-1], phase[1:]
                    )[0, 1]

                    ### estimate smoothness with difference
                    stat_array[0][f"res_{comp}_diff"] = np.abs(
                        np.median(np.diff(res))
                    )
                    stat_array[0][f"phase_{comp}_diff"] = np.abs(
                        np.median(np.diff(phase))
                    )

                    ### compute tipper
                    if ii == 0 and self.t_object is not None:
                        tcomp = self.t_dict[(0, jj)]
                        t_index = np.nonzero(
                            self.t_object.amplitude[:, 0, jj]
                        )[0]
                        bad_points_t = self.locate_bad_tipper_points(
                            self.t_object.amplitude[:, 0, jj]
                        )
                        stat_array[0][f"bad_points_tipper_{tcomp}"] = max(
                            [1, len(bad_points_t)]
                        )
                        t_index = np.delete(
                            t_index, np.isin(t_index, bad_points_t)
                        )
                        if t_index.size == 0:
                            continue
                        else:
                            tip_f = self.t_object.frequency[t_index]
                            if len(tip_f) < 2:
                                logger.warning(
                                    f"Could not compute stats for T{comp}"
                                )
                                break
                            tmag = self.t_object.amplitude[t_index, 0, jj]
                            if self.t_object.amplitude_error is not None:
                                tmag_error = self.t_object.amplitude_error[
                                    t_index, 0, jj
                                ]
                            else:
                                tmag_error = np.zeros_like(tmag)

                            if flip:
                                tmag = tmag[::-1]
                                tmag_error = tmag_error[::-1]
                                tip_f = tip_f[::-1]

                            tip_t = np.r_[
                                (tip_f[0],) * (k + 1),
                                [min(1, tip_f.mean())],
                                (tip_f[-1],) * (k + 1),
                            ]
                            try:
                                ls_tmag = interpolate.make_lsq_spline(
                                    tip_f, tmag, tip_t, k
                                )
                                stat_array[0][f"tipper_{tcomp}_fit"] = np.std(
                                    tmag - ls_tmag(tip_f)
                                )
                            except (
                                ValueError,
                                np.linalg.LinAlgError,
                            ) as error:
                                stat_array[0][f"tipper_{tcomp}_fit"] = np.NaN
                                logger.error(f"T{tcomp}: {error}")
                            stat_array[0][
                                f"tipper_{tcomp}_std"
                            ] = tmag_error.mean()
                            stat_array[0][
                                f"tipper_{tcomp}_corr"
                            ] = np.corrcoef(tmag[0:-1], tmag[1:])[0, 1]
                            stat_array[0][f"tipper_{tcomp}_diff"] = np.std(
                                np.diff(tmag)
                            ) / abs(np.mean(np.diff(tmag)))

        ### write file
        df = pd.DataFrame(stat_array)
        df = df.replace(0, np.NAN)

        return df

    def estimate_data_quality(self, stat_df):
        """
        Convert the statistical estimates into the rating between 0-5 given
        a certain criteria.

        .. note:: To change the criteria change self.stat_limits

        :param stat_df: Dataframe of the statistics
        :type stat_df: pandas.DataFrame

        :param stat_fn: name of .csv file of statistics
        :type stat_fn: string

        :returns: a dataframe of the converted statistics
        :rtype: pandas.DataFrame

        .. note:: Writes a file to the tf_dir named tf_quality_estimate.csv
        """
        if stat_df is None:
            raise ValueError("No DataFrame to analyze")

        ### make a copy of the data fram to put quality factors in
        qual_df = pd.DataFrame(
            np.zeros(
                stat_df.shape[0],
                dtype=[(key, float) for key in sorted(self.types)],
            ),
            index=stat_df.index,
        )
        for col in qual_df.columns:
            qual_df[col].values[:] = np.NaN

        ### loop over quality factors
        for qkey in self.stat_limits.keys():
            for column in qual_df.columns:
                if qkey in column:
                    for ckey, cvalues in self.stat_limits[qkey].items():
                        qual_df[column][
                            (stat_df[column] > cvalues[0])
                            & (stat_df[column] <= cvalues[1])
                        ] = ckey

        return qual_df

    def summarize_data_quality(
        self,
        quality_df,
        weights={
            "bad": 0.35,
            "corr": 0.2,
            "diff": 0.2,
            "std": 0.2,
            "fit": 0.05,
        },
    ):
        """
        Summarize the data quality into a single number for each station.

        :param quality_df: Dataframe of the quality factors
        :type quality_df: pandas.DataFrame

        :param quality_fn: name of .csv file of quality factors
        :type quality_fn: string

        :returns: a dataframe of the  summarized quality factors
        :rtype: pandas.DataFrame

        .. note:: Writes a file to the tf_dir named tf_quality.csv
        """
        if quality_df is None:
            raise ValueError("No DataFrame to analyze")

        ### compute median value
        ### need to weight things differently
        bad_df = quality_df[
            [col for col in quality_df.columns if "bad" in col]
        ]
        diff_df = quality_df[
            [col for col in quality_df.columns if "diff" in col]
        ]
        fit_df = quality_df[
            [col for col in quality_df.columns if "fit" in col]
        ]
        corr_df = quality_df[
            [col for col in quality_df.columns if "corr" in col]
        ]
        std_df = quality_df[
            [col for col in quality_df.columns if "std" in col]
        ]

        qf_df = np.nansum(
            np.array(
                [
                    weights["bad"] * np.nanmedian(bad_df, axis=1),
                    weights["corr"] * np.nanmedian(corr_df, axis=1),
                    weights["diff"] * np.nanmedian(diff_df, axis=1),
                    weights["std"] * np.nanmedian(std_df, axis=1),
                    weights["fit"] * np.nanmedian(fit_df, axis=1),
                ]
            )
        )

        qf_df = qf_df.round()
        return qf_df

    def estimate_quality_factor(
        self,
        weights={
            "bad": 0.35,
            "corr": 0.2,
            "diff": 0.2,
            "std": 0.2,
            "fit": 0.05,
        },
    ):
        """
        Convenience function doing all the steps to estimate quality factor
        """

        qualities_df = self.estimate_data_quality(self.compute_statistics())
        return self.summarize_data_quality(
            quality_df=qualities_df, weights=weights
        )


# =============================================================================
# Test
# =============================================================================
# edi_dir = r"c:\Users\jpeacock\Documents\edi_folders\imush_edi"
# q = EMTFStats()
# stat_df = q.compute_statistics(edi_dir)
# q_df = q.estimate_data_quality(stat_df=stat_df)
# s_df = q.summarize_data_quality(q_df)
