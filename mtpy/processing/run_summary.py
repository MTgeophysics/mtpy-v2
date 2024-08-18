"""

This module contains the RunSummary class.

This is a helper class that summarizes the Runs in an mth5.

TODO: This class and methods could be replaced by methods in MTH5.

Functionality of RunSummary()
1. User can get a list of local_station options, which correspond to unique pairs
of values: (survey,  station)

2. User can see all possible ways of processing the data:
- one list per (survey,  station) pair in the run_summary

Some of the following functionalities may end up in KernelDataset:
3. User can select local_station
-this can trigger a reduction of runs to only those that are from the local staion
and simultaneous runs at other stations
4. Given a local station, a list of possible reference stations can be generated
5. Given a remote reference station, a list of all relevent runs, truncated to
maximize coverage of the local station runs is generated
6. Given such a "restricted run list", runs can be dropped
7. Time interval endpoints can be changed


Development Notes:
    TODO: consider adding methods:
     - drop_runs_shorter_than": removes short runs from summary
     - fill_gaps_by_time_interval": allows runs to be merged if gaps between 
       are short
     - fill_gaps_by_run_names": allows runs to be merged if gaps between are 
       short
    TODO: Consider whether this should return a copy or modify in-place when
    querying the df.

"""

# =============================================================================
# Imports
# =============================================================================
import copy
from typing import Optional, Union

import pandas as pd
from loguru import logger

from mtpy.processing import RUN_SUMMARY_COLUMNS, MINI_SUMMARY_COLUMNS

import mth5
from mth5.utils.helpers import initialize_mth5

# =============================================================================


class RunSummary:
    """
    Class to contain a run-summary table from one or more mth5s.

    WIP: For the full MMT case this may need modification to a channel based
    summary.


    """

    def __init__(
        self,
        input_dict: Optional[Union[dict, None]] = None,
        df: Optional[Union[pd.DataFrame, None]] = None,
    ):
        """
        Constructor

        Parameters
        ----------
        kwargs
        """
        self.column_dtypes = [str, str, pd.Timestamp, pd.Timestamp]
        self._input_dict = input_dict
        self.df = df
        self._mini_summary_columns = MINI_SUMMARY_COLUMNS

    def __str__(self):
        return str(self.mini_summary.head(-1))

    def __repr__(self):
        return self.__str__()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        """
        Make sure the data frame is set properly with proper column names

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if value is None:
            self._df = None
            return

        if not isinstance(value, pd.DataFrame):
            msg = f"Need to set df with a Pandas.DataFrame not type({type(value)})"
            logger.error(msg)

            raise TypeError(msg)

        need_columns = []
        for col in RUN_SUMMARY_COLUMNS:
            if not col in value.columns:
                need_columns.append(col)
        if need_columns:
            msg = f"DataFrame needs columns {', '.join(need_columns)}"
            logger.error(msg)
            raise ValueError(msg)
        self._df = value

    def clone(self):
        """
        2022-10-20:
        Cloning may be causing issues with extra instances of open h5 files ...

        """
        return copy.deepcopy(self)

    def from_mth5s(self, mth5_list) -> list:
        """Iterates over mth5s in list and creates one big dataframe
        summarizing the runs
        """
        run_summary_df = extract_run_summaries_from_mth5s(mth5_list)
        self.df = run_summary_df

    def _warn_no_data_runs(self):
        if False in self.df.has_data.values:
            for row in self.df[self.df.has_data == False].itertuples():
                logger.warning(
                    f"Found no data run in row {row.Index}: "
                    f"survey: {row.survey}, station: {row.station}, run: {row.run}"
                )
            logger.info("To drop no data runs use `drop_no_data_rows`")

    @property
    def mini_summary(self) -> pd.DataFrame:
        """shows the dataframe with only a few columns for readbility"""
        return self.df[self._mini_summary_columns]

    @property
    def print_mini_summary(self) -> str:
        """Calls minisummary through logger so it is formatted."""
        logger.info(self.mini_summary)

    def drop_no_data_rows(self) -> bool:
        """
        Drops rows marked `has_data` = False and resets the index of self.df

        """
        self.df = self.df[self.df.has_data]
        self.df.reset_index(drop=True, inplace=True)

    def set_sample_rate(self, sample_rate: float, inplace: bool = False):
        """
        Set the sample rate so that the run summary represents all runs for
        a single sample rate.

        :param inplace: DESCRIPTION, defaults to True
        :type inplace: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if sample_rate not in self.df.sample_rate.values:
            msg = (
                f"Sample rate {sample_rate} is not in RunSummary. Unique "
                f"values are {self.df.sample_rate.unique()}"
            )
            logger.error(msg)
            raise ValueError(msg)
        if inplace:
            self.df = self.df[self.df.sample_rate == sample_rate]
        else:
            new_rs = self.clone()
            new_rs.df = new_rs.df[new_rs.df.sample_rate == sample_rate]
            return new_rs


def extract_run_summary_from_mth5(
    mth5_obj, summary_type: Optional[str] = "run"
):
    """
    Given a single mth5 object, get the channel_summary and compress it to a
    run_summary.

    Development Notes:
    TODO: Move this into MTH5 or replace with MTH5 built-in run_summary method.

    Parameters
    ----------
    mth5_obj: mth5.mth5.MTH5
        The initialized mth5 object that will be interrogated
    summary_type: str
        One of ["run", "channel"].  Returns a run summary or a channel summary

    Returns
    -------
    out_df: pd.Dataframe
        Table summarizing the available runs in the input mth5_obj
    """

    if summary_type == "run":
        out_df = mth5_obj.run_summary
    else:
        out_df = mth5_obj.channel_summary.to_dataframe()
    out_df["mth5_path"] = str(mth5_obj.filename)
    return out_df


def extract_run_summaries_from_mth5s(
    mth5_list, summary_type="run", deduplicate=True
):
    """
    Given a list of mth5's, iterate over them, extracting run_summaries and
    merging into one big table.

    Development Notes:
    ToDo: Move this method into mth5? or mth5_helpers?
    ToDo: Make this a class so that the __repr__ is a nice visual representation
    of the
    df, like what channel summary does in mth5
    - 2022-05-28 Modified to allow this method to accept mth5 objects as well
    as the
    already supported types of pathlib.Path or str


    In order to drop duplicates I used the solution here:
    https://stackoverflow.com/questions/43855462/pandas-drop-duplicates-method-not-working-on-dataframe-containing-lists

    Parameters
    ----------
    mth5_paths: list
        paths or strings that point to mth5s
    summary_type: string
        one of ["channel", "run"]
        "channel" returns concatenated channel summary,
        "run" returns concatenated run summary,
    deduplicate: bool
        Default is True, deduplicates the summary_df

    Returns
    -------
    super_summary: pd.DataFrame
        Given a list of mth5s, a dataframe of all available runs

    """
    dfs = len(mth5_list) * [None]

    for i, mth5_elt in enumerate(mth5_list):
        if isinstance(mth5_elt, mth5.mth5.MTH5):
            mth5_obj = mth5_elt
        else:  # mth5_elt is a path or a string
            mth5_obj = initialize_mth5(mth5_elt, mode="a")

        df = extract_run_summary_from_mth5(mth5_obj, summary_type=summary_type)

        # close it back up if you opened it
        if not isinstance(mth5_elt, mth5.mth5.MTH5):
            mth5_obj.close_mth5()
        dfs[i] = df

    # merge all summaries into a super_summary
    super_summary = pd.concat(dfs)
    super_summary.reset_index(drop=True, inplace=True)

    # drop rows that correspond to TFs:
    run_rows = super_summary.sample_rate != 0
    super_summary = super_summary[run_rows]
    super_summary.reset_index(drop=True, inplace=True)

    if deduplicate:
        keep_indices = super_summary.astype(str).drop_duplicates().index
        super_summary = super_summary.loc[keep_indices]
        super_summary.reset_index(drop=True, inplace=True)
    return super_summary
