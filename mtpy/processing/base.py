# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:13:07 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from mth5.processing.kernel_dataset import KernelDataset
from mth5.processing.run_summary import RunSummary


# =============================================================================


class BaseProcessing(KernelDataset):
    """
    Base processing class containing paths to various files.

    Attributes
    ----------
    config : object or None
        Processing configuration object.
    run_summary : RunSummary or None
        Run summary object containing metadata about processing runs.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize BaseProcessing.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to KernelDataset.

        """
        super().__init__(**kwargs)

        self.config = None
        self.run_summary = None

    @property
    def mth5_list(self) -> list[str]:
        """
        Get list of MTH5 file paths.

        Returns
        -------
        list[str]
            List of MTH5 file paths to get run summary from.

        Raises
        ------
        ValueError
            If no MTH5 file paths are set.

        """
        mth5_list = []
        if self.has_local_mth5():
            mth5_list.append(self.local_mth5_path)
        if self.has_remote_mth5():
            if self.local_mth5_path != self.remote_mth5_path:
                mth5_list.append(self.remote_mth5_path)

        if len(mth5_list) == 0:
            raise ValueError("No MTH5 file paths set. Return list is empty.")
        return mth5_list

    @property
    def run_summary(self) -> RunSummary | None:
        """
        Get the run summary object.

        Returns
        -------
        RunSummary or None
            Run summary containing metadata about processing runs.

        """
        return self._run_summary

    @run_summary.setter
    def run_summary(self, value: RunSummary | None) -> None:
        """
        Set the run summary object.

        Parameters
        ----------
        value : RunSummary or None
            Run summary to set, or None to clear.

        """
        if value is None:
            self._run_summary = None
        else:
            self._run_summary = RunSummary(df=value.df)

    def get_run_summary(self) -> RunSummary:
        """
        Get the RunSummary object from MTH5 files.

        Returns
        -------
        RunSummary
            Run summary object created from MTH5 file list.

        """
        run_summary = RunSummary()
        run_summary.from_mth5s(self.mth5_list)
        return run_summary

    def has_run_summary(self) -> bool:
        """
        Check if run summary is set.

        Returns
        -------
        bool
            True if run_summary is not None, False otherwise.

        """
        if self.run_summary is None:
            return False
        return True

    def has_kernel_dataset(self) -> bool:
        """
        Check if kernel dataset is set.

        Returns
        -------
        bool
            True if df attribute is not None, False otherwise.

        """
        if self.df is not None:
            return True
        return False
