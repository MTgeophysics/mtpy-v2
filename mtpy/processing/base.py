# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:13:07 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

from mtpy.processing.run_summary import RunSummary
from mtpy.processing.kernel_dataset import KernelDataset

# =============================================================================


class BaseProcessing(KernelDataset):
    """Base processing class contains path to various files."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config = None
        self.run_summary = None

    @property
    def mth5_list(self):
        """Mth5 list.
        :return: List of mth5 to get run summary from.
        :rtype: list
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
    def run_summary(self):
        """Run summary object.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        return self._run_summary

    @run_summary.setter
    def run_summary(self, value):
        """Set run summary.
        :param value: DESCRIPTION.
        :type value: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if value is None:
            self._run_summary = None
        else:
            self._run_summary = RunSummary(df=value.df)

    def get_run_summary(self):
        """Get the RunSummary object.
        :param local: DESCRIPTION, defaults to True.
        :type local: TYPE, optional
        :param remote: DESCRIPTION, defaults to True.
        :type remote: TYPE, optional
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        run_summary = RunSummary()
        run_summary.from_mth5s(self.mth5_list)
        return run_summary

    def has_run_summary(self):
        """Check to see if run summary is set.
        :return: DESCRIPTION.
        :rtype: TYPE
        """
        if self.run_summary is None:
            return False
        return True

    def has_kernel_dataset(self):
        """Test if has kernel dataset.
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        if self.df is not None:
            return True
        return False
