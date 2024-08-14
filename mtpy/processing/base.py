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
    """
    Base processing class contains path to various files
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config = None

    @property
    def mth5_list(self):
        """

        :return: list of mth5 to get run summary from
        :rtype: list

        """
        mth5_list = []
        if self.has_local_mth5():
            mth5_list.append(self.local_mth5_path)
        if self.has_remote_mth5():
            mth5_list.append(self.remote_mth5_path)

        if len(mth5_list) == 0:
            raise ValueError("No MTH5 file paths set. Return list is empty.")
        return mth5_list

    def get_run_summary(self):
        """
        Get the RunSummary object

        :param local: DESCRIPTION, defaults to True
        :type local: TYPE, optional
        :param remote: DESCRIPTION, defaults to True
        :type remote: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        run_summary = RunSummary()
        run_summary.from_mth5s(self.mth5_list)
        return run_summary

    def has_kernel_dataset(self):
        """
        test if has kernel dataset

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.kernel_dataset.df is not None:
            return True
        return False
