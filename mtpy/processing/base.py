# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:13:07 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mtpy.processing.run_summary import RunSummary

# =============================================================================


class BaseProcessing:
    """
    Base processing class contains path to various files
    """

    def __init__(self, **kwargs):
        self.local_mth5_path = None
        self.remote_mth5_path = None
        self.config = None

    @property
    def local_mth5_path(self):
        return self._local_mth5_path

    @local_mth5_path.setter
    def local_mth5_path(self, value):
        self._local_mth5_path = self.set_path(value)

    def has_local_mth5(self):
        """test if local mth5 exists"""
        if self.local_mth5_path is None:
            return False
        else:
            return self.local_mth5_path.exists()

    @property
    def remote_mth5_path(self):
        return self._local_mth5_path

    @local_mth5_path.setter
    def remote_mth5_path(self, value):
        self._local_mth5_path = self.set_path(value)

    def has_remote_mth5(self):
        """test if remote mth5 exists"""
        if self.remote_mth5_path is None:
            return False
        else:
            return self.remote_mth5_path.exists()

    @property
    def mth5_list(self):
        """
        list of mth5's as [local, remote]

        :return: DESCRIPTION
        :rtype: TYPE

        """
        mth5_list = []
        if self.has_local_mth5():
            mth5_list.append(self.local_mth5_path)
            if self.has_remote_mth5():
                if self.local_mth5_path != self.remote_mth5_path:
                    mth5_list.append(self.remote_mth5_path)
        return mth5_list

    @classmethod
    def set_path(self, value):
        return_path = None
        if value is not None:
            if isinstance(value, (str, Path)):
                return_path = Path(value)
                if not return_path.exists():
                    raise IOError(f"Cannot find file: {return_path}")
            else:
                raise ValueError(f"Cannot convert type{type(value)} to Path")

        return return_path

    def get_run_summary(self, local=True, remote=True):
        """
        Get the RunSummary object

        :param local: DESCRIPTION, defaults to True
        :type local: TYPE, optional
        :param remote: DESCRIPTION, defaults to True
        :type remote: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not self.has_local_mth5():
            raise IOError("Local MTH5 path must be set with a valid file path.")
        run_summary = RunSummary()
        mth
