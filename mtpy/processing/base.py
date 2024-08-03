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
from mtpy.processing.kernel_dataset import KernelDataset

# =============================================================================


class BaseProcessing:
    """
    Base processing class contains path to various files
    """

    def __init__(self, **kwargs):
        self.local_station_id = None
        self.remote_station_id = None
        self.local_mth5_path = None
        self.remote_mth5_path = None
        self.config = None
        self.run_summary = RunSummary()
        self.kernel_dataset = KernelDataset()

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def local_station_id(self):
        return self._local_station_id

    @local_station_id.setter
    def local_station_id(self, value):
        if value is None:
            self._local_station_id = None
        else:
            try:
                self._local_station_id = str(value)
            except ValueError:
                raise ValueError(
                    f"Bad type {type(value)}. "
                    "Cannot convert local_station_id value to string."
                )

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
    def remote_station_id(self):
        return self._remote_station_id

    @remote_station_id.setter
    def remote_station_id(self, value):
        if value is None:
            self._remote_station_id = None
        else:
            try:
                self._remote_station_id = str(value)
            except ValueError:
                raise ValueError(
                    f"Bad type {type(value)}. "
                    "Cannot convert remote_station_id value to string."
                )

    @property
    def remote_mth5_path(self):
        return self._remote_mth5_path

    @remote_mth5_path.setter
    def remote_mth5_path(self, value):
        self._remote_mth5_path = self.set_path(value)

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
        else:
            raise IOError("Local MTH5 path must be set with a valid file path.")
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

    def get_run_summary(self, inplace: bool = True):
        """
        Get the RunSummary object

        :param local: DESCRIPTION, defaults to True
        :type local: TYPE, optional
        :param remote: DESCRIPTION, defaults to True
        :type remote: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if inplace:
            self.run_summary.from_mth5s(self.mth5_list)
        else:
            run_summary = RunSummary()
            run_summary.from_mth5s(self.mth5_list)
            return run_summary

    def has_run_summary(self):
        """
        test if has run summary

        :return: DESCRIPTION
        :rtype: TYPE

        """
        if self.run_summary.df is not None:
            return True
        return False

    def get_kernel_dataset(self, inplace: bool = True):
        """

        :param inplace: DESCRIPTION, defaults to True
        :type inplace: bool, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if not self.has_run_summary():
            self.get_run_summary()
        if inplace:

            self.kernel_dataset.from_run_summary(
                self.run_summary, self.local_station_id, self.remote_station_id
            )
        else:
            kds = KernelDataset()
            kds.from_run_summary(
                self.run_summary, self.local_station_id, self.remote_station_id
            )
            return kds

    def has_kernel_dataset(self):
        """
        test if has kernel dataset

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.kernel_dataset.df is not None:
            return True
        return False
