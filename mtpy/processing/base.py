# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:13:07 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

# =============================================================================


class BaseProcessing:
    """
    Base processing class contains path to various files
    """

    def __init__(self, **kwargs):
        self.local_mth5_path = None
        self.remote_mth5_path = None

    @property
    def local_mth5_path(self):
        return self._local_mth5_path

    @local_mth5_path.setter
    def local_mth5_path(self, value):
        self._local_mth5_path = self.set_path(value)

    @property
    def remote_mth5_path(self):
        return self._local_mth5_path

    @local_mth5_path.setter
    def remote_mth5_path(self, value):
        self._local_mth5_path = self.set_path(value)

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
