# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:11:42 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import warnings
from pathlib import Path
from loguru import logger
import pandas as pd

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5

from mth5.helpers import close_open_files
from mth5.mth5 import MTH5

from mt_metadata.utils.mttime import MTime

from mtpy import MT
from mtpy.processing.base import BaseProcessing

warnings.filterwarnings("ignore")
# =============================================================================


class AuroraProcessing(BaseProcessing):
    """
    convenience class to process with Aurora

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_config(self):
        pass

    def build_kernel_dataset(self):
        pass

    def process(self):
        pass
