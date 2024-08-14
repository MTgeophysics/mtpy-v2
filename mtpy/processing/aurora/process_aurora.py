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
import numpy as np

from aurora.config.config_creator import ConfigCreator
from aurora.pipelines.process_mth5 import process_mth5
from aurora.config.metadata import Processing

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
        self.merge_dictionary = {
            1: {"period_min": 4, "period_max": 30000},
            4: {"period_min": 1, "period_max": 30000},
            50: {"period_min": 15, "period_max": 10000},
            150: {"period_min": 30, "period_max": 3000},
            256: {"period_min": 64, "period_max": 100},
            1024: {"period_min": 1.0 / 256, "period_max": 1.0 / 2.6},
            4096: {"period_min": 1.0 / 1024, "period_max": 1.0 / 26},
            24000: {"period_min": 1.0 / 6000, "period_max": 1.0 / 187.5},
        }

        self.default_window_parameters = {
            "high": {
                "window.overlap": 256,
                "window.num_samples": 1024,
                "window.type": "dpss",
                "window.additional_args": {"alpha": 2.5},
            },
            "low": {
                "window.overlap": 64,
                "window.num_samples": 128,
                "window.type": "dpss",
                "window.additional_args": {"alpha": 2.5},
            },
        }

    def create_config(self, decimation_kwargs={}, **kwargs):
        """

        decimation kwargs can include information about window,

        :return: DESCRIPTION
        :rtype: aurora.config

        """
        if self.has_kernel_dataset():
            cc = ConfigCreator()
            config = cc.create_from_kernel_dataset(self, **kwargs)
            if self.sample_rate > 1000:
                decimation_kwargs.update(
                    self.default_window_parameters["high"]
                )
            else:
                decimation_kwargs.update(self.default_window_parameters["low"])
            self._set_decimation_level_parameters(config, **decimation_kwargs)
            return config
        else:
            raise ValueError(
                "Cannot make config because KernelDataset has not been set yet."
            )

    def _set_decimation_level_parameters(self, config, **kwargs):
        """
        set decimation level parameters

        :param config: DESCRIPTION
        :type config: TYPE
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        for decimation in config.decimations:
            for key, value in kwargs.items():
                decimation.set_attr_from_name(key, value)

    def build_kernel_dataset(
        self,
        run_summary=None,
        local_station_id=None,
        remote_station_id=None,
        sample_rate=None,
    ):
        """
        Build KernelDataset
        """

        if run_summary is None:
            if not self.has_run_summary():
                self.run_summary = self.get_run_summary()

        if sample_rate is not None:
            run_summary = self.run_summary.set_sample_rate(sample_rate)

        self.from_run_summary(run_summary, local_station_id, remote_station_id)

    def process_single_sample_rate(
        self, sample_rate, config=None, kernel_dataset=None
    ):
        """
        process data

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if kernel_dataset is None:
            if not self.has_run_summary():
                self.run_summary = self.get_run_summary()
            run_summary = self.run_summary.set_sample_rate(sample_rate)
            self.from_run_summary(run_summary)
            kernel_dataset = self.clone()
        if config is None:
            config = self.create_config()

        try:
            tf_obj = process_mth5(config, kernel_dataset)
        except Exception as error:
            close_open_files()
            logger.exception(error)
            logger.error(f"Skipping sample_rate {sample_rate}")
            return None

        mt_obj = MT(survey_metadata=tf_obj.survey_metadata)
        mt_obj._transfer_function = tf_obj._transfer_function

        return mt_obj

    def process(
        self, sample_rates, processing_dict=None, merge=True, save_to_mth5=True
    ):
        """
        If you provide just the sample rates, then at each sample rate a
        KernelDataset will be created as well as a subsequent config object
        which are then used to process the data.

        If processing_dict is set then the processing will loop through the
        dictionary and use the provided config and kernel datasets.

        process all runs for all sample rates and the combine the transfer
        function according to merge_dict.

        Need to figure out a way to adjust the config per sample rate.  Maybe
        A dictionary of configs keyed by sample rate.

        processing_dict = {sample_rate: {
            "config": config object,
            "kernel_dataset": KernelDataset object,
            }

        :param config: dictionary of configuration keyed by sample rates.
        :type config: dictionary
        :param merge: DESCRIPTION, defaults to True
        :type merge: TYPE, optional
        :param save_to_mth5: DESCRIPTION, defaults to True
        :type save_to_mth5: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if processing_dict is None:
            if isinstance(sample_rates, (int, float)):
                sample_rates = [sample_rates]
            elif isinstance(sample_rates, (list, tuple, np.ndarray)):
                sample_rates = list(sample_rates)
            else:
                raise TypeError(
                    "Sample rates are incorrect type. Expected an int or "
                    f"list not {type(sample_rates)}"
                )

            tf_processed = dict(
                [(sr, {"processed": False, "tf": None}) for sr in sample_rates]
            )

            for sr in sample_rates:
                mt_obj = self._process_single_sample_rate(sr)
                if mt_obj is not None:
                    tf_processed[sr]["processed"] = True
                    tf_processed[sr]["tf"] = mt_obj

            if merge:
                ### merge transfer functions according to merge dict
                pass

    def merge_transfer_functions(self, tf_dict):
        """
        merge transfer functions according to merge dict

        :param tf_dict: DESCRIPTION
        :type tf_dict: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass
