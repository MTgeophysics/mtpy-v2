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
from mtpy.processing.kernel_dataset import KernelDataset
from mtpy.processing.base import BaseProcessing

warnings.filterwarnings("ignore")
# =============================================================================


class AuroraProcessing(BaseProcessing):
    """Convenience class to process with Aurora


    .. code-block:: python

        from mtpy.processing.aurora.process_aurora import AuroraProcessing

        ap = AuroraProcessing()

        # set local station and path to MTH5
        ap.local_station_id = "mt01"
        ap.local_mth5_path = "/path/to/local_mth5.h5"

        # set remote station and path to MTH5
        ap.remote_station_id = "rr01"
        ap.remote_mth5_path = "/path/to/remote_mth5.h5"

        # process single sample rate
        tf_obj = ap.process_single_sample_rate(sample_rate=1)

        # process multiple sample rates, merge them all together and
        # save transfer functions to the local MTH5
        tf_processed_dict = ap.process(
            sample_rates=[4096, 1],
            merge=True,
            save_to_mth5=True
            ).
    """

    def __init__(self, **kwargs):
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

        self._processing_dict_keys = ["config", "kernel_dataset"]

        super().__init__(**kwargs)

    def _get_merge_df(self):
        """A datafram containing the periods to use for each sample rate."""

        return pd.DataFrame(
            {
                "sample_rate": list(self.merge_dictionary.keys()),
                "period_min": [
                    mgd["period_min"] for mgd in self.merge_dictionary.values()
                ],
                "period_max": [
                    mgd["period_max"] for mgd in self.merge_dictionary.values()
                ],
            }
        )

    def create_config(
        self, kernel_dataset=None, decimation_kwargs={}, **kwargs
    ):
        """Decimation kwargs can include information about window,.
        :return: DESCRIPTION.
        :rtype: aurora.config
        """
        if kernel_dataset is None:
            if self.has_kernel_dataset():
                cc = ConfigCreator()
                config = cc.create_from_kernel_dataset(self, **kwargs)
                if self.sample_rate > 1000:
                    decimation_kwargs.update(
                        self.default_window_parameters["high"]
                    )
                else:
                    decimation_kwargs.update(
                        self.default_window_parameters["low"]
                    )
                self._set_decimation_level_parameters(
                    config, **decimation_kwargs
                )
                return config
            else:
                raise ValueError(
                    "Cannot make config because KernelDataset has not been set yet."
                )
        else:
            cc = ConfigCreator()
            config = cc.create_from_kernel_dataset(kernel_dataset, **kwargs)
            if kernel_dataset.sample_rate > 1000:
                decimation_kwargs.update(
                    self.default_window_parameters["high"]
                )
            else:
                decimation_kwargs.update(self.default_window_parameters["low"])
            self._set_decimation_level_parameters(config, **decimation_kwargs)
            return config

    def _set_decimation_level_parameters(self, config, **kwargs):
        """Set decimation level parameters.
        :param config: DESCRIPTION.
        :type config: TYPE
        :param **kwargs: DESCRIPTION.
        :type **kwargs: TYPE
        :return: DESCRIPTION.
        :rtype: TYPE
        """

        for decimation in config.decimations:
            for key, value in kwargs.items():
                decimation.set_attr_from_name(key, value)

    def _initialize_kernel_dataset(self, sample_rate=None):
        """Make an initial kernel dataset."""

        if not self.has_run_summary():
            self.run_summary = self.get_run_summary()

        if sample_rate is not None:
            run_summary = self.run_summary.set_sample_rate(sample_rate)
        else:
            # have to use a single sample rate otherwise an error is thrown.
            run_summary = self.run_summary.set_sample_rate(
                self.run_summary.df.sample_rate.unique()[0]
            )

        self.from_run_summary(run_summary)

    def create_kernel_dataset(
        self,
        run_summary=None,
        local_station_id=None,
        remote_station_id=None,
        sample_rate=None,
    ):
        """This can be a stane alone method and return a kds, or create in place
        Build KernelDataset
        """

        if run_summary is None:
            if not self.has_run_summary():
                run_summary = self.get_run_summary()
            else:
                run_summary = self.run_summary

        if sample_rate is not None:
            run_summary = run_summary.set_sample_rate(sample_rate)

        self.from_run_summary(
            run_summary,
            local_station_id=local_station_id,
            remote_station_id=remote_station_id,
            sample_rate=sample_rate,
        )
        return self.clone()

    def process_single_sample_rate(
        self, sample_rate, config=None, kernel_dataset=None
    ):
        """
        Process a single sample rate

        :param sample_rate: sample rate of time series data
        :type sample_rate: float
        :param config: configuration file, defaults to None
        :type config: aurora.config, optional
        :param kernel_dataset: Kerenel dataset to define what data to process,
          defaults to None
        :type kernel_dataset: mtpy.processing.KernelDataset, optional
        :return: transfer function
        :rtype: mtpy.MT

        """

        if kernel_dataset is None:
            kernel_dataset = self.create_kernel_dataset(
                local_station_id=self.local_station_id,
                remote_station_id=self.remote_station_id,
                sample_rate=sample_rate,
            )
        if config is None:
            config = self.create_config(kernel_dataset=kernel_dataset)

        try:
            tf_obj = process_mth5(config, kernel_dataset)
        except Exception as error:
            close_open_files()
            logger.exception(error)
            logger.error(f"Skipping sample_rate {sample_rate}")
            return

        tf_obj.tf_id = self.processing_id

        # copy to an MT object
        mt_obj = MT(survey_metadata=tf_obj.survey_metadata)
        mt_obj.channel_nomenclature = tf_obj.channel_nomenclature
        mt_obj._transfer_function = tf_obj._transfer_function

        return mt_obj

    def process(
        self,
        sample_rates=None,
        processing_dict=None,
        merge=True,
        save_to_mth5=True,
    ):
        """
        Need to either provide a list of sample rates to process or
        a processing dictionary.

        If you provide just the sample rates, then at each sample rate a
        KernelDataset will be created as well as a subsequent config object
        which are then used to process the data.

        If processing_dict is set then the processing will loop through the
        dictionary and use the provided config and kernel datasets.

        The processing dict has the following form

        .. code_block:: python

            processing_dict = {sample_rate: {
                "config": config object,
                "kernel_dataset": KernelDataset object,
                }

        If merge is True then all runs for all sample rates are combined into
        a single function according to merge_dict.

        If `save_to_mth5` is True then the transfer functions are saved to
        the local MTH5.


        :param sample_rates: list of sample rates to process, defaults to None
        :type sample_rates: float or list, optional
        :param processing_dict: processing dictionary as described above,
         defaults to None
        :type processing_dict: dict, optional
        :param merge: [ True | False ] True merges all sample rates into a
         single transfer function according to the merge_dict, defaults to True
        :type merge: bool, optional
        :param save_to_mth5: [ True | False ] save transfer functions to the
         local MTH5, defaults to True
        :type save_to_mth5: TYPE, optional
        :raises ValueError: If neither sample rates nor processing dict are
         provided
        :raises TypeError: If the provided processing dictionary is not
         the correct format
        :return: dictionary of each sample rate processed in the form of
         {sample_rate: {'processed': bool, 'tf': MT}}
        :rtype: dict

        """

        if sample_rates is None and processing_dict is None:
            raise ValueError(
                "Must set either sample rates or processing_dict."
            )

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
                mt_obj = self.process_single_sample_rate(sr)
                if mt_obj is not None:
                    tf_processed[sr]["processed"] = True
                    tf_processed[sr]["tf"] = mt_obj
        else:
            self._validate_processing_dict(processing_dict)

            tf_processed = dict(
                [
                    (sr, {"processed": False, "tf": None})
                    for sr in processing_dict.keys()
                ]
            )
            for key, pdict in processing_dict.items():
                mt_obj = self.process_single_sample_rate(
                    key,
                    config=pdict["config"],
                    kernel_dataset=pdict["kernel_dataset"],
                )
                if mt_obj is not None:
                    tf_processed[key][
                        "processed"
                    ] = mt_obj.has_transfer_function()
                    tf_processed[key]["tf"] = mt_obj

        processed = self._validate_tf_processed_dict(tf_processed)
        if len(processed.keys()) > 1:
            if merge:
                ### merge transfer functions according to merge dict
                combined_tf = self.merge_transfer_functions(processed)
                combined_tf_id = self.local_station_id
                if self.remote_station_id:
                    combined_tf_id += f"_rr_{self.remote_station_id}"
                combined_tf_id += "_combined"
                combined_tf.tf_id = combined_tf_id
                processed["combined"] = {"processed": True, "tf": combined_tf}

        if save_to_mth5:
            ### add tf to local MTH5
            self._add_tf_to_local_mth5(processed)

        return processed

    def _validate_config(self, config):
        """Validate config."""
        if not isinstance(config, Processing):
            raise TypeError(
                "Config must be a aurora.config.metadata.Processing object. "
                f"Got type {type(config)}"
            )

    def _validate_kernel_dataset(self, kernel_dataset):
        """Validate kernel dataset."""
        if not isinstance(kernel_dataset, KernelDataset):
            raise TypeError(
                "Config must be a mtpy.processing.KernelDataset object. "
                f"Got type {type(kernel_dataset)}"
            )

    def _validate_processing_dict(self, processing_dict):
        """Validate the processing dict to make sure it is in the correct format.

        :param processing_dict: processing dictionary
        :type processing_dict: dict
        """
        error_msg = "Format is {sample_rate: {'config': config object, "
        "'kernel_dataset': KernelDataset object}"
        if not isinstance(processing_dict, dict):
            raise TypeError(
                "Input processing_dict must be a dictionary. "
                f"Got type {type(processing_dict)}."
            )

        for key, pdict in processing_dict.items():
            if not isinstance(pdict, dict):
                raise TypeError(
                    "Input processing_dict must be a dictionary. "
                    f"Got type {type(pdict)}. " + error_msg
                )
            if sorted(self._processing_dict_keys) != sorted(pdict.keys()):
                raise KeyError(
                    "Processing dict can only have keys "
                    f"{self._processing_dict_keys}. " + error_msg
                )

            self._validate_config(pdict["config"])
            self._validate_kernel_dataset(pdict["kernel_dataset"])

    def _validate_tf_processed_dict(self, tf_dict):
        """Pick out processed transfer functions from a given processed dict,
        which may have some sample rates that did not process for whatever
        reason.

        :param tf_dict: dictionary of processed transfer functions.
        :type tf_dict: dict
        :return: dicionary of trasnfer functions for which processed=True
        :rtype: dict
        """

        new_dict = {}
        for key, p_dict in tf_dict.items():
            if p_dict["processed"]:
                new_dict[key] = p_dict
            else:
                logger.warning(
                    f"Sample rate {key} was not processed correctly. Check log."
                )

        if new_dict == {}:
            raise ValueError("No Transfer Functions were processed.")
        return new_dict

    def _add_tf_to_local_mth5(self, tf_dict):
        """Add transfer function to MTH5.

        :param tf_dict: dictionary of transfer functions
        :type tf_dict: dict
        """

        with MTH5() as m:
            m.open_mth5(self.local_mth5_path)
            for p_dict in tf_dict.values():
                m.add_transfer_function(p_dict["tf"])

    def _get_merge_tf_list(self, tf_dict):
        """Merge transfer functions according to merge dict.

        :param tf_dict: dictionary of transfer functions.
        :type tf_dict: dict
        :return: list of transfer functions with appropriate min/max period
         to merge
        :rtype: list
        """

        merge_df = self._get_merge_df()
        merge_list = []
        for key, pdict in self._validate_tf_processed_dict(tf_dict).items():
            if key in merge_df.sample_rate.tolist():
                row = merge_df.loc[merge_df.sample_rate == key]
                period_min = row.period_min.iloc[0]
                period_max = row.period_max.iloc[0]
            else:
                period_min = pdict["tf"].period.min()
                period_max = pdict["tf"].period.max()

            merge_list.append(
                {
                    "tf": pdict["tf"],
                    "period_min": period_min,
                    "period_max": period_max,
                }
            )

        return merge_list

    def merge_transfer_functions(self, tf_dict):
        """Merge transfer functions according to AuroraProcessing.merge_dict

        :param tf_dict: dictionary of transfer functions
        :type tf_dict: dict
        :return: merged transfer function.
        :rtype: mtpy.MT
        """

        merge_list = self._get_merge_tf_list(tf_dict)

        if len(merge_list) > 1:
            return merge_list[0]["tf"].merge(
                merge_list[1:], period_max=merge_list[0]["period_max"]
            )
        else:
            return merge_list[0]["tf"]
