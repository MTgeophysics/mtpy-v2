# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:11:42 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from aurora.config.config_creator import ConfigCreator
from aurora.config.metadata import Processing
from aurora.pipelines.process_mth5 import process_mth5
from loguru import logger
from mt_metadata.features import StridingWindowCoherence
from mt_metadata.features.weights import (
    ChannelWeightSpec,
    FeatureWeightSpec,
    TaperMonotonicWeightKernel,
)
from mth5.helpers import close_open_files
from mth5.mth5 import MTH5
from mth5.processing.kernel_dataset import KernelDataset

from mtpy import MT
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

    def __init__(self, **kwargs) -> None:
        """
        Initialize AuroraProcessing with default merge and window parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to BaseProcessing.

        """
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
                "stft.window.overlap": 256,
                "stft.window.num_samples": 1024,
                "stft.window.type": "dpss",
                "stft.window.additional_args": {"alpha": 2.5},
            },
            "low": {
                "stft.window.overlap": 64,
                "stft.window.num_samples": 128,
                "stft.window.type": "dpss",
                "stft.window.additional_args": {"alpha": 2.5},
            },
        }

        self._processing_dict_keys = ["config", "kernel_dataset"]

        super().__init__(**kwargs)

    def _get_merge_df(self) -> pd.DataFrame:
        """
        Get a DataFrame containing the periods to use for each sample rate.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: sample_rate, period_min, period_max.

        """
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

    def add_simple_coherence_weights(self, **kwargs) -> list[ChannelWeightSpec]:
        """
        Add coherence weights using the channel weight spec.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        list[ChannelWeightSpec]
            List of channel weight specifications with coherence features.

        """
        channel_weight_specs = []
        for channel in [
            ("ex", ["ex", "hy"]),
            ("ey", ["ey", "hx"]),
            ("hz", ["hx", "hx"]),
        ]:
            station_1 = self.local_station_id
            station_2 = self.local_station_id
            if channel[0] in ["hz"]:
                station_2 = self.remote_station_id
            cws = ChannelWeightSpec(
                combination_style="multiplication",
                output_channels=[channel[0]],
                feature_weight_specs=[
                    FeatureWeightSpec(
                        feature_name="coherence",
                        # time domain coherence estimation
                        feature=StridingWindowCoherence(
                            channel_1=channel[1][0],
                            channel_2=channel[1][1],
                            station_1=station_1,
                            station_2=station_2,
                            # the window is set to the stft window internally.
                            # window=Window(
                            #     type="hann",
                            #     num_samples=256,
                            #     overlap=128
                            #     )
                        ),
                        # how to weight the coherence, could be a list of different tapers
                        weight_kernels=[
                            TaperMonotonicWeightKernel(
                                style="taper",
                                half_window_style="hann",
                                threshold="low cut",
                                transition_lower_bound=kwargs.get(
                                    "transition_lower_bound", 0.6
                                ),
                                transition_upper_bound=kwargs.get(
                                    "transition_upper_bound", 0.9
                                ),
                            )
                        ],
                    )
                ],
            )
            channel_weight_specs.append(cws)
        return channel_weight_specs

    def create_config(
        self,
        kernel_dataset: KernelDataset | None = None,
        decimation_kwargs: dict = {},
        add_coherence_weights: bool = False,
        **kwargs,
    ) -> Processing:
        """
        Create Aurora processing configuration.

        Parameters
        ----------
        kernel_dataset : KernelDataset or None, optional
            Kernel dataset defining processing runs, by default None.
        decimation_kwargs : dict, optional
            Decimation parameters including window settings, by default {}.
        add_coherence_weights : bool, optional
            Whether to add coherence-based weights, by default True.
        **kwargs : dict
            Additional configuration parameters.

        Returns
        -------
        Processing
            Aurora configuration object.

        Raises
        ------
        ValueError
            If kernel_dataset is None and no kernel dataset exists.

        """
        if kernel_dataset is None:
            if self.has_kernel_dataset():
                if self.sample_rate > 1000:
                    decimation_kwargs.update(self.default_window_parameters["high"])
                else:
                    decimation_kwargs.update(self.default_window_parameters["low"])

            else:
                raise ValueError(
                    "Cannot make config because KernelDataset has not been set yet."
                )
        else:
            if kernel_dataset.sample_rate > 1000:
                decimation_kwargs.update(self.default_window_parameters["high"])
            else:
                decimation_kwargs.update(self.default_window_parameters["low"])
        # need to pass the number of samples in the window to correctly set the bands
        kwargs["num_samples_window"] = decimation_kwargs["stft.window.num_samples"]

        cc = ConfigCreator()
        config = cc.create_from_kernel_dataset(kernel_dataset, **kwargs)
        self._set_decimation_level_parameters(
            config, add_coherence_weights=add_coherence_weights, **decimation_kwargs
        )

        return config

    def _set_decimation_level_parameters(
        self, config: Processing, add_coherence_weights: bool = False, **kwargs
    ) -> None:
        """
        Set decimation level parameters for all decimation bands.

        Parameters
        ----------
        config : Processing
            Aurora configuration object to modify.
        add_coherence_weights : bool, optional
            Whether to add coherence-based channel weights, by default True.
        **kwargs : dict
            Key-value pairs to update in each decimation level.

        """

        for decimation in config.decimations:
            for key, value in kwargs.items():
                decimation.update_attribute(key, value)
            if add_coherence_weights:
                channel_weight_specs = self.add_simple_coherence_weights()
                decimation.channel_weight_specs = channel_weight_specs

    def _initialize_kernel_dataset(self, sample_rate: float | None = None) -> None:
        """
        Initialize a kernel dataset.

        Parameters
        ----------
        sample_rate : float or None, optional
            Sample rate to use, by default None (uses first available).

        """

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
        run_summary: RunSummary | None = None,
        local_station_id: str | None = None,
        remote_station_id: str | None = None,
        sample_rate: float | None = None,
    ) -> KernelDataset:
        """
        Build and return a KernelDataset.

        Parameters
        ----------
        run_summary : RunSummary or None, optional
            Run summary to use, by default None (creates from MTH5).
        local_station_id : str or None, optional
            Local station identifier, by default None.
        remote_station_id : str or None, optional
            Remote reference station identifier, by default None.
        sample_rate : float or None, optional
            Sample rate to filter runs, by default None.

        Returns
        -------
        KernelDataset
            Kernel dataset defining processing configuration.

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
        self,
        sample_rate: float,
        config: Processing | None = None,
        kernel_dataset: KernelDataset | None = None,
        plot: bool = False,
    ) -> MT | None:
        """
        Process a single sample rate to generate transfer functions.

        Parameters
        ----------
        sample_rate : float
            Sample rate of time series data to process.
        config : Processing or None, optional
            Aurora configuration object, by default None (creates from kernel_dataset).
        kernel_dataset : KernelDataset or None, optional
            Kernel dataset defining processing runs, by default None (creates from run summary).

        Returns
        -------
        MT or None
            Transfer function object, or None if processing fails.

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
            tf_obj = process_mth5(config, kernel_dataset, show_plot=plot)
        except Exception as error:
            close_open_files()
            logger.exception(error)
            logger.error(f"Skipping sample_rate {sample_rate}")
            return

        tf_obj.tf_id = self.processing_id

        # copy to an MT object using deep copy to avoid metadata references
        mt_obj = MT()
        mt_obj.survey_metadata.update(tf_obj.survey_metadata)
        mt_obj.station_metadata.update(tf_obj.station_metadata)
        mt_obj.channel_nomenclature = tf_obj.channel_nomenclature
        mt_obj._transfer_function = tf_obj._transfer_function

        return mt_obj

    def process(
        self,
        sample_rates: float | list[float] | None = None,
        processing_dict: (
            dict[float, dict[str, Processing | KernelDataset]] | None
        ) = None,
        merge: bool = True,
        save_to_mth5: bool = True,
        plot: bool = False,
    ) -> dict[float | str, dict[str, bool | MT]]:
        """
        Process magnetotelluric data at multiple sample rates.

        Parameters
        ----------
        sample_rates : float, list of float, or None, optional
            Sample rate(s) to process, by default None.
        processing_dict : dict or None, optional
            Dictionary mapping sample rates to config and kernel_dataset.
            Format: {sample_rate: {'config': Processing, 'kernel_dataset': KernelDataset}}
            By default None.
        merge : bool, optional
            Whether to merge all sample rates into a single transfer function
            according to merge_dict, by default True.
        save_to_mth5 : bool, optional
            Whether to save transfer functions to local MTH5 file, by default True.

        Returns
        -------
        dict[float or str, dict[str, bool or MT]]
            Dictionary with sample rates and 'combined' as keys, each containing
            {'processed': bool, 'tf': MT or None}.

        Raises
        ------
        ValueError
            If neither sample_rates nor processing_dict is provided.
        TypeError
            If sample_rates or processing_dict is not the correct format.

        Notes
        -----
        If merge is True and multiple sample rates are processed, a 'combined'
        key is added with the merged transfer function.

        Examples
        --------
        >>> ap = AuroraProcessing()
        >>> ap.local_station_id = "mt01"
        >>> ap.local_mth5_path = "data.h5"
        >>> results = ap.process(sample_rates=[1, 4], merge=True)

        """

        if sample_rates is None and processing_dict is None:
            raise ValueError("Must set either sample rates or processing_dict.")

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
                try:
                    mt_obj = self.process_single_sample_rate(sr)
                except Exception as e:
                    logger.error(e)
                    logger.error(f"Skipping sample rate {sr}")
                    logger.exception(e)
                    continue

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
                logger.info(f"Processing sample rate {key}.")
                try:
                    mt_obj = self.process_single_sample_rate(
                        key,
                        config=pdict["config"],
                        kernel_dataset=pdict["kernel_dataset"],
                        plot=plot,
                    )
                except Exception as e:
                    logger.error(e)
                    logger.error(f"Skipping sample rate {key}")
                    logger.exception(e)
                    continue
                if mt_obj is not None:
                    tf_processed[key]["processed"] = mt_obj.has_transfer_function()
                    tf_processed[key]["tf"] = mt_obj
                logger.info(f"Finished processing sample rate {key}.")

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
        else:
            processed["combined"] = processed[list(processed.keys())[0]]

        if save_to_mth5:
            ### add tf to local MTH5
            self._add_tf_to_local_mth5(processed)

        return processed

    def _validate_config(self, config: Processing) -> None:
        """
        Validate configuration object type.

        Parameters
        ----------
        config : Processing
            Configuration object to validate.

        Raises
        ------
        TypeError
            If config is not a Processing object.

        """
        if not isinstance(config, Processing):
            raise TypeError(
                "Config must be a aurora.config.metadata.Processing object. "
                f"Got type {type(config)}"
            )

    def _validate_kernel_dataset(self, kernel_dataset: KernelDataset) -> None:
        """
        Validate kernel dataset object type.

        Parameters
        ----------
        kernel_dataset : KernelDataset
            Kernel dataset object to validate.

        Raises
        ------
        TypeError
            If kernel_dataset is not a KernelDataset object.

        """
        if not isinstance(kernel_dataset, KernelDataset):
            raise TypeError(
                "Config must be a mtpy.processing.KernelDataset object. "
                f"Got type {type(kernel_dataset)}"
            )

    def _validate_processing_dict(
        self, processing_dict: dict[float, dict[str, Processing | KernelDataset]]
    ) -> None:
        """
        Validate the processing dictionary format.

        Parameters
        ----------
        processing_dict : dict
            Dictionary mapping sample rates to config and kernel_dataset.

        Raises
        ------
        TypeError
            If processing_dict or its values are not dictionaries.
        KeyError
            If required keys are missing from processing dictionary.

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

    def _validate_tf_processed_dict(
        self, tf_dict: dict[float, dict[str, bool | MT]]
    ) -> dict[float, dict[str, bool | MT]]:
        """
        Filter processed transfer functions from processing dictionary.

        Parameters
        ----------
        tf_dict : dict
            Dictionary of processed transfer functions with format:
            {sample_rate: {'processed': bool, 'tf': MT or None}}.

        Returns
        -------
        dict
            Dictionary containing only successfully processed transfer functions.

        Raises
        ------
        ValueError
            If no transfer functions were processed successfully.

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

    def _add_tf_to_local_mth5(
        self, tf_dict: dict[float | str, dict[str, bool | MT]]
    ) -> None:
        """
        Add transfer functions to the local MTH5 file.

        Parameters
        ----------
        tf_dict : dict
            Dictionary of transfer functions to add.

        """

        with MTH5() as m:
            m.open_mth5(self.local_mth5_path)
            for p_dict in tf_dict.values():
                m.add_transfer_function(p_dict["tf"])

    def _get_merge_tf_list(
        self, tf_dict: dict[float, dict[str, bool | MT]]
    ) -> list[dict[str, MT | float]]:
        """
        Prepare transfer functions list for merging with period constraints.

        Parameters
        ----------
        tf_dict : dict
            Dictionary of processed transfer functions.

        Returns
        -------
        list[dict]
            List of dictionaries containing transfer functions and their
            period min/max values for merging.

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

    def merge_transfer_functions(
        self, tf_dict: dict[float, dict[str, bool | MT]]
    ) -> MT:
        """
        Merge multiple transfer functions according to merge_dict.

        Parameters
        ----------
        tf_dict : dict
            Dictionary of transfer functions to merge.

        Returns
        -------
        MT
            Merged transfer function combining all sample rates.

        """

        merge_list = self._get_merge_tf_list(tf_dict)

        if len(merge_list) > 1:
            return merge_list[0]["tf"].merge(
                merge_list[1:], period_max=merge_list[0]["period_max"]
            )
        else:
            return merge_list[0]["tf"]
