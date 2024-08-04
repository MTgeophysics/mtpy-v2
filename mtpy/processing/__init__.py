RUN_SUMMARY_COLUMNS = [
    "survey",
    "station",
    "run",
    "start",
    "end",
    "sample_rate",
    "input_channels",
    "output_channels",
    "remote",
    "mth5_path",
    "duration",
    "has_data",
]

KERNEL_DATASET_COLUMNS = RUN_SUMMARY_COLUMNS + [
    "channel_scale_factors",
    "fc",
]
