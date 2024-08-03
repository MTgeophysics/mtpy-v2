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
]

KERNEL_DATASET_COLUMNS = RUN_SUMMARY_COLUMNS + [
    "channel_scale_factors",
    "duration",
    "fc",
]
