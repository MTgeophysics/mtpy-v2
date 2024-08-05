from mth5 import RUN_SUMMARY_COLUMNS

ADDED_KERNEL_DATASET_COLUMNS = [
    "fc",
    "remote",
    "run_dataarray",
    "stft",
    "mth5_obj",
]

KERNEL_DATASET_COLUMNS = RUN_SUMMARY_COLUMNS + ADDED_KERNEL_DATASET_COLUMNS

MINI_SUMMARY_COLUMNS = [
    "survey",
    "station",
    "run",
    "start",
    "end",
    "duration",
]
