from mth5 import RUN_SUMMARY_DTYPE, RUN_SUMMARY_COLUMNS

ADDED_KERNEL_DATASET_DTYPE = [
    ("fc", bool),
    ("remote", bool),
    ("run_dataarray", object),
    ("stft", object),
    ("mth5_obj", object),
]
ADDED_KERNEL_DATASET_COLUMNS = [
    entry[0] for entry in ADDED_KERNEL_DATASET_DTYPE
]

KERNEL_DATASET_DTYPE = RUN_SUMMARY_DTYPE + ADDED_KERNEL_DATASET_DTYPE
KERNEL_DATASET_COLUMNS = [entry[0] for entry in KERNEL_DATASET_DTYPE]

MINI_SUMMARY_COLUMNS = [
    "survey",
    "station",
    "run",
    "start",
    "end",
    "duration",
]

from .run_summary import RunSummary
from .kernel_dataset import KernelDataset
from .aurora.process_aurora import AuroraProcessing

__all__ = ["RunSummary", "KernelDataset", "AuroraProcessing"]
