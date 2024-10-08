from .transfer_function import Z
from .transfer_function import Tipper
from .transfer_function import PhaseTensor
from .mt_location import MTLocation
from .mt_stations import MTStations
from .mt_dataframe import MTDataFrame

__all__ = [
    "Z",
    "Tipper",
    "PhaseTensor",
    "MTLocation",
    "MTStations",
    "MTDataFrame",
]

# coordinate reference frames
COORDINATE_REFERENCE_FRAME_OPTIONS = {
    "+": "ned",
    "-": "enu",
    "ned": "ned",
    "enu": "enu",
    "exp(+ i\\omega t)": "ned",
    "exp(+i\\omega t)": "ned",
    "exp(- i\\omega t)": "enu",
    "exp(-i\\omega t)": "enu",
}
