"""xarray Dataset accessor for transfer-function representations."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from . import IMPEDANCE_UNITS
from .pt import PhaseTensor
from .tf_helpers import (
    compute_phase,
    compute_phase_error,
    compute_phase_tensor,
    compute_phase_tensor_error,
    compute_pt_alpha,
    compute_pt_alpha_error,
    compute_pt_azimuth,
    compute_pt_azimuth_error,
    compute_pt_beta,
    compute_pt_beta_error,
    compute_pt_det,
    compute_pt_det_error,
    compute_pt_eccentricity,
    compute_pt_eccentricity_error,
    compute_pt_ellipticity,
    compute_pt_ellipticity_error,
    compute_pt_phimax,
    compute_pt_phimax_error,
    compute_pt_phimin,
    compute_pt_phimin_error,
    compute_pt_skew,
    compute_pt_skew_error,
    compute_pt_trace,
    compute_pt_trace_error,
    compute_resistivity,
    compute_resistivity_error,
    compute_tipper_amp_phase_error,
    compute_tipper_amplitude,
    compute_tipper_angle_error,
    compute_tipper_angle_imag,
    compute_tipper_angle_real,
    compute_tipper_magnitude_error,
    compute_tipper_magnitude_imag,
    compute_tipper_magnitude_real,
    compute_tipper_phase,
)
from .tipper import Tipper
from .z import Z


@xr.register_dataset_accessor("tf")
class TFDatasetAccessor:
    """Accessor exposing transfer-function views from an xarray Dataset.

    Notes
    -----
    This first implementation focuses on impedance (Z) access. Tipper and
    phase tensor support can be layered on top of the same validation and
    channel-selection helpers.
    """

    _REQUIRED_VARIABLES = (
        "transfer_function",
        "transfer_function_error",
        "transfer_function_model_error",
    )

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj
        self._cache: dict[str, Any] = {}

    def validate(self) -> bool:
        """Validate that required transfer-function variables and coords exist."""
        for variable in self._REQUIRED_VARIABLES:
            if variable not in self._obj.data_vars:
                raise KeyError(f"Dataset missing required variable: {variable}")

        tf = self._obj["transfer_function"]
        for dim in ("period", "output", "input"):
            if dim not in tf.dims:
                raise KeyError(
                    "Dataset transfer_function must have dims "
                    "('period', 'output', 'input')"
                )

        if "period" not in self._obj.coords:
            raise KeyError("Dataset missing required coord: period")

        return True

    @property
    def frequency(self) -> np.ndarray:
        """Frequency array derived from period coordinates."""
        period = np.asarray(self._obj.coords["period"].values, dtype=float)
        return 1.0 / period

    @property
    def impedance_units(self) -> str:
        """Impedance units for accessor outputs."""
        units = str(self._obj.attrs.get("impedance_units", "mt")).lower()
        if units not in IMPEDANCE_UNITS:
            return "mt"
        return units

    @staticmethod
    def _pick_channel_labels(
        available: list[Any], candidates: list[str], required: int
    ) -> list[Any] | None:
        """Pick ordered coordinate labels using preferred candidate names."""
        channel_map = {str(label).lower(): label for label in available}
        selected: list[Any] = []
        for candidate in candidates:
            key = candidate.lower()
            if key in channel_map and channel_map[key] not in selected:
                selected.append(channel_map[key])
            if len(selected) == required:
                return selected
        return None

    def _impedance_dataarray(self, variable: str) -> xr.DataArray:
        """Return impedance subset from a transfer-function variable."""
        self.validate()

        da = self._obj[variable]
        outputs = list(self._obj.coords["output"].values)
        inputs = list(self._obj.coords["input"].values)

        out_labels = self._pick_channel_labels(outputs, ["ex", "ey", "x", "y"], 2)
        in_labels = self._pick_channel_labels(inputs, ["hx", "hy", "x", "y"], 2)

        if out_labels is None and da.sizes.get("output", 0) == 2:
            out_labels = list(self._obj.coords["output"].values[:2])
        if in_labels is None and da.sizes.get("input", 0) == 2:
            in_labels = list(self._obj.coords["input"].values[:2])

        if out_labels is None or in_labels is None:
            raise ValueError(
                "Could not determine impedance channels from output/input coords."
            )

        return da.sel(output=out_labels, input=in_labels)

    def _tipper_dataarray(self, variable: str) -> xr.DataArray:
        """Return tipper subset from a transfer-function variable."""
        self.validate()

        da = self._obj[variable]
        outputs = list(self._obj.coords["output"].values)
        inputs = list(self._obj.coords["input"].values)

        out_labels = self._pick_channel_labels(outputs, ["hz", "z"], 1)
        in_labels = self._pick_channel_labels(inputs, ["hx", "hy", "x", "y"], 2)

        if out_labels is None and da.sizes.get("output", 0) == 1:
            out_labels = list(self._obj.coords["output"].values[:1])
        if in_labels is None and da.sizes.get("input", 0) == 2:
            in_labels = list(self._obj.coords["input"].values[:2])

        if out_labels is None or in_labels is None:
            raise ValueError(
                "Could not determine tipper channels from output/input coords."
            )

        return da.sel(output=out_labels, input=in_labels)

    @staticmethod
    def _unit_factor(units: str) -> float:
        """Return conversion factor to convert internal mt units to requested units."""
        key = units.lower()
        if key not in IMPEDANCE_UNITS:
            raise ValueError(
                f"{units} is not an acceptable unit for impedance. "
                f"Options are {list(IMPEDANCE_UNITS.keys())}."
            )
        return float(IMPEDANCE_UNITS[key])

    def z(self, units: str | None = None) -> np.ndarray:
        """Return impedance tensor array with requested units."""
        resolved_units = self.impedance_units if units is None else units.lower()
        factor = self._unit_factor(resolved_units)
        return self._impedance_dataarray("transfer_function").values / factor

    def z_error(self, units: str | None = None) -> np.ndarray:
        """Return impedance standard-deviation errors with requested units."""
        resolved_units = self.impedance_units if units is None else units.lower()
        factor = self._unit_factor(resolved_units)
        return self._impedance_dataarray("transfer_function_error").values / factor

    def z_model_error(self, units: str | None = None) -> np.ndarray:
        """Return impedance model errors with requested units."""
        resolved_units = self.impedance_units if units is None else units.lower()
        factor = self._unit_factor(resolved_units)
        return (
            self._impedance_dataarray("transfer_function_model_error").values / factor
        )

    def tipper(self) -> np.ndarray:
        """Return tipper tensor array."""
        return self._tipper_dataarray("transfer_function").values

    def tipper_error(self) -> np.ndarray:
        """Return tipper standard-deviation errors."""
        return self._tipper_dataarray("transfer_function_error").values

    def tipper_model_error(self) -> np.ndarray:
        """Return tipper model errors."""
        return self._tipper_dataarray("transfer_function_model_error").values

    def _z_mt(self) -> np.ndarray:
        """Return impedance tensor in internal mt units."""
        return self._impedance_dataarray("transfer_function").values

    def _z_error_mt(self) -> np.ndarray:
        """Return impedance error tensor in internal mt units."""
        return self._impedance_dataarray("transfer_function_error").values

    def _z_model_error_mt(self) -> np.ndarray:
        """Return impedance model-error tensor in internal mt units."""
        return self._impedance_dataarray("transfer_function_model_error").values

    @staticmethod
    def _get_component(comp: str, array: np.ndarray | None) -> np.ndarray | None:
        """Return one impedance-derived tensor component from a 2x2 array stack."""
        if array is None:
            return None

        index_dict = {"x": 0, "y": 1}
        ii = index_dict[comp[-2]]
        jj = index_dict[comp[-1]]
        return array[:, ii, jj]

    def to_z(self, units: str | None = None) -> Z:
        """Build a :class:`Z` object lazily from Dataset-backed transfer functions."""
        resolved_units = self.impedance_units if units is None else units.lower()
        z_object = Z(
            z=self.z(units=resolved_units),
            z_error=self.z_error(units=resolved_units),
            z_model_error=self.z_model_error(units=resolved_units),
            frequency=self.frequency,
            units=resolved_units,
        )
        return z_object

    def to_tipper(self) -> Tipper:
        """Build a :class:`Tipper` object from Dataset-backed transfer functions."""
        return Tipper(
            tipper=self.tipper(),
            tipper_error=self.tipper_error(),
            tipper_model_error=self.tipper_model_error(),
            frequency=self.frequency,
        )

    def to_pt(self) -> PhaseTensor:
        """Build a :class:`PhaseTensor` object from Dataset-backed impedance."""
        return PhaseTensor(
            z=self.z(),
            z_error=self.z_error(),
            z_model_error=self.z_model_error(),
            frequency=self.frequency,
        )

    @property
    def resistivity(self) -> np.ndarray | None:
        """Apparent resistivity tensor computed directly from Dataset values."""
        return compute_resistivity(self._z_mt(), self.frequency)

    @property
    def phase(self) -> np.ndarray | None:
        """Impedance phase tensor (degrees) computed directly from Dataset values."""
        return compute_phase(self._z_mt())

    @property
    def resistivity_error(self) -> np.ndarray | None:
        """Apparent resistivity error computed directly from Dataset values."""
        return compute_resistivity_error(
            self._z_mt(),
            self._z_error_mt(),
            self.frequency,
        )

    @property
    def phase_error(self) -> np.ndarray | None:
        """Impedance phase error (degrees) computed directly from Dataset values."""
        return compute_phase_error(self._z_mt(), self._z_error_mt())

    @property
    def resistivity_model_error(self) -> np.ndarray | None:
        """Apparent resistivity model error computed directly from Dataset values."""
        return compute_resistivity_error(
            self._z_mt(),
            self._z_model_error_mt(),
            self.frequency,
        )

    @property
    def phase_model_error(self) -> np.ndarray | None:
        """Impedance phase model error (degrees) computed directly from Dataset values."""
        return compute_phase_error(self._z_mt(), self._z_model_error_mt())

    @property
    def res_error_xx(self) -> np.ndarray | None:
        """Apparent resistivity error of the xx impedance component."""
        return self._get_component("xx", self.resistivity_error)

    @property
    def res_error_xy(self) -> np.ndarray | None:
        """Apparent resistivity error of the xy impedance component."""
        return self._get_component("xy", self.resistivity_error)

    @property
    def res_error_yx(self) -> np.ndarray | None:
        """Apparent resistivity error of the yx impedance component."""
        return self._get_component("yx", self.resistivity_error)

    @property
    def res_error_yy(self) -> np.ndarray | None:
        """Apparent resistivity error of the yy impedance component."""
        return self._get_component("yy", self.resistivity_error)

    @property
    def res_model_error_xx(self) -> np.ndarray | None:
        """Apparent resistivity model error of the xx impedance component."""
        return self._get_component("xx", self.resistivity_model_error)

    @property
    def res_model_error_xy(self) -> np.ndarray | None:
        """Apparent resistivity model error of the xy impedance component."""
        return self._get_component("xy", self.resistivity_model_error)

    @property
    def res_model_error_yx(self) -> np.ndarray | None:
        """Apparent resistivity model error of the yx impedance component."""
        return self._get_component("yx", self.resistivity_model_error)

    @property
    def res_model_error_yy(self) -> np.ndarray | None:
        """Apparent resistivity model error of the yy impedance component."""
        return self._get_component("yy", self.resistivity_model_error)

    @property
    def res_xx(self) -> np.ndarray | None:
        """Apparent resistivity of the xx impedance component."""
        return self._get_component("xx", self.resistivity)

    @property
    def res_xy(self) -> np.ndarray | None:
        """Apparent resistivity of the xy impedance component."""
        return self._get_component("xy", self.resistivity)

    @property
    def res_yx(self) -> np.ndarray | None:
        """Apparent resistivity of the yx impedance component."""
        return self._get_component("yx", self.resistivity)

    @property
    def res_yy(self) -> np.ndarray | None:
        """Apparent resistivity of the yy impedance component."""
        return self._get_component("yy", self.resistivity)

    @property
    def phase_xx(self) -> np.ndarray | None:
        """Phase of the xx impedance component in degrees."""
        return self._get_component("xx", self.phase)

    @property
    def phase_xy(self) -> np.ndarray | None:
        """Phase of the xy impedance component in degrees."""
        return self._get_component("xy", self.phase)

    @property
    def phase_yx(self) -> np.ndarray | None:
        """Phase of the yx impedance component in degrees."""
        return self._get_component("yx", self.phase)

    @property
    def phase_yy(self) -> np.ndarray | None:
        """Phase of the yy impedance component in degrees."""
        return self._get_component("yy", self.phase)

    @property
    def phase_error_xx(self) -> np.ndarray | None:
        """Phase error of the xx impedance component in degrees."""
        return self._get_component("xx", self.phase_error)

    @property
    def phase_error_xy(self) -> np.ndarray | None:
        """Phase error of the xy impedance component in degrees."""
        return self._get_component("xy", self.phase_error)

    @property
    def phase_error_yx(self) -> np.ndarray | None:
        """Phase error of the yx impedance component in degrees."""
        return self._get_component("yx", self.phase_error)

    @property
    def phase_error_yy(self) -> np.ndarray | None:
        """Phase error of the yy impedance component in degrees."""
        return self._get_component("yy", self.phase_error)

    @property
    def phase_model_error_xx(self) -> np.ndarray | None:
        """Phase model error of the xx impedance component in degrees."""
        return self._get_component("xx", self.phase_model_error)

    @property
    def phase_model_error_xy(self) -> np.ndarray | None:
        """Phase model error of the xy impedance component in degrees."""
        return self._get_component("xy", self.phase_model_error)

    @property
    def phase_model_error_yx(self) -> np.ndarray | None:
        """Phase model error of the yx impedance component in degrees."""
        return self._get_component("yx", self.phase_model_error)

    @property
    def phase_model_error_yy(self) -> np.ndarray | None:
        """Phase model error of the yy impedance component in degrees."""
        return self._get_component("yy", self.phase_model_error)

    @property
    def tipper_amplitude(self) -> np.ndarray | None:
        """Tipper amplitude derived from the accessor tipper view."""
        return compute_tipper_amplitude(self.tipper())

    @property
    def tipper_phase(self) -> np.ndarray | None:
        """Tipper phase in degrees derived from the accessor tipper view."""
        return compute_tipper_phase(self.tipper())

    @property
    def tipper_amplitude_error(self) -> np.ndarray | None:
        """Tipper amplitude error derived from the accessor tipper view."""
        return compute_tipper_amp_phase_error(self.tipper(), self.tipper_error())[0]

    @property
    def tipper_phase_error(self) -> np.ndarray | None:
        """Tipper phase error in degrees derived from the accessor tipper view."""
        return compute_tipper_amp_phase_error(self.tipper(), self.tipper_error())[1]

    @property
    def tipper_amplitude_model_error(self) -> np.ndarray | None:
        """Tipper amplitude model error derived from Dataset values."""
        return compute_tipper_amp_phase_error(self.tipper(), self.tipper_model_error())[
            0
        ]

    @property
    def tipper_phase_model_error(self) -> np.ndarray | None:
        """Tipper phase model error in degrees derived from Dataset values."""
        return compute_tipper_amp_phase_error(self.tipper(), self.tipper_model_error())[
            1
        ]

    @property
    def tipper_mag_real(self) -> np.ndarray | None:
        """Tipper real-component magnitude."""
        return compute_tipper_magnitude_real(self.tipper())

    @property
    def tipper_mag_imag(self) -> np.ndarray | None:
        """Tipper imaginary-component magnitude."""
        return compute_tipper_magnitude_imag(self.tipper())

    @property
    def tipper_angle_real(self) -> np.ndarray | None:
        """Tipper real-component angle in degrees."""
        return compute_tipper_angle_real(self.tipper())

    @property
    def tipper_angle_imag(self) -> np.ndarray | None:
        """Tipper imaginary-component angle in degrees."""
        return compute_tipper_angle_imag(self.tipper())

    @property
    def tipper_mag_error(self) -> np.ndarray | None:
        """Tipper magnitude error derived from Dataset values."""
        return compute_tipper_magnitude_error(self.tipper_error())

    @property
    def tipper_angle_error(self) -> np.ndarray | None:
        """Tipper angle error in degrees derived from Dataset values."""
        return compute_tipper_angle_error(self.tipper_error())

    @property
    def tipper_mag_model_error(self) -> np.ndarray | None:
        """Tipper magnitude model error derived from Dataset values."""
        return compute_tipper_magnitude_error(self.tipper_model_error())

    @property
    def tipper_angle_model_error(self) -> np.ndarray | None:
        """Tipper angle model error in degrees derived from Dataset values."""
        return compute_tipper_angle_error(self.tipper_model_error())

    @property
    def pt(self) -> np.ndarray | None:
        """Phase tensor array derived from impedance channels."""
        return compute_phase_tensor(self._z_mt())

    @property
    def pt_error(self) -> np.ndarray | None:
        """Phase tensor error array derived from impedance channels."""
        return compute_phase_tensor_error(self._z_mt(), self._z_error_mt())

    @property
    def pt_model_error(self) -> np.ndarray | None:
        """Phase tensor model error array derived from impedance channels."""
        return compute_phase_tensor_error(self._z_mt(), self._z_model_error_mt())

    @property
    def pt_phimin(self) -> np.ndarray | None:
        """Minimum phase angle of the phase tensor in degrees."""
        return compute_pt_phimin(self.pt)

    @property
    def pt_phimin_error(self) -> np.ndarray | None:
        """Minimum phase angle error of the phase tensor in degrees."""
        return compute_pt_phimin_error(self.pt, self.pt_error)

    @property
    def pt_phimin_model_error(self) -> np.ndarray | None:
        """Minimum phase angle model error of the phase tensor in degrees."""
        return compute_pt_phimin_error(self.pt, self.pt_model_error)

    @property
    def pt_phimax(self) -> np.ndarray | None:
        """Maximum phase angle of the phase tensor in degrees."""
        return compute_pt_phimax(self.pt)

    @property
    def pt_phimax_error(self) -> np.ndarray | None:
        """Maximum phase angle error of the phase tensor in degrees."""
        return compute_pt_phimax_error(self.pt, self.pt_error)

    @property
    def pt_phimax_model_error(self) -> np.ndarray | None:
        """Maximum phase angle model error of the phase tensor in degrees."""
        return compute_pt_phimax_error(self.pt, self.pt_model_error)

    @property
    def pt_trace(self) -> np.ndarray | None:
        """Trace of the phase tensor."""
        return compute_pt_trace(self.pt)

    @property
    def pt_trace_error(self) -> np.ndarray | None:
        """Trace error of the phase tensor."""
        return compute_pt_trace_error(self.pt_error)

    @property
    def pt_trace_model_error(self) -> np.ndarray | None:
        """Trace model error of the phase tensor."""
        return compute_pt_trace_error(self.pt_model_error)

    @property
    def pt_alpha(self) -> np.ndarray | None:
        """Principal axis angle of the phase tensor in degrees."""
        return compute_pt_alpha(self.pt)

    @property
    def pt_alpha_error(self) -> np.ndarray | None:
        """Principal axis angle error of the phase tensor in degrees."""
        return compute_pt_alpha_error(self.pt, self.pt_error)

    @property
    def pt_alpha_model_error(self) -> np.ndarray | None:
        """Principal axis angle model error of the phase tensor in degrees."""
        return compute_pt_alpha_error(self.pt, self.pt_model_error)

    @property
    def pt_beta(self) -> np.ndarray | None:
        """3D-dimensionality angle of the phase tensor in degrees."""
        return compute_pt_beta(self.pt)

    @property
    def pt_beta_error(self) -> np.ndarray | None:
        """3D-dimensionality angle error of the phase tensor in degrees."""
        return compute_pt_beta_error(self.pt, self.pt_error)

    @property
    def pt_beta_model_error(self) -> np.ndarray | None:
        """3D-dimensionality angle model error of the phase tensor in degrees."""
        return compute_pt_beta_error(self.pt, self.pt_model_error)

    @property
    def pt_azimuth(self) -> np.ndarray | None:
        """Phase tensor azimuth in degrees."""
        return compute_pt_azimuth(self.pt)

    @property
    def pt_azimuth_error(self) -> np.ndarray | None:
        """Phase tensor azimuth error in degrees."""
        return compute_pt_azimuth_error(self.pt, self.pt_error)

    @property
    def pt_azimuth_model_error(self) -> np.ndarray | None:
        """Phase tensor azimuth model error in degrees."""
        return compute_pt_azimuth_error(self.pt, self.pt_model_error)

    @property
    def pt_skew(self) -> np.ndarray | None:
        """Phase tensor skew in degrees."""
        return compute_pt_skew(self.pt)

    @property
    def pt_skew_error(self) -> np.ndarray | None:
        """Phase tensor skew error in degrees."""
        return compute_pt_skew_error(self.pt, self.pt_error)

    @property
    def pt_skew_model_error(self) -> np.ndarray | None:
        """Phase tensor skew model error in degrees."""
        return compute_pt_skew_error(self.pt, self.pt_model_error)

    @property
    def pt_det(self) -> np.ndarray | None:
        """Determinant of the phase tensor."""
        return compute_pt_det(self.pt)

    @property
    def pt_det_error(self) -> np.ndarray | None:
        """Determinant error of the phase tensor."""
        return compute_pt_det_error(self.pt, self.pt_error)

    @property
    def pt_det_model_error(self) -> np.ndarray | None:
        """Determinant model error of the phase tensor."""
        return compute_pt_det_error(self.pt, self.pt_model_error)

    @property
    def pt_ellipticity(self) -> np.ndarray | None:
        """Phase tensor ellipticity."""
        return compute_pt_ellipticity(self.pt)

    @property
    def pt_ellipticity_error(self) -> np.ndarray | None:
        """Phase tensor ellipticity error."""
        return compute_pt_ellipticity_error(self.pt, self.pt_error)

    @property
    def pt_ellipticity_model_error(self) -> np.ndarray | None:
        """Phase tensor ellipticity model error."""
        return compute_pt_ellipticity_error(self.pt, self.pt_model_error)

    @property
    def pt_eccentricity(self) -> np.ndarray | None:
        """Phase tensor eccentricity."""
        return compute_pt_eccentricity(self.pt)

    @property
    def pt_eccentricity_error(self) -> np.ndarray | None:
        """Phase tensor eccentricity error."""
        return compute_pt_eccentricity_error(self.pt, self.pt_error)

    @property
    def pt_eccentricity_model_error(self) -> np.ndarray | None:
        """Phase tensor eccentricity model error."""
        return compute_pt_eccentricity_error(self.pt, self.pt_model_error)

    @property
    def pt_only1d(self) -> np.ndarray | None:
        """Phase tensor expressed in the 1D convenience form."""
        if self.pt is None:
            return None

        pt_1d = self.pt.copy()
        pt_1d[:, 0, 1] = 0
        pt_1d[:, 1, 0] = 0

        mean_1d = 0.5 * (pt_1d[:, 0, 0] + pt_1d[:, 1, 1])
        pt_1d[:, 0, 0] = mean_1d
        pt_1d[:, 1, 1] = mean_1d
        return pt_1d

    @property
    def pt_only2d(self) -> np.ndarray | None:
        """Phase tensor expressed in the 2D convenience form."""
        if self.pt is None:
            return None

        pt_2d = self.pt.copy()
        pt_2d[:, 0, 1] = 0
        pt_2d[:, 1, 0] = 0
        pt_2d[:, 0, 0] = self.pt_phimax[:]
        pt_2d[:, 1, 1] = self.pt_phimin[:]
        return pt_2d

    @property
    def phase_tensor(self) -> Any:
        """Phase tensor object derived from the accessor impedance view."""
        return self.to_z().phase_tensor

    @property
    def invariants(self) -> Any:
        """Impedance invariants object derived from the accessor impedance view."""
        return self.to_z().invariants

    def remove_ss(
        self,
        reduce_res_factor_x: float | list | np.ndarray = 1.0,
        reduce_res_factor_y: float | list | np.ndarray = 1.0,
        units: str | None = None,
        as_dataset: bool = False,
    ) -> xr.Dataset | Z:
        """Apply static-shift correction using :class:`Z.remove_ss`.

        Parameters
        ----------
        reduce_res_factor_x : float or array-like, optional
            Static-shift correction factor for x-polarized rows.
        reduce_res_factor_y : float or array-like, optional
            Static-shift correction factor for y-polarized rows.
        units : str, optional
            Impedance unit system passed to :meth:`to_z`.
        as_dataset : bool, optional
            If True, return corrected transfer-function Dataset. If False,
            return a corrected :class:`Z` object.
        """
        z_out = self.to_z(units=units).remove_ss(
            reduce_res_factor_x=reduce_res_factor_x,
            reduce_res_factor_y=reduce_res_factor_y,
            inplace=False,
        )
        if as_dataset:
            ds_out = z_out.to_xarray()
            ds_out.attrs.update(dict(self._obj.attrs))
            ds_out.attrs["impedance_units"] = z_out.units
            return ds_out
        return z_out

    def remove_distortion(
        self,
        distortion_tensor: np.ndarray | None = None,
        distortion_error_tensor: np.ndarray | None = None,
        n_frequencies: int | None = None,
        comp: str = "det",
        only_2d: bool = False,
        units: str | None = None,
        as_dataset: bool = False,
    ) -> xr.Dataset | Z:
        """Apply galvanic distortion correction using :class:`Z.remove_distortion`."""
        z_out = self.to_z(units=units).remove_distortion(
            distortion_tensor=distortion_tensor,
            distortion_error_tensor=distortion_error_tensor,
            n_frequencies=n_frequencies,
            comp=comp,
            only_2d=only_2d,
            inplace=False,
        )
        if as_dataset:
            ds_out = z_out.to_xarray()
            ds_out.attrs.update(dict(self._obj.attrs))
            ds_out.attrs["impedance_units"] = z_out.units
            return ds_out
        return z_out

    def estimate_dimensionality(
        self, skew_threshold: float = 5, eccentricity_threshold: float = 0.1
    ) -> np.ndarray:
        """Estimate 1D/2D/3D dimensionality from impedance-derived PT metrics."""
        return self.to_z().estimate_dimensionality(
            skew_threshold=skew_threshold,
            eccentricity_threshold=eccentricity_threshold,
        )

    def estimate_distortion(
        self,
        n_frequencies: int | None = None,
        comp: str = "det",
        only_2d: bool = False,
        clockwise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate galvanic distortion tensor using :class:`Z.estimate_distortion`."""
        return self.to_z().estimate_distortion(
            n_frequencies=n_frequencies,
            comp=comp,
            only_2d=only_2d,
            clockwise=clockwise,
        )

    def estimate_depth_of_investigation(self) -> Any:
        """Estimate depth of investigation using :class:`Z` implementation."""
        return self.to_z().estimate_depth_of_investigation()
