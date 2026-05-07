# -*- coding: utf-8 -*-
"""Tests for the xarray transfer-function accessor."""

from __future__ import annotations

import numpy as np
import xarray as xr

from mtpy.core.transfer_function import MT_TO_OHM_FACTOR
from mtpy.core.transfer_function.accessor import TFDatasetAccessor
from mtpy.core.transfer_function.pt import PhaseTensor
from mtpy.core.transfer_function.tipper import Tipper
from mtpy.core.transfer_function.z import Z


class TestTFAccessorZ:
    """Validate Dataset.tf accessor behavior for impedance (Z)."""

    def test_accessor_registered(self):
        z = Z(
            z=np.ones((2, 2, 2), dtype=complex),
            z_error=np.ones((2, 2, 2), dtype=float),
            z_model_error=np.ones((2, 2, 2), dtype=float),
            frequency=np.array([1.0, 0.1]),
        )
        ds = z.to_xarray()

        assert hasattr(ds, "tf")
        assert ds.tf.validate()

    def test_z_roundtrip_mt_units(self):
        z = Z(
            z=np.array(
                [
                    [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
                    [[2.0 + 1.0j, 4.0 + 3.0j], [6.0 + 5.0j, 8.0 + 7.0j]],
                ],
                dtype=complex,
            ),
            z_error=np.full((2, 2, 2), 0.1, dtype=float),
            z_model_error=np.full((2, 2, 2), 0.2, dtype=float),
            frequency=np.array([1.0, 0.1]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        assert np.allclose(ds.tf.z(), z.z)
        assert np.allclose(ds.tf.z_error(), z.z_error)
        assert np.allclose(ds.tf.z_model_error(), z.z_model_error)

        z_from_accessor = ds.tf.to_z()
        assert np.allclose(z_from_accessor.z, z.z)
        assert np.allclose(z_from_accessor.z_error, z.z_error)
        assert np.allclose(z_from_accessor.z_model_error, z.z_model_error)
        assert np.allclose(z_from_accessor.frequency, z.frequency)

    def test_z_unit_scaling_to_ohm(self):
        z_mt = np.array(
            [[[10.0 + 0.0j, 20.0 + 0.0j], [30.0 + 0.0j, 40.0 + 0.0j]]],
            dtype=complex,
        )
        z = Z(
            z=z_mt,
            z_error=np.ones((1, 2, 2), dtype=float),
            z_model_error=np.ones((1, 2, 2), dtype=float),
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()

        expected_ohm = z_mt / MT_TO_OHM_FACTOR
        assert np.allclose(ds.tf.z(units="ohm"), expected_ohm)

    def test_channel_selection_from_full_tf_dataset(self):
        period = np.array([1.0, 10.0], dtype=float)
        output = np.array(["ex", "ey", "hz"], dtype=object)
        input_ = np.array(["hx", "hy"], dtype=object)

        tf = np.array(
            [
                [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [9 + 9j, 10 + 10j]],
                [[5 + 5j, 6 + 6j], [7 + 7j, 8 + 8j], [11 + 11j, 12 + 12j]],
            ],
            dtype=complex,
        )
        tf_err = np.ones_like(tf, dtype=float)
        tf_model_err = 2 * np.ones_like(tf, dtype=float)

        ds = xr.Dataset(
            data_vars={
                "transfer_function": (("period", "output", "input"), tf),
                "transfer_function_error": (("period", "output", "input"), tf_err),
                "transfer_function_model_error": (
                    ("period", "output", "input"),
                    tf_model_err,
                ),
            },
            coords={"period": period, "output": output, "input": input_},
            attrs={"impedance_units": "mt"},
        )

        z_view = ds.tf.z()
        assert z_view.shape == (2, 2, 2)
        assert np.allclose(z_view, tf[:, :2, :])
        assert np.allclose(ds.tf.frequency, np.array([1.0, 0.1]))

    def test_resistivity_phase_parity_with_z(self):
        z = Z(
            z=np.array(
                [
                    [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
                    [[2.0 + 1.0j, 4.0 + 3.0j], [6.0 + 5.0j, 8.0 + 7.0j]],
                ],
                dtype=complex,
            ),
            z_error=np.full((2, 2, 2), 0.1, dtype=float),
            z_model_error=np.full((2, 2, 2), 0.2, dtype=float),
            frequency=np.array([1.0, 0.1]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        assert np.allclose(ds.tf.resistivity, z.resistivity)
        assert np.allclose(ds.tf.phase, z.phase)
        assert np.allclose(ds.tf.resistivity_error, z.resistivity_error)
        assert np.allclose(ds.tf.phase_error, z.phase_error)
        assert np.allclose(ds.tf.resistivity_model_error, z.resistivity_model_error)
        assert np.allclose(ds.tf.phase_model_error, z.phase_model_error)
        assert np.allclose(ds.tf.res_xx, z.res_xx)
        assert np.allclose(ds.tf.res_xy, z.res_xy)
        assert np.allclose(ds.tf.res_yx, z.res_yx)
        assert np.allclose(ds.tf.res_yy, z.res_yy)
        assert np.allclose(ds.tf.res_error_xx, z.res_error_xx)
        assert np.allclose(ds.tf.res_error_xy, z.res_error_xy)
        assert np.allclose(ds.tf.res_error_yx, z.res_error_yx)
        assert np.allclose(ds.tf.res_error_yy, z.res_error_yy)
        assert np.allclose(ds.tf.res_model_error_xx, z.res_model_error_xx)
        assert np.allclose(ds.tf.res_model_error_xy, z.res_model_error_xy)
        assert np.allclose(ds.tf.res_model_error_yx, z.res_model_error_yx)
        assert np.allclose(ds.tf.res_model_error_yy, z.res_model_error_yy)
        assert np.allclose(ds.tf.phase_xx, z.phase_xx)
        assert np.allclose(ds.tf.phase_xy, z.phase_xy)
        assert np.allclose(ds.tf.phase_yx, z.phase_yx)
        assert np.allclose(ds.tf.phase_yy, z.phase_yy)
        assert np.allclose(ds.tf.phase_error_xx, z.phase_error_xx)
        assert np.allclose(ds.tf.phase_error_xy, z.phase_error_xy)
        assert np.allclose(ds.tf.phase_error_yx, z.phase_error_yx)
        assert np.allclose(ds.tf.phase_error_yy, z.phase_error_yy)
        assert np.allclose(ds.tf.phase_model_error_xx, z.phase_model_error_xx)
        assert np.allclose(ds.tf.phase_model_error_xy, z.phase_model_error_xy)
        assert np.allclose(ds.tf.phase_model_error_yx, z.phase_model_error_yx)
        assert np.allclose(ds.tf.phase_model_error_yy, z.phase_model_error_yy)

    def test_direct_properties_do_not_require_to_z(self, monkeypatch):
        z = Z(
            z=np.array(
                [[[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]],
                dtype=complex,
            ),
            z_error=np.full((1, 2, 2), 0.1, dtype=float),
            z_model_error=np.full((1, 2, 2), 0.2, dtype=float),
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        def fail_to_z(self, units=None):
            raise AssertionError("to_z should not be called")

        monkeypatch.setattr(TFDatasetAccessor, "to_z", fail_to_z)

        assert ds.tf.resistivity is not None
        assert ds.tf.phase is not None
        assert ds.tf.resistivity_error is not None
        assert ds.tf.phase_error is not None
        assert ds.tf.resistivity_model_error is not None
        assert ds.tf.phase_model_error is not None

    def test_remove_ss_wrapper_returns_z_and_dataset(self):
        z = Z(
            z=np.array(
                [[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]],
                dtype=complex,
            ),
            z_error=np.ones((1, 2, 2), dtype=float),
            z_model_error=np.ones((1, 2, 2), dtype=float),
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        z_out = ds.tf.remove_ss(0.5, 1.5, as_dataset=False)
        assert isinstance(z_out, Z)

        ds_out = ds.tf.remove_ss(0.5, 1.5, as_dataset=True)
        assert isinstance(ds_out, xr.Dataset)
        assert np.allclose(ds_out.tf.z(), z_out.z)

    def test_with_z_updates_dataset_from_arrays(self):
        z = Z(
            z=np.ones((2, 2, 2), dtype=complex),
            z_error=np.ones((2, 2, 2), dtype=float),
            z_model_error=np.ones((2, 2, 2), dtype=float),
            frequency=np.array([1.0, 0.1]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        z_new = 2.0 * z.z
        ds_updated = ds.tf.with_z(
            z=z_new,
            z_error=0.5 * np.ones((2, 2, 2), dtype=float),
            z_model_error=0.25 * np.ones((2, 2, 2), dtype=float),
            frequency=z.frequency,
            units="mt",
            inplace=False,
        )

        assert np.allclose(ds.tf.z(), z.z)
        assert np.allclose(ds_updated.tf.z(), z_new)

    def test_with_z_updates_dataset_inplace(self):
        z = Z(
            z=np.ones((1, 2, 2), dtype=complex),
            z_error=np.ones((1, 2, 2), dtype=float),
            z_model_error=np.ones((1, 2, 2), dtype=float),
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        z_new = np.array([[[3 + 1j, 4 + 2j], [5 + 3j, 6 + 4j]]], dtype=complex)
        ds.tf.with_z(
            z=z_new,
            z_error=np.ones((1, 2, 2), dtype=float),
            z_model_error=np.ones((1, 2, 2), dtype=float),
            frequency=z.frequency,
            inplace=True,
        )
        assert np.allclose(ds.tf.z(), z_new)

    def test_with_tipper_updates_dataset_from_object(self):
        tipper = Tipper(
            tipper=np.ones((1, 1, 2), dtype=complex),
            tipper_error=np.ones((1, 1, 2), dtype=float) * 0.01,
            tipper_model_error=np.ones((1, 1, 2), dtype=float) * 0.03,
            frequency=np.array([1.0]),
        )
        ds = tipper.to_xarray()

        new_tipper = Tipper(
            tipper=(2 + 0.5j) * np.ones((1, 1, 2), dtype=complex),
            tipper_error=np.ones((1, 1, 2), dtype=float) * 0.02,
            tipper_model_error=np.ones((1, 1, 2), dtype=float) * 0.04,
            frequency=np.array([1.0]),
        )

        ds_updated = ds.tf.with_tipper(tipper_obj=new_tipper)
        assert np.allclose(ds_updated.tf.tipper(), new_tipper.tipper)
        assert np.allclose(ds_updated.tf.tipper_error(), new_tipper.tipper_error)
        assert np.allclose(
            ds_updated.tf.tipper_model_error(), new_tipper.tipper_model_error
        )

    def test_with_res_phase_updates_dataset(self):
        z = Z(
            z=np.ones((1, 2, 2), dtype=complex),
            z_error=np.ones((1, 2, 2), dtype=float),
            z_model_error=np.ones((1, 2, 2), dtype=float),
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        resistivity = np.full((1, 2, 2), 100.0, dtype=float)
        phase = np.full((1, 2, 2), 45.0, dtype=float)
        res_error = np.full((1, 2, 2), 2.0, dtype=float)
        phase_error = np.full((1, 2, 2), 1.0, dtype=float)

        ds_updated = ds.tf.with_res_phase(
            resistivity=resistivity,
            phase=phase,
            frequency=np.array([1.0]),
            res_error=res_error,
            phase_error=phase_error,
            inplace=False,
        )

        z_expected = Z(units="mt")
        z_expected.set_resistivity_phase(
            resistivity=resistivity,
            phase=phase,
            frequency=np.array([1.0]),
            res_error=res_error,
            phase_error=phase_error,
        )

        assert np.allclose(ds_updated.tf.z(), z_expected.z)
        assert np.allclose(ds_updated.tf.z_error(), z_expected.z_error)

    def test_with_z_rejects_mixed_object_and_array_inputs(self):
        z = Z(
            z=np.ones((1, 2, 2), dtype=complex),
            z_error=np.ones((1, 2, 2), dtype=float),
            z_model_error=np.ones((1, 2, 2), dtype=float),
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()

        try:
            ds.tf.with_z(z_obj=z, z=z.z, frequency=z.frequency)
        except ValueError as error:
            assert "either z_obj or z" in str(error)
        else:
            raise AssertionError("Expected ValueError for mixed z inputs")

    def test_rotate_matches_z_for_impedance_dataset(self):
        z = Z(
            z=np.array(
                [[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]],
                dtype=complex,
            ),
            z_error=np.ones((1, 2, 2), dtype=float) * 0.1,
            z_model_error=np.ones((1, 2, 2), dtype=float) * 0.2,
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        expected = z.rotate(30, inplace=False)
        ds_rot = ds.tf.rotate(30, inplace=False)

        assert np.allclose(ds_rot.tf.z(), expected.z)
        assert np.allclose(ds_rot.tf.z_error(), expected.z_error)
        assert np.allclose(ds_rot.tf.z_model_error(), expected.z_model_error)

    def test_rotate_matches_tipper_for_tipper_dataset(self):
        tipper = Tipper(
            tipper=np.array([[[1 + 0.5j, 2 + 0.25j]]], dtype=complex),
            tipper_error=np.ones((1, 1, 2), dtype=float) * 0.01,
            tipper_model_error=np.ones((1, 1, 2), dtype=float) * 0.03,
            frequency=np.array([1.0]),
        )
        ds = tipper.to_xarray()

        expected = tipper.rotate(45, inplace=False)
        ds_rot = ds.tf.rotate(45, inplace=False)

        assert np.allclose(ds_rot.tf.tipper(), expected.tipper)
        assert np.allclose(ds_rot.tf.tipper_error(), expected.tipper_error)
        assert np.allclose(ds_rot.tf.tipper_model_error(), expected.tipper_model_error)

    def test_rotate_inplace_updates_dataset(self):
        z = Z(
            z=np.array(
                [[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]],
                dtype=complex,
            ),
            z_error=np.ones((1, 2, 2), dtype=float) * 0.1,
            z_model_error=np.ones((1, 2, 2), dtype=float) * 0.2,
            frequency=np.array([1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        z_original = ds.tf.z().copy()
        ds.tf.rotate(20, inplace=True)
        assert not np.allclose(ds.tf.z(), z_original)

    def test_interpolate_matches_z_for_impedance_dataset(self):
        z = Z(
            z=np.array(
                [
                    [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
                    [[2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j]],
                    [[3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]],
                ],
                dtype=complex,
            ),
            z_error=np.ones((3, 2, 2), dtype=float) * 0.1,
            z_model_error=np.ones((3, 2, 2), dtype=float) * 0.2,
            frequency=np.array([10.0, 1.0, 0.1]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"
        new_periods = np.array([0.1, 0.5, 1.0, 5.0, 10.0], dtype=float)

        z_expected = z.interpolate(new_periods, inplace=False, method="linear")
        ds_interp = ds.tf.interpolate(new_periods, inplace=False, method="linear")

        assert np.allclose(ds_interp.tf.z(), z_expected.z, equal_nan=True)
        assert np.allclose(ds_interp.tf.z_error(), z_expected.z_error, equal_nan=True)
        assert np.allclose(
            ds_interp.tf.z_model_error(),
            z_expected.z_model_error,
            equal_nan=True,
        )
        assert np.allclose(ds_interp.period.values, new_periods)

    def test_interpolate_inplace_updates_periods(self):
        z = Z(
            z=np.array(
                [
                    [[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]],
                    [[2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j]],
                    [[3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]],
                ],
                dtype=complex,
            ),
            z_error=np.ones((3, 2, 2), dtype=float) * 0.1,
            z_model_error=np.ones((3, 2, 2), dtype=float) * 0.2,
            frequency=np.array([10.0, 1.0, 0.1]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        result = ds.tf.interpolate(
            np.array([0.1, 0.5, 1.0, 5.0, 10.0], dtype=float),
            inplace=True,
            method="linear",
        )

        assert result is None
        assert ds.sizes["period"] == 5
        assert np.allclose(ds.period.values, np.array([0.1, 0.5, 1.0, 5.0, 10.0]))
        assert ds["transfer_function"].shape[0] == 5

    def test_dimensionality_and_distortion_helpers(self):
        z = Z(
            z=np.array(
                [
                    [
                        [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                        [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
                    ],
                    [
                        [-1.420305 - 1.02897j, 603.44306 + 814.4988j],
                        [-10.96444 - 21.4191j, 1.95081 + 1.52367j],
                    ],
                ],
                dtype=complex,
            ),
            z_error=np.ones((2, 2, 2), dtype=float),
            z_model_error=np.ones((2, 2, 2), dtype=float),
            frequency=np.array([10.0, 1.0]),
            units="mt",
        )
        ds = z.to_xarray()
        ds.attrs["impedance_units"] = "mt"

        dim = ds.tf.estimate_dimensionality()
        assert isinstance(dim, np.ndarray)
        assert dim.size == 2

        distortion, distortion_error = ds.tf.estimate_distortion()
        assert isinstance(distortion, np.ndarray)
        assert isinstance(distortion_error, np.ndarray)

    def test_tipper_accessor_parity(self):
        tipper = Tipper(
            tipper=np.ones((1, 1, 2), dtype=complex)
            + 0.25j * np.ones((1, 1, 2), dtype=complex),
            tipper_error=np.ones((1, 1, 2), dtype=float) * 0.01,
            tipper_model_error=np.ones((1, 1, 2), dtype=float) * 0.03,
            frequency=np.array([1.0]),
        )
        ds = tipper.to_xarray()

        assert np.allclose(ds.tf.tipper(), tipper.tipper)
        assert np.allclose(ds.tf.tipper_error(), tipper.tipper_error)
        assert np.allclose(ds.tf.tipper_model_error(), tipper.tipper_model_error)
        assert np.allclose(ds.tf.t_zx, tipper.tipper[:, 0, 0])
        assert np.allclose(ds.tf.t_zy, tipper.tipper[:, 0, 1])
        assert np.allclose(ds.tf.t_zx_error, tipper.tipper_error[:, 0, 0])
        assert np.allclose(ds.tf.t_zy_error, tipper.tipper_error[:, 0, 1])
        assert np.allclose(ds.tf.t_zx_model_error, tipper.tipper_model_error[:, 0, 0])
        assert np.allclose(ds.tf.t_zy_model_error, tipper.tipper_model_error[:, 0, 1])
        assert np.allclose(ds.tf.tipper_amplitude, tipper.amplitude)
        assert np.allclose(ds.tf.tipper_phase, tipper.phase)
        assert np.allclose(ds.tf.tipper_amplitude_error, tipper.amplitude_error)
        assert np.allclose(ds.tf.tipper_phase_error, tipper.phase_error)
        assert np.allclose(
            ds.tf.tipper_amplitude_model_error, tipper.amplitude_model_error
        )
        assert np.allclose(ds.tf.tipper_phase_model_error, tipper.phase_model_error)
        assert np.allclose(ds.tf.tipper_mag_real, tipper.mag_real)
        assert np.allclose(ds.tf.tipper_mag_imag, tipper.mag_imag)
        assert np.allclose(ds.tf.tipper_angle_real, tipper.angle_real)
        assert np.allclose(ds.tf.tipper_angle_imag, tipper.angle_imag)
        assert np.allclose(ds.tf.tipper_mag_error, tipper.mag_error)
        assert np.allclose(ds.tf.tipper_angle_error, tipper.angle_error)
        assert np.allclose(ds.tf.tipper_mag_model_error, tipper.mag_model_error)
        assert np.allclose(ds.tf.tipper_angle_model_error, tipper.angle_model_error)

    def test_direct_tipper_properties_do_not_require_to_tipper(self, monkeypatch):
        tipper = Tipper(
            tipper=np.ones((1, 1, 2), dtype=complex)
            + 0.25j * np.ones((1, 1, 2), dtype=complex),
            tipper_error=np.ones((1, 1, 2), dtype=float) * 0.01,
            tipper_model_error=np.ones((1, 1, 2), dtype=float) * 0.03,
            frequency=np.array([1.0]),
        )
        ds = tipper.to_xarray()

        def fail_to_tipper(self):
            raise AssertionError("to_tipper should not be called")

        monkeypatch.setattr(TFDatasetAccessor, "to_tipper", fail_to_tipper)

        assert ds.tf.t_zx is not None
        assert ds.tf.t_zy is not None
        assert ds.tf.t_zx_error is not None
        assert ds.tf.t_zy_error is not None
        assert ds.tf.t_zx_model_error is not None
        assert ds.tf.t_zy_model_error is not None
        assert ds.tf.tipper_amplitude is not None
        assert ds.tf.tipper_phase is not None
        assert ds.tf.tipper_amplitude_error is not None
        assert ds.tf.tipper_phase_error is not None
        assert ds.tf.tipper_amplitude_model_error is not None
        assert ds.tf.tipper_phase_model_error is not None
        assert ds.tf.tipper_mag_real is not None
        assert ds.tf.tipper_mag_imag is not None
        assert ds.tf.tipper_angle_real is not None
        assert ds.tf.tipper_angle_imag is not None
        assert ds.tf.tipper_mag_error is not None
        assert ds.tf.tipper_angle_error is not None
        assert ds.tf.tipper_mag_model_error is not None
        assert ds.tf.tipper_angle_model_error is not None

    def test_pt_accessor_parity(self):
        z_values = np.array([[[0, 1 + 1j], [-1 - 1j, 0]]])
        z_errors = np.array([[[0.1, 0.05], [0.05, 0.1]]])
        pt = PhaseTensor(
            z=z_values,
            z_error=z_errors,
            z_model_error=z_errors,
            frequency=np.array([1.0]),
        )
        ds = Z(
            z=z_values,
            z_error=z_errors,
            z_model_error=z_errors,
            frequency=np.array([1.0]),
            units="mt",
        ).to_xarray()
        ds.attrs["impedance_units"] = "mt"

        assert np.allclose(ds.tf.pt, pt.pt)
        assert np.allclose(ds.tf.pt_error, pt.pt_error)
        assert np.allclose(ds.tf.pt_model_error, pt.pt_model_error)
        assert np.allclose(ds.tf.pt_xx, pt.pt[:, 0, 0])
        assert np.allclose(ds.tf.pt_xy, pt.pt[:, 0, 1])
        assert np.allclose(ds.tf.pt_yx, pt.pt[:, 1, 0])
        assert np.allclose(ds.tf.pt_yy, pt.pt[:, 1, 1])
        assert np.allclose(ds.tf.pt_error_xx, pt.pt_error[:, 0, 0])
        assert np.allclose(ds.tf.pt_error_xy, pt.pt_error[:, 0, 1])
        assert np.allclose(ds.tf.pt_error_yx, pt.pt_error[:, 1, 0])
        assert np.allclose(ds.tf.pt_error_yy, pt.pt_error[:, 1, 1])
        assert np.allclose(ds.tf.pt_model_error_xx, pt.pt_model_error[:, 0, 0])
        assert np.allclose(ds.tf.pt_model_error_xy, pt.pt_model_error[:, 0, 1])
        assert np.allclose(ds.tf.pt_model_error_yx, pt.pt_model_error[:, 1, 0])
        assert np.allclose(ds.tf.pt_model_error_yy, pt.pt_model_error[:, 1, 1])
        assert np.allclose(ds.tf.pt_trace, pt.trace)
        assert np.allclose(ds.tf.pt_trace_error, pt.trace_error)
        assert np.allclose(ds.tf.pt_trace_model_error, pt.trace_model_error)
        assert np.allclose(ds.tf.pt_alpha, pt.alpha)
        assert np.allclose(ds.tf.pt_alpha_error, pt.alpha_error, equal_nan=True)
        assert np.allclose(
            ds.tf.pt_alpha_model_error, pt.alpha_model_error, equal_nan=True
        )
        assert np.allclose(ds.tf.pt_beta, pt.beta)
        assert np.allclose(ds.tf.pt_beta_error, pt.beta_error)
        assert np.allclose(ds.tf.pt_beta_model_error, pt.beta_model_error)
        assert np.allclose(ds.tf.pt_det, pt.det)
        assert np.allclose(ds.tf.pt_det_error, pt.det_error)
        assert np.allclose(ds.tf.pt_det_model_error, pt.det_model_error)
        assert np.allclose(ds.tf.pt_phimin, pt.phimin)
        assert np.allclose(ds.tf.pt_phimin_error, pt.phimin_error, equal_nan=True)
        assert np.allclose(
            ds.tf.pt_phimin_model_error, pt.phimin_model_error, equal_nan=True
        )
        assert np.allclose(ds.tf.pt_phimax, pt.phimax)
        assert np.allclose(ds.tf.pt_phimax_error, pt.phimax_error, equal_nan=True)
        assert np.allclose(
            ds.tf.pt_phimax_model_error, pt.phimax_model_error, equal_nan=True
        )
        assert np.allclose(ds.tf.pt_azimuth, pt.azimuth)
        assert np.allclose(ds.tf.pt_azimuth_error, pt.azimuth_error, equal_nan=True)
        assert np.allclose(
            ds.tf.pt_azimuth_model_error, pt.azimuth_model_error, equal_nan=True
        )
        assert np.allclose(ds.tf.pt_skew, pt.skew)
        assert np.allclose(ds.tf.pt_skew_error, pt.skew_error, equal_nan=True)
        assert np.allclose(
            ds.tf.pt_skew_model_error, pt.skew_model_error, equal_nan=True
        )
        assert np.allclose(ds.tf.pt_ellipticity, pt.ellipticity)
        assert np.allclose(
            ds.tf.pt_ellipticity_error, pt.ellipticity_error, equal_nan=True
        )
        assert np.allclose(
            ds.tf.pt_ellipticity_model_error,
            pt.ellipticity_model_error,
            equal_nan=True,
        )
        assert np.allclose(ds.tf.pt_eccentricity, pt.eccentricity)
        assert np.allclose(
            ds.tf.pt_eccentricity_error, pt.eccentricity_error, equal_nan=True
        )
        assert np.allclose(
            ds.tf.pt_eccentricity_model_error,
            pt.eccentricity_model_error,
            equal_nan=True,
        )
        assert np.allclose(ds.tf.pt_only1d, pt.only1d)
        assert np.allclose(ds.tf.pt_only2d, pt.only2d)

    def test_direct_pt_properties_do_not_require_to_pt(self, monkeypatch):
        z_values = np.array([[[0, 1 + 1j], [-1 - 1j, 0]]])
        z_errors = np.array([[[0.1, 0.05], [0.05, 0.1]]])
        ds = Z(
            z=z_values,
            z_error=z_errors,
            z_model_error=z_errors,
            frequency=np.array([1.0]),
            units="mt",
        ).to_xarray()
        ds.attrs["impedance_units"] = "mt"

        def fail_to_pt(self):
            raise AssertionError("to_pt should not be called")

        monkeypatch.setattr(TFDatasetAccessor, "to_pt", fail_to_pt)

        assert ds.tf.pt is not None
        assert ds.tf.pt_error is not None
        assert ds.tf.pt_model_error is not None
        assert ds.tf.pt_xx is not None
        assert ds.tf.pt_xy is not None
        assert ds.tf.pt_yx is not None
        assert ds.tf.pt_yy is not None
        assert ds.tf.pt_error_xx is not None
        assert ds.tf.pt_error_xy is not None
        assert ds.tf.pt_error_yx is not None
        assert ds.tf.pt_error_yy is not None
        assert ds.tf.pt_model_error_xx is not None
        assert ds.tf.pt_model_error_xy is not None
        assert ds.tf.pt_model_error_yx is not None
        assert ds.tf.pt_model_error_yy is not None
        assert ds.tf.pt_trace is not None
        assert ds.tf.pt_trace_error is not None
        assert ds.tf.pt_trace_model_error is not None
        assert ds.tf.pt_alpha is not None
        assert ds.tf.pt_alpha_error is not None
        assert ds.tf.pt_alpha_model_error is not None
        assert ds.tf.pt_beta is not None
        assert ds.tf.pt_beta_error is not None
        assert ds.tf.pt_beta_model_error is not None
        assert ds.tf.pt_det is not None
        assert ds.tf.pt_det_error is not None
        assert ds.tf.pt_det_model_error is not None
        assert ds.tf.pt_phimin is not None
        assert ds.tf.pt_phimin_error is not None
        assert ds.tf.pt_phimin_model_error is not None
        assert ds.tf.pt_phimax is not None
        assert ds.tf.pt_phimax_error is not None
        assert ds.tf.pt_phimax_model_error is not None
        assert ds.tf.pt_azimuth is not None
        assert ds.tf.pt_azimuth_error is not None
        assert ds.tf.pt_azimuth_model_error is not None
        assert ds.tf.pt_skew is not None
        assert ds.tf.pt_skew_error is not None
        assert ds.tf.pt_skew_model_error is not None
        assert ds.tf.pt_ellipticity is not None
        assert ds.tf.pt_ellipticity_error is not None
        assert ds.tf.pt_ellipticity_model_error is not None
        assert ds.tf.pt_eccentricity is not None
        assert ds.tf.pt_eccentricity_error is not None
        assert ds.tf.pt_eccentricity_model_error is not None
        assert ds.tf.pt_only1d is not None
        assert ds.tf.pt_only2d is not None
