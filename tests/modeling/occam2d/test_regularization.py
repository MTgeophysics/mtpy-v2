# -*- coding: utf-8 -*-
"""Tests for occam2d Regularization.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.modeling.occam2d import Regularization


@pytest.fixture
def station_locations() -> np.ndarray:
    """Provide unsorted station locations to verify inherited mesh sorting."""
    return np.array([1500.0, 0.0, 800.0, 300.0], dtype=float)


@pytest.fixture
def regularization(station_locations: np.ndarray) -> Regularization:
    """Create a Regularization object from station locations."""
    return Regularization(station_locations=station_locations.copy())


class TestRegularizationBasics:
    """Basic initialization and text representation."""

    def test_defaults_without_station_locations(self):
        reg = Regularization()

        assert reg.model_columns is None
        assert reg.model_rows is None
        assert reg.binding_offset is None
        assert reg.reg_fn is None
        assert reg.reg_basename == "Occam2DModel"
        assert reg.model_name == "model made by mtpy.modeling.occam2d"
        assert reg.description == "simple Inversion"
        assert reg.num_param is None
        assert reg.num_free_param is None
        assert reg.statics_fn == "none"
        assert reg.prejudice_fn == "none"

    def test_init_with_station_locations_builds_objects(
        self, regularization: Regularization
    ):
        assert regularization.x_nodes is not None
        assert regularization.z_nodes is not None
        assert regularization.model_columns is not None
        assert regularization.model_rows is not None
        assert regularization.binding_offset is not None

    def test_str_contains_expected_lines(self, regularization: Regularization):
        value = str(regularization)

        assert "Regularization Parameters" in value
        assert "binding offset" in value
        assert "number layers" in value
        assert "number of parameters" in value
        assert "number of free param" in value


class TestRegularizationBuild:
    """Regularization grid building and derived values."""

    def test_build_regularization_outputs(
        self, regularization: Regularization, subtests
    ):
        with subtests.test("rows_and_columns"):
            assert len(regularization.model_rows) > 0
            assert len(regularization.model_columns) > 0
            assert len(regularization.model_rows) == len(regularization.model_columns)

        with subtests.test("parameter_counts"):
            assert regularization.num_param > 0
            assert regularization.num_free_param > 0
            assert regularization.num_free_param <= regularization.num_param

        with subtests.test("binding_offset_finite"):
            assert np.isfinite(regularization.binding_offset)

        with subtests.test("top_row_column_sum"):
            # First regularization row spans all horizontal mesh cells (+1 legacy boundary).
            assert (
                sum(regularization.model_columns[0]) == regularization.x_nodes.size + 1
            )

    def test_block_cell_centres_and_indices_shape(
        self, regularization: Regularization, subtests
    ):
        with subtests.test("cell_centres"):
            assert regularization.cell_centres.shape[0] == regularization.num_param
            assert regularization.cell_centres.shape[1] == 2

        with subtests.test("block_indices_shape"):
            assert regularization.model_block_indices.shape == (
                regularization.z_nodes.size,
                regularization.x_nodes.size,
            )

        with subtests.test("block_indices_bounds"):
            assert regularization.model_block_indices.min() == 0
            assert (
                regularization.model_block_indices.max() == regularization.num_param - 1
            )

    def test_get_num_free_params_reduces_when_fixed(
        self, regularization: Regularization
    ):
        initial_num_free = regularization.num_free_param
        regularization.mesh_values[0, 0, :] = regularization.air_key
        regularization.get_num_free_params()

        assert regularization.num_free_param < initial_num_free


class TestRegularizationSerialization:
    """Write/read regularization file behavior."""

    def test_write_regularization_file(
        self, regularization: Regularization, tmp_path, worker_id: str
    ):
        mesh_basename = f"Occam2DMesh_reg_{worker_id}"
        reg_basename = f"Occam2DModel_reg_{worker_id}"

        regularization.write_mesh_file(save_path=tmp_path, basename=mesh_basename)
        regularization.write_regularization_file(
            save_path=tmp_path,
            reg_basename=reg_basename,
        )

        assert regularization.reg_fn.exists()
        assert regularization.reg_fn.stat().st_size > 0

    def test_write_read_round_trip(
        self, regularization: Regularization, tmp_path, worker_id: str, subtests
    ):
        mesh_basename = f"Occam2DMesh_roundtrip_{worker_id}"
        reg_basename = f"Occam2DModel_roundtrip_{worker_id}"

        regularization.write_mesh_file(save_path=tmp_path, basename=mesh_basename)
        regularization.write_regularization_file(
            save_path=tmp_path,
            reg_basename=reg_basename,
        )

        reloaded = Regularization()
        reloaded.read_regularization_file(regularization.reg_fn)

        with subtests.test("binding_offset"):
            assert reloaded.binding_offset == pytest.approx(
                regularization.binding_offset
            )

        with subtests.test("row_count"):
            assert len(reloaded.model_rows) == len(regularization.model_rows)

        with subtests.test("column_count"):
            assert len(reloaded.model_columns) == len(regularization.model_columns)

        with subtests.test("rows_equal"):
            assert reloaded.model_rows == regularization.model_rows

        with subtests.test("columns_equal"):
            assert reloaded.model_columns == regularization.model_columns

        with subtests.test("mesh_filename"):
            assert reloaded.mesh_fn.name == regularization.mesh_fn.name
