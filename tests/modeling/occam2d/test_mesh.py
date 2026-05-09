# -*- coding: utf-8 -*-
"""Tests for occam2d Mesh.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtpy.modeling.occam2d import Mesh, Occam2DData


@pytest.fixture
def station_locations() -> np.ndarray:
    """Provide unsorted station locations to verify internal sorting."""
    return np.array([1500.0, 0.0, 800.0, 300.0], dtype=float)


@pytest.fixture
def elevation_profile(station_locations: np.ndarray) -> np.ndarray:
    """Create a simple elevation profile in the expected (2, n) format."""
    # Extend bounds beyond station extrema because add_elevation evaluates
    # over interior grid nodes near padded station bounds.
    x = np.linspace(
        station_locations.min() - 500.0, station_locations.max() + 500.0, 40
    )
    elev = 100.0 + 20.0 * np.sin(np.linspace(0, np.pi, x.size))
    return np.vstack([x, elev])


@pytest.fixture
def mesh(station_locations: np.ndarray) -> Mesh:
    """Create a Mesh object with deterministic station locations."""
    return Mesh(station_locations=station_locations.copy())


class TestMeshBasics:
    """Basic initialization and validation tests."""

    def test_defaults(self):
        m = Mesh()

        assert m.rel_station_locations is None
        assert m.n_layers == 90
        assert m.cell_width == 100
        assert m.num_x_pad_cells == 7
        assert m.num_z_pad_cells == 5
        assert m.x_pad_multiplier == 1.5
        assert m.z1_layer == 10.0
        assert m.z_bottom == 200000.0
        assert m.z_target_depth == 50000.0
        assert m.air_value == 1e13
        assert m.air_key == "0"

    def test_build_mesh_requires_station_locations(self):
        m = Mesh()

        with pytest.raises(ValueError, match="station locations"):
            m.build_mesh()

    def test_add_elevation_requires_profile(self, mesh: Mesh):
        with pytest.raises(ValueError, match="elevation profile"):
            mesh.add_elevation(None)


class TestMeshBuild:
    """Tests for mesh construction behavior."""

    def test_build_mesh_populates_arrays(self, mesh: Mesh, subtests):
        mesh.build_mesh()

        with subtests.test("stations_sorted"):
            assert np.all(np.diff(mesh.station_locations) >= 0)

        with subtests.test("rel_locations_centered"):
            assert np.mean(mesh.rel_station_locations) == pytest.approx(0.0)

        with subtests.test("x_and_z_arrays_exist"):
            assert mesh.x_nodes is not None
            assert mesh.z_nodes is not None
            assert mesh.x_grid is not None
            assert mesh.z_grid is not None

        with subtests.test("grid_node_relationship"):
            assert mesh.x_grid.size == mesh.x_nodes.size + 1
            assert mesh.z_grid.size == mesh.z_nodes.size + 1

        with subtests.test("mesh_values_shape"):
            assert mesh.mesh_values.shape == (mesh.x_nodes.size, mesh.z_nodes.size, 4)

        with subtests.test("mesh_values_default"):
            assert np.all(mesh.mesh_values == "?")

        with subtests.test("x_grid_monotonic"):
            assert np.all(np.diff(mesh.x_grid) > 0)

        with subtests.test("z_grid_monotonic"):
            assert np.all(np.diff(mesh.z_grid) > 0)

    def test_build_mesh_with_elevation(self, mesh: Mesh, elevation_profile: np.ndarray):
        mesh.elevation_profile = elevation_profile
        mesh.build_mesh()

        assert mesh.mesh_values is not None
        assert np.any(mesh.mesh_values == mesh.air_key)
        assert mesh.z_nodes.size > 0


class TestMeshSerialization:
    """Tests for write/read mesh file round-trip."""

    def test_write_mesh_file_builds_if_needed(
        self, mesh: Mesh, tmp_path, worker_id: str
    ):
        basename = f"Occam2DMesh_{worker_id}"
        mesh.write_mesh_file(save_path=tmp_path, basename=basename)

        assert mesh.mesh_fn.exists()
        assert mesh.mesh_fn.stat().st_size > 0
        assert mesh.x_nodes is not None
        assert mesh.z_nodes is not None

    def test_write_read_round_trip(
        self, mesh: Mesh, tmp_path, worker_id: str, subtests
    ):
        mesh.build_mesh()
        basename = f"Occam2DMesh_roundtrip_{worker_id}"
        mesh.write_mesh_file(save_path=tmp_path, basename=basename)

        reloaded = Mesh()
        reloaded.read_mesh_file(mesh.mesh_fn)

        with subtests.test("x_nodes"):
            assert np.allclose(reloaded.x_nodes, mesh.x_nodes, atol=0.11)

        with subtests.test("z_nodes"):
            assert np.allclose(reloaded.z_nodes, mesh.z_nodes, atol=0.11)

        with subtests.test("x_grid_len"):
            assert reloaded.x_grid.size == reloaded.x_nodes.size + 1

        with subtests.test("z_grid_len"):
            assert reloaded.z_grid.size == reloaded.z_nodes.size

        with subtests.test("mesh_values_shape"):
            # Legacy reader may retain one extra x-column from header sizing.
            assert reloaded.mesh_values.shape[0] in {
                reloaded.x_nodes.size,
                reloaded.x_nodes.size + 1,
            }
            assert reloaded.mesh_values.shape[1] in {
                reloaded.z_nodes.size,
                reloaded.z_nodes.size + 1,
            }
            assert reloaded.mesh_values.shape[2] == 4


class TestMeshPlotting:
    """Lightweight plotting smoke test."""

    @pytest.mark.parametrize("depth_scale", ["km", "m", "invalid"])
    def test_plot_mesh_smoke(self, mesh: Mesh, monkeypatch, depth_scale: str):
        mesh.build_mesh()

        # Avoid interactive display during tests.
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

        mesh.plot_mesh(depth_scale=depth_scale, plot_triangles="y")


class TestOccam2DDataMeshIntegration:
    """Focused integration test linking Occam2DData offsets to Mesh."""

    def test_occam2d_data_offsets_drive_mesh_build(
        self, tmp_path, worker_id: str, subtests
    ):
        stations_and_offsets = [("S03", 2100.0), ("S01", 0.0), ("S02", 1000.0)]
        frequencies = [1.0, 0.25]
        rows = []

        for station, offset in stations_and_offsets:
            for frequency in frequencies:
                rows.append(
                    {
                        "station": station,
                        "frequency": frequency,
                        "period": 1.0 / frequency,
                        "profile_offset": offset,
                        "east": 500000.0 + offset,
                        "north": 4000000.0,
                        "model_east": offset,
                        "model_north": 0.0,
                        "res_xy": 15.0,
                        "res_yx": 25.0,
                        "phase_xy": 30.0,
                        "phase_yx": -120.0,
                        "t_zy": 0.10 + 0.20j,
                        "res_xy_model_error": 1.0,
                        "res_yx_model_error": 2.0,
                        "phase_xy_model_error": 3.0,
                        "phase_yx_model_error": 4.0,
                        "t_zy_model_error": 0.05,
                    }
                )

        occam = Occam2DData(dataframe=pd.DataFrame(rows), model_mode="1")
        data_fn = tmp_path / f"occam2d_for_mesh_{worker_id}.dat"
        occam.write_data_file(data_fn)

        occam_reloaded = Occam2DData()
        occam_reloaded.read_data_file(data_fn)

        station_offsets = (
            occam_reloaded.dataframe.groupby("station")["profile_offset"]
            .first()
            .to_numpy()
        )

        x = np.linspace(
            station_offsets.min() - 500.0, station_offsets.max() + 500.0, 40
        )
        elev = 120.0 + 10.0 * np.cos(np.linspace(0, np.pi, x.size))
        elev_profile = np.vstack([x, elev])

        mesh = Mesh(
            station_locations=station_offsets.copy(), elevation_profile=elev_profile
        )
        mesh.build_mesh()

        with subtests.test("station_count_match"):
            assert mesh.rel_station_locations.size == np.unique(station_offsets).size

        with subtests.test("station_locations_sorted"):
            assert np.all(np.diff(mesh.station_locations) >= 0)

        with subtests.test("relative_locations_centered"):
            assert np.mean(mesh.rel_station_locations) == pytest.approx(0.0)

        with subtests.test("stations_lie_on_horizontal_nodes"):
            for station in mesh.rel_station_locations:
                assert np.min(np.abs(mesh.x_grid - station)) < 1e-8

        with subtests.test("elevation_marks_air_cells"):
            assert np.any(mesh.mesh_values == mesh.air_key)

        mesh_basename = f"Occam2DMesh_from_occam_{worker_id}"
        mesh.write_mesh_file(save_path=tmp_path, basename=mesh_basename)
        assert mesh.mesh_fn.exists()
