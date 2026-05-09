# -*- coding: utf-8 -*-
"""Tests for occam2d Startup.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mtpy.modeling.occam2d.startup import Startup


@pytest.fixture
def startup(tmp_path) -> Startup:
    """Create a Startup object with minimal valid inputs set."""
    s = Startup()
    s.save_path = tmp_path
    s.data_fn = tmp_path / "OccamDataFile.dat"
    s.model_fn = tmp_path / "Occam2DModel"
    s.param_count = 8
    return s


class TestStartupBasics:
    """Basic initialization and defaults."""

    def test_defaults(self):
        s = Startup()

        assert s.startup_basename == "Occam2DStartup"
        assert s.startup_fn is None
        assert s.model_fn is None
        assert s.data_fn is None
        assert s.format == "OCCAMITER_FLEX"
        assert s.description == "startup created by mtpy"
        assert s.iterations_to_run == 20
        assert s.roughness_type == 1
        assert s.target_misfit == 1.0
        assert s.diagonal_penalties == 0
        assert s.stepsize_count == 8
        assert s.model_limits is None
        assert s.model_value_steps is None
        assert s.debug_level == 1
        assert s.iteration == 0
        assert s.lagrange_value == 5.0
        assert s.roughness_value == 1e10
        assert s.misfit_value == 1000
        assert s.misfit_reached == 0
        assert s.param_count is None
        assert s.resistivity_start == 2
        assert s.model_values is None


class TestStartupValidation:
    """Input validation tests for required attributes."""

    def test_write_requires_data_fn(self):
        s = Startup()
        s.model_fn = Path("Occam2DModel")
        s.param_count = 4

        with pytest.raises(ValueError, match="data file name"):
            s.write_startup_file()

    def test_write_requires_model_fn(self, tmp_path):
        s = Startup()
        s.data_fn = tmp_path / "OccamDataFile.dat"
        s.param_count = 4

        with pytest.raises(ValueError, match="model/regularization file name"):
            s.write_startup_file()

    def test_write_requires_param_count(self, tmp_path):
        s = Startup()
        s.data_fn = tmp_path / "OccamDataFile.dat"
        s.model_fn = tmp_path / "Occam2DModel"

        with pytest.raises(ValueError, match="number of model parameters"):
            s.write_startup_file()

    def test_model_values_length_mismatch_raises(self, startup: Startup):
        startup.model_values = np.ones(3)

        with pytest.raises(ValueError, match="param count"):
            startup.write_startup_file()


class TestStartupWriting:
    """Startup file write behavior and output structure."""

    def test_write_startup_file_creates_file(self, startup: Startup, worker_id: str):
        startup.write_startup_file(startup_basename=f"Occam2DStartup_{worker_id}")

        assert startup.startup_fn.exists()
        assert startup.startup_fn.stat().st_size > 0

    def test_write_populates_default_model_values(self, startup: Startup):
        startup.resistivity_start = 3.25
        startup.write_startup_file()

        assert startup.model_values is not None
        assert startup.model_values.shape == (startup.param_count,)
        assert np.allclose(startup.model_values, 3.25)

    def test_write_uses_explicit_model_values(self, startup: Startup):
        values = np.linspace(1.0, 2.4, startup.param_count)
        startup.model_values = values
        startup.write_startup_file()

        assert np.allclose(startup.model_values, values)

    def test_write_with_external_paths_keeps_full_paths(self, tmp_path, worker_id: str):
        save_dir = tmp_path / f"save_{worker_id}"
        data_dir = tmp_path / f"data_{worker_id}"
        model_dir = tmp_path / f"model_{worker_id}"
        save_dir.mkdir()
        data_dir.mkdir()
        model_dir.mkdir()

        s = Startup()
        s.save_path = save_dir
        s.data_fn = data_dir / "OccamDataFile.dat"
        s.model_fn = model_dir / "Occam2DModel"
        s.param_count = 4
        s.write_startup_file()

        contents = s.startup_fn.read_text()
        assert str(s.data_fn) in contents
        assert str(s.model_fn) in contents

    def test_write_with_local_paths_uses_filenames(self, tmp_path, worker_id: str):
        s = Startup()
        s.save_path = tmp_path
        s.data_fn = tmp_path / f"OccamDataFile_{worker_id}.dat"
        s.model_fn = tmp_path / f"Occam2DModel_{worker_id}"
        s.param_count = 4
        s.write_startup_file()

        contents = s.startup_fn.read_text()
        assert s.data_fn.name in contents
        assert s.model_fn.name in contents

    def test_startup_file_contains_header_and_values(self, startup: Startup, subtests):
        startup.write_startup_file()

        lines = startup.startup_fn.read_text().splitlines()

        with subtests.test("header_keys"):
            expected_keys = [
                "Format:",
                "Description:",
                "Model File:",
                "Data File:",
                "Date/Time:",
                "Iterations to run:",
                "Target Misfit:",
                "Roughness Type:",
                "Diagonal Penalties:",
                "Stepsize Cut Count:",
                "Debug Level:",
                "Iteration:",
                "Lagrange Value:",
                "Roughness Value:",
                "Misfit Value:",
                "Misfit Reached:",
                "Param Count:",
            ]
            for key in expected_keys:
                assert any(line.startswith(key) for line in lines)

        with subtests.test("model_values_written"):
            # Last lines contain numeric startup model values.
            value_lines = [
                line
                for line in lines
                if line.strip() and line.strip()[0] in "-0123456789"
            ]
            assert len(value_lines) > 0
