# -*- coding: utf-8 -*-
"""Tests for Simpeg1D inversion recipe."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from mt_metadata import TF_EDI_EMPOWER

from mtpy import MT
from mtpy.modeling.simpeg.recipes.inversion_1d import Simpeg1D


@pytest.fixture(scope="module")
def mt_dataframe() -> pd.DataFrame:
    """Load a real MT EDI and expose a reusable DataFrame for tests."""
    mt_obj = MT(TF_EDI_EMPOWER)
    mt_obj.read()
    mt_obj.compute_model_z_errors()
    return mt_obj.to_dataframe(impedance_units="mt").dataframe.copy(deep=True)


@pytest.fixture
def simpeg_1d_factory(mt_dataframe):
    """Build fresh Simpeg1D objects per test to keep tests parallel-safe."""

    def _build(**kwargs) -> Simpeg1D:
        return Simpeg1D(mt_dataframe=mt_dataframe.copy(deep=True), **kwargs)

    return _build


@pytest.mark.unit
def test_mode_builds_sub_dataframe(simpeg_1d_factory, subtests):
    """All supported modes should build non-empty inversion data."""
    for mode in ["te", "tm", "det"]:
        with subtests.test(mode=mode):
            simpeg_1d = simpeg_1d_factory(mode=mode)

            assert simpeg_1d.mode == mode
            assert not simpeg_1d._sub_df.empty
            assert list(simpeg_1d._sub_df.columns) == [
                "frequency",
                "res",
                "res_error",
                "phase",
                "phase_error",
            ]


@pytest.mark.unit
def test_mode_data_phase_is_in_inversion_branch(simpeg_1d_factory, subtests):
    """Prepared phase data should be in the expected SimPEG inversion branch."""
    for mode in ["te", "tm", "det"]:
        with subtests.test(mode=mode):
            simpeg_1d = simpeg_1d_factory(mode=mode)
            phase = simpeg_1d._sub_df["phase"].to_numpy()
            assert np.all((phase >= -180.0) & (phase <= -90.0))


@pytest.mark.unit
def test_invalid_mode_raises(simpeg_1d_factory):
    """Invalid modes should fail fast with ValueError."""
    with pytest.raises(ValueError, match="not in accetable modes"):
        simpeg_1d_factory(mode="bad_mode")


@pytest.mark.unit
def test_data_and_error_vector_lengths(simpeg_1d_factory):
    """Flattened data and error vectors should track 2 values per frequency."""
    simpeg_1d = simpeg_1d_factory(mode="te")

    assert simpeg_1d.data.size == 2 * simpeg_1d.frequencies.size
    assert simpeg_1d.data_error.size == 2 * simpeg_1d.frequencies.size


@pytest.mark.unit
def test_phase_for_plotting_conversion(simpeg_1d_factory):
    """Plotting helper should map inversion-phase branch to first quadrant."""
    simpeg_1d = simpeg_1d_factory(mode="det")

    phase_in = np.array([-180.0, -135.0, -90.0, 20.0])
    phase_out = simpeg_1d._phase_for_plotting(phase_in)

    assert np.allclose(phase_out, np.array([0.0, 45.0, 90.0, 20.0]))


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="SimPEG default Pardiso solver crashes on Windows in this environment",
)
def test_run_fixed_layer_inversion_te_smoke(simpeg_1d_factory):
    """TE mode should run a short inversion and produce per-iteration output."""
    simpeg_1d = simpeg_1d_factory(mode="te", n_layers=12)

    simpeg_1d.run_fixed_layer_inversion(
        maxIter=2,
        maxIterCG=2,
        coolingRate=1,
        chi_factor=1,
    )

    assert simpeg_1d.output_dict is not None
    assert len(simpeg_1d.output_dict) > 0
