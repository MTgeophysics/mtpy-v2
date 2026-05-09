# -*- coding: utf-8 -*-
"""Tests for Occam2DData.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtpy import MTData
from mtpy.modeling.occam2d import Occam2DData


@pytest.fixture
def occam_dataframe() -> pd.DataFrame:
    """Create a small synthetic dataframe for Occam2D write/read tests."""
    rows = []
    stations = [("S01", 0.0), ("S02", 1000.0), ("S03", 2100.0)]
    frequencies = [1.0, 0.25, 0.05]

    for station, offset in stations:
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
                    "res_xy": 10.0,
                    "res_yx": 25.0,
                    "phase_xy": 35.0,
                    "phase_yx": -120.0,
                    "t_zy": 0.10 + 0.20j,
                    "res_xy_model_error": 1.0,
                    "res_yx_model_error": 2.0,
                    "phase_xy_model_error": 3.0,
                    "phase_yx_model_error": 4.0,
                    "t_zy_model_error": 0.05,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def occam_data(occam_dataframe: pd.DataFrame) -> Occam2DData:
    """Build an Occam2DData instance with all components enabled."""
    return Occam2DData(
        dataframe=occam_dataframe,
        model_mode="1",
        profile_origin=(500000.0, 4000000.0),
        profile_angle=15.0,
        geoelectric_strike=20.0,
    )


@pytest.fixture
def occam_data_file(tmp_path, worker_id: str, occam_data: Occam2DData):
    """Write a worker-unique Occam2D data file for read/serialization tests."""
    data_fn = tmp_path / f"occam2d_data_{worker_id}.dat"
    occam_data.write_data_file(data_fn)
    return data_fn


class TestOccam2DDataBasic:
    """Basic object and property behavior."""

    def test_defaults(self):
        ocd = Occam2DData()

        assert ocd.model_mode == "1"
        assert ocd.error_type == "floor"
        assert ocd.profile_origin == (0, 0)
        assert ocd.profile_angle == 0
        assert ocd.geoelectric_strike == 0
        assert ocd.n_stations == 0
        assert ocd.n_frequencies == 0

    def test_data_filename_property(self):
        ocd = Occam2DData()
        assert ocd.data_filename is None

        ocd.data_filename = "test_data.dat"
        assert str(ocd.data_filename).endswith("test_data.dat")

    def test_str_representation(self, occam_data: Occam2DData):
        value = str(occam_data)

        assert "Occam2D Data" in value
        assert "Number of Stations" in value
        assert "Number of Frequencies" in value
        assert "Number of data" in value


class TestOccam2DDataHelpers:
    """Helper method behavior and parsing logic."""

    def test_get_model_locations(self):
        ocd = Occam2DData()
        east, north = ocd._get_model_locations(1000.0, 30.0)

        assert east == pytest.approx(866.0254037844386)
        assert north == pytest.approx(500.0)

    def test_read_title_string_parsing(self, subtests):
        ocd = Occam2DData()
        ocd._read_title_string(
            "Example Title, Profile=12.5 deg, Strike=20.0 deg, Origin=(10.0, 20.0)"
        )

        with subtests.test("title"):
            assert ocd.title == "Example Title"

        with subtests.test("profile_angle"):
            assert ocd.profile_angle == pytest.approx(12.5)

        with subtests.test("geoelectric_strike"):
            assert ocd.geoelectric_strike == pytest.approx(20.0)

        with subtests.test("profile_origin"):
            assert ocd.profile_origin == (10.0, 20.0)

    @pytest.mark.parametrize(
        "inv_list,expected_mode",
        [
            ([1, 2, 3, 4, 5, 6], "log_all"),
            ([9, 2, 10, 6], "te_tm"),
            ([3, 4], "tip"),
            ([9, 2], "te"),
        ],
    )
    def test_match_inv_list_to_mode(self, inv_list, expected_mode):
        ocd = Occam2DData()
        assert ocd._match_inv_list_to_mode(inv_list) == expected_mode


class TestOccam2DDataSerialization:
    """Read/write serialization and error handling."""

    def test_write_data_file_creates_file(self, occam_data_file):
        assert occam_data_file.exists()
        assert occam_data_file.stat().st_size > 0

    def test_write_then_read_round_trip(
        self, occam_dataframe, occam_data_file, subtests
    ):
        reloaded = Occam2DData()
        reloaded.read_data_file(occam_data_file)

        input_sorted = occam_dataframe.sort_values(["station", "period"]).reset_index(
            drop=True
        )
        output_sorted = reloaded.dataframe.sort_values(
            ["station", "period"]
        ).reset_index(drop=True)

        with subtests.test("sizes"):
            assert reloaded.n_stations == input_sorted.station.nunique()
            assert reloaded.n_frequencies == input_sorted.frequency.nunique()

        with subtests.test("model_mode_detected"):
            assert reloaded.model_mode == "log_all"

        atol_by_col = {
            # Resistivity values are written in log10 with fixed precision,
            # then reconstructed, so a small round-off is expected.
            "res_xy": 5e-3,
            "res_yx": 5e-3,
            "phase_xy": 1e-6,
            "phase_yx": 1e-6,
        }
        for col in ["res_xy", "res_yx", "phase_xy", "phase_yx"]:
            with subtests.test(f"column_{col}"):
                assert np.allclose(
                    output_sorted[col].to_numpy(),
                    input_sorted[col].to_numpy(),
                    atol=atol_by_col[col],
                )

        with subtests.test("tipper_complex"):
            assert np.allclose(
                output_sorted["t_zy"].to_numpy(),
                input_sorted["t_zy"].to_numpy(),
            )

        with subtests.test("errors"):
            atol_by_error_col = {
                "res_xy_model_error": 5e-4,
                "res_yx_model_error": 5e-4,
                "phase_xy_model_error": 1e-6,
                "phase_yx_model_error": 1e-6,
            }
            for col in [
                "res_xy_model_error",
                "res_yx_model_error",
                "phase_xy_model_error",
                "phase_yx_model_error",
            ]:
                assert np.allclose(
                    output_sorted[col].to_numpy(),
                    input_sorted[col].to_numpy(),
                    atol=atol_by_error_col[col],
                )

    def test_read_data_file_missing_raises(self, tmp_path):
        ocd = Occam2DData()
        missing_fn = tmp_path / "does_not_exist.dat"

        with pytest.raises(ValueError, match="Could not find"):
            ocd.read_data_file(missing_fn)

    def test_get_data_block_empty_dataframe_raises(self):
        ocd = Occam2DData()

        with pytest.raises(ValueError, match="empty dataframe"):
            ocd._get_data_block()


class TestMTDataOccam2DIntegration:
    """Integration tests for MTData to/from Occam2D conversion."""

    def test_mtdata_to_occam2d_writes_file(self, mt_from_edi, tmp_path, worker_id):
        md = MTData()
        md.add_station(mt_from_edi)

        data_fn = tmp_path / f"mtdata_to_occam2d_{worker_id}.dat"
        occam_obj = md.to_occam2d(data_filename=data_fn, model_mode="1")

        assert isinstance(occam_obj, Occam2DData)
        assert data_fn.exists()
        assert data_fn.stat().st_size > 0
        assert occam_obj.n_stations == md.n_stations

    def test_mtdata_occam2d_round_trip(
        self, mt_from_edi, tmp_path, worker_id, subtests
    ):
        md_in = MTData()
        md_in.add_station(mt_from_edi)

        data_fn = tmp_path / f"mtdata_occam2d_roundtrip_{worker_id}.dat"
        md_in.to_occam2d(data_filename=data_fn, model_mode="1")

        md_out = MTData()
        md_out.from_occam2d(data_fn)

        in_df = (
            md_in.to_dataframe()
            .sort_values(["station", "period"])
            .reset_index(drop=True)
        )
        out_df = (
            md_out.to_dataframe()
            .sort_values(["station", "period"])
            .reset_index(drop=True)
        )

        with subtests.test("station_count"):
            assert md_out.n_stations == md_in.n_stations

        with subtests.test("survey_assignment_default_data"):
            assert md_out.survey_ids == ["data"]

        with subtests.test("required_columns_present"):
            for col in ["res_xy", "res_yx", "phase_xy", "phase_yx", "t_zy"]:
                assert col in out_df.columns

        with subtests.test("resistivity_round_trip"):
            # Resistivity goes through log10 formatting in Occam2D files.
            assert np.allclose(
                out_df["res_xy"].to_numpy(),
                in_df["res_xy"].to_numpy(),
                rtol=5e-4,
                atol=1e-2,
            )
            assert np.allclose(
                out_df["res_yx"].to_numpy(),
                in_df["res_yx"].to_numpy(),
                rtol=5e-4,
                atol=1e-2,
            )

        with subtests.test("phase_and_tipper_round_trip"):
            assert np.allclose(
                out_df["phase_xy"].to_numpy(),
                in_df["phase_xy"].to_numpy(),
                atol=1e-3,
            )
            assert np.allclose(
                out_df["phase_yx"].to_numpy(),
                in_df["phase_yx"].to_numpy(),
                atol=1e-3,
            )
            assert np.allclose(
                out_df["t_zy"].to_numpy(),
                in_df["t_zy"].to_numpy(),
                atol=1e-4,
            )

    def test_mtdata_from_occam2d_model_file_type(
        self, mt_from_edi, tmp_path, worker_id
    ):
        md_in = MTData()
        md_in.add_station(mt_from_edi)

        data_fn = tmp_path / f"mtdata_occam2d_model_{worker_id}.dat"
        md_in.to_occam2d(data_filename=data_fn, model_mode="1")

        md_model = MTData()
        md_model.from_occam2d(data_fn, file_type="model")

        assert md_model.n_stations == md_in.n_stations
        assert md_model.survey_ids == ["model"]
