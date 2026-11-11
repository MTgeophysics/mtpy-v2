"""Tests for preparing phase tensor map data."""

import numpy as np
import pytest

from mtpy import MT
from mtpy.core import MTData, Tipper
from mtpy.core.transfer_function import PhaseTensor


def make_tree():
    mt = MT(station="st01", survey="survey_a", period=np.array([0.01, 1.0, 100.0]))
    mt.latitude = 38.8
    mt.longitude = -118.2
    z = np.array([[0.2 - 1j, 3 + 4j], [-10 - 2j, -1 + 2j]])
    t = np.array([[0.2 - 0.1j, 0.3 - 0.2j]])
    mt.impedance = np.tile(z, (3, 1, 1))
    mt.tipper = np.tile(t, (3, 1, 1))
    mt.impedance_model_error = np.full((3, 2, 2), 0.2)
    mt.tipper_model_error = np.full((3, 1, 2), 0.02)
    tree = MTData()
    tree.add_station(mt)
    return tree, mt


def test_map_period_grid_covers_range_with_four_intervals_per_decade():
    tree, _ = make_tree()
    grid = tree.get_phase_tensor_map_periods()
    assert len(grid) == 17
    np.testing.assert_allclose(grid[[0, -1]], [0.01, 100])
    np.testing.assert_allclose(np.diff(np.log10(grid)), 0.25)


def test_map_period_grid_rejects_empty_data():
    with pytest.raises(ValueError, match="positive periods"):
        MTData().get_phase_tensor_map_periods()


def test_map_period_grid_handles_one_period_on_a_decade():
    mt = MT(station="st01", survey="survey_a", period=np.array([1.0]))
    mt.impedance = np.array([[[0.2 - 1j, 3 + 4j], [-10 - 2j, -1 + 2j]]])
    mt.tipper = np.array([[[0.2 - 0.1j, 0.3 - 0.2j]]])
    tree = MTData()
    tree.add_station(mt)
    np.testing.assert_array_equal(tree.get_phase_tensor_map_periods(), [1.0])
    result = tree.to_phase_tensor_map_dataframe()
    assert np.isfinite(result.phimin.iloc[0])
    assert np.isfinite(result.mag_real.iloc[0])


def test_map_dataframe_preserves_surveys_and_per_station_coverage():
    tree, mt = make_tree()
    other = MT(station="st01", survey="survey_b", period=np.array([1.0, 10.0]))
    other.latitude = 40
    other.longitude = -120
    other.impedance = np.tile(mt.Z.z[0], (2, 1, 1))
    tree.add_station(other)

    frame = tree.to_phase_tensor_map_dataframe(np.array([0.1, 1.0, 10.0, 100.0]))
    assert len(frame) == 8
    a = frame[frame.survey == "survey_a"]
    b = frame[frame.survey == "survey_b"]
    assert a.phimin.notna().all()
    assert b.phimin.notna().tolist() == [False, True, True, False]
    assert b.mag_real.isna().all()
    assert a.station_path.iloc[0] != b.station_path.iloc[0]


def test_map_dataframe_interpolates_once_and_matches_transfer_functions(monkeypatch):
    tree, mt = make_tree()
    calls = []
    interpolate = tree.interpolate

    def wrapped(*args, **kwargs):
        calls.append(args[0].copy())
        return interpolate(*args, **kwargs)

    monkeypatch.setattr(tree, "interpolate", wrapped)
    periods = np.array([0.001, 0.01, 0.1, 1.0, 100.0, 1000.0])
    result = tree.to_phase_tensor_map_dataframe(periods)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], periods)
    np.testing.assert_array_equal(result.period, periods)
    assert result.station.eq("st01").all()
    assert result.survey.eq("survey_a").all()
    assert result.latitude.eq(38.8).all()
    assert result.longitude.eq(-118.2).all()
    assert result.loc[[0, 5], ["phimin", "mag_real"]].isna().all().all()

    expected_pt = PhaseTensor(z=mt.Z.z[:1])
    expected_tip = Tipper(tipper=mt.Tipper.tipper[:1])
    for row in result.iloc[1:5].itertuples():
        assert row.phimin == pytest.approx(expected_pt.phimin[0])
        assert row.phimax == pytest.approx(expected_pt.phimax[0])
        assert row.azimuth == pytest.approx(expected_pt.azimuth[0])
        assert row.beta == pytest.approx(expected_pt.beta[0])
        assert row.det == pytest.approx(expected_pt.det[0])
        assert row.mag_real == pytest.approx(expected_tip.mag_real[0])
        assert row.angle_imag == pytest.approx(expected_tip.angle_imag[0])


def test_map_dataframe_computes_phase_tensor_after_interpolation():
    tree, mt = make_tree()
    z = mt.impedance.values.copy()
    z[1] *= 1 + 0.7j
    z[2] *= 1 - 0.4j
    mt.impedance = z
    tree.add_station(mt, overwrite=True)

    periods = np.array([0.1, 10.0])
    expected_tree = tree.interpolate(periods, inplace=False, bounds_error=False)
    expected_mt = expected_tree.get_station(
        list(expected_tree._iter_station_paths())[0], as_mt=True
    )
    expected_pt = PhaseTensor(z=expected_mt.Z.z)
    frame = tree.to_phase_tensor_map_dataframe(periods)
    np.testing.assert_allclose(frame.phimin, expected_pt.phimin)
    np.testing.assert_allclose(frame.phimax, expected_pt.phimax)
    np.testing.assert_allclose(frame.azimuth, expected_pt.azimuth)


@pytest.mark.parametrize("periods", [[], [0], [-1, 1], [1, 1], [2, 1], [np.nan]])
def test_map_dataframe_rejects_invalid_periods(periods):
    tree, _ = make_tree()
    with pytest.raises(ValueError, match="Map periods"):
        tree.to_phase_tensor_map_dataframe(periods)
