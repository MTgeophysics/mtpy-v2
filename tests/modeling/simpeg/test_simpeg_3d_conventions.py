# -*- coding: utf-8 -*-
"""
Coordinate conventions of Simpeg3DData.

mtpy (EDI) components are x = north, y = east, z = down; SimPEG's are
x = east, y = north, z = up. Swapping x and y swaps the indices of every
element, and flipping z negates the tipper, H_z / H_h. These tests need no
data files.
"""

# =============================================================================
# Imports
# =============================================================================
import discretize
import numpy as np
import pandas as pd
import pytest
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem

from mtpy.modeling.simpeg.data_3d import Simpeg3DData

# =============================================================================

COMPONENTS = ["z_xx", "z_xy", "z_yx", "z_yy", "t_zx", "t_zy"]


def _dataframe(stations=((0.0, 0.0),), periods=(0.01, 0.1)):
    """Distinct values per component, so a swapped or unsigned one shows."""
    rows = []
    for index, (east, north) in enumerate(stations):
        for period in periods:
            row = {
                "station": f"s{index}",
                "period": period,
                "east": east,
                "north": north,
                "elevation": 0.0,
                "model_east": east,
                "model_north": north,
                "model_elevation": 0.0,
            }
            for k, comp in enumerate(COMPONENTS):
                row[comp] = complex(k + 1, 10 * (k + 1))
            rows.append(row)
    return pd.DataFrame(rows)


def _receivers(simpeg_data):
    return simpeg_data.survey.source_list[0].receiver_list


def test_tipper_receivers_are_tipper():
    """Point3DTipper raises NotImplementedError in current SimPEG."""
    receivers = _receivers(Simpeg3DData(_dataframe()))
    tippers = [rx for rx in receivers if isinstance(rx, nsem.receivers.Tipper)]
    assert len(tippers) == 4
    assert sorted(rx.orientation for rx in tippers) == ["zx", "zx", "zy", "zy"]


@pytest.mark.parametrize(
    ("component", "orientation", "receiver_type"),
    [
        ("z_xx", "yy", nsem.receivers.Impedance),
        ("z_xy", "yx", nsem.receivers.Impedance),
        ("z_yx", "xy", nsem.receivers.Impedance),
        ("z_yy", "xx", nsem.receivers.Impedance),
        ("t_zx", "zy", nsem.receivers.Tipper),
        ("t_zy", "zx", nsem.receivers.Tipper),
    ],
)
def test_single_component_survey_measures_that_component(
    component, orientation, receiver_type
):
    """With one component selected, the survey's receivers must be the SimPEG
    element that component maps to, as ``to_rec_array`` labels its data."""
    kwargs = {f"invert_{c}": c == component for c in COMPONENTS}
    simpeg_data = Simpeg3DData(_dataframe(), **kwargs)
    receivers = _receivers(simpeg_data)

    assert [rx.orientation for rx in receivers] == [orientation, orientation]
    assert all(isinstance(rx, receiver_type) for rx in receivers)
    assert f"{component[0]}{orientation}" in simpeg_data.to_rec_array().dtype.names


def test_rec_array_swaps_impedance_and_negates_tipper():
    df = _dataframe()
    rec = Simpeg3DData(df).to_rec_array()

    np.testing.assert_array_equal(rec["zyx"], df["z_xy"])
    np.testing.assert_array_equal(rec["zxy"], df["z_yx"])
    np.testing.assert_array_equal(rec["zyy"], df["z_xx"])
    np.testing.assert_array_equal(rec["zxx"], df["z_yy"])
    np.testing.assert_array_equal(rec["tzy"], -df["t_zx"])
    np.testing.assert_array_equal(rec["tzx"], -df["t_zy"])


def test_rec_array_leaves_the_dataframe_unchanged():
    df = _dataframe()
    before = df.copy()
    Simpeg3DData(df).to_rec_array()
    pd.testing.assert_frame_equal(df, before)


def test_data_object_carries_the_negated_tipper():
    df = _dataframe()
    data = Simpeg3DData(df).get_simpeg_data_object()
    src = data.survey.source_list[0]
    tzx_real = next(
        rx
        for rx in src.receiver_list
        if isinstance(rx, nsem.receivers.Tipper)
        and rx.orientation == "zx"
        and rx.component == "real"
    )
    np.testing.assert_allclose(data[src, tzx_real], -df["t_zy"].to_numpy()[0].real)


def test_tipper_sign_matches_simpeg_physics():
    """A conductor to the east of the station. In mtpy's convention the real
    induction arrow (Re T_zx, Re T_zy) points away from a conductor, so
    Re T_zy < 0. The SimPEG receiver for mtpy t_zy is its tzx, and mtpy t_zy is
    minus that value."""
    cs, n_core, n_pad = 100.0, 10, 5
    h = [(cs, n_pad, -1.5), (cs, n_core), (cs, n_pad, 1.5)]
    mesh = discretize.TensorMesh([h, h, h], "CCC")
    x, y, z = mesh.cell_centers.T
    sigma = np.where(z > 0, 1e-8, 1e-2)
    sigma_1d = np.where(mesh.cell_centers_z > 0, 1e-8, 1e-2)
    in_block = (x > 150) & (x < 450) & (np.abs(y) < 300) & (z < 0) & (z > -400)
    sigma[in_block] = 1.0

    kwargs = {f"invert_{c}": c == "t_zy" for c in COMPONENTS}
    df = _dataframe(stations=((-150.0, 0.0),), periods=(0.1,))
    simpeg_data = Simpeg3DData(df, **kwargs)
    survey = simpeg_data.survey
    real_rx = survey.source_list[0].receiver_list[0]
    assert real_rx.orientation == "zx" and real_rx.component == "real"

    simulation = nsem.simulation.Simulation3DPrimarySecondary(
        mesh,
        survey=survey,
        sigmaMap=maps.IdentityMap(mesh),
        sigmaPrimary=sigma_1d,
        forward_only=True,
    )
    d = simulation.dpred(sigma)
    mtpy_t_zy_real = -d[0]
    assert mtpy_t_zy_real < -0.01
