# -*- coding: utf-8 -*-
"""Tests for BokehPlotBaseProfile behavior and class wiring."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.plotting


@pytest.fixture(scope="session")
def bokeh_plot_base_profile_class():
    pytest.importorskip("param")
    from mtpy.imaging.bokeh_plots import BokehPlotBaseProfile

    return BokehPlotBaseProfile


def test_profile_class_hierarchy_and_defaults(bokeh_plot_base_profile_class):
    from mtpy.imaging.bokeh_plots import BokehPlotBase

    assert issubclass(bokeh_plot_base_profile_class, BokehPlotBase)

    obj = bokeh_plot_base_profile_class(mt_data={})
    assert obj.profile_reverse is False
    assert obj.x_stretch == pytest.approx(5000)
    assert obj.y_stretch == pytest.approx(1000)
    assert obj.y_scale == "period"


def test_palette_registry_available_on_profile_base(bokeh_plot_base_profile_class):
    obj = bokeh_plot_base_profile_class(mt_data={})

    assert "magma" in obj.palette_options
    assert "magma_r" in obj.palette_options
    assert "rainbow" in obj.palette_options
    assert "rainbow_r" in obj.palette_options
    assert obj.palette_options["magma_r"] == list(
        reversed(obj.palette_options["magma"])
    )


def test_profile_plot_classes_inherit_bokeh_profile_base(bokeh_plot_base_profile_class):
    from mtpy.imaging.bokeh_plots import (
        PlotPhaseTensorPseudoSection,
        PlotResPhasePseudoSection,
    )

    assert issubclass(PlotResPhasePseudoSection, bokeh_plot_base_profile_class)
    assert issubclass(PlotPhaseTensorPseudoSection, bokeh_plot_base_profile_class)


class _DummyMT:
    def __init__(self, station, survey, offset):
        self.station = station
        self.survey = survey
        self.profile_offset = offset
        self.rotation_angle = 0


class _DummyMTData:
    def __init__(self):
        self._objects = {
            "/Surveys/s/Stations/a": _DummyMT("a", "s", 1.0),
            "/Surveys/s/Stations/b": _DummyMT("b", "s", 3.0),
        }

    def _iter_station_paths(self):
        return iter(self._objects.keys())

    def get_station(self, station_path, as_mt=False):
        if as_mt:
            return self._objects[station_path]
        raise ValueError("Expected as_mt=True in this dummy")


def test_get_mt_objects_and_offset_sign(bokeh_plot_base_profile_class):
    mt_data = _DummyMTData()
    obj = bokeh_plot_base_profile_class(mt_data=mt_data)

    mt_objects = obj._get_mt_objects()
    assert len(mt_objects) == 2

    obj.x_stretch = 2.0
    obj.profile_reverse = False
    assert obj._get_offset(mt_objects[0]) == pytest.approx(2.0)

    obj.profile_reverse = True
    assert obj._get_offset(mt_objects[0]) == pytest.approx(-2.0)


def test_rotation_angle_setter_applies_to_mt_objects(bokeh_plot_base_profile_class):
    mt_data = _DummyMTData()
    obj = bokeh_plot_base_profile_class(mt_data=mt_data)

    obj.rotation_angle = 15.0

    for mt_obj in mt_data._objects.values():
        assert mt_obj.rotation_angle == pytest.approx(15.0)
    assert obj.rotation_angle == pytest.approx(15.0)


def test_palette_from_name_falls_back_to_turbo(bokeh_plot_base_profile_class):
    obj = bokeh_plot_base_profile_class(mt_data={})

    default_palette = obj.palette_options["turbo"]
    assert obj.palette_from_name("no_such_palette") == default_palette
    assert obj.palette_from_name(None) == default_palette
    assert obj.palette_from_name("Magma") == obj.palette_options["magma"]
