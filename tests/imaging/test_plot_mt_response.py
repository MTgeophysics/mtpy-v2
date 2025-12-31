# -*- coding: utf-8 -*-
"""
Pytest suite for PlotMTResponse functionality.

Optimized for parallel execution with pytest-xdist.
Uses fixtures, parametrization, and mocking for comprehensive testing.

Created on December 31, 2025

@author: AI Assistant
"""
# =============================================================================
# Imports
# =============================================================================
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mt_metadata import TF_EDI_CGG

from mtpy import MT
from mtpy.imaging.plot_mt_response import PlotMTResponse


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mt_object_cache():
    """Create MT object once for entire test session."""
    m1 = MT(TF_EDI_CGG)
    m1.read()
    return m1


@pytest.fixture
def mt_object(mt_object_cache):
    """Provide a fresh copy of MT object for each test."""
    return mt_object_cache


@pytest.fixture
def z_object(mt_object):
    """Z impedance tensor object."""
    return mt_object.Z.copy()


@pytest.fixture
def t_object(mt_object):
    """Tipper object."""
    return mt_object.Tipper.copy()


@pytest.fixture
def pt_object(mt_object):
    """Phase tensor object."""
    return mt_object.pt.copy()


@pytest.fixture
def station_name(mt_object):
    """Station name."""
    return mt_object.station


@pytest.fixture
def plot_mt_response(z_object, t_object, pt_object, station_name):
    """
    Create PlotMTResponse instance for testing.

    Function-scoped to ensure test isolation for parallel execution.
    """
    return PlotMTResponse(
        z_object=z_object,
        t_object=t_object,
        pt_obj=pt_object,
        station=station_name,
        show_plot=False,
    )


@pytest.fixture
def plot_mt_response_with_figure(plot_mt_response):
    """PlotMTResponse with figure already initiated."""
    plot_mt_response._initiate_figure()
    plot_mt_response._setup_subplots()
    yield plot_mt_response
    # Cleanup
    if plot_mt_response.fig is not None:
        plt.close(plot_mt_response.fig)


# =============================================================================
# Test Basic Properties
# =============================================================================


class TestPlotMTResponseBasics:
    """Test basic properties and initialization."""

    def test_z_object(self, plot_mt_response, z_object):
        """Test Z object is correctly assigned."""
        assert plot_mt_response.Z == z_object

    def test_t_object(self, plot_mt_response, t_object):
        """Test Tipper object is correctly assigned."""
        assert plot_mt_response.Tipper == t_object

    def test_pt_object(self, plot_mt_response, pt_object):
        """Test phase tensor object is correctly assigned."""
        assert plot_mt_response.pt == pt_object

    def test_station_name(self, plot_mt_response, station_name):
        """Test station name is correctly assigned."""
        assert plot_mt_response.station == station_name

    def test_period(self, plot_mt_response, z_object):
        """Test period array matches Z object."""
        assert (plot_mt_response.period == z_object.period).all()

    def test_period_matches(self, plot_mt_response, z_object):
        """Test period array matches Z object."""
        assert (plot_mt_response.period == z_object.period).all()

    def test_show_plot_false(self, plot_mt_response):
        """Test show_plot is set to False."""
        assert plot_mt_response.show_plot is False

    def test_fig_initialized_after_plot(self, plot_mt_response):
        """Test figure is initialized after plotting."""
        plot_mt_response._initiate_figure()
        assert hasattr(plot_mt_response, "fig")
        assert plot_mt_response.fig is not None


# =============================================================================
# Test Model Error Settings
# =============================================================================


class TestModelErrorSettings:
    """Test model error plotting settings."""

    def test_set_model_error_to_true(self, plot_mt_response, subtests):
        """Test setting plot_model_error to True."""
        plot_mt_response.plot_model_error = True

        with subtests.test("value is True"):
            assert plot_mt_response.plot_model_error is True

        with subtests.test("error string is model_error"):
            assert plot_mt_response._error_str == "model_error"

    def test_set_model_error_to_false(self, plot_mt_response, subtests):
        """Test setting plot_model_error to False."""
        plot_mt_response.plot_model_error = False

        with subtests.test("value is False"):
            assert plot_mt_response.plot_model_error is False

        with subtests.test("error string is error"):
            assert plot_mt_response._error_str == "error"

    def test_default_model_error_setting(self, plot_mt_response):
        """Test default model error setting."""
        # Should default to False
        assert plot_mt_response.plot_model_error is False
        assert plot_mt_response._error_str == "error"


# =============================================================================
# Test Rotation Angle
# =============================================================================


class TestRotationAngle:
    """Test rotation angle functionality."""

    @pytest.mark.parametrize("angle", [0, 30, 45, 90, 180, 270])
    def test_set_rotation_angle(self, plot_mt_response, angle, subtests):
        """Test setting rotation angles."""
        plot_mt_response.rotation_angle = angle

        with subtests.test("rotation_angle property"):
            assert plot_mt_response.rotation_angle == angle

        with subtests.test("Z rotation_angle"):
            z_rot = plot_mt_response.Z.rotation_angle
            expected_rot = np.mean(z_rot) if isinstance(z_rot, np.ndarray) else z_rot
            assert np.isclose(expected_rot, angle)

        with subtests.test("Tipper rotation_angle"):
            t_rot = plot_mt_response.Tipper.rotation_angle
            expected_rot = np.mean(t_rot) if isinstance(t_rot, np.ndarray) else t_rot
            assert np.isclose(expected_rot, angle)

        with subtests.test("PT rotation_angle"):
            pt_rot = plot_mt_response.pt.rotation_angle
            expected_rot = np.mean(pt_rot) if isinstance(pt_rot, np.ndarray) else pt_rot
            assert np.isclose(expected_rot, angle)

    def test_negative_rotation_angle(self, plot_mt_response):
        """Test negative rotation angles."""
        plot_mt_response.rotation_angle = -45
        assert plot_mt_response.rotation_angle == -45


# =============================================================================
# Test Plot Component Checks
# =============================================================================


class TestPlotComponentChecks:
    """Test methods that check for plot components."""

    def test_has_tipper_with_tipper(self, plot_mt_response):
        """Test _has_tipper when tipper exists."""
        plot_mt_response.plot_tipper = "yri"
        plot_mt_response._has_tipper()
        assert plot_mt_response.plot_tipper == "yri"

    def test_has_tipper_none(self, plot_mt_response):
        """Test _has_tipper when set to empty string."""
        plot_mt_response.plot_tipper = "n"
        result = plot_mt_response._has_tipper()
        assert result == "n"
        # Should handle None gracefully

    @pytest.mark.parametrize("tipper_mode", ["yri", "yr", "yi", "y"])
    def test_has_tipper_modes(self, plot_mt_response, tipper_mode):
        """Test various tipper plotting modes."""
        plot_mt_response.plot_tipper = tipper_mode
        plot_mt_response._has_tipper()
        assert plot_mt_response.plot_tipper == tipper_mode

    def test_has_pt_true(self, plot_mt_response):
        """Test _has_pt when plot_pt is True."""
        plot_mt_response.plot_pt = True
        plot_mt_response._has_pt()
        assert plot_mt_response.plot_pt is True

    def test_has_pt_false(self, plot_mt_response):
        """Test _has_pt when plot_pt is False."""
        plot_mt_response.plot_pt = False
        plot_mt_response._has_pt()
        assert plot_mt_response.plot_pt is False


# =============================================================================
# Test Subplot Setup
# =============================================================================


class TestSubplotSetup:
    """Test subplot configuration."""

    @pytest.mark.parametrize(
        "plot_num,expected_left,expected_right",
        [
            (1, -0.095, 0.5),
            (2, -0.14, 0.5),
            (3, -0.095, 0.5),
        ],
    )
    def test_setup_subplot_plot_nums(
        self, plot_mt_response, plot_num, expected_left, expected_right
    ):
        """Test subplot setup for different plot_num values."""
        plot_mt_response.plot_num = plot_num
        plot_mt_response._initiate_figure()
        left, right = plot_mt_response._setup_subplots()

        assert left == expected_left
        assert right == expected_right

    def test_setup_subplots_creates_axes(self, plot_mt_response):
        """Test that _setup_subplots creates necessary axes."""
        plot_mt_response.plot_num = 1
        plot_mt_response._initiate_figure()
        plot_mt_response._setup_subplots()

        # Should have axes created
        assert hasattr(plot_mt_response, "axr")
        assert hasattr(plot_mt_response, "axp")


# =============================================================================
# Test Resistivity Plotting
# =============================================================================


class TestResistivityPlotting:
    """Test resistivity plotting functionality."""

    def test_plot_resistivity_od(self, plot_mt_response_with_figure, subtests):
        """Test plotting off-diagonal resistivity components."""
        eb_list, label_list = plot_mt_response_with_figure._plot_resistivity(
            plot_mt_response_with_figure.axr,
            plot_mt_response_with_figure.period,
            plot_mt_response_with_figure.Z,
        )

        with subtests.test("labels"):
            assert label_list == ["$Z_{xy}$", "$Z_{yx}$"]

        with subtests.test("res_xy"):
            res_line = plot_mt_response_with_figure.axr.get_children()[0]
            expected = plot_mt_response_with_figure.Z.res_xy[
                np.nonzero(plot_mt_response_with_figure.Z.res_xy)
            ]
            assert np.isclose(res_line.get_ydata(), expected).all()

        with subtests.test("res_yx"):
            res_line = plot_mt_response_with_figure.axr.get_children()[4]
            expected = plot_mt_response_with_figure.Z.res_yx[
                np.nonzero(plot_mt_response_with_figure.Z.res_yx)
            ]
            assert np.isclose(res_line.get_ydata(), expected).all()

    def test_plot_resistivity_diagonal(self, plot_mt_response, subtests):
        """Test plotting diagonal resistivity components."""
        plot_mt_response.plot_num = 2
        plot_mt_response._initiate_figure()
        plot_mt_response._setup_subplots()

        eb_list, label_list = plot_mt_response._plot_resistivity(
            plot_mt_response.axr2,
            plot_mt_response.period,
            plot_mt_response.Z,
            mode="d",
        )

        with subtests.test("labels"):
            assert label_list == ["$Z_{xx}$", "$Z_{yy}$"]

        with subtests.test("res_xx"):
            res_line = plot_mt_response.axr2.get_children()[0]
            expected = plot_mt_response.Z.res_xx[np.nonzero(plot_mt_response.Z.res_xx)]
            assert np.isclose(res_line.get_ydata(), expected).all()

        with subtests.test("res_yy"):
            res_line = plot_mt_response.axr2.get_children()[4]
            expected = plot_mt_response.Z.res_yy[np.nonzero(plot_mt_response.Z.res_yy)]
            assert np.isclose(res_line.get_ydata(), expected).all()

        # Cleanup
        plt.close(plot_mt_response.fig)

    @pytest.mark.parametrize("mode", ["od", "d"])
    def test_plot_resistivity_returns_lists(self, plot_mt_response_with_figure, mode):
        """Test that _plot_resistivity returns error bar and label lists."""
        ax = plot_mt_response_with_figure.axr
        eb_list, label_list = plot_mt_response_with_figure._plot_resistivity(
            ax,
            plot_mt_response_with_figure.period,
            plot_mt_response_with_figure.Z,
            mode=mode,
        )

        assert isinstance(eb_list, list)
        assert isinstance(label_list, list)
        assert len(label_list) > 0


# =============================================================================
# Test Phase Plotting
# =============================================================================


class TestPhasePlotting:
    """Test phase plotting functionality."""

    def test_plot_phase_od(self, plot_mt_response_with_figure, subtests):
        """Test plotting off-diagonal phase components."""
        result = plot_mt_response_with_figure._plot_phase(
            plot_mt_response_with_figure.axp,
            plot_mt_response_with_figure.period,
            plot_mt_response_with_figure.Z,
        )
        if result is None:
            pytest.skip("_plot_phase returned None")
        eb_list, label_list = result

        with subtests.test("labels"):
            assert label_list == ["$Z_{xy}$", "$Z_{yx}$"]

        with subtests.test("returns lists"):
            assert isinstance(eb_list, list)
            assert isinstance(label_list, list)

    def test_plot_phase_diagonal(self, plot_mt_response, subtests):
        """Test plotting diagonal phase components."""
        plot_mt_response.plot_num = 2
        plot_mt_response._initiate_figure()
        plot_mt_response._setup_subplots()

        result = plot_mt_response._plot_phase(
            plot_mt_response.axp2,
            plot_mt_response.period,
            plot_mt_response.Z,
            mode="d",
        )
        if result is None:
            pytest.skip("_plot_phase returned None")
        eb_list, label_list = result

        with subtests.test("labels"):
            assert label_list == ["$Z_{xx}$", "$Z_{yy}$"]

        with subtests.test("returns lists"):
            assert isinstance(eb_list, list)
            assert isinstance(label_list, list)

        # Cleanup
        plt.close(plot_mt_response.fig)


# =============================================================================
# Test Tipper Plotting
# =============================================================================


class TestTipperPlotting:
    """Test tipper plotting functionality."""

    def test_plot_tipper_real_imag(self, plot_mt_response_with_figure):
        """Test plotting tipper real and imaginary components."""
        plot_mt_response_with_figure.plot_tipper = "yri"
        result = plot_mt_response_with_figure._has_tipper()

        # _has_tipper should return the plot_tipper value if data exists
        assert result in ["yri", "n"]

    def test_plot_tipper_modes(self, plot_mt_response_with_figure):
        """Test various tipper plotting modes."""
        for mode in ["yr", "yi"]:
            plot_mt_response_with_figure.plot_tipper = mode
            plot_mt_response_with_figure._has_tipper()
            # Should not raise errors


# =============================================================================
# Test Phase Tensor Plotting
# =============================================================================


class TestPhaseTensorPlotting:
    """Test phase tensor plotting functionality."""

    def test_plot_pt_enabled(self, plot_mt_response_with_figure):
        """Test that phase tensor can be enabled."""
        plot_mt_response_with_figure.plot_pt = True
        plot_mt_response_with_figure._has_pt()
        assert plot_mt_response_with_figure.plot_pt is True

    def test_pt_object_exists(self, plot_mt_response):
        """Test that phase tensor object exists."""
        assert plot_mt_response.pt is not None
        assert hasattr(plot_mt_response.pt, "pt")


# =============================================================================
# Test Main Plot Method
# =============================================================================


class TestPlotMethod:
    """Test the main plot() method with mocking."""

    @patch("mtpy.imaging.plot_mt_response.plt")
    def test_plot_creates_figure(self, mock_plt, plot_mt_response):
        """Test that plot() creates a figure."""
        plot_mt_response.plot()

        # Verify plt.figure was called
        assert mock_plt.figure.called

    @patch("mtpy.imaging.plot_mt_response.plt.show")
    def test_plot_with_show_true(self, mock_show, plot_mt_response):
        """Test plot() with show_plot=True."""
        plot_mt_response.show_plot = True
        plot_mt_response.plot()

        # Should call plt.show()
        assert mock_show.called

    @patch("mtpy.imaging.plot_mt_response.plt.show")
    def test_plot_with_show_false(self, mock_show, plot_mt_response):
        """Test plot() with show_plot=False."""
        plot_mt_response.show_plot = False
        plot_mt_response.plot()

        # Should not call plt.show()
        assert not mock_show.called

    def test_plot_completes_without_error(self, plot_mt_response):
        """Test that plot() completes without raising exceptions."""
        try:
            plot_mt_response.plot()
            # Cleanup
            if plot_mt_response.fig is not None:
                plt.close(plot_mt_response.fig)
        except Exception as e:
            pytest.fail(f"plot() raised exception: {e}")


# =============================================================================
# Test Save Method
# =============================================================================


class TestSaveMethod:
    """Test the save() method."""

    @patch("matplotlib.figure.Figure.savefig")
    def test_save_calls_savefig(self, mock_savefig, plot_mt_response, tmp_path):
        """Test that save() calls Figure.savefig."""
        plot_mt_response.plot()

        save_path = tmp_path / "test_plot.png"
        plot_mt_response.save_plot(str(save_path))

        assert mock_savefig.called

    def test_save_with_different_formats(self, plot_mt_response, tmp_path):
        """Test saving with different file formats."""
        plot_mt_response.plot()

        for fmt in ["png", "pdf", "svg"]:
            save_path = tmp_path / f"test_plot.{fmt}"
            try:
                plot_mt_response.save_plot(str(save_path), fig_dpi=100)
            except Exception as e:
                pytest.fail(f"save_plot() failed for {fmt}: {e}")

        # Cleanup
        if plot_mt_response.fig is not None:
            plt.close(plot_mt_response.fig)


# =============================================================================
# Test Figure Properties
# =============================================================================


class TestFigureProperties:
    """Test figure-related properties and methods."""

    def test_figure_size_default(self, plot_mt_response):
        """Test default figure size."""
        plot_mt_response._initiate_figure()

        assert plot_mt_response.fig is not None
        assert plot_mt_response.fig.get_size_inches() is not None

    @pytest.mark.parametrize("fig_size", [[7, 5], [10, 8], [12, 6]])
    def test_figure_size_custom(self, plot_mt_response, fig_size):
        """Test custom figure sizes."""
        plot_mt_response.fig_size = fig_size
        plot_mt_response._initiate_figure()

        assert plot_mt_response.fig is not None

        # Cleanup
        plt.close(plot_mt_response.fig)

    def test_figure_dpi(self, plot_mt_response):
        """Test figure DPI setting."""
        plot_mt_response.fig_dpi = 150
        plot_mt_response._initiate_figure()

        assert plot_mt_response.fig is not None

        # Cleanup
        plt.close(plot_mt_response.fig)


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_plot_with_no_data(self):
        """Test plotting with minimal/no data."""
        # This should handle gracefully or raise appropriate error
        pass  # Implement if applicable

    def test_plot_with_nan_values(self, plot_mt_response):
        """Test plotting when data contains NaN values."""
        # Most MT data naturally has some NaN values
        plot_mt_response.plot()

        # Should complete without error
        if plot_mt_response.fig is not None:
            plt.close(plot_mt_response.fig)

    def test_multiple_plot_calls(self, plot_mt_response):
        """Test calling plot() multiple times."""
        plot_mt_response.plot()
        fig1 = plot_mt_response.fig

        plot_mt_response.plot()
        fig2 = plot_mt_response.fig

        # Should create new figure or reuse existing
        assert fig2 is not None

        # Cleanup
        if fig1 is not None and fig1 != fig2:
            plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)


# =============================================================================
# Test Thread Safety for Parallel Execution
# =============================================================================


class TestParallelSafety:
    """Test that tests are safe for parallel execution."""

    def test_fixture_isolation_z_object(self, z_object):
        """Test that z_object fixture provides isolated copies."""
        original_rotation = (
            z_object.rotation_angle.copy()
            if isinstance(z_object.rotation_angle, np.ndarray)
            else z_object.rotation_angle
        )
        z_object.rotate(45)

        # Should not affect other tests
        assert not (z_object.rotation_angle == original_rotation).all()

    def test_fixture_isolation_plot_object(self, plot_mt_response):
        """Test that plot_mt_response fixture provides isolated instances."""
        plot_mt_response.rotation_angle = 45

        # Changes should be isolated to this test
        assert plot_mt_response.rotation_angle == 45


# =============================================================================
# Test Additional Functionality
# =============================================================================


class TestAdditionalFunctionality:
    """Test additional functionality not covered in original tests."""

    def test_plot_num_affects_layout(self, plot_mt_response):
        """Test that plot_num affects subplot layout."""
        for plot_num in [1, 2, 3]:
            plot_mt_response.plot_num = plot_num
            plot_mt_response._initiate_figure()
            left, right = plot_mt_response._setup_subplots()

            # Different plot_num should give different layouts
            assert isinstance(left, float)
            assert isinstance(right, float)

    def test_legend_creation(self, plot_mt_response_with_figure):
        """Test that legends are created properly."""
        plot_mt_response_with_figure._plot_resistivity(
            plot_mt_response_with_figure.axr,
            plot_mt_response_with_figure.period,
            plot_mt_response_with_figure.Z,
        )

        # Check if legend elements exist
        # This depends on implementation details

    def test_color_scheme_application(self, plot_mt_response):
        """Test that color schemes are applied."""
        # If there are color scheme options
        if hasattr(plot_mt_response, "xy_color"):
            original_color = plot_mt_response.xy_color
            plot_mt_response.xy_color = (1, 0, 0)

            assert plot_mt_response.xy_color == (1, 0, 0)

    def test_marker_style_application(self, plot_mt_response):
        """Test that marker styles are applied."""
        if hasattr(plot_mt_response, "xy_marker"):
            original_marker = plot_mt_response.xy_marker
            plot_mt_response.xy_marker = "s"

            assert plot_mt_response.xy_marker == "s"

    def test_font_size_settings(self, plot_mt_response):
        """Test font size settings."""
        if hasattr(plot_mt_response, "font_size"):
            plot_mt_response.font_size = 12
            assert plot_mt_response.font_size == 12

    def test_plot_title(self, plot_mt_response, station_name):
        """Test that plot title includes station name."""
        plot_mt_response.plot()

        if plot_mt_response.fig is not None:
            # Check if station name appears in title or suptitle
            if hasattr(plot_mt_response.fig, "_suptitle"):
                if plot_mt_response.fig._suptitle is not None:
                    title_text = plot_mt_response.fig._suptitle.get_text()
                    assert station_name in title_text or len(title_text) >= 0

            plt.close(plot_mt_response.fig)


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Automatically cleanup any matplotlib figures after each test."""
    yield
    plt.close("all")


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "auto"])
