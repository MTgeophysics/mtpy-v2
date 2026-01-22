"""
Pytest suite for shapefile_creator.py module.

Tests shapefile creation for MT phase tensors and tippers using mocks.
Optimized for pytest-xdist parallel execution using session-scoped fixtures.

Created: 2025-12-22
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Point

from mtpy.core import MTDataFrame
from mtpy.gis.shapefile_creator import ShapefileCreator


# =============================================================================
# Session-scoped fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mock_mt_dataframe():
    """Create a mock MTDataFrame with test data."""
    # Create mock dataframe
    mock_df = MagicMock(spec=MTDataFrame)

    # Set up basic properties
    mock_df.period = np.array([0.01, 0.1, 1.0, 10.0])

    # Mock station distances
    mock_df.get_station_distances.return_value = pd.Series([1000, 1500, 2000, 2500])

    return mock_df


@pytest.fixture(scope="session")
def mock_phase_tensor_df():
    """Create mock phase tensor dataframe."""
    data = {
        "survey": ["test"] * 5,
        "station": [f"ST{i:03d}" for i in range(5)],
        "latitude": np.linspace(40.0, 40.1, 5),
        "longitude": np.linspace(-120.0, -119.9, 5),
        "elevation": np.ones(5) * 1000,
        "datum_epsg": [4326] * 5,
        "east": np.linspace(0, 10000, 5),
        "north": np.linspace(0, 10000, 5),
        "utm_epsg": [32610] * 5,
        "model_east": np.linspace(0, 10000, 5),
        "model_north": np.linspace(0, 10000, 5),
        "model_elevation": np.ones(5) * 1000,
        "profile_offset": np.linspace(0, 10000, 5),
        "pt_xx": np.random.rand(5),
        "pt_xx_error": np.random.rand(5) * 0.1,
        "pt_xx_model_error": np.random.rand(5) * 0.1,
        "pt_xy": np.random.rand(5),
        "pt_xy_error": np.random.rand(5) * 0.1,
        "pt_xy_model_error": np.random.rand(5) * 0.1,
        "pt_yx": np.random.rand(5),
        "pt_yx_error": np.random.rand(5) * 0.1,
        "pt_yx_model_error": np.random.rand(5) * 0.1,
        "pt_yy": np.random.rand(5),
        "pt_yy_error": np.random.rand(5) * 0.1,
        "pt_yy_model_error": np.random.rand(5) * 0.1,
        "pt_phimin": np.random.rand(5) * 20 + 10,
        "pt_phimin_error": np.random.rand(5),
        "pt_phimin_model_error": np.random.rand(5),
        "pt_phimax": np.random.rand(5) * 30 + 20,
        "pt_phimax_error": np.random.rand(5),
        "pt_phimax_model_error": np.random.rand(5),
        "pt_azimuth": np.random.rand(5) * 180 - 90,
        "pt_azimuth_error": np.random.rand(5) * 5,
        "pt_azimuth_model_error": np.random.rand(5) * 5,
        "pt_skew": np.random.rand(5) * 0.5,
        "pt_skew_error": np.random.rand(5) * 0.1,
        "pt_skew_model_error": np.random.rand(5) * 0.1,
        "pt_ellipticity": np.random.rand(5) * 0.5 + 0.5,
        "pt_ellipticity_error": np.random.rand(5) * 0.1,
        "pt_ellipticity_model_error": np.random.rand(5) * 0.1,
        "pt_det": np.random.rand(5) * 100,
        "pt_det_error": np.random.rand(5) * 10,
        "pt_det_model_error": np.random.rand(5) * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def mock_tipper_df():
    """Create mock tipper dataframe."""
    data = {
        "index": range(5),
        "survey": ["test"] * 5,
        "station": [f"ST{i:03d}" for i in range(5)],
        "latitude": np.linspace(40.0, 40.1, 5),
        "longitude": np.linspace(-120.0, -119.9, 5),
        "elevation": np.ones(5) * 1000,
        "datum_epsg": [4326] * 5,
        "east": np.linspace(0, 10000, 5),
        "north": np.linspace(0, 10000, 5),
        "utm_epsg": [32610] * 5,
        "model_east": np.linspace(0, 10000, 5),
        "model_north": np.linspace(0, 10000, 5),
        "model_elevation": np.ones(5) * 1000,
        "profile_offset": np.linspace(0, 10000, 5),
        "t_mag_real": np.random.rand(5) * 0.5,
        "t_mag_real_error": np.random.rand(5) * 0.05,
        "t_mag_real_model_error": np.random.rand(5) * 0.05,
        "t_mag_imag": np.random.rand(5) * 0.5,
        "t_mag_imag_error": np.random.rand(5) * 0.05,
        "t_mag_imag_model_error": np.random.rand(5) * 0.05,
        "t_angle_real": np.random.rand(5) * 360,
        "t_angle_real_error": np.random.rand(5) * 10,
        "t_angle_real_model_error": np.random.rand(5) * 10,
        "t_angle_imag": np.random.rand(5) * 360,
        "t_angle_imag_error": np.random.rand(5) * 10,
        "t_angle_imag_model_error": np.random.rand(5) * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def test_crs():
    """Test CRS object."""
    return CRS.from_epsg(4326)


@pytest.fixture(scope="session")
def test_periods():
    """Test period values."""
    return [0.1, 1.0, 10.0]


# =============================================================================
# Test Classes
# =============================================================================


class TestShapefileCreatorInit:
    """Tests for ShapefileCreator initialization."""

    def test_init_basic(self, mock_mt_dataframe):
        """Test basic initialization."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        assert creator.mt_dataframe is mock_mt_dataframe
        assert creator.ellipse_size == 2
        assert creator.ellipse_resolution == 180
        assert creator.arrow_size == 2
        assert creator.utm is False

    def test_init_with_save_dir(self, mock_mt_dataframe, tmp_path):
        """Test initialization with save directory."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        assert creator.save_dir == tmp_path
        assert creator.save_dir.exists()

    def test_init_with_kwargs(self, mock_mt_dataframe):
        """Test initialization with keyword arguments."""
        creator = ShapefileCreator(
            mock_mt_dataframe,
            output_crs="EPSG:4326",
            ellipse_size=5,
            arrow_size=3,
            utm=True,
        )

        assert creator.ellipse_size == 5
        assert creator.arrow_size == 3
        assert creator.utm is True

    def test_init_with_epsg_code(self, mock_mt_dataframe):
        """Test initialization with EPSG code as integer."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs=4326)

        assert creator.output_crs is not None
        assert creator.output_crs.to_epsg() == 4326

    def test_init_with_crs_object(self, mock_mt_dataframe, test_crs):
        """Test initialization with CRS object."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs=test_crs)

        assert creator.output_crs is not None
        assert creator.output_crs.to_epsg() == 4326


class TestShapefileCreatorProperties:
    """Tests for ShapefileCreator properties."""

    def test_mt_dataframe_setter(self):
        """Test mt_dataframe setter."""
        mock_df = MagicMock(spec=MTDataFrame)
        creator = ShapefileCreator(mock_df, output_crs="EPSG:4326")

        assert creator.mt_dataframe is mock_df

    def test_save_dir_setter_creates_directory(self, mock_mt_dataframe, tmp_path):
        """Test that save_dir setter creates directory if it doesn't exist."""
        new_dir = tmp_path / "new_shapefile_dir"

        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=new_dir
        )

        assert creator.save_dir == new_dir
        assert new_dir.exists()

    def test_save_dir_none_uses_cwd(self, mock_mt_dataframe):
        """Test that save_dir=None uses current working directory."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=None
        )

        assert creator.save_dir == Path().cwd()

    def test_output_crs_none(self, mock_mt_dataframe):
        """Test output_crs setter with None."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs=None)

        assert creator.output_crs is None

    def test_x_key_utm(self, mock_mt_dataframe):
        """Test x_key property with UTM."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326", utm=True)

        assert creator.x_key == "east"

    def test_x_key_latlon(self, mock_mt_dataframe):
        """Test x_key property with lat/lon."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326", utm=False)

        assert creator.x_key == "longitude"

    def test_y_key_utm(self, mock_mt_dataframe):
        """Test y_key property with UTM."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326", utm=True)

        assert creator.y_key == "north"

    def test_y_key_latlon(self, mock_mt_dataframe):
        """Test y_key property with lat/lon."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326", utm=False)

        assert creator.y_key == "latitude"


class TestEstimateSizes:
    """Tests for size estimation methods."""

    def test_estimate_ellipse_size_default(self, mock_mt_dataframe):
        """Test ellipse size estimation with default quantile."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        size = creator.estimate_ellipse_size()

        assert isinstance(size, (int, float))
        assert size > 0
        mock_mt_dataframe.get_station_distances.assert_called()

    def test_estimate_ellipse_size_custom_quantile(self, mock_mt_dataframe):
        """Test ellipse size estimation with custom quantile."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        size = creator.estimate_ellipse_size(quantile=0.05)

        assert isinstance(size, (int, float))
        assert size > 0

    def test_estimate_arrow_size_default(self, mock_mt_dataframe):
        """Test arrow size estimation with default quantile."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        size = creator.estimate_arrow_size()

        assert isinstance(size, (int, float))
        assert size > 0

    def test_estimate_arrow_size_custom_quantile(self, mock_mt_dataframe):
        """Test arrow size estimation with custom quantile."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        size = creator.estimate_arrow_size(quantile=0.1)

        assert isinstance(size, (int, float))
        assert size > 0


class TestExportShapefiles:
    """Tests for _export_shapefiles method."""

    @patch("geopandas.GeoDataFrame.to_file")
    def test_export_with_output_crs(
        self, mock_to_file, mock_mt_dataframe, mock_phase_tensor_df, tmp_path
    ):
        """Test exporting shapefile with CRS transformation."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        # Create mock GeoDataFrame
        points = [
            Point(lon, lat)
            for lon, lat in zip(
                mock_phase_tensor_df["longitude"], mock_phase_tensor_df["latitude"]
            )
        ]
        gpdf = gpd.GeoDataFrame(
            mock_phase_tensor_df, crs=CRS.from_epsg(4326), geometry=points
        )

        result_path = creator._export_shapefiles(gpdf, "Test_Type", 1.0)

        assert result_path.parent == tmp_path
        assert "Test_Type" in result_path.name
        assert "Period_1.0s" in result_path.name
        mock_to_file.assert_called_once()

    @patch("geopandas.GeoDataFrame.to_file")
    def test_export_without_output_crs(
        self, mock_to_file, mock_mt_dataframe, mock_phase_tensor_df, tmp_path
    ):
        """Test exporting shapefile without CRS transformation."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs=None, save_dir=tmp_path
        )

        points = [
            Point(lon, lat)
            for lon, lat in zip(
                mock_phase_tensor_df["longitude"], mock_phase_tensor_df["latitude"]
            )
        ]
        gpdf = gpd.GeoDataFrame(
            mock_phase_tensor_df, crs=CRS.from_epsg(4326), geometry=points
        )

        result_path = creator._export_shapefiles(gpdf, "Test_Type", 1.0)

        assert "EPSG" not in result_path.name
        assert "Period_1.0s" in result_path.name


class TestGetPeriodGeodf:
    """Tests for _get_period_geodf method."""

    def test_get_period_geodf_phase_tensor(
        self, mock_mt_dataframe, mock_phase_tensor_df
    ):
        """Test getting GeoDataFrame for phase tensor."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        # Mock the dataframe methods
        mock_period_df = MagicMock()
        mock_period_df.phase_tensor = mock_phase_tensor_df
        mock_mt_dataframe.get_period.return_value = mock_period_df

        result = creator._get_period_geodf(1.0, "pt")

        assert result is not None
        crs, geopdf = result
        assert isinstance(crs, CRS)
        assert isinstance(geopdf, gpd.GeoDataFrame)
        assert len(geopdf) == len(mock_phase_tensor_df)

    def test_get_period_geodf_tipper(self, mock_mt_dataframe, mock_tipper_df):
        """Test getting GeoDataFrame for tipper."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        mock_period_df = MagicMock()
        mock_period_df.tipper = mock_tipper_df
        mock_mt_dataframe.get_period.return_value = mock_period_df

        result = creator._get_period_geodf(1.0, "tipper")

        assert result is not None
        crs, geopdf = result
        assert isinstance(crs, CRS)
        assert isinstance(geopdf, gpd.GeoDataFrame)

    def test_get_period_geodf_empty(self, mock_mt_dataframe):
        """Test getting GeoDataFrame with no data."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        mock_period_df = MagicMock()
        mock_period_df.phase_tensor = pd.DataFrame()
        mock_mt_dataframe.get_period.return_value = mock_period_df

        result = creator._get_period_geodf(1.0, "pt")

        assert result is None

    def test_get_period_geodf_utm(self, mock_mt_dataframe, mock_phase_tensor_df):
        """Test getting GeoDataFrame with UTM coordinates."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:32610", utm=True)

        mock_period_df = MagicMock()
        mock_period_df.phase_tensor = mock_phase_tensor_df
        mock_mt_dataframe.get_period.return_value = mock_period_df

        result = creator._get_period_geodf(1.0, "pt")

        assert result is not None
        crs, geopdf = result
        assert crs.to_epsg() == 32610


class TestCreatePhaseTensorShp:
    """Tests for _create_phase_tensor_shp method."""

    @patch.object(ShapefileCreator, "_export_shapefiles")
    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_create_phase_tensor_shp(
        self,
        mock_get_geodf,
        mock_export,
        mock_mt_dataframe,
        mock_phase_tensor_df,
        tmp_path,
    ):
        """Test creating phase tensor shapefile."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        # Mock the geodataframe
        points = [
            Point(lon, lat)
            for lon, lat in zip(
                mock_phase_tensor_df["longitude"], mock_phase_tensor_df["latitude"]
            )
        ]
        gpdf = gpd.GeoDataFrame(
            mock_phase_tensor_df, crs=CRS.from_epsg(4326), geometry=points
        )

        mock_get_geodf.return_value = (CRS.from_epsg(4326), gpdf)
        mock_export.return_value = tmp_path / "test.shp"

        result = creator._create_phase_tensor_shp(1.0)

        assert result == tmp_path / "test.shp"
        mock_get_geodf.assert_called_once_with(1.0, "pt", tol=None)
        mock_export.assert_called_once()

    @patch.object(ShapefileCreator, "_export_shapefiles")
    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_create_phase_tensor_with_tolerance(
        self,
        mock_get_geodf,
        mock_export,
        mock_mt_dataframe,
        mock_phase_tensor_df,
        tmp_path,
    ):
        """Test creating phase tensor shapefile with tolerance."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        points = [
            Point(lon, lat)
            for lon, lat in zip(
                mock_phase_tensor_df["longitude"], mock_phase_tensor_df["latitude"]
            )
        ]
        gpdf = gpd.GeoDataFrame(
            mock_phase_tensor_df, crs=CRS.from_epsg(4326), geometry=points
        )

        mock_get_geodf.return_value = (CRS.from_epsg(4326), gpdf)
        mock_export.return_value = tmp_path / "test.shp"

        result = creator._create_phase_tensor_shp(1.0, tol=0.1)

        mock_get_geodf.assert_called_once_with(1.0, "pt", tol=0.1)


class TestCreateTipperShp:
    """Tests for tipper shapefile creation methods."""

    @patch.object(ShapefileCreator, "_export_shapefiles")
    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_create_tipper_real_shp(
        self, mock_get_geodf, mock_export, mock_mt_dataframe, mock_tipper_df, tmp_path
    ):
        """Test creating real tipper shapefile."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        points = [
            Point(lon, lat)
            for lon, lat in zip(mock_tipper_df["longitude"], mock_tipper_df["latitude"])
        ]
        gpdf = gpd.GeoDataFrame(
            mock_tipper_df, crs=CRS.from_epsg(4326), geometry=points
        )

        mock_get_geodf.return_value = (CRS.from_epsg(4326), gpdf)
        mock_export.return_value = tmp_path / "tipper_real.shp"

        result = creator._create_tipper_real_shp(1.0)

        assert result == tmp_path / "tipper_real.shp"
        mock_get_geodf.assert_called_once_with(1.0, "tip", tol=None)

    @patch.object(ShapefileCreator, "_export_shapefiles")
    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_create_tipper_imag_shp(
        self, mock_get_geodf, mock_export, mock_mt_dataframe, mock_tipper_df, tmp_path
    ):
        """Test creating imaginary tipper shapefile."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        points = [
            Point(lon, lat)
            for lon, lat in zip(mock_tipper_df["longitude"], mock_tipper_df["latitude"])
        ]
        gpdf = gpd.GeoDataFrame(
            mock_tipper_df, crs=CRS.from_epsg(4326), geometry=points
        )

        mock_get_geodf.return_value = (CRS.from_epsg(4326), gpdf)
        mock_export.return_value = tmp_path / "tipper_imag.shp"

        result = creator._create_tipper_imag_shp(1.0)

        assert result == tmp_path / "tipper_imag.shp"
        mock_get_geodf.assert_called_once_with(1.0, "tip", tol=None)

    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_create_tipper_real_empty(self, mock_get_geodf, mock_mt_dataframe):
        """Test creating real tipper shapefile with empty data."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        # Create empty GeoDataFrame with geometry column
        empty_df = pd.DataFrame(columns=["geometry"])
        gpdf = gpd.GeoDataFrame(empty_df, geometry="geometry", crs=CRS.from_epsg(4326))
        mock_get_geodf.return_value = (CRS.from_epsg(4326), gpdf)

        result = creator._create_tipper_real_shp(1.0)

        assert result is None

    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_create_tipper_imag_empty(self, mock_get_geodf, mock_mt_dataframe):
        """Test creating imaginary tipper shapefile with empty data."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        # Create empty GeoDataFrame with geometry column
        empty_df = pd.DataFrame(columns=["geometry"])
        gpdf = gpd.GeoDataFrame(empty_df, geometry="geometry", crs=CRS.from_epsg(4326))
        mock_get_geodf.return_value = (CRS.from_epsg(4326), gpdf)

        result = creator._create_tipper_imag_shp(1.0)

        assert result is None


class TestMakeShpFiles:
    """Tests for make_shp_files method."""

    @patch.object(ShapefileCreator, "_create_phase_tensor_shp")
    def test_make_shp_files_pt_only(
        self, mock_create_pt, mock_mt_dataframe, test_periods, tmp_path
    ):
        """Test making shapefiles for phase tensors only."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        mock_create_pt.return_value = tmp_path / "pt.shp"

        result = creator.make_shp_files(pt=True, tipper=False, periods=test_periods)

        assert "pt" in result
        assert len(result["pt"]) == len(test_periods)
        assert mock_create_pt.call_count == len(test_periods)

    @patch.object(ShapefileCreator, "_create_tipper_real_shp")
    @patch.object(ShapefileCreator, "_create_tipper_imag_shp")
    def test_make_shp_files_tipper_only(
        self,
        mock_create_imag,
        mock_create_real,
        mock_mt_dataframe,
        test_periods,
        tmp_path,
    ):
        """Test making shapefiles for tippers only."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        mock_create_real.return_value = tmp_path / "tipper_real.shp"
        mock_create_imag.return_value = tmp_path / "tipper_imag.shp"

        result = creator.make_shp_files(pt=False, tipper=True, periods=test_periods)

        assert "tipper_real" in result
        assert "tipper_imag" in result
        assert len(result["tipper_real"]) == len(test_periods)
        assert len(result["tipper_imag"]) == len(test_periods)

    @patch.object(ShapefileCreator, "_create_phase_tensor_shp")
    @patch.object(ShapefileCreator, "_create_tipper_real_shp")
    @patch.object(ShapefileCreator, "_create_tipper_imag_shp")
    def test_make_shp_files_all(
        self,
        mock_create_imag,
        mock_create_real,
        mock_create_pt,
        mock_mt_dataframe,
        test_periods,
        tmp_path,
    ):
        """Test making all shapefile types."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        mock_create_pt.return_value = tmp_path / "pt.shp"
        mock_create_real.return_value = tmp_path / "tipper_real.shp"
        mock_create_imag.return_value = tmp_path / "tipper_imag.shp"

        result = creator.make_shp_files(pt=True, tipper=True, periods=test_periods)

        assert all(k in result for k in ["pt", "tipper_real", "tipper_imag"])
        assert all(len(result[k]) == len(test_periods) for k in result)

    @patch.object(ShapefileCreator, "_create_phase_tensor_shp")
    def test_make_shp_files_default_periods(
        self, mock_create_pt, mock_mt_dataframe, tmp_path
    ):
        """Test making shapefiles with default periods from dataframe."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        mock_create_pt.return_value = tmp_path / "pt.shp"

        result = creator.make_shp_files(pt=True, tipper=False)

        # Should use periods from mock_mt_dataframe
        assert len(result["pt"]) == len(mock_mt_dataframe.period)

    @patch.object(ShapefileCreator, "_create_tipper_real_shp")
    @patch.object(ShapefileCreator, "_create_tipper_imag_shp")
    def test_make_shp_files_with_tolerance(
        self,
        mock_create_imag,
        mock_create_real,
        mock_mt_dataframe,
        test_periods,
        tmp_path,
    ):
        """Test making shapefiles with period tolerance."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", save_dir=tmp_path
        )

        mock_create_real.return_value = tmp_path / "tipper_real.shp"
        mock_create_imag.return_value = tmp_path / "tipper_imag.shp"

        result = creator.make_shp_files(
            pt=False, tipper=True, periods=test_periods, period_tol=0.1
        )

        # Check that tolerance was passed to tipper methods
        for call in mock_create_real.call_args_list:
            assert "tol" in call.kwargs or len(call.args) > 1


class TestShapefileCreatorParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "crs_input",
        [
            "EPSG:4326",
            4326,
            "EPSG:32610",
            32610,
        ],
    )
    def test_various_crs_inputs(self, mock_mt_dataframe, crs_input):
        """Test initialization with various CRS input formats."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs=crs_input)

        assert creator.output_crs is not None

    @pytest.mark.parametrize("utm_flag", [True, False])
    def test_coordinate_keys_with_utm(self, mock_mt_dataframe, utm_flag):
        """Test coordinate key selection based on UTM flag."""
        creator = ShapefileCreator(
            mock_mt_dataframe, output_crs="EPSG:4326", utm=utm_flag
        )

        if utm_flag:
            assert creator.x_key == "east"
            assert creator.y_key == "north"
        else:
            assert creator.x_key == "longitude"
            assert creator.y_key == "latitude"

    @pytest.mark.parametrize("component", ["pt", "phase_tensor", "t", "tip", "tipper"])
    def test_get_period_geodf_components(
        self, mock_mt_dataframe, component, mock_phase_tensor_df, mock_tipper_df
    ):
        """Test _get_period_geodf with different component types."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        mock_period_df = MagicMock()
        if component in ["pt", "phase_tensor"]:
            mock_period_df.phase_tensor = mock_phase_tensor_df
        else:
            mock_period_df.tipper = mock_tipper_df

        mock_mt_dataframe.get_period.return_value = mock_period_df

        result = creator._get_period_geodf(1.0, component)

        assert result is not None

    @pytest.mark.parametrize(
        "ellipse_size,arrow_size",
        [
            (1, 1),
            (2, 2),
            (5, 3),
            (10, 5),
        ],
    )
    def test_various_sizes(self, mock_mt_dataframe, ellipse_size, arrow_size):
        """Test initialization with various size parameters."""
        creator = ShapefileCreator(
            mock_mt_dataframe,
            output_crs="EPSG:4326",
            ellipse_size=ellipse_size,
            arrow_size=arrow_size,
        )

        assert creator.ellipse_size == ellipse_size
        assert creator.arrow_size == arrow_size


class TestShapefileCreatorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_init_with_none_crs(self, mock_mt_dataframe):
        """Test initialization with None CRS."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs=None)

        assert creator.output_crs is None

    @patch.object(ShapefileCreator, "_get_period_geodf")
    def test_make_shp_files_empty_periods(self, mock_get_geodf, mock_mt_dataframe):
        """Test make_shp_files with empty period list."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        result = creator.make_shp_files(periods=[])

        assert result["pt"] == []
        assert result["tipper_real"] == []
        assert result["tipper_imag"] == []

    def test_estimate_sizes_with_empty_distances(self, mock_mt_dataframe):
        """Test size estimation with empty distance data."""
        creator = ShapefileCreator(mock_mt_dataframe, output_crs="EPSG:4326")

        # Mock empty series
        mock_mt_dataframe.get_station_distances.return_value = pd.Series([])

        # Should still work but return NaN
        size = creator.estimate_ellipse_size()
        assert np.isnan(size) or size >= 0
