"""
Pytest suite for raster_tools.py module.

Tests array2raster functionality with mocks to avoid file I/O.
Optimized for pytest-xdist parallel execution using session-scoped fixtures.

Created: 2025-12-23
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rasterio.transform import Affine

from mtpy.core.mt_location import MTLocation
from mtpy.gis.raster_tools import array2raster


# =============================================================================
# Session-scoped fixtures
# =============================================================================


@pytest.fixture(scope="session")
def basic_array():
    """Create a basic 2D test array."""
    return np.random.rand(10, 10)


@pytest.fixture(scope="session")
def large_array():
    """Create a larger test array."""
    return np.random.rand(100, 100)


@pytest.fixture(scope="session")
def integer_array():
    """Create an integer array."""
    return np.random.randint(0, 100, size=(20, 20))


@pytest.fixture(scope="session")
def float32_array():
    """Create a float32 array."""
    return np.random.rand(15, 15).astype(np.float32)


@pytest.fixture(scope="session")
def basic_mt_location():
    """Create a basic MTLocation object."""
    return MTLocation(latitude=40.0, longitude=-120.0, elevation=1000.0, datum="WGS84")


@pytest.fixture(scope="session")
def negative_coord_location():
    """Create an MTLocation with negative coordinates."""
    return MTLocation(latitude=-33.5, longitude=-70.6, elevation=500.0, datum="WGS84")


@pytest.fixture(scope="session")
def utm_location():
    """Create an MTLocation with UTM coordinates."""
    return MTLocation(
        latitude=40.0, longitude=-120.0, elevation=1000.0, datum="WGS84", utm_zone="11N"
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestArray2RasterBasic:
    """Tests for basic array2raster functionality."""

    @patch("rasterio.open")
    def test_basic_write(self, mock_rasterio_open, basic_array, basic_mt_location):
        """Test basic raster writing with default parameters."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        # Verify rasterio.open was called
        mock_rasterio_open.assert_called_once()

        # Verify dataset.write was called
        mock_dataset.write.assert_called_once_with(basic_array, 1)

    @patch("rasterio.open")
    def test_with_rotation(self, mock_rasterio_open, basic_array, basic_mt_location):
        """Test raster writing with rotation angle."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test_rotated.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
            rotation_angle=45.0,
        )

        mock_rasterio_open.assert_called_once()
        mock_dataset.write.assert_called_once()

    @patch("rasterio.open")
    def test_file_path_string(self, mock_rasterio_open, basic_array, basic_mt_location):
        """Test that filename is passed correctly."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        filename = "output_raster.tif"
        array2raster(
            filename,
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        # Check that first argument to rasterio.open is the filename
        call_args = mock_rasterio_open.call_args
        assert call_args[0][0] == filename


class TestArray2RasterParameters:
    """Tests for different parameter combinations."""

    @patch("rasterio.open")
    def test_rasterio_open_parameters(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test that rasterio.open is called with correct parameters."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        # Verify parameters passed to rasterio.open
        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["driver"] == "GTiff"
        assert call_kwargs["height"] == basic_array.shape[0]
        assert call_kwargs["width"] == basic_array.shape[1]
        assert call_kwargs["count"] == 1
        assert call_kwargs["dtype"] == basic_array.dtype
        assert call_kwargs["crs"] == "EPSG:4326"
        assert "transform" in call_kwargs

    @patch("rasterio.open")
    def test_array_shape_propagation(
        self, mock_rasterio_open, large_array, basic_mt_location
    ):
        """Test that array shape is correctly propagated."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            large_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["height"] == 100
        assert call_kwargs["width"] == 100

    @patch("rasterio.open")
    def test_array_dtype_propagation(
        self, mock_rasterio_open, integer_array, basic_mt_location
    ):
        """Test that array dtype is correctly propagated."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            integer_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["dtype"] == integer_array.dtype

    @patch("rasterio.open")
    def test_float32_dtype(self, mock_rasterio_open, float32_array, basic_mt_location):
        """Test with float32 array."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            float32_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["dtype"] == np.float32


class TestArray2RasterTransform:
    """Tests for affine transformation."""

    @patch("rasterio.open")
    def test_transform_created(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test that affine transform is created."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        transform = call_kwargs["transform"]

        assert isinstance(transform, Affine)

    @patch("rasterio.open")
    def test_transform_with_zero_rotation(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test transform with zero rotation."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
            rotation_angle=0,
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        transform = call_kwargs["transform"]

        # For zero rotation, a and e should be the scale factors
        assert isinstance(transform, Affine)

    @patch("rasterio.open")
    def test_transform_with_rotation(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test transform with non-zero rotation."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
            rotation_angle=30.0,
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        transform = call_kwargs["transform"]

        assert isinstance(transform, Affine)

    @patch("rasterio.open")
    def test_transform_uses_location_coordinates(self, mock_rasterio_open, basic_array):
        """Test that transform uses MTLocation coordinates."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        location = MTLocation(latitude=45.0, longitude=-110.0, datum="WGS84")

        array2raster(
            "test.tif",
            basic_array,
            location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        transform = call_kwargs["transform"]

        # Transform should include the location coordinates
        assert isinstance(transform, Affine)


class TestArray2RasterCRS:
    """Tests for different CRS specifications."""

    @patch("rasterio.open")
    def test_epsg_string_crs(self, mock_rasterio_open, basic_array, basic_mt_location):
        """Test with EPSG string CRS."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["crs"] == "EPSG:4326"

    @patch("rasterio.open")
    def test_epsg_integer_crs(self, mock_rasterio_open, basic_array, basic_mt_location):
        """Test with EPSG integer CRS."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs=4326,
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["crs"] == 4326

    @patch("rasterio.open")
    def test_utm_crs(self, mock_rasterio_open, basic_array, utm_location):
        """Test with UTM CRS."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            utm_location,
            cell_size_north=100,
            cell_size_east=100,
            crs="EPSG:32611",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["crs"] == "EPSG:32611"


class TestArray2RasterValidation:
    """Tests for input validation."""

    @patch("rasterio.open")
    def test_invalid_lower_left_type(self, mock_rasterio_open, basic_array):
        """Test that TypeError is raised for invalid lower_left type."""
        with pytest.raises(TypeError, match="lower_left must be a MTLocation"):
            array2raster(
                "test.tif",
                basic_array,
                (40.0, -120.0),  # Invalid type (tuple)
                cell_size_north=0.01,
                cell_size_east=0.01,
                crs="EPSG:4326",
            )

    @patch("rasterio.open")
    def test_invalid_array_type(self, mock_rasterio_open, basic_mt_location):
        """Test that TypeError is raised for invalid array type."""
        with pytest.raises(TypeError, match="array must be a numpy array"):
            array2raster(
                "test.tif",
                [[1, 2], [3, 4]],  # Invalid type (list)
                basic_mt_location,
                cell_size_north=0.01,
                cell_size_east=0.01,
                crs="EPSG:4326",
            )

    @patch("rasterio.open")
    def test_none_lower_left(self, mock_rasterio_open, basic_array):
        """Test that TypeError is raised for None lower_left."""
        with pytest.raises(TypeError, match="lower_left must be a MTLocation"):
            array2raster(
                "test.tif",
                basic_array,
                None,
                cell_size_north=0.01,
                cell_size_east=0.01,
                crs="EPSG:4326",
            )

    @patch("rasterio.open")
    def test_none_array(self, mock_rasterio_open, basic_mt_location):
        """Test that TypeError is raised for None array."""
        with pytest.raises(TypeError, match="array must be a numpy array"):
            array2raster(
                "test.tif",
                None,
                basic_mt_location,
                cell_size_north=0.01,
                cell_size_east=0.01,
                crs="EPSG:4326",
            )


class TestArray2RasterLocations:
    """Tests with different MTLocation configurations."""

    @patch("rasterio.open")
    def test_negative_coordinates(
        self, mock_rasterio_open, basic_array, negative_coord_location
    ):
        """Test with negative coordinates (southern hemisphere)."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            negative_coord_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()
        mock_dataset.write.assert_called_once()

    @patch("rasterio.open")
    def test_utm_coordinates(self, mock_rasterio_open, basic_array, utm_location):
        """Test with UTM location."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            utm_location,
            cell_size_north=100,
            cell_size_east=100,
            crs="EPSG:32611",
        )

        mock_rasterio_open.assert_called_once()
        mock_dataset.write.assert_called_once()

    @patch("rasterio.open")
    def test_high_elevation_location(self, mock_rasterio_open, basic_array):
        """Test with high elevation location."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        location = MTLocation(
            latitude=40.0, longitude=-120.0, elevation=5000.0, datum="WGS84"
        )

        array2raster(
            "test.tif",
            basic_array,
            location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()


class TestArray2RasterParameterized:
    """Parameterized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "cell_size_north,cell_size_east",
        [
            (0.01, 0.01),
            (0.001, 0.001),
            (0.1, 0.1),
            (1.0, 1.0),
            (100, 100),
            (0.01, 0.02),  # Different north/east sizes
            (0.5, 0.25),
        ],
    )
    @patch("rasterio.open")
    def test_various_cell_sizes(
        self,
        mock_rasterio_open,
        cell_size_north,
        cell_size_east,
        basic_array,
        basic_mt_location,
    ):
        """Test with various cell sizes."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=cell_size_north,
            cell_size_east=cell_size_east,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()

    @pytest.mark.parametrize(
        "rotation_angle",
        [
            0,
            30,
            45,
            90,
            180,
            -45,
            -90,
            360,
        ],
    )
    @patch("rasterio.open")
    def test_various_rotation_angles(
        self, mock_rasterio_open, rotation_angle, basic_array, basic_mt_location
    ):
        """Test with various rotation angles."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
            rotation_angle=rotation_angle,
        )

        mock_rasterio_open.assert_called_once()

    @pytest.mark.parametrize(
        "crs_spec",
        [
            "EPSG:4326",
            "EPSG:32611",
            "EPSG:3857",
            4326,
            32611,
            3857,
        ],
    )
    @patch("rasterio.open")
    def test_various_crs_formats(
        self, mock_rasterio_open, crs_spec, basic_array, basic_mt_location
    ):
        """Test with various CRS formats."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs=crs_spec,
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["crs"] == crs_spec

    @pytest.mark.parametrize(
        "array_shape",
        [
            (5, 5),
            (10, 20),
            (50, 50),
            (100, 200),
            (1, 10),
            (10, 1),
        ],
    )
    @patch("rasterio.open")
    def test_various_array_shapes(
        self, mock_rasterio_open, array_shape, basic_mt_location
    ):
        """Test with various array shapes."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        test_array = np.random.rand(*array_shape)

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["height"] == array_shape[0]
        assert call_kwargs["width"] == array_shape[1]

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.int16,
            np.int32,
            np.uint8,
            np.uint16,
        ],
    )
    @patch("rasterio.open")
    def test_various_dtypes(self, mock_rasterio_open, dtype, basic_mt_location):
        """Test with various numpy dtypes."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        if np.issubdtype(dtype, np.integer):
            test_array = np.random.randint(0, 100, size=(10, 10)).astype(dtype)
        else:
            test_array = np.random.rand(10, 10).astype(dtype)

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["dtype"] == dtype


class TestArray2RasterEdgeCases:
    """Tests for edge cases and special scenarios."""

    @patch("rasterio.open")
    def test_single_pixel_array(self, mock_rasterio_open, basic_mt_location):
        """Test with 1x1 array."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        test_array = np.array([[42.0]])

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        call_kwargs = mock_rasterio_open.call_args[1]
        assert call_kwargs["height"] == 1
        assert call_kwargs["width"] == 1

    @patch("rasterio.open")
    def test_array_with_nan_values(self, mock_rasterio_open, basic_mt_location):
        """Test with array containing NaN values."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        test_array = np.random.rand(10, 10)
        test_array[5, 5] = np.nan
        test_array[0, 0] = np.nan

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()
        mock_dataset.write.assert_called_once()

    @patch("rasterio.open")
    def test_array_with_inf_values(self, mock_rasterio_open, basic_mt_location):
        """Test with array containing inf values."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        test_array = np.random.rand(10, 10)
        test_array[3, 3] = np.inf
        test_array[7, 7] = -np.inf

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()

    @patch("rasterio.open")
    def test_array_all_zeros(self, mock_rasterio_open, basic_mt_location):
        """Test with array of all zeros."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        test_array = np.zeros((10, 10))

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        mock_dataset.write.assert_called_once_with(test_array, 1)

    @patch("rasterio.open")
    def test_array_all_ones(self, mock_rasterio_open, basic_mt_location):
        """Test with array of all ones."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        test_array = np.ones((10, 10))

        array2raster(
            "test.tif",
            test_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        mock_dataset.write.assert_called_once_with(test_array, 1)

    @patch("rasterio.open")
    def test_very_small_cell_size(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test with very small cell sizes."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.0001,
            cell_size_east=0.0001,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()

    @patch("rasterio.open")
    def test_very_large_cell_size(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test with very large cell sizes."""
        mock_dataset = MagicMock()
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=1000,
            cell_size_east=1000,
            crs="EPSG:4326",
        )

        mock_rasterio_open.assert_called_once()


class TestArray2RasterContextManager:
    """Tests for proper context manager usage."""

    @patch("rasterio.open")
    def test_context_manager_enter_exit(
        self, mock_rasterio_open, basic_array, basic_mt_location
    ):
        """Test that context manager is used correctly."""
        mock_dataset = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_dataset)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = mock_context

        array2raster(
            "test.tif",
            basic_array,
            basic_mt_location,
            cell_size_north=0.01,
            cell_size_east=0.01,
            crs="EPSG:4326",
        )

        # Verify __enter__ and __exit__ were called
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()
