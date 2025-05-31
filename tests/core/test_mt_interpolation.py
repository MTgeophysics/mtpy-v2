import pytest
import numpy as np
import xarray as xr
from mtpy.core.transfer_function.base import TFBase


class TestTFBaseInterpolate:
    @pytest.fixture
    def sample_tf(self):
        """Create a sample transfer function with some NaN values"""
        frequencies = np.logspace(-3, 3, 20)
        periods = 1.0 / frequencies

        tf = np.ones((len(periods), 2, 2), dtype=complex) * (1 + 1j)
        # Add some pattern to make interpolation more interesting
        for i_index in range(len(periods)):
            tf[i_index, 0, 0] = complex(
                np.sin(i_index / 5) + 1, np.cos(i_index / 5) + 1
            )
            tf[i_index, 0, 1] = complex(
                np.sin(i_index / 3) + 2, np.cos(i_index / 4) + 2
            )
            tf[i_index, 1, 0] = complex(
                np.sin(i_index / 4) + 3, np.cos(i_index / 3) + 3
            )
            tf[i_index, 1, 1] = complex(
                np.sin(i_index / 2) + 4, np.cos(i_index / 6) + 4
            )

        # Insert some NaN values
        tf[2, 0, 0] = np.nan + 1j * np.nan
        tf[3, 0, 1] = np.nan + 1j * np.nan
        tf[15, 1, 0] = np.nan + 1j * np.nan

        # Create transfer function object
        tf_base = TFBase(tf=tf, frequency=frequencies)
        return tf_base

    def test_interpolate_same_periods(self, sample_tf):
        """Test that interpolating to the same periods returns identical values"""
        original_periods = sample_tf.period
        interpolated = sample_tf.interpolate(original_periods)

        # Check that all finite values are preserved
        original_data = sample_tf._dataset.transfer_function.values
        interp_data = interpolated._dataset.transfer_function.values

        # Create masks for non-NaN values in both datasets
        orig_mask = np.isfinite(original_data)
        interp_mask = np.isfinite(interp_data)

        # Verify masks are different, interpolated should fill in nans
        assert not np.array_equal(orig_mask, interp_mask)

        # Check values (allowing for small numerical differences)
        np.testing.assert_allclose(
            original_data[orig_mask], interp_data[orig_mask], rtol=1e-10
        )

    def test_interpolate_different_periods(self, sample_tf):
        """Test interpolation to new set of periods"""
        # Create new period range (with more points)
        new_periods = np.logspace(
            np.log10(sample_tf.period.min()), np.log10(sample_tf.period.max()), 40
        )

        interpolated = sample_tf.interpolate(new_periods)

        # Check shape of interpolated result
        assert interpolated._dataset.transfer_function.shape == (40, 2, 2)
        assert np.array_equal(interpolated._dataset.period.values, new_periods)

        # Check that no new NaN values are introduced for periods within the original range
        for i_index in range(2):
            for j_index in range(2):
                # Get values from original that aren't NaN
                orig_tf = sample_tf._dataset.transfer_function.values[
                    :, i_index, j_index
                ]
                valid_indices = np.isfinite(orig_tf)
                valid_periods = sample_tf.period[valid_indices]

                # Check interpolated values within original valid period range
                in_range_mask = (interpolated.period >= valid_periods.min()) & (
                    interpolated.period <= valid_periods.max()
                )
                interp_values = interpolated._dataset.transfer_function.values[
                    in_range_mask, i_index, j_index
                ]

                # Ensure no NaNs within the interpolated range
                assert not np.any(np.isnan(interp_values))

    def test_interpolate_extrapolate(self, sample_tf):
        """Test extrapolation beyond original period range"""
        # Create new period range that extends beyond original
        extended_periods = np.logspace(
            np.log10(sample_tf.period.min() / 10),  # Extend lower
            np.log10(sample_tf.period.max() * 10),  # Extend higher
            30,
        )

        # Without extrapolation
        no_extrap = sample_tf.interpolate(extended_periods, extrapolate=False)

        # With extrapolation
        with_extrap = sample_tf.interpolate(extended_periods, extrapolate=True)

        # Check that values outside original range are NaN when extrapolate=False
        outside_range = (extended_periods < sample_tf.period.min()) | (
            extended_periods > sample_tf.period.max()
        )
        for i_index in range(2):
            for j_index in range(2):
                no_extrap_values = no_extrap._dataset.transfer_function.values[
                    outside_range, i_index, j_index
                ]
                assert np.all(np.isnan(no_extrap_values))

                # With extrapolation, values should be finite (for components that were originally finite)
                with_extrap_values = with_extrap._dataset.transfer_function.values[
                    outside_range, i_index, j_index
                ]
                if not np.all(
                    np.isnan(
                        sample_tf._dataset.transfer_function.values[:, i_index, j_index]
                    )
                ):
                    assert not np.all(np.isnan(with_extrap_values))

    def test_different_interpolation_methods(self, sample_tf):
        """Test different interpolation methods"""
        new_periods = np.logspace(
            np.log10(sample_tf.period.min()), np.log10(sample_tf.period.max()), 25
        )

        # Try different methods
        methods = [
            "linear",
            "cubic",
            "nearest",
            "slinear",
            "pchip",
            "spline",
            "akima",
            "polynomial",
        ]

        for method in methods:
            interpolated = sample_tf.interpolate(new_periods, method=method)

            # Check shape and period values
            assert interpolated._dataset.transfer_function.shape == (25, 2, 2)
            assert np.array_equal(interpolated._dataset.period.values, new_periods)

            # Basic check that we have non-NaN values where expected
            for i_index in range(2):
                for j_index in range(2):
                    orig_tf = sample_tf._dataset.transfer_function.values[
                        :, i_index, j_index
                    ]
                    valid_indices = np.isfinite(orig_tf)
                    if np.any(valid_indices):
                        valid_periods = sample_tf.period[valid_indices]
                        in_range = (new_periods >= valid_periods.min()) & (
                            new_periods <= valid_periods.max()
                        )
                        # Check that we have values in the valid range
                        interp_values = interpolated._dataset.transfer_function.values[
                            in_range, i_index, j_index
                        ]
                        assert not np.all(np.isnan(interp_values))

    def test_inplace_interpolation(self, sample_tf):
        """Test inplace interpolation"""
        new_periods = np.logspace(
            np.log10(sample_tf.period.min()), np.log10(sample_tf.period.max()), 15
        )

        # Save a copy of the original
        original = sample_tf.copy()

        # Perform inplace interpolation
        result = sample_tf.interpolate(new_periods, inplace=True)

        # Check that result is None (inplace)
        assert result is None

        # Check that the original object was modified
        assert sample_tf._dataset.transfer_function.shape == (15, 2, 2)
        assert np.array_equal(sample_tf._dataset.period.values, new_periods)
        assert not np.array_equal(
            original._dataset.period.values, sample_tf._dataset.period.values
        )

    def test_all_nan_component(self):
        """Test handling a component that's all NaN values"""
        frequencies = np.logspace(-3, 3, 10)
        periods = 1.0 / frequencies

        tf = np.ones((len(periods), 2, 2), dtype=complex) * (1 + 1j)
        # Make one component all NaN
        tf[:, 0, 1] = np.nan + 1j * np.nan

        tf_base = TFBase(tf=tf, frequency=frequencies)

        new_periods = np.logspace(-3, 3, 20)
        interpolated = tf_base.interpolate(new_periods)

        # Check that all-NaN component remains all-NaN
        assert np.all(np.isnan(interpolated._dataset.transfer_function.values[:, 0, 1]))

        # Check that other components are properly interpolated
        assert not np.all(
            np.isnan(interpolated._dataset.transfer_function.values[:, 0, 0])
        )
        assert not np.all(
            np.isnan(interpolated._dataset.transfer_function.values[:, 1, 0])
        )
        assert not np.all(
            np.isnan(interpolated._dataset.transfer_function.values[:, 1, 1])
        )


if __name__ == "__main__":
    pytest.main([__file__])
