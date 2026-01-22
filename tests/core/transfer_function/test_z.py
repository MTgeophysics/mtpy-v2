# -*- coding: utf-8 -*-
"""
Pytest version of Z (impedance tensor) tests

Created on Tue Nov  8 13:04:38 2022

@author: jpeacock
"""
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mtpy.core.transfer_function import MT_TO_OHM_FACTOR
from mtpy.core.transfer_function.z import Z


# =============================================================================
# Session Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def empty_z():
    """Empty Z object for testing initialization."""
    return Z()


@pytest.fixture(scope="session")
def z_res_phase_data():
    """Sample resistivity and phase data for testing."""
    return {
        "resistivity": np.array([[[5.0, 100.0], [100.0, 5.0]]]),
        "phase": np.array([[[90.0, 45.0], [-135.0, -90.0]]]),
        "resistivity_error": np.array([[0.1, 2], [2, 0.1]]),
        "phase_error": np.array([[0.573, 0.573], [0.573, 0.573]]),
        "resistivity_model_error": np.array([[0.1, 2], [2, 0.1]]),
        "phase_model_error": np.array([[0.573, 0.573], [0.573, 0.573]]),
        "frequency": np.array([1]),
    }


@pytest.fixture(scope="session")
def z_with_res_phase(z_res_phase_data):
    """Z object initialized with resistivity and phase."""
    z = Z()
    z.set_resistivity_phase(
        z_res_phase_data["resistivity"],
        z_res_phase_data["phase"],
        z_res_phase_data["frequency"],
        res_error=z_res_phase_data["resistivity_error"],
        phase_error=z_res_phase_data["phase_error"],
        res_model_error=z_res_phase_data["resistivity_model_error"],
        phase_model_error=z_res_phase_data["phase_model_error"],
    )
    return z


@pytest.fixture(scope="session")
def z_for_static_shift():
    """Z object for static shift removal tests."""
    z = Z()
    z.z = np.array([[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]])
    return z


@pytest.fixture(scope="session")
def z_for_distortion():
    """Z object for distortion removal tests."""
    z_array = np.array([[[0.1 - 0.1j, 10.0 + 10.0j], [-10.0 - 10.0j, -0.1 + 0.1j]]])
    z = Z()
    z.z = z_array

    distortion = np.array(np.real([[0.5, 0.1], [0.2, 0.6]]))

    dz = Z()
    dz.z = np.array(np.dot(distortion, z_array)).reshape((1, 2, 2))

    return {"z": z, "dz": dz, "distortion": distortion}


@pytest.fixture(scope="session")
def z_for_invariants():
    """Z object for invariants testing."""
    return Z(
        z=np.array(
            [
                [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
            ]
        )
    )


@pytest.fixture(scope="session")
def z_for_analysis():
    """Z object for analysis testing (multiple periods)."""
    return Z(
        z=np.array(
            [
                [
                    [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
                    [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
                ],
                [
                    [-1.420305 - 1.02897j, 603.44306 + 814.4988j],
                    [-10.96444 - 21.4191j, 1.95081 + 1.52367j],
                ],
                [
                    [-70.420305 - 111.02897j, 3.44306 + 214.4988j],
                    [-19.96444 - 56.4191j, 81.95081 + 314.52367j],
                ],
            ]
        ),
        frequency=np.array([10, 1, 0.1]),
    )


@pytest.fixture(scope="session")
def z_units_data():
    """Z data for units testing."""
    z_mt = np.array(
        [
            [-7.420305 - 15.02897j, 53.44306 + 114.4988j],
            [-49.96444 - 116.4191j, 11.95081 + 21.52367j],
        ]
    )
    z_ohm = z_mt / MT_TO_OHM_FACTOR
    return {"z_mt": z_mt, "z_ohm": z_ohm}


# =============================================================================
# Tests for Empty Z Initialization
# =============================================================================


class TestZInitialize:
    """Tests for empty Z initialization."""

    def test_n_periods(self, empty_z):
        assert empty_z.n_periods == 0

    def test_is_empty(self, empty_z):
        assert empty_z._is_empty()

    def test_has_tf(self, empty_z):
        assert not empty_z._has_tf()

    def test_has_tf_error(self, empty_z):
        assert not empty_z._has_tf_error()

    def test_has_tf_model_error(self, empty_z):
        assert not empty_z._has_tf_model_error()

    @pytest.mark.parametrize(
        "attr",
        [
            "z",
            "z_error",
            "z_model_error",
            "resistivity",
            "phase",
            "resistivity_error",
            "phase_error",
            "resistivity_model_error",
            "phase_model_error",
        ],
    )
    def test_empty_properties(self, empty_z, attr):
        """Test that all Z properties are None when empty."""
        assert getattr(empty_z, attr) is None

    def test_units_default(self, empty_z):
        assert empty_z.units == "mt"


# =============================================================================
# Tests for Z with Resistivity and Phase
# =============================================================================


class TestZSetResPhase:
    """Tests for Z object with resistivity and phase data."""

    def test_is_empty(self, z_with_res_phase):
        assert not z_with_res_phase._is_empty()

    def test_has_tf(self, z_with_res_phase):
        assert z_with_res_phase._has_tf()

    def test_has_tf_error(self, z_with_res_phase):
        assert z_with_res_phase._has_tf_error()

    def test_has_tf_model_error(self, z_with_res_phase):
        assert z_with_res_phase._has_tf_model_error()

    def test_resistivity(self, z_with_res_phase, z_res_phase_data):
        assert np.isclose(
            z_with_res_phase.resistivity, z_res_phase_data["resistivity"]
        ).all()

    def test_phase(self, z_with_res_phase, z_res_phase_data):
        assert np.isclose(z_with_res_phase.phase, z_res_phase_data["phase"]).all()

    def test_resistivity_error(self, z_with_res_phase, z_res_phase_data):
        assert np.isclose(
            z_res_phase_data["resistivity_error"],
            z_with_res_phase.resistivity_error,
            atol=0.001,
        ).all()

    def test_phase_error(self, z_with_res_phase, z_res_phase_data):
        assert np.isclose(
            z_with_res_phase.phase_error,
            z_res_phase_data["phase_error"],
        ).all()

    def test_resistivity_model_error(self, z_with_res_phase, z_res_phase_data):
        assert np.isclose(
            z_res_phase_data["resistivity_model_error"],
            z_with_res_phase.resistivity_model_error,
            atol=0.001,
        ).all()

    def test_phase_model_error(self, z_with_res_phase, z_res_phase_data):
        assert np.isclose(
            z_with_res_phase.phase_model_error,
            z_res_phase_data["phase_model_error"],
        ).all()

    def test_rotation_plus_5_ned(self, z_with_res_phase):
        """Test rotation +5 degrees in NED coordinate system."""
        zr = z_with_res_phase.rotate(5, coordinate_reference_frame="ned")
        assert np.isclose(zr.phase_tensor.azimuth[0], 40, atol=0.1)
        assert np.isclose(zr.invariants.strike[0], 40, atol=0.1)

    def test_rotation_minus_5_ned(self, z_with_res_phase):
        """Test rotation -5 degrees in NED coordinate system."""
        zr = z_with_res_phase.rotate(-5, coordinate_reference_frame="ned")
        assert np.isclose(zr.phase_tensor.azimuth[0], 50, atol=0.1)
        assert np.isclose(zr.invariants.strike[0], 50, atol=0.1)

    def test_rotation_plus_5_enu(self, z_with_res_phase):
        """Test rotation +5 degrees in ENU coordinate system."""
        zr = z_with_res_phase.rotate(5, coordinate_reference_frame="enu")
        assert np.isclose(zr.phase_tensor.azimuth[0], 50, atol=0.1)
        assert np.isclose(zr.invariants.strike[0], 50, atol=0.1)

    def test_rotation_minus_5_enu(self, z_with_res_phase):
        """Test rotation -5 degrees in ENU coordinate system."""
        zr = z_with_res_phase.rotate(-5, coordinate_reference_frame="enu")
        assert np.isclose(zr.phase_tensor.azimuth[0], 40, atol=0.1)
        assert np.isclose(zr.invariants.strike[0], 40, atol=0.1)

    def test_rotation_angle(self, z_res_phase_data):
        """Test that rotation angle is stored correctly."""
        # Create a fresh Z object to avoid cumulative rotations from other tests
        z = Z()
        z.set_resistivity_phase(
            z_res_phase_data["resistivity"],
            z_res_phase_data["phase"],
            z_res_phase_data["frequency"],
        )
        zr = z.rotate(40)
        assert np.all(zr.rotation_angle == np.array([40]))


# =============================================================================
# Tests for Static Shift Removal
# =============================================================================


class TestRemoveStaticShift:
    """Tests for static shift removal functionality."""

    @pytest.fixture(scope="class")
    def ss_factors(self):
        """Static shift factors."""
        return {"ss_x": 0.5, "ss_y": 1.5}

    @pytest.fixture(scope="class")
    def expected_ss_z(self):
        """Expected Z after static shift removal."""
        return np.array(
            [
                [
                    [0.14142136 - 0.14142136j, 14.14213562 + 14.14213562j],
                    [-8.16496581 - 8.16496581j, -0.08164966 + 0.08164966j],
                ]
            ]
        )

    def test_remove_ss(self, z_for_static_shift, ss_factors, expected_ss_z):
        """Test static shift removal produces expected result."""
        new_z = z_for_static_shift.remove_ss(
            ss_factors["ss_x"], ss_factors["ss_y"], inplace=False
        )
        assert np.allclose(new_z.z, expected_ss_z)

    def test_ss_factors(self, z_for_static_shift, ss_factors):
        """Test that static shift factors are correctly applied."""
        new_z = z_for_static_shift.remove_ss(
            ss_factors["ss_x"], ss_factors["ss_y"], inplace=False
        )
        assert np.allclose(
            (z_for_static_shift.z / new_z.z) ** 2,
            np.array([[[0.5 + 0.0j, 0.5 + 0.0j], [1.5 - 0.0j, 1.5 - 0.0j]]]),
        )

    def test_set_factor_fail_single(self, z_for_static_shift):
        """Test that invalid single factor raises ValueError."""
        with pytest.raises(ValueError):
            z_for_static_shift.remove_ss("k")

    def test_set_factor_fail_too_many(self, z_for_static_shift):
        """Test that too many factors raises ValueError."""
        with pytest.raises(ValueError):
            z_for_static_shift.remove_ss([1, 2, 3])


# =============================================================================
# Tests for Distortion Removal
# =============================================================================


class TestRemoveDistortion:
    """Tests for distortion removal functionality."""

    def test_remove_distortion(self, z_for_distortion):
        """Test that distortion removal recovers original Z."""
        new_z = z_for_distortion["dz"].remove_distortion(z_for_distortion["distortion"])
        assert np.allclose(new_z.z, z_for_distortion["z"].z)

    def test_fail_bad_input_shape_too_many(self, z_for_distortion):
        """Test that invalid shape (too many dimensions) raises ValueError."""
        with pytest.raises(ValueError):
            z_for_distortion["z"].remove_distortion(np.random.rand(4, 3, 3, 3))

    def test_fail_bad_input_shape_not_z_shape(self, z_for_distortion):
        """Test that invalid shape (wrong dimensions) raises ValueError."""
        with pytest.raises(ValueError):
            z_for_distortion["z"].remove_distortion(np.random.rand(4, 3, 3))

    def test_fail_bad_input_singular_matrix(self, z_for_distortion):
        """Test that singular matrix raises ValueError."""
        with pytest.raises(ValueError):
            z_for_distortion["z"].remove_distortion(np.array([[0, 1], [0, 0]]))


# =============================================================================
# Tests for Z Invariants
# =============================================================================


class TestInvariants:
    """Tests for Z invariants calculations."""

    @pytest.mark.parametrize(
        "attr,expected",
        [
            ("normalizing_real", 51.753349),
            ("normalizing_imag", 115.504607),
            ("anisotropic_real", 0.190142),
            ("anisotropic_imag", 0.158448),
            ("electric_twist", 0.071840),
            ("phase_distortion", -0.015665),
            ("dimensionality", 0.0503231),
            ("structure_3d", -0.180011),
            ("strike", 17.266664),
            ("strike_error", 5.185204),
        ],
    )
    def test_invariant_values(self, z_for_invariants, attr, expected):
        """Test that invariant calculations match expected values."""
        actual = getattr(z_for_invariants.invariants, attr)[0]
        assert np.isclose(actual, expected, atol=1e-5)

    def test_estimate_dimensionality(self, z_for_invariants):
        """Test dimensionality estimation."""
        assert np.all(z_for_invariants.estimate_dimensionality() == np.array([1]))

    def test_rotate_plus_5(self, z_for_invariants):
        """Test that rotation +5 degrees updates strike correctly."""
        new_z = z_for_invariants.rotate(5)
        assert np.isclose(new_z.invariants.strike[0], 12.266664, atol=1e-5)

    def test_rotate_minus_5(self, z_for_invariants):
        """Test that rotation -5 degrees updates strike correctly."""
        new_z = z_for_invariants.rotate(-5)
        assert np.isclose(new_z.invariants.strike[0], 22.266664, atol=1e-5)


# =============================================================================
# Tests for Z Analysis
# =============================================================================


class TestZAnalysis:
    """Tests for Z analysis methods."""

    def test_estimate_dimensionality(self, z_for_analysis):
        """Test dimensionality estimation for multiple periods."""
        dim = z_for_analysis.estimate_dimensionality()
        assert np.all(dim == np.array([1, 2, 3]))

    def test_estimate_distortion(self, z_for_analysis):
        """Test distortion estimation."""
        d, d_err = z_for_analysis.estimate_distortion()

        # Expected distortion matrix (using R A R^-1 rotation)
        expected_d = np.array([[0.99586079, 1.17387806], [0.09359908, 1.1030191]])
        expected_d_err = np.array([[0.57735027, 0.57735027], [0.57735027, 0.57735027]])

        assert np.isclose(d, expected_d).all()
        assert np.isclose(d_err, expected_d_err).all()

    def test_remove_distortion(self, z_for_analysis):
        """Test distortion removal on analysis data."""
        new_z = z_for_analysis.remove_distortion()

        # Expected Z after distortion removal (using R A R^-1)
        expected_z = np.array(
            [
                [
                    [
                        51.05044024 + 121.47184822j,
                        45.43885369 + 102.19531416j,
                    ],
                    [
                        -49.62988816 - 115.85361851j,
                        6.9788229 + 10.84140998j,
                    ],
                ],
                [
                    [
                        11.43488341 + 24.28579241j,
                        670.98211541 + 906.97720461j,
                    ],
                    [
                        -10.91072189 - 21.47943583j,
                        -55.16903678 - 75.5821594j,
                    ],
                ],
                [
                    [
                        -54.86571355 - 56.88761188j,
                        -93.46996499 - 134.14819458j,
                    ],
                    [
                        -13.44406414 - 46.32238172j,
                        82.22841591 + 296.53141905j,
                    ],
                ],
            ]
        )

        assert np.isclose(new_z.z, expected_z).all()

    def test_depth_of_investigation(self, z_for_analysis):
        """Test depth of investigation calculation."""
        doi = z_for_analysis.estimate_depth_of_investigation()

        expected_results = {
            "depth_det": np.array([1987.75038069, 24854.87498141, 283705.23967805]),
            "depth_xy": np.array([2011.03691158, 161332.55006745, 341429.42016186]),
            "depth_yx": np.array([2016.30231674, 3829.64228158, 95249.86168927]),
            "depth_min": np.array([1987.75038069, 3829.64228158, 95249.86168927]),
            "depth_max": np.array([2016.30231674, 161332.55006745, 341429.42016186]),
        }

        for key, expected in expected_results.items():
            assert np.allclose(doi[key], expected)

        assert np.allclose(doi["period"], z_for_analysis.period)


# =============================================================================
# Tests for Unit Conversions
# =============================================================================


class TestUnits:
    """Tests for Z unit conversions between MT and Ohm."""

    def test_initialize_with_units_ohm(self, z_units_data):
        """Test initialization with Ohm units."""
        z_obj = Z(z=z_units_data["z_ohm"], units="ohm")

        # Internal storage should be in MT units
        assert np.allclose(
            z_units_data["z_mt"], z_obj._dataset.transfer_function.values
        )
        # Output should be in Ohm units
        assert np.allclose(z_units_data["z_ohm"], z_obj.z)
        assert z_obj.units == "ohm"

    def test_initialize_with_units_mt(self, z_units_data):
        """Test initialization with MT units."""
        z_obj = Z(z=z_units_data["z_mt"], units="mt")

        # Internal storage and output should both be in MT units
        assert np.allclose(
            z_units_data["z_mt"], z_obj._dataset.transfer_function.values
        )
        assert np.allclose(z_units_data["z_mt"], z_obj.z)
        assert z_obj.units == "mt"

    def test_units_change_ohm_to_mt(self, z_units_data):
        """Test changing units from Ohm to MT."""
        z_obj = Z(z=z_units_data["z_ohm"], units="ohm")
        z_obj.units = "mt"

        # Internal storage should remain in MT units
        assert np.allclose(
            z_units_data["z_mt"], z_obj._dataset.transfer_function.values
        )
        # Output should now be in MT units
        assert np.allclose(z_units_data["z_mt"], z_obj.z)
        assert z_obj.units == "mt"

    def test_units_change_mt_to_ohm(self, z_units_data):
        """Test changing units from MT to Ohm."""
        z_obj = Z(z=z_units_data["z_mt"], units="mt")
        z_obj.units = "ohm"

        # Internal storage should remain in MT units
        assert np.allclose(
            z_units_data["z_mt"], z_obj._dataset.transfer_function.values
        )
        # Output should now be in Ohm units
        assert np.allclose(z_units_data["z_ohm"], z_obj.z)
        assert z_obj.units == "ohm"

    def test_set_unit_fail_bad_type(self, z_units_data):
        """Test that invalid unit type raises TypeError."""
        z_obj = Z(z=z_units_data["z_mt"])
        with pytest.raises(TypeError):
            z_obj.units = 4

    def test_set_unit_fail_bad_choice(self, z_units_data):
        """Test that invalid unit choice raises ValueError."""
        z_obj = Z(z=z_units_data["z_mt"])
        with pytest.raises(ValueError):
            z_obj.units = "ants"

    def test_phase_tensor_equal(self, z_units_data):
        """Test that phase tensor is unit-independent."""
        z_ohm = Z(z=z_units_data["z_ohm"], units="ohm")
        z_mt = Z(z=z_units_data["z_mt"], units="mt")

        assert np.allclose(z_ohm.phase_tensor.pt, z_mt.phase_tensor.pt)

    def test_resistivity_phase_equal(self, z_units_data):
        """Test that resistivity and phase are unit-independent."""
        z_ohm = Z(z=z_units_data["z_ohm"], units="ohm")
        z_mt = Z(z=z_units_data["z_mt"], units="mt")

        assert np.allclose(z_ohm.resistivity, z_mt.resistivity)
        assert np.allclose(z_ohm.phase, z_mt.phase)


# =============================================================================
# Additional Tests for Enhanced Coverage
# =============================================================================


class TestZMultiplePeriods:
    """Tests for Z with multiple periods."""

    @pytest.fixture(scope="class")
    def multi_period_z(self):
        """Z object with multiple periods."""
        n_periods = 10
        z_array = np.random.rand(n_periods, 2, 2) + 1j * np.random.rand(n_periods, 2, 2)
        frequency = np.logspace(-3, 3, n_periods)
        return Z(z=z_array, frequency=frequency)

    def test_n_periods(self, multi_period_z):
        assert multi_period_z.n_periods == 10

    def test_z_shape(self, multi_period_z):
        assert multi_period_z.z.shape == (10, 2, 2)

    def test_resistivity_shape(self, multi_period_z):
        assert multi_period_z.resistivity.shape == (10, 2, 2)

    def test_phase_shape(self, multi_period_z):
        assert multi_period_z.phase.shape == (10, 2, 2)


class TestZRotation:
    """Tests for Z rotation functionality."""

    @pytest.fixture(scope="class")
    def rotation_z(self, z_res_phase_data):
        """Z object for rotation tests."""
        z = Z()
        z.set_resistivity_phase(
            z_res_phase_data["resistivity"],
            z_res_phase_data["phase"],
            z_res_phase_data["frequency"],
        )
        return z

    @pytest.mark.parametrize("angle", [0, 15, 30, 45, 90, 180, -45, -90])
    def test_rotation_various_angles(self, rotation_z, angle):
        """Test rotation at various angles produces valid results."""
        rotated = rotation_z.rotate(angle)

        assert rotated is not None
        assert rotated._has_tf()
        assert rotated.z is not None

    def test_rotation_zero_preserves_data(self, rotation_z):
        """Test that 0 degree rotation preserves original data."""
        rotated = rotation_z.rotate(0)

        # Should be very close to original (within numerical precision)
        assert np.allclose(rotation_z.z, rotated.z, rtol=1e-10)

    def test_rotation_360_equivalent_to_zero(self, rotation_z):
        """Test that 360 degree rotation is equivalent to 0."""
        rotated_0 = rotation_z.rotate(0)
        rotated_360 = rotation_z.rotate(360)

        assert np.allclose(rotated_0.z, rotated_360.z, rtol=1e-10)

    @pytest.mark.parametrize("coord_frame", ["ned", "enu"])
    def test_rotation_coordinate_frames(self, rotation_z, coord_frame):
        """Test rotation in different coordinate frames."""
        rotated = rotation_z.rotate(45, coordinate_reference_frame=coord_frame)

        assert rotated is not None
        assert rotated._has_tf()


class TestZErrorHandling:
    """Tests for Z error handling and edge cases."""

    def test_empty_z_operations_resistivity(self):
        """Test that empty Z returns None for resistivity."""
        z = Z()
        assert z.resistivity is None

    def test_empty_z_operations_phase(self):
        """Test that empty Z returns None for phase."""
        z = Z()
        assert z.phase is None

    def test_z_with_only_tf(self):
        """Test Z with only transfer function, no errors."""
        z = Z(z=np.ones((1, 2, 2)) + 1j * np.ones((1, 2, 2)))

        assert z._has_tf()
        assert not z._has_tf_error()
        assert not z._has_tf_model_error()
        assert z.resistivity is not None
        assert z.phase is not None

    def test_z_with_tf_and_error(self):
        """Test Z with transfer function and errors."""
        z = Z(
            z=np.ones((1, 2, 2)) + 1j * np.ones((1, 2, 2)),
            z_error=np.ones((1, 2, 2)) * 0.1,
        )

        assert z._has_tf()
        assert z._has_tf_error()
        assert not z._has_tf_model_error()


class TestZCopy:
    """Tests for Z copy functionality."""

    def test_copy_creates_independent_object(self, z_with_res_phase):
        """Test that copy creates an independent object."""
        z_copy = z_with_res_phase.copy()

        # Should be equal but not the same object
        assert z_copy == z_with_res_phase
        assert z_copy is not z_with_res_phase

        # Modifying copy should not affect original
        z_copy.frequency = np.array([10])
        assert not np.allclose(z_copy.frequency, z_with_res_phase.frequency)


class TestZFrequencyPeriod:
    """Tests for frequency and period relationship in Z."""

    def test_frequency_period_relationship(self):
        """Test that frequency and period are reciprocals."""
        frequency = np.logspace(-3, 3, 10)
        z = Z(frequency=frequency)

        assert np.allclose(z.frequency, 1.0 / z.period)
        assert np.allclose(z.period, 1.0 / z.frequency)

    def test_set_frequency(self):
        """Test setting frequency updates period."""
        z = Z(frequency=np.array([1, 2, 3]))
        new_freq = np.array([10, 20, 30])
        z.frequency = new_freq

        assert np.allclose(z.frequency, new_freq)
        assert np.allclose(z.period, 1.0 / new_freq)

    def test_set_period(self):
        """Test setting period updates frequency."""
        z = Z(frequency=np.array([1, 2, 3]))
        new_period = np.array([0.1, 0.2, 0.3])
        z.period = new_period

        assert np.allclose(z.period, new_period)
        assert np.allclose(z.frequency, 1.0 / new_period)


@pytest.mark.parametrize(
    "n_periods,frequency_range",
    [
        (1, (1, 1)),
        (5, (-1, 1)),
        (10, (-3, 3)),
        (20, (-3, 3)),
    ],
)
def test_z_various_sizes(n_periods, frequency_range):
    """Test Z with various period sizes."""
    z_array = np.random.rand(n_periods, 2, 2) + 1j * np.random.rand(n_periods, 2, 2)
    frequency = np.logspace(frequency_range[0], frequency_range[1], n_periods)

    z = Z(z=z_array, frequency=frequency)

    assert z.z.shape == (n_periods, 2, 2)
    assert z.n_periods == n_periods
    assert z._has_tf()
    assert z.resistivity.shape == (n_periods, 2, 2)
    assert z.phase.shape == (n_periods, 2, 2)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
