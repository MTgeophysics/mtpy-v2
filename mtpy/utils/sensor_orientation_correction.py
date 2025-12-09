# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 23:28:58 2022

@author: jpeacock
"""

import cmath

# =============================================================================
# Imports
# =============================================================================
import copy
import math

import numpy as np


# =============================================================================


def correct4sensor_orientation(Z_prime, Bx=0, By=90, Ex=0, Ey=90, Z_prime_error=None):
    """Correct a Z-array for wrong orientation of the sensors.

    Assume, E' is measured by sensors orientated with the angles
        E'x: a
        E'y: b

    Assume, B' is measured by sensors orientated with the angles
        B'x: c
        B'y: d

    With those data, one obtained the impedance tensor Z':
        E' = Z' * B'

    Now we define change-of-basis matrices T,U so that
        E = T * E'
        B = U * B'

    =>   T contains the expression of the E'-basis in terms of E
    (the standard basis)
    and  U contains the expression of the B'-basis in terms of B
    (the standard basis)
    The respective expressions for E'x-basis vector and E'y-basis
    vector are the columns of T.
    The respective expressions for B'x-basis vector and B'y-basis
    vector are the columns of U.

    We obtain the impedance tensor in default coordinates as:

    E' = Z' * B' => T^(-1) * E = Z' * U^(-1) * B
                 => E = T * Z' * U^(-1) * B
                 => Z = T * Z' * U^(-1)
    :param Z_prime_error:
        Defaults to None.
    :param Ey:
        Defaults to 90.
    :param Ex:
        Defaults to 0.
    :param By:
        Defaults to 90.
    :param Bx:
        Defaults to 0.
    :param Z_prime: Impedance tensor to be adjusted.
    :return s: Adjusted impedance tensor.
    :rtype s: np.ndarray(Z_prime.shape, dtype='complex')
    :return s: Impedance tensor standard deviation in
        default orientation.
    :rtype s: np.ndarray(Z_prime.shape, dtype='real')
    """
    try:
        if len(Z_prime.shape) != 2:
            raise
        if Z_prime.shape != (2, 2):
            raise
        if Z_prime.dtype not in ["complex", "float", "int"]:
            raise
        Z_prime = np.matrix(Z_prime)
    except:
        raise MTpyError_input_arguments(
            "ERROR - Z array not valid!" + "Must be 2x2 complex array"
        )
    if Z_prime_error is not None:
        try:
            if len(Z_prime_error.shape) != 2:
                raise
            if Z_prime_error.shape != (2, 2):
                raise
            if Z_prime_error.dtype not in ["float", "int"]:
                raise
        except:
            raise MTpyError_input_arguments(
                "ERROR - Z-error array not" + "valid! Must be 2x2 real array"
            )
    T = np.matrix(np.zeros((2, 2)))
    U = np.matrix(np.zeros((2, 2)))

    dummy1 = cmath.rect(1, math.radians(Ex))

    T[0, 0] = np.real(dummy1)
    T[1, 0] = np.imag(dummy1)
    dummy2 = cmath.rect(1, math.radians(Ey))
    T[0, 1] = np.real(dummy2)
    T[1, 1] = np.imag(dummy2)

    dummy3 = cmath.rect(1, math.radians(Bx))
    U[0, 0] = np.real(dummy3)
    U[1, 0] = np.imag(dummy3)
    dummy4 = cmath.rect(1, math.radians(By))
    U[0, 1] = np.real(dummy4)
    U[1, 1] = np.imag(dummy4)

    try:
        z_arr = np.array(np.dot(T, np.dot(Z_prime, U.I)))
    except:
        raise MTpyError_input_arguments(
            "ERROR - Given angles do not"
            + "define basis for 2 dimensions - cannot convert Z'"
        )
    z_err_arr = copy.copy(Z_prime_error)

    # TODO: calculate error propagation

    return z_arr, z_err_arr
