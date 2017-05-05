# -*- coding: utf-8 -*-
#
from numpy import log10, pi

mu0 = pi * 4.0e-7

melting_point = 1511.0

# From figure 1 in [3], we take
#     chi_v = -1.23e-10 m^3/kg,
# so with
#     mu = mu0 * (1+chi_v),
# this gives
magnetic_permeability = mu0 * (1.0 - 1.23e-10)

electrical_conductivity = 3.0e-7
# Blakemore :cite:`blakemore1961gallium` suggests an intrinsic
# conductivity of :math:`3.0 \\times 10^{-7} / (Ohm m)`.


def density(T):
    # [5]
    return 5.32e3 - 9.91e-2 * T


def specific_heat_capacity(T):
    # [5]
    return 0.302e3 + 8.1e-2 * T


def thermal_conductivity(T):
    # # [5]
    # return 1e2 * 208 * T**(-1.09)
    # [6]
    return 1e2 * 10**(2.318 - 1.09 * log10(T))
