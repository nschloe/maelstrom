# -*- coding: utf-8 -*-
#
from numpy import pi

mu0 = pi * 4.0e-7

magnetic_permeability = mu0
density = 1.75e3
electrical_conductivity = 52.0e3


def thermal_conductivity(T):
    return 87.607 - 8.52865e-2 * T + 3.835e-5 * T ** 2 + 5.143515e-9 * T ** 3
