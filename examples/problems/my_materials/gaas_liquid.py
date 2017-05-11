# -*- coding: utf-8 -*-
#
from numpy import pi
mu0 = pi * 4.0e-7

# Gallium arsenide.
# [1] https://en.wikipedia.org/wiki/Gallium_arsenide
# [4] https://en.wikipedia.org/wiki/Electrical_conductivity
#
melting_point = 1511.0

magnetic_permeability = mu0 * (1.0 - 0.85e-10)
'''From figure 1 in :cite:`MRS:GaAs`, we take
.. math::
    \chi_v = -0.85\\times 10^{-10} \\frac{m^3}{kg},
so with :math:`\mu = \mu_0 (1+\chi_v)`
(https://en.wikipedia.org/wiki/Magnetic_susceptibility#Definition_of_volume_susceptibility),
we get the value.
'''

electrical_conductivity = 7.9e5
'''Blakemore :cite:`blakemore1961gallium` suggests an intrinsic
conductivity of :math:`3.0 \\times 10^{-7} / (Ohm m)`. However, "the
electrical conductivity in GaAs is in general determined by charge carriers
provided by impurities" :cite:`MRS:GaAs` and can vary roughly between
:math:`10^{-5}` and :math:`10^8` :cite:`Chu:SMP`.
This particular value is from private communication with Robert Luce.
'''

specific_heat_capacity = 0.434e3
'''Data from :cite:`Jordan85`.
'''

thermal_conductivity = 17.8
''':cite:`Jordan80` estimates the conductivity at the melting point by
means of relating it to the conductivity at melting point for the solid
state, i.e.,
.. math::
    K_s &= 10^2 \\times 10^{2.318 - 1.09  \log_{10}1511.0},\\\\
    K_l &= 2.5 K_s \\approx 17.8 \\frac{W}{m\cdot K}.
:cite:`NH92` also quotes this data (fig. 6) and suggests that the
conductivity is nearly independent of the temperature when liquid. The same
figure also suggests that this estimation is rather on the lower end of
possible values.  Moreover, :cite:`NH92` says:
"Although it is not clear whether the Wiedemann--Franz law is applicable to
molten GaAs, thermal conductivity estimated by this law is
:math:`29.2 W/m/K`."
'''


def density(T):
    '''Data from :cite:`Jordan85`, :cite:`Jordan80`.
    '''
    return 7.33e3 - 1.07 * T


def dynamic_viscosity(T):
    '''Data from :cite:`Jordan85`, :cite:`KH87`.
    '''
    return max(1.4e-3, 0.1 * 10**(-8.049 + 9814.0/T))
