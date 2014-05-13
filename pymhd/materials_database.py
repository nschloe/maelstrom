# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyMHD.
#
#  PyMHD is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PyMHD is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyMHD.  If not, see <http://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------
# Sources:
#
# Magnetic permeability.
# https://en.wikipedia.org/wiki/Permeability_(electromagnetism)#Values_for_some_common_materials
# http://en.wikipedia.org/wiki/Vacuum_permeability
# http://vk1od.net/antenna/conductors/loss.htm
# https://en.wikipedia.org/wiki/Diamagnetism
#
# Electrical conductivity.
# https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity
#
# Density in kg/m^3 (liquid density for silver).
#
# Dynamic viscosity in (Pa s) (equivalent to Ns/m^2, or
# kg/(ms)).
#  * http://www.engineeringtoolbox.com/air-absolute-kinematic-viscosity-d_601.html
#  * Copper: Its Trade, Manufacture, Use and Environmental Status.
#    books.google.de/books?isbn=0871706563
#
# Molar mass in g/mol.
#
# Molar heat capacity in J/(mol K).
#  * http://en.wikipedia.org/wiki/Heat_capacity
#
# Specific heat capacity in J/(kg K).
# The specific heat is calculated as
#    shc = (molar heat capacity) / (molar mass)
# With the figures from above, this pretty much coincides with the data
# from
# <http://www2.ucdsb.on.ca/tiss/stretton/database/Specific_Heat_Capacity_Table.html>.
#
# Thermal conductivity in W/(m K).
#  * http://en.wikipedia.org/wiki/List_of_thermal_conductivities
#
# Thermal diffusivity in m^2/s.
#   td = (thermal conductivity) / (density*(specific heat capacity))
#   * https://en.wikipedia.org/wiki/Thermal_diffusivity
#
'''
From thermal expansion coefficient to density
---------------------------------------------

For the thermal expansion coefficient :math:`\\alpha`, we have

.. math::
    \\alpha V = \\frac{dV}{dT},

so with

.. math::
    \\frac{d\\rho}{dT}
    = \\frac{d(m/V)}{dT}
    = \\frac{-\\frac{dV}{dT}m}{V^2}
    = -\\alpha \\rho

it follows

.. math::
    \\rho(T) = C \exp(-A(T)),

where :math:`\\frac{dA}{dT}=\\alpha`.
'''

from numpy import pi, log, log10, exp

mu0 = pi * 4.0e-7


def get_material(string):
    '''Material factory.
    '''
    if string == 'air':
        return Air
    if string == 'argon':
        return Argon
    elif string == 'GaAs (liquid)':
        return GaAsLiquid
    elif string == 'GaAs (solid)':
        return GaAsSolid
    elif string == 'graphite EK90':
        return GraphiteEK90
    elif string == 'graphite EK98':
        return GraphiteEK98
    elif string == 'porcelain':
        return Porcelain
    else:
        raise ValueError('Illegal material \'%s\'.' % string)
    return


class Air(object):
    # [1] http://en.wikipedia.org/wiki/Density_of_air
    # [2] https://en.wikipedia.org/wiki/Heat_capacity#Table_of_specific_heat_capacities
    # [3] http://www.engineeringtoolbox.com/air-properties-d_156.html
    # [4] http://profmaster.blogspot.ide/2009/01/thermal-conductivity-of-air-vs.html
    # [5] http://www.ohio.edu/mechanical/thermo/property_tables/air/air_Cp_Cv.html
    # [6] https://en.wikipedia.org/wiki/Permeability_(electromagnetism)#Values_for_some_common_materials

    # [6]
    magnetic_permeability = 1.00000037 * mu0
    electrical_conductivity = 0.0

    @staticmethod
    def density(T):
        # [1]
        return 1.2922 * 273.15/T

    @staticmethod
    def specific_heat_capacity(T):
        # least-squares fit from [5]
        return 2.196e-13 * T**4 \
            - 8.916e-10 * T**3 \
            + 1.234e-06 * T**2 \
            - 0.0004807 * T \
            + 1.06

    @staticmethod
    def thermal_conductivity(T):
        # [4]
        return 1.5207e-11 * T**3 \
            - 4.8574e-8 * T**2 \
            + 1.0184e-4 * T \
            - 0.00039333


class Argon(object):
    magnetic_permeability = mu0
    electrical_conductivity = 0.0

    @staticmethod
    def density(T):
        '''https://en.wikipedia.org/wiki/Argon tells us that for 0°C, 101.325
        kPa, the density is 1.784 kg/m^3.  Assuming then that only density and
        temperature change, the ideal gas law :math:`PV = nRT` gives us the
        complete formula.
        '''
        return 1.784 * 273.15 / T


class BoronTrioxide(object):
    # [1] https://en.wikipedia.org/wiki/Boron_trioxide
    #
    # [1]
    melting_point = 723.0
    # [1]
    boiling_point = 2133.0

    # TODO include temperature dependence from [2]
    density = 1.5e3
    ''':cite:`NMH65`.'''

    # TODO include temperature dependence from :cite:`Setze57`
    # 30.0 * 4.184 / 69.6182e-3 =
    specific_heat_capacity = 1.802e3
    ''':cite:`Setze57`.'''

    electrical_conductivity = 1.0 / 2.2e6
    ''':cite:`Setze57` (value for :math:`1000K`).'''

    ## TODO fill in proper value
    #thermal_conductivity = 27.0
    # This is the value for boron, cf.
    # <http://www.americanelements.com/bb.html>.

    # TODO find out actual value
    magnetic_permeability = mu0


class Copper(object):
    # [1] http://en.wikipedia.org/wiki/Copper
    # [3] http://aries.ucsd.edu/LIB/PROPS/PANOS/cu.html
    # [4] Numerical modeling of induction heating of long workpieces;
    #     Chaboudez et al.;
    #     IEEE Transactions on Magnetics, 30 (6), Nov. 1994;
    #     <https://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=334291>.
    # [5] https://en.wikipedia.org/wiki/Permeability_(electromagnetism)#Values_for_some_common_materials
    #
    # [5]
    magnetic_permeability = 0.999994*mu0

    @staticmethod
    def electrical_conductivity(T):
        '''The least-squares fit from :cite:`Mat79` is

        .. math::
            \sigma(T) = \\frac{1.0}{- 0.7477\\times 10^{-8}
                                    + 0.007792\\times 10^{-8} T
                                    }.

        In better accordance with [1] (for :math:`T=293`) is the expression
        from [4],

        .. math::
            \sigma(T) = \\frac{1.0}{-3.033\\times 10^{-9}
                                    + 68.85\\times 10^{-12} T
                                    - 6.72\\times 10^{-15} T^2
                                    + 8.56\\times 10^{-18} T^3
                                    }.
        '''
        return 1.0 / (-3.033e-9 + 68.85e-12*T - 6.72e-15*T**2 + 8.56e-18*T**3)

    @staticmethod
    def density(T):
        '''[3] specifies the thermal expansion coefficient as

        .. math::
           \\alpha(T) = 10^{-6} (13.251
                                 + 6.903\\times 10^{-3} T
                                 + 8.5306\\times 10^{-7} T^2
                                 ),

        so (with :math:`\\rho(293)=8.96\\times 10^3`) the density is derived.
        '''
        return 8.9975852012753705e3 \
            * exp(-1.0e-6*(13.251 * T
                           + 6.903e-3 / 2.0 * T**2
                           + 8.5306e-7 / 3.0 * T**3
                           ))

    @staticmethod
    def specific_heat_capacity(T):
        # [3]
        return 316.21 + 0.3177*T - 3.4936e-4*T**2 + 1.661e-7*T**3

    @staticmethod
    def thermal_conductivity(T):
        # [3]
        return 420.75 - 6.8493e-2 * T


class GaAsLiquid(object):
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

    @staticmethod
    def density(T):
        '''Data from :cite:`Jordan85`, :cite:`Jordan80`.
        '''
        return 7.33e3 - 1.07 * T

    @staticmethod
    def dynamic_viscosity(T):
        '''Data from :cite:`Jordan85`, :cite:`KH87`.
        '''
        return max(1.4e-3, 0.1 * 10**(-8.049 + 9814.0/T))


class GaAsSolid(object):

    melting_point = 1511.0

    # From figure 1 in [3], we take
    #     chi_v = -1.23e-10 m^3/kg,
    # so with
    #    mu = mu0 * (1+chi_v),
    # this gives
    magnetic_permeability = mu0 * (1.0 - 1.23e-10)

    electrical_conductivity = 3.0e-7
    '''Blakemore :cite:`blakemore1961gallium` suggests an intrinsic
    conductivity of :math:`3.0 \\times 10^{-7} / (Ohm m)`.
    '''

    @staticmethod
    def density(T):
        # [5]
        return 5.32e3 - 9.91e-2 * T

    @staticmethod
    def specific_heat_capacity(T):
        # [5]
        return 0.302e3 + 8.1e-2 * T

    @staticmethod
    def thermal_conductivity(T):
        ## [5]
        #return 1e2 * 208 * T**(-1.09)
        # [6]
        return 1e2 * 10**(2.318 - 1.09 * log10(T))


class SiC(object):
    # Silicon carbide.
    # [1] https://en.wikipedia.org/wiki/Silicon_carbide
    # [2] http://www.rgpballs.com/srv/sheet.ashx?cid=20&lng=EN
    # [3] https://en.wikipedia.org/wiki/Silicon_carbide#Structure_and_properties
    # [4] http://www.ioffe.rssi.ru/SVA/NSM/Semicond/SiC/thermal.html
    # [6] Electrical conductivity of silicon carbide composites and fibers;
    #     R. Scholz, F. dos Santos Marques, B. Riccardi;
    #     <https://www.sciencedirect.com/science/article/pii/S0022311502009509>.
    # [7] http://www.ceramics.nist.gov/srd/summary/scdscs.htm
    # [8] Y.S. Touloukian and E.M. Buyko;
    #     Specific Heat-Nonmetallic Solids;
    #     Vol. 5 of Thermophysical Properties of Matter: The TPRC Data Series;
    #     edited by Y.S. Touloukian and C.Y. Ho;
    #     New York, 1970.
    #
    # [1]
    melting_point = 3003.0
    # [2]
    magnetic_permeability = mu0
    # The specific heat is temperature-dependent, see [5], [7], [8].
    # Take the value for 1500 degrees Celsius here.
    # TODO check out [8]
    specific_heat_capacity = 1.336e3

    # TODO respect temperature-dependence here
    electrical_conductivity = 300.0
    '''Electrical conductivity is highly temperature-dependent, see
    :cite:`SSR02`. Take an approximate value for :math:`T=1000K` here.
    '''

    @staticmethod
    def density(T):
        '''
        [3], [4], single-crystal 3C-SiC
        A quadratic least-squares fit for the data from [4]
        (single-crystal 3C-SiC) gives

        .. math::
            \\alpha(T) = -1.05 \\times 10^{-12} T^2
                         + 3.717 \\times 10^{-9} T
                         + 2.314 \\times 10^{-6}.

        With :math:`\\rho(293)=3.21 \\times 10^3`, this gives
        '''
        return 3.2126613855078426e3 \
            * exp(1.05e-12/3.0*T**3 - 3.717e-09/2.0*T**2 - 2.314e-06*T)

    @staticmethod
    def thermal_conducivity(T):
        ''':cite:`NMH97`.
        '''
        return 61.1e3 / (T - 115.0)

    @staticmethod
    def thermal_diffusivity(T):
        ''':cite:`NMH97`.
        '''
        return 14.6e-3 / (T - 207.0)


class CarbonSteel(object):
    # [1] https://en.wikipedia.org/wiki/Carbon_steel
    # [2] https://en.wikipedia.org/wiki/Permeability_(electromagnetism)#Values_for_some_common_materials
    # [3] https://en.wikipedia.org/wiki/List_of_thermal_conductivities
    # [4] https://en.wikipedia.org/wiki/Heat_capacity#Table_of_specific_heat_capacities
    #
    # [1]
    magnetic_permeability = 100*mu0
    density = 7.85e3
    # [3]
    thermal_conductivity = 50.0
    # stainless steel @293K:
    electrical_conductivity = 1.180e6
    # [4]
    specific_heat_capacity = 0.466e3


class Graphite(object):
    # [1] https://en.wikipedia.org/wiki/Graphite
    # [2] https://en.wikipedia.org/wiki/Carbon
    # [3] The electric and magnetic properties of graphite;
    #     R.R. Haering;
    #     1957;
    #     http://digitool.library.mcgill.ca/R/?func=dbin-jump-full&object_id=111169&local_base=GEN01-MCG02
    # [4] http://www.ndt-ed.org/GeneralResources/MaterialProperties/ET/ET_matlprop_Misc_Matls.htm
    # [5] http://chemistry.stackexchange.com/questions/820/electrical-conductivity-of-graphite
    # [6] Thermal and Electrical Conductivity of Graphite and Carbon at Low
    #     Temperatures,
    #     Robert A. Buerschaper.
    # [7] http://chemistry.stackexchange.com/questions/820/electrical-conductivity-of-graphite
    # [8] http://www.engineeringtoolbox.com/specific-heat-solids-d_154.html
    # [9] http://www.azom.com/article.aspx?ArticleID=1630
    # [10] A fine-grained, isotropic graphite for use as NBS thermophysical
    #      property RM's from 5 to 2500 K;
    #      Jerome G. Hust;
    #      <http://www.nist.gov/srm/upload/SP260-89.PDF>.
    # [12] http://www.engineeringtoolbox.com/linear-expansion-coefficients-d_95.html
    # [14] http://aries.ucsd.edu/LIB/PROPS/PANOS/c.html
    #
    # Carbon doesn't actually melt at atmospheric pressure,
    # but sublimes.
    melting_point = 3900.0
    magnetic_permeability = 0.999984 * mu0

    @staticmethod
    def electrical_conductivity(T):
        '''Data from :cite:`KP:2003:TNI`.
        '''
        return 1e6 / (28.9 - 18.8 * exp(-(log(T/1023.0)/2.37)**2))

    @staticmethod
    def density(T):
        ''':cite:`Morgan72` gives the thermal expansion coefficients parallel
        and perpendicular to the basal plane.
        '''
        # TODO better temperature dependence
        return 2.267e3 / (1 + 30.0e-6*(T-298.0))

    @staticmethod
    def specific_heat_capacity(T):
        '''Data from :cite:`BM73`.
        '''
        return 4.184e3 * (0.538657 + 9.11129e-6*T - 90.2725/T
                          - 43449.3/T**2 + 1.59309e7/T**3
                          - 1.43688e9/T**4
                          )

    @staticmethod
    def thermal_conductivity(T):
        # Data from [9] (except for the outlier at T=2000K),
        # perpendicular to the layers.
        # Quadratic least-squares fit of 1/c_p.
        return 1.0 / (-9.797e-08*T**2 + 0.0007809*T - 0.05741)


class Silver(object):
    # http://en.wikipedia.org/wiki/Silver
    magnetic_permeability = 0.999974*mu0
    electrical_conductivity = 6.30e7
    density = 9.32e3  # liquid density
    # http://www.convertunits.com/molarmass/Silver
    molar_mass = 107.8682
    molar_heat_capacity = 24.9
    specific_heat_capacity = 0.240e3
    thermal_conductivity = 406.0
    thermal_diffusivity = 1.6563e-4


class Water(object):
    # [1] https://en.wikipedia.org/wiki/Water_(molecule)#Density_of_water_and_ice
    # [2] https://en.wikipedia.org/wiki/Viscosity#Water
    # [3] https://en.wikipedia.org/wiki/Heat_capacity#Table_of_specific_heat_capacities
    # [4] https://en.wikipedia.org/wiki/List_of_thermal_conductivities
    # [5] Standard reference data for the thermal conductivity of water;
    #     J. Phys. Chem. Ref. Data 24, 1377 (1995);
    #     <http://www.nist.gov/data/PDFfiles/jpcrd493.pdf>.
    # [6] http://www.tainstruments.co.jp/application/pdf/Thermal_Library/Applications_Notes/TN015.PDF
    # [7] https://en.wikipedia.org/wiki/Properties_of_water#Density_of_water_and_ice
    @staticmethod
    def density(T):
        # Least-squares fit from [7].
        return - 1.407e-07 * (T - 273.15)**4 \
            + 4.387e-05 * (T - 273.15)**3 \
            - 0.00767 * (T - 273.15)**2 \
            + 0.05444 * (T - 273.15) \
            + 999.9

    @staticmethod
    def dynamic_viscosity(T):
        # [2]
        return 2.414e-5 * 10**(247.8 / (T - 140))

    @staticmethod
    def specific_heat_capacity(T):
        # Least-squares fit from [6].
        return 3.22e-06*T**4 - 0.004298*T**3 \
            + 2.155*T**2 - 480.7*T \
            + 4.439e+04

    @staticmethod
    def thermal_conducitivity(T):
        # [5]
        return 0.6065 * (-1.48445 + 4.12292*(T/298.15) - 1.63866*(T/298.15)**2)


class GraphiteEK90(object):
    magnetic_permeability = mu0
    density = 1.75e3
    electrical_conductivity = 52.0e3

    @staticmethod
    def thermal_conductivity(T):
        return 87.607 \
            - 8.52865e-2 * T \
            + 3.835e-5 * T**2 \
            + 5.143515e-9 * T**3


class GraphiteEK98(object):
    thermal_conductivity = 90.0
    density = 1.85e3
    electrical_conductivity = 71.42857e3


class Porcelain(object):
    magnetic_permeability = mu0
    # http://www.engineeringtoolbox.com/density-solids-d_1265.html
    density = 2.3e3
    # http://www.engineeringtoolbox.com/resistance-resisitivity-d_1382.html
    electrical_conductivity = 0.0
