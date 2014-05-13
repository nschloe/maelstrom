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
'''
Compute least-squares fit for a given data set.
'''
import numpy as np


def _density_water():
    # Data from
    # https://en.wikipedia.org/wiki/Properties_of_water#Density_of_water_and_ice
    x = [0, 4, 10, 15, 20, 22, 25, 30, 40, 60, 80, 100]
    y = [999.8395, 999.9720, 999.7026, 999.1026, 998.2071, 997.7735, 997.0479,
         995.6502, 992.2, 983.2, 971.8, 958.4]
    return x, y


def _spec_heat_water():
    '''
    Specific heat data of water.
    '''
    # Data source:
    # http://www.tainstruments.co.jp/application/pdf/Thermal_Library/Applications_Notes/TN015.PDF
    x = np.array(range(0, 101)) + 273.15
    y = [4.2177, 4.2141, 4.2107, 4.2077, 4.2048, 4.2022, 4.1999, 4.1977,
         4.1957, 4.1939, 4.1922, 4.1907, 4.1893, 4.1880, 4.1869, 4.1858,
         4.1849, 4.1840, 4.1832, 4.1825, 4.1819, 4.1813, 4.1808, 4.1804,
         4.1800, 4.1796, 4.1793, 4.1790, 4.1788, 4.1786, 4.1785, 4.1784,
         4.1783, 4.1783, 4.1782, 4.1782, 4.1783, 4.1783, 4.1784, 4.1785,
         4.1786, 4.1787, 4.1789, 4.1791, 4.1792, 4.1795, 4.1797, 4.1799,
         4.1802, 4.1804, 4.1807, 4.1810, 4.1814, 4.1817, 4.1820, 4.1824,
         4.1828, 4.1832, 4.1836, 4.1840, 4.1844, 4.1849, 4.1853, 4.1858,
         4.1863, 4.1868, 4.1874, 4.1879, 4.1885, 4.1890, 4.1896, 4.1902,
         4.1908, 4.1915, 4.1921, 4.1928, 4.1935, 4.1942, 4.1949, 4.1957,
         4.1964, 4.1972, 4.1980, 4.1988, 4.1997, 4.2005, 4.2014, 4.2023,
         4.2032, 4.2042, 4.2051, 4.2061, 4.2071, 4.2081, 4.2092, 4.2103,
         4.2114, 4.2125, 4.2136, 4.2148, 4.2160]
    y = np.array(y) * 1.0e3
    return x, y


def _therm_cond_graphite():
    # thermal conductivity of graphite from
    # http://aries.ucsd.edu/LIB/PROPS/PANOS/c.html
    x = [300, 400, 500, 600, 645, 800, 1000, 1200, 1500,
         2500, 3000]
    y = [5.7, 4.09, 3.49, 2.68, 2.45, 2.01, 1.60, 1.34, 1.08,
         0.81, 0.70]
    y = 1 / np.array(y)
    #from matplotlib import pyplot as pp
    #pp.plot(x, 1/np.array(y), 'o')
    #pp.show()
    return x, y


def _spec_heat_sic():
    # specific heat of SiC,
    # <http://www.ceramics.nist.gov/srd/summary/scdscs.htm>.
    x = [20, 500, 1000, 1200, 1400, 1500]
    x = np.array(x) + 273.15
    y = [715, 1086, 1240, 1282, 1318, 1336]
    return x, y


def _vol_exp_sic():
    # volume expansion coefficient of SiC
    x = [500, 600, 900, 1500, 2100]
    y = [3.8e-6, 4.3e-6, 4.8e-6, 5.5e-6, 5.5e-6]
    return x, y


def _elec_res_copper():
    # electrical resistivity of copper:
    # http://www.nist.gov/data/PDFfiles/jpcrd155.pdf
    x = [273.15, 293, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100,
         1200, 1300, 1357.6]
    y = [1.543, 1.678, 1.725, 2.063, 2.402, 3.090, 3.792, 4.514, 5.262, 6.041,
         6.858, 7.717, 8.626, 9.592, 10.171]
    return x, y


def _main():
    #x, y = _spec_heat_water()
    x, y = _density_water()

    # form the Vandermonde matrix
    degree = 4
    A = np.vander(x, degree+1)

    # find the x that minimizes the norm of Ax-y
    (coeffs, residuals, rank, sing_vals) = np.linalg.lstsq(A, y)

    print('Singular values of the Vandermonde matrix:')
    print (sing_vals)
    print
    print('Rank: %d' % rank)
    print
    if rank < degree+1:
        raise RuntimeError('Ill-conditioned problem. Byes.')

    # create a polynomial using coefficients
    f = np.poly1d(coeffs)
    print(f)

    # f(x) == np.dot(A,coeffs)
    res = (f(x) - y)
    print
    print('||r||_1   = %e' % np.linalg.norm(res, 1))
    print('||r||_2   = %e' % np.linalg.norm(res, 2))
    print('||r||_inf = %e' % np.linalg.norm(res, np.inf))

    from matplotlib import pyplot as pp

    # least-squares polynomial
    X = np.linspace(min(x), max(x), 100)
    pp.plot(X, f(X), '-')

    # data points
    pp.plot(x, y, 'o')

    pp.show()
    return


if __name__ == '__main__':
    _main()
