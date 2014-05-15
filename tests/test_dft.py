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
import numpy

from pymhd import dft


def _test_dft():
    # Generate random data set.
    t0 = 1.0
    t1 = 2.0
    n = 49
    t = numpy.linspace(t0, t1, n)
    data = numpy.random.rand(n)

    freqs, X = dft.uniform_dft(t1 - t0, data)

    #t_new = numpy.linspace(t0, t1, n, endpoint=False)
    #print t_new
    data2 = numpy.zeros(n)
    for i, freq in enumerate(freqs):
        alpha = X[i] * numpy.exp(1j * 2*numpy.pi * freq * (t - t0))
        data2 += alpha.real

    import matplotlib.pyplot as plt
    plt.plot(t, data - data2)
    #plt.plot(t, data)
    #plt.plot(t, data2)
    plt.show()

    return


if __name__ == '__main__':
    _test_dft()
