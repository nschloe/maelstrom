# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of Maelstrom.
#
#  Maelstrom is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Maelstrom is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Maelstrom.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy

from maelstrom import dft


def _test_dft():
    # Test that Fourier-transformed data can be recovered nicely.
    t0 = 1.0
    t1 = 2.0
    n = 9
    t = numpy.linspace(t0, t1, n)
    data = numpy.random.rand(n)

    freqs, X = dft.uniform_dft(t1 - t0, data)

    data2 = numpy.zeros(n, dtype=complex)
    for i, freq in enumerate(freqs):
        alpha = X[i] * numpy.exp(1j * 2*numpy.pi * freq * (t - t0))
        data2 += alpha

    print(data)
    print(data2)
    import matplotlib.pyplot as plt
    plt.plot(t, data - data2)
    #plt.plot(t, data)
    #plt.plot(t, data2)
    plt.show()

    return


if __name__ == '__main__':
    _test_dft()
