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


def uniform_dft(time_interval_length, data):
    '''Discrete Fourier Transform of real-valued data, interpreted
    for a uniform time series over an interval of length time_interval_length.

    The original data can be recovered from the output of this function by

    data = numpy.zeros(n)
    for i, freq in enumerate(freqs):
        alpha = X[i] * numpy.exp(1j * 2*numpy.pi * freq * (t-t0))
        data += alpha.real
    '''
    X = numpy.fft.rfft(data)
    n = len(data)
    # The input data is assumed to cover the entire time interval, i.e.,
    # including start and end point. The data produced from RFFT however
    # assumes that the end point is excluded. Hence, stretch the
    # time_interval_length such that cutting off the end point results in the
    # interval of length time_interval_length.
    time_interval_length *= n / float(n - 1)
    freqs = numpy.array([i / time_interval_length for i in range(n//2 + 1)])
    #
    # Note that this definition of the frequencies slightly differs from the
    # output of np.fft.freqs which is
    #
    #     freqs = [i / time_interval_length / n for i in range(n//2 + 1)].
    #
    # Also note that the angular frequency is  omega = 2*pi*freqs.
    #
    # With RFFT, the amplitudes need to be scaled by a factor of 2.
    X /= n
    X[1:-1] *= 2
    if n % 2 != 0:
        X[-1] *= 2
    assert(len(freqs) == len(X))
    return freqs, X
