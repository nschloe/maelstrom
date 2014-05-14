#!/usr/bin/env python
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
import numpy as np


def uniform_dft(t0, t1, theta):
    '''Discrete Fourier Transform for a uniform time series.
    '''
    X = np.fft.rfft(theta)
    n = len(theta)
    # When doing proper FT, the data points theta are composed of
    #
    #     theta[i] += 1.0/n * X[k].real * np.cos(2*np.pi * n * freqs[k] * ti) \
    #               - 1.0/n * X[k].imag * np.sin(2*np.pi * n * freqs[k] * ti).
    #
    # The imaginary part vanishes since theta is real-valued (which is why we
    # effectively use rfft here).
    # The ti are given by
    #
    #     ti = (float(i) / n) * (t1 - t0)
    #
    # and the ordinary frequencies <https://en.wikipedia.org/wiki/Sine_wave>
    # by
    #
    freqs = np.array([i / (t1 - t0) for i in range(n//2 + 1)])
    #
    # Note that this definition differs from the output of np.fft.freqs which
    # is
    #     freqs = [i / (t[-1] - t[0]) / n for i in range(n//2 + 1)].
    #
    # Also note that the angular frequency is  omega = 2*pi*freqs.
    #
    # With RFFT, the amplitudes are scaled by a factor of 2. Compare with
    # plot_ft_approximation below.
    X /= n
    X[1:-1] *= 2
    if n % 2 != 0:
        X[-1] *= 2

    assert(len(freqs) == len(X))
    return freqs, X
