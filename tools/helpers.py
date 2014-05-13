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


def dft(t, theta):
    '''Discrete Fourier Transform for time series.
    '''
    # As of now, we cannot compute nonuniform Fourier series, so show a plot of
    # the stretched version.
    n = len(t)
    t_uniform = np.linspace(t[0], t[-1], n)
    # Create a modified temperature array by interpolation of the actual data
    # to the uniform grid. This is done to make NumpPy's FFT work.
    # Note that there are a number of nonuniform FFT libraries, notably
    #
    #     * http://www.cims.nyu.edu/cmcl/nufft/nufft.html
    #     * http://www-user.tu-chemnitz.de/~potts/nfft/
    #
    # and a Python frontend
    #
    #    * https://github.com/ghisvail/pyNFFT.
    #
    # TODO use one of those
    #
    theta_interp = np.interp(t_uniform, t, theta)

    X = np.fft.rfft(theta_interp)
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
    freqs = np.array([i / (t[-1] - t[0]) for i in range(n//2 + 1)])
    #
    # Note that this definition differs from the output of np.fft.freqs which
    # is
    #     freqs = [i / (t[-1] - t[0]) / n for i in range(n//2 + 1)].
    #
    # Also note that the angular frequency is  omega = 2*pi*freqs.
    #
    # With RFFT, the amplitudes are scaled by a factor of 2. Compare with
    # plot_ft_approximation below.
    X_scaled = X.copy()
    X_scaled /= n
    X_scaled[1:-1] *= 2
    if n % 2 != 0:
        X_scaled[-1] *= 2
    return t_uniform, freqs, X_scaled, theta_interp
