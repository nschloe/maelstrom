# -*- coding: utf-8 -*-
#
import numpy

import dft


def test_dft():
    # Test that Fourier-transformed data can be recovered nicely.
    t0 = 1.0
    t1 = 2.0
    n = 9
    t = numpy.linspace(t0, t1, n)
    data = numpy.random.rand(n)

    freqs, X = dft.uniform_dft(t1 - t0, data)

    data2 = numpy.zeros(n, dtype=complex)
    for x, freq in zip(X, freqs):
        alpha = x * numpy.exp(1j * 2 * numpy.pi * freq * (t - t0))
        data2 += alpha

    assert (abs(data - data2.real) < 1.0e-14).all()

    return


if __name__ == "__main__":
    test_dft()
