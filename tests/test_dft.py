# -*- coding: utf-8 -*-
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
    for x, freq in zip(X, freqs):
        alpha = x * numpy.exp(1j * 2*numpy.pi * freq * (t - t0))
        data2 += alpha

    print('Original data:')
    print(data)
    print
    print('Reconverted data:')
    print('Real part:')
    print(data2.real)
    print('Imaginary part:')
    print(data2.imag)
    import matplotlib.pyplot as plt
    plt.plot(t, data - data2.real)
    #plt.plot(t, data)
    #plt.plot(t, data2)
    plt.show()

    return


if __name__ == '__main__':
    _test_dft()
