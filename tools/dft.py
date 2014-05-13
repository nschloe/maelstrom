import numpy as np
from matplotlib import pyplot as pp
from mpltools import style
style.use('ggplot')

n = 1001
t0 = 0.1
t1 = 0.3
t = np.linspace(t0, t1, n)
#y = np.cos(2*np.pi * 23 * x) * np.exp(-np.pi*x**2)
#    + 0.1 * np.cos(2*np.pi * 97 * x) \
#    + 0.5 * np.cos(2*np.pi * 13 * x)
x = np.cos(2*np.pi * 23 * t)

#pp.plot(x, y, label='original')
pp.plot(x, label='original')

real = True
if real:
    X = np.fft.rfft(x)
    # Available in SciPy 0.13.0
    #freqs = np.fft.rfftfreq(n, d=t1-t0)
    freqs = [k / (t1 - t0) / n for k in range(n//2 + 1)]

    # Inverse Fourier transform just for playing around with it.
    proper = True
    if proper:
        t = np.array([(float(i) / n) * (t1 - t0) for i in range(n)])
        assert(X[0].imag < 1.0e-14)
        assert(freqs[0] < 1.0e-14)
        xx = X[0].real / n * np.exp(-1j * 2*np.pi * n * freqs[0] * t)
        for k in range(1, (n+1)//2):
            alpha = X[k] / n * np.exp(1j * 2*np.pi * n * freqs[k] * t)
            xx += 2.0 * alpha.real
        if n % 2 == 0:
            assert(X[-1].imag < 1.0e-14)
            print X[0], freqs[0]
            print X[-1], freqs[-1], np.exp(-1j * 2*np.pi * n * freqs[-1] * t)
            xx += X[-1].real / n * np.exp(-1j * 2*np.pi * n * freqs[-1] * t)
    else:
    # Alternative: Extend the arrays to match complex-valued FFT.
        xx = np.zeros(n, dtype=complex)
        if n % 2 == 0:
            X2 = np.concatenate((X[:-1], [z.conjugate() for z in reversed(X[1:])]))
            freqs2 = freqs[:-1] + [-f for f in reversed(freqs[1:])]
        else:
            X2 = np.concatenate((X, [z.conjugate() for z in reversed(X[1:])]))
            freqs2 = freqs + [-f for f in reversed(freqs[1:])]
        #for i in range(n):
        #    ti = (float(i) / n) * (t1 - t0)
        #    for k in range(-n//2, n//2):
        #        xx[i] += 1.0/n * X2[k] * np.exp(1j * 2*np.pi * n * freqs2[k] * ti)
        assert(all(xx.imag < 1.0e-15))
        xx = xx.real

else:
    X = np.fft.fft(x)
    print X
    ## (Slow) Fourier transform
    #X = np.zeros(n, dtype=complex)
    #for i in range(n):
    #    for k in range(n):
    #        X[i] += x[k] * np.exp(-2*np.pi * 1j * i*k / n)

    #m = len(amplitudes)
    #X /= n

    freqs = np.fft.fftfreq(n, d=t1-t0)
    print freqs

    # Inverse Fourier transform
    xx = np.zeros(n, dtype=complex)
    for i in range(n):
        ti = (float(i) / n) * (t1 - t0)
        for k in range(-n//2, n//2):
            # xx[i] += 1.0/n * X[k] * np.exp(2*np.pi * 1j * i*k / n)
            #xx[i] += 1.0/n * X[k] * np.exp(1j * 2*np.pi * n * freq * ti)
            xx[i] += 1.0/n * X[k].real * np.cos(2*np.pi * n * freqs[k] * ti) \
                - 1.0/n * X[k].imag * np.sin(2*np.pi * n * freqs[k] * ti) \
                + 1j * 1.0/n * X[k].real * np.sin(2*np.pi * n * freqs[k] * ti) \
                + 1j * 1.0/n * X[k].imag * np.cos(2*np.pi * n * freqs[k] * ti)
    assert(all(xx.imag < 1.0e-15))
    xx = xx.real


pp.plot(xx, '--', label='FT approximation')
pp.legend()
assert(all(abs(x - xx) < 1.0e-12))

#assert(n//2 + 1 == m)

pp.figure()
pp.plot(freqs, X.real / n, '.', label='real')
pp.plot(freqs, X.imag / n, '.', label='imag')
pp.plot(freqs, abs(X) / n, '.', label='abs')
pp.legend()

pp.show()
