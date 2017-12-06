"""
Plots pertaining to the network.

"""

import os
import numpy as np


def phase_space(csvi, npz_data: os.PathLike='data.R.npz'):
    import pylab as pl
    opt = len(csvi['x']) == 1
    npz = np.load(npz_data)
    tr = lambda A: np.transpose(A, (0, 2, 1))
    x, z = tr(csvi['x']), tr(csvi['z'])
    tau0 = npz['tau0']
    X, Z = np.mgrid[-5.0:5.0:50j, -5.0:5.0:50j]
    dX = (npz['I1'] + 1.0) - X**3.0 - 2.0*X**2.0 - Z
    x0mean = csvi['x0'].mean(axis=0)
    Kmean = csvi['K'].mean(axis=0)
    def nullclines(i):
        pl.contour(X, Z, dX, 0, colors='r')
        dZ = (1.0/tau0) * (4.0 * (X - x0mean[i])) - Z - Kmean*(-npz['Ic'][i]*(1.8 + X))
        pl.contour(X, Z, dZ, 0, colors='b')
    for i in range(x.shape[-1]):
        pl.subplot(2, 3, i + 1)
        if opt:
            pl.plot(x[0, :, i], z[0, :, i], 'k', alpha=0.5)
        else:
            for j in range(1 if opt else 10):
                pl.plot(x[-j, :, i], z[-j, :, i], 'k', alpha=0.2, linewidth=0.5)
        nullclines(i)
        pl.grid(True)
        pl.xlabel('x(t)')
        pl.ylabel('z(t)')
        pl.title(f'node {i}')
