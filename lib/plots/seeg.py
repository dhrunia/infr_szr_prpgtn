"""
Plots of sEEG time series & sensor geometry.

"""

import os
import numpy as np


def plot_bip(t, bip, contacts_bip):
    import pylab as pl
    tm = ((np.r_[:len(t)] % 10) == 0)
    pl.plot(t[tm], bip[:, tm].T / 800 + np.r_[:len(bip)], 'k', linewidth=0.05)
    pl.axis('tight')
    pl.yticks(np.r_[:len(bip)], [f'{contacts_bip[i][0]}' for i in np.r_[:len(bip)]])
    pl.ylim((-1, len(bip) + 1));
    pl.xlim((0, t[-1]))
    pl.grid(1, linewidth=0.3, alpha=0.4)
    pl.title('Bipolar sEEG')


def plot_envelope(te, isort, iother, lbenv, lbenv_all, contacts_bip):
    import pylab as pl
    pl.figure(figsize=(8, 6))
    pl.subplot(211)
    pl.plot(te, lbenv.T / lbenv.max() + np.r_[:len(lbenv)], 'k')
    pl.yticks(np.r_[:len(lbenv)], [contacts_bip[i][0] for i in isort])
    pl.xlabel('Time (s)'), pl.ylabel('Seizure Sensors')
    pl.grid(1, linestyle='--', alpha=0.3)
    pl.title('log bipolar sEEG envelope (`lbenv`)')
    for i, (t0, t1) in enumerate([(0, 50), (60, 100), (120, 250)]):
        tm = (te > t0) * (te < t1)
        pl.subplot(2, 3, i + 4)
        pl.hist(lbenv[:, tm].flat[:], np.r_[0.0:6.0:30j], normed=True, orientation='horizontal')
        pl.hist(lbenv_all[iother[:, None], tm].flat[:], np.r_[0.0:6.0:30j], normed=True, orientation='horizontal')
        # pl.ylim([0, 1])
        pl.title(f'Time {t0} - {t1}s')
        pl.legend(('Seizure Sensors', 'Others'))
        pl.ylabel('p(lbenv)'), pl.xlabel('lbenv')
        pl.grid(1, linestyle='--', alpha=0.3)
    pl.tight_layout()
    pl.show()


def ppc_seeg(csvi, skip=0, npz_data: os.PathLike='data.R.npz'):
    from numpy import newaxis, log, exp, load
    from pylab import subplot, plot, title, xlabel, grid

    npz = load(npz_data)

    x = csvi['x'][skip:, :]
    gain = npz['gain']

    yh0, yh1 = csvi['amplitude'][skip:, newaxis] * log(gain.dot(exp(x))) + csvi['offset'][skip:, newaxis]
    y0, y1 = npz['seeg_log_power'].T

    # TODO generalize
    subplot(211)
    plot(yh0.T, 'k', alpha=0.1, label='PPC SLP')
    plot(y0.T, 'b', label='SLP')
    title("Sensor B"), xlabel('Time point'), grid(1)

    subplot(212)
    plot(yh1.T, 'k', alpha=0.1, label='PPC SLP')
    plot(y1.T, 'b', label='SLP')
    title("Sensor TP'"), xlabel('Time point'), grid(1)


def violin_x0(csv, skip=0, x0c=-1.8, x0lim=(-6, 0), per_chain=False):
    from pylab import subplot, axhline, violinplot, ylim, legend, xlabel, title
    if not per_chain:
        from ..io.stan import merge_csv_data
        csv = [merge_csv_data(*csv, skip=skip)]
    for i, csvi in enumerate(csv):
        subplot(1, len(csv), i + 1)
        axhline(x0c, color='r');
        violinplot(csvi['x0'])
        ylim(x0lim)
        legend((f'x0 < {x0c} healthy', 'p(x0)',)), xlabel('Region'), title(f'Chain {i+1}')


def source_sensors(cxyz: np.ndarray, rxyz: np.ndarray):
    from pylab import subplot, plot
    for i, (j, k) in enumerate([(0, 1), (1, 2), (0, 2)]):
        subplot(3, 1, i + 1)
        plot(cxyz[:, j], cxyz[:, k], 'bo')
        plot(rxyz[:, j], rxyz[:, k], 'rx')