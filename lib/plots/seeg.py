"""
Plots of sEEG time series & sensor geometry.

"""

import os


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