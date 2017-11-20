#!/usr/bin/env python3
"""
Analyze and plot results of model

"""

#%%
import argparse
#import matplotlib as mpl; mpl.use('agg')
#%pylab inline
from pylab import (
    subplot, bar, title, r_, violinplot, xlabel, imshow,
    figure, plot, percentile, fill_between, log, exp, legend, ylabel,
    xticks, yticks, tight_layout, ylim, show, text, savefig,
    contour
)
import pylab as pl
import numpy as np
import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='+')
    for flag in 'junk time_series phase_space conn figx all report'.split():
        parser.add_argument('--'+flag, action="store_true")
    parser.add_argument('--skip')
    args = parser.parse_args()
    return args
#%%
args = parse_args()
print(args)
#%%
npz = np.load('data.R.npz')
Ic = npz['Ic']
ps = npz['seeg_log_power']
W_ = npz['SC']
nodes = npz['nodes']
gain = npz['gain']
data = {k:v[args.skip:] for k, v in common.parse_csv(args.csv).items()}
#%%
for key, val in data.items():
    print(key, val.shape)
print(data['x0'].shape, 'samples')
try:
    # ODE model has xz, SDE does not
    data['x'] = data['xz'][..., 0]
    data['z'] = data['xz'][..., 1]
except:
    pass
opt = data['x0'].shape[0] == 1
n = data['x0'].shape[1]
#%%
# junk
if args.junk or args.all:
    subplot(611), bar(r_[:n], Ic); title('Ic')
    subplot(612)
    if opt:
        bar(r_[:n], data['x0'][0]); title('x0')
    else:
        violinplot(data['x0']);
    subplot(312), imshow(data['x'][0], interpolation='none', aspect='auto');
    subplot(313), imshow(data['FC'][0], interpolation='none', aspect='auto');
    if opt:
        kstr = 'K = %f' % (data['K'][0], )
    else:
        kstr = 'K = %f %f' % (data['K'].mean(), data['K'].std())
    xlabel(kstr)
    savefig('fig-junk.png')

# time series
if args.time_series or args.all:
    figure()
    subplot(211)
    plot(ps, 'k', alpha=0.2)
    subplot(212)
    if opt:
        plot(data['x'][0], 'k')
        plot(data['z'][0], 'r')
        title(data['time_scale'][0])
    else:
        xlo, xhi = percentile(data['x'], [5, 95], axis=0)
        t = r_[:xlo.shape[0]]
        for lo, hi in zip(xlo.T, xhi.T):
            fill_between(t, lo, hi, alpha=0.2, facecolor='k', edgecolor='none')

        xlo, xhi = percentile(data['z'], [5, 95], axis=0)
        t = r_[:xlo.shape[0]]
        for lo, hi in zip(xlo.T, xhi.T):
            fill_between(t, lo, hi, alpha=0.2, facecolor='r', edgecolor='none')
    tight_layout()
    savefig('fig-time-series.png')

# phase space
if args.phase_space or args.all:
    figure()
    x, z = data['x'], data['z']
    tau0 = npz['tau0']
    """
    real dx = (I1 + 1.0) - pow(x[i, t],3.0) - 2.0 * pow(x[i, t], 2.0) - z[i, t];
    real gx = FC[i,] * (x[,t] - x[i,t]) - Ic[i] * (1.8 + x[i,t]);
    real dz = inv(tau0) * (4 * (x[i, t] - x0[i]) - z[i, t] - K * gx);
    """
    X, Z = np.mgrid[-5.0:5.0:50j, -5.0:5.0:50j]
    dX = (npz['I1'] + 1.0) - X**3.0 - 2.0*X**2.0 - Z
    x0mean = data['x0'].mean(axis=0)
    Kmean = data['K'].mean(axis=0)
    def nullclines(i):
        contour(X, Z, dX, 0, colors='r')
        dZ = (1.0/tau0) * (4.0 * (X - x0mean[i])) - Z - Kmean*(-npz['Ic'][i]*(1.8 + X))
        contour(X, Z, dZ, 0, colors='b')
    for i in range(x.shape[-1]):
        subplot(2, 3, i + 1)
        if opt:
            plot(x[0, :, i], z[0, :, i], 'k', alpha=0.5)
        else:
            for j in range(1 if opt else 10):
                plot(x[-j, :, i], z[-j, :, i], 'k', alpha=0.2, linewidth=0.5)
        nullclines(i)
        pl.grid(True)
        xlabel('x(t)')
        ylabel('z(t)')
        title(f'node {i}')
    tight_layout()
    savefig('fig-phase-space.png')

if 0:
    figure()
    pl.hist(data['sigma'], 30)

# connectivity
if args.conn or args.all:
    im = lambda i: imshow(i, interpolation='none', aspect='auto', cmap='binary')
    figure(); subplot(121); im(W_); subplot(122); im(data['FC'].mean(axis=0));
    savefig('fig-connectivity.png')

# figure X
if args.figx or args.all:
    figure(figsize=(6, 4))
    showpicks = r_[0, 1]
    amp, offset = [data[_].reshape((-1, 1, 1)) for _ in 'amplitude offset'.split()]
    ps_est =  amp*(log(exp(data['x']).dot(gain.T)) + offset)
    pelo, pehi = percentile(ps_est, [5, 95], axis=0)
    t = r_[:pelo.shape[0]]
    subplot(211)
    for i, (lo, hi) in enumerate(zip(pelo.T, pehi.T)):
        if i in showpicks:
            fill_between(t, lo, hi, alpha=0.5, facecolor='b', edgecolor='none')
    plot(ps[:, showpicks], 'k', alpha=0.5)
    leg = []
    for i in [78, 14]:
        cn, ci = common.contacts[i]
        leg.append(cn+str(ci))
    legend(leg)
    xlabel('Time (s)'); ylabel('sEEG(t)');
    subplot(212)
    violinplot(data['x0'], widths=0.7);
    node_abbrev = []
    for i in nodes:
        rn = common.reg_names[i]
        ra = ''.join([p[0] for p in rn.split('-')[1:] if p]).upper()
        node_abbrev.append(ra)
        print (ra, ' '.join(rn.split('-')))
    xticks(r_[1:len(nodes)+1], node_abbrev, rotation=30)
    ylabel("$P(x_0)$")
    ylim([-4.5, -0.5])
    tight_layout()
    text(-2, 5, "A", size=14)
    text(-2, 0, "B", size=14)
    savefig('fig-x.png', dpi=300)

if args.report or args.all:
    with open('report.txt', 'w') as fd:
        fd.write("index,x0mean,x0median,abbrev,name\n")
        fmt = "%d,%f,%f,%s,%s\n"
        for i, idx in enumerate(nodes):
            i_x0 = data['x0'][:, i]
            args = (
                idx, np.mean(i_x0), np.median(i_x0), node_abbrev[i],
                common.reg_names[idx]
            )
            fd.write(fmt % args)


show()
