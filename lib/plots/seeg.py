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
    pl.yticks(np.r_[:len(bip)],
              [f'{contacts_bip[i][0]}' for i in np.r_[:len(bip)]])
    pl.ylim((-1, len(bip) + 1))
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
        pl.hist(lbenv[:, tm].flat[:],
                np.r_[0.0:6.0:30j],
                normed=True,
                orientation='horizontal')
        pl.hist(lbenv_all[iother[:, None], tm].flat[:],
                np.r_[0.0:6.0:30j],
                normed=True,
                orientation='horizontal')
        # pl.ylim([0, 1])
        pl.title(f'Time {t0} - {t1}s')
        pl.legend(('Seizure Sensors', 'Others'))
        pl.ylabel('p(lbenv)'), pl.xlabel('lbenv')
        pl.grid(1, linestyle='--', alpha=0.3)
    pl.tight_layout()
    pl.show()


def ppc_seeg(csvi, skip=0, npz_data: os.PathLike = 'data.R.npz'):
    from numpy import newaxis, log, exp, load
    from pylab import subplot, plot, title, xlabel, grid
    npz = load(npz_data)
    x = csvi['x'][skip:, :]
    gain = npz['gain']
    ppc = csvi['amplitude'][skip:, newaxis] * log(gain.dot(
        exp(x))) + csvi['offset'][skip:, newaxis]
    emp = npz['seeg_log_power'].T
    for i, (yh, y) in enumerate(zip(ppc, emp)):
        subplot(len(ppc), 1, i + 1)
        plot(yh.T, 'k', alpha=0.1, label='PPC SLP')
        plot(y.T, 'b', label='SLP')
        grid(1)


def violin_x0(csv, skip=0, x0c=-1.8, x0lim=(-6, 0), per_chain=False):
    from pylab import subplot, axhline, violinplot, ylim, legend, xlabel, title
    if not per_chain:
        from ..io.stan import merge_csv_data
        csv = [merge_csv_data(*csv, skip=skip)]
    for i, csvi in enumerate(csv):
        subplot(1, len(csv), i + 1)
        axhline(x0c, color='r')
        violinplot(csvi['x0'])
        ylim(x0lim)
        legend((
            f'x0 < {x0c} healthy',
            'p(x0)',
        )), xlabel('Region'), title(f'Chain {i+1}')


def source_sensors(cxyz: np.ndarray, rxyz: np.ndarray):
    from pylab import subplot, plot
    for i, (j, k) in enumerate([(0, 1), (1, 2), (0, 2)]):
        subplot(3, 1, i + 1)
        plot(cxyz[:, j], cxyz[:, k], 'bo')
        plot(rxyz[:, j], rxyz[:, k], 'rx')


def seeg_elecs(json_fname, tvbzip_file, seegxyz, ez_idx, pz_idx, out_fig=''):
    from ..io import tvb
    from ..io import seeg
    import zipfile
    import matplotlib.pyplot as plt
    import json

    with open(json_fname) as fp:
        bad_chnls = json.load(fp)['bad_channels']

    roi_cntrs, roi_lbls = tvb.read_roi_cntrs(tvbzip_file)
    contacts = seeg.read_contacts(seegxyz)
    contactsxyz = np.array([
        contacts[ch_name] for ch_name in contacts.keys()
        if ch_name not in bad_chnls
    ])

    fig = plt.figure(figsize=(10, 10))
    labels = ['L --- R', 'P --- A', 'I --- S']

    # reg_color = ['royalblue' if c == 1 else 'darkblue' for c in cortical]
    reg_color = ['black' for c in roi_lbls]
    for idx in ez_idx:
        reg_color[idx] = 'xkcd:red'
    for idx in pz_idx:
        reg_color[idx] = 'xkcd:rust'

    for pos, id1, id2 in [(111, 0, 1)
                          ]:  #[(221, 0, 1), (224, 1, 2), (223, 0, 2)]:
        ax = fig.add_subplot(pos)
        ax.scatter(roi_cntrs[:, id1],
                   roi_cntrs[:, id2],
                   color=reg_color,
                   alpha=0.8,
                   s=40)
        for idx, name in enumerate(roi_lbls):
            ax.annotate(str(idx), (roi_cntrs[idx, id1], roi_cntrs[idx, id2]))
        ax.scatter(contactsxyz[:, id1],
                   contactsxyz[:, id2],
                   color='black',
                   s=10)
        ax.set_xlabel(labels[id1])
        ax.set_ylabel(labels[id2])
        ax.set_aspect('equal')

    # ax = fig.add_subplot(222, projection='3d')
    # ax.scatter(regions.xyz[:, 0], regions.xyz[:, 1], regions.xyz[:, 2], color=reg_color, s=40)
    # for name, idxs in contacts.electrodes.items():
    #     ax.scatter(contacts.xyz[idxs, 0], contacts.xyz[idxs, 1], contacts.xyz[idxs, 2], s=10, label=name)
    # ax.set_xlabel(labels[0])
    # ax.set_ylabel(labels[1])
    # ax.set_zlabel(labels[2])
    # ax.set_aspect('equal')

    # plt.legend(loc='upper right')
    plt.tight_layout()
    if (out_fig):
        plt.savefig(out_fig)


def plot_gain(gain_mat):
    from matplotlib import colors, cm
    import matplotlib.pyplot as plt

    norm = colors.LogNorm(gain_mat.min(), gain_mat.max())
    im = plt.imshow(gain_mat, norm=norm, cmap=cm.jet)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Gain Matrix', fontsize=13.0)
    plt.xlabel('Node')
    plt.ylabel('Sensor')


def plot_slp(slp,
             save_dir=None,
             fig_name=None,
             ax=None,
             figsize=(7, 6),
             title='SEEG log power',
             clim=None):
    import matplotlib.pyplot as plt
    import lib.utils.consts as consts

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120)
    if clim is None:
        clim = {'min': np.min(slp), 'max': np.max(slp)}
    fig = ax.get_figure()
    im = ax.imshow(slp.T, interpolation=None, aspect='auto', cmap='inferno')
    im.set_clim(clim['min'], clim['max'])
    ax.set_title(title, fontsize=consts.FS_LARGE)
    ax.set_xlabel('Time', fontsize=consts.FS_LARGE)
    ax.set_ylabel('Sensor', fontsize=consts.FS_LARGE)
    ax.tick_params(labelsize=consts.FS_MED)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, fraction=0.1)
    cbar.ax.tick_params(labelsize=consts.FS_MED)
    if fig_name is not None:
        plt.savefig(f'{save_dir}/{fig_name}', facecolor='white')
