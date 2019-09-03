
# origin preprocessing used for NeuroImage paper, based on specgram,
# a short window FFT. Deprecated in preference for envelope technique.

import os
import numpy as np

from ..io.stan import rdump
from .base import BasePreproc


def log_power_change(fs, seeg, flo=10.0, nfft=1024, tb=10.0):
    """
    Compute log power change using a short window spectrogram.

    :param fs: sampling frequency
    :param seeg: sEEG dataset
    :param flo: high pass frequency
    :param nfft: FFT size to use
    :param tb: period to use for baseline correction
    :return:
    """
    import pylab as pl
    pl.ioff()
    ps = []
    assert len(seeg) > 1
    for chan in seeg:
        P, F, T, _ = pl.specgram(chan, NFFT=nfft, Fs=fs)
        p = P[F > flo].sum(axis=0)
        p /= p[T < tb].mean()
        ps.append(pl.detrend_linear(np.log(p)))
    return T, np.array(ps)


def prep_stan_data(seeg):
    # load sEEG, pick channels, analyze
    # seeg = load_complex()
    from numpy import r_, argsort, unique, loadtxt, newaxis, c_, setxor1d
    fn = 'seedpsd.npz'
    if not os.path.exists(fn):
        print('recomputing log power change')
        T, ps = log_power_change(512.0, seeg, nfft=512)
        np.savez(fn, T=T, ps=ps)
    npz = np.load(fn)
    T, ps = npz['T'], npz['ps']
    picks = r_[78, 14] # r_[78, 23, 54, 14, 35, 100]
    T -= T[0]
    dt = T[1] - T[0]
    print('dt based on psd is ', dt)
    def plot_ps():
        from pylab import figure, plot, yticks, savefig
        figure()
        plot(T, ps[picks].T + r_[:len(picks)]*10, 'k-x', linewidth=0.5)
        yticks(r_[:len(picks)]*10, [str(_) for _ in picks])
        savefig('prepdata-ps.png')
    plot_ps()
    # lead field, select nodes
    Vr = np.load('data/Vr.npy')
    nodes = []#158, 157, 147]
    for i in picks:
        pick_nodes = argsort(Vr[i])[-3:].tolist()
        nodes.extend(pick_nodes)
        print(f"{i} {contacts[i]} picks")
        for pick_node in pick_nodes:
            print(f'\t{reg_names[pick_node]}')
    nodes = unique(nodes)
    Vr_ = Vr[picks][:, nodes]
    Vr_ /= Vr_.sum(axis=1)[:, newaxis]
    print('reduced gain matrix shape is ', Vr_.shape)
    # reduce connectivity & compute appropriate constant
    W = loadtxt('data/weights.txt')
    iconst = setxor1d(r_[:W.shape[0]], nodes)
    for i in range(W.shape[0]):
        W[i, i] = 0.0
    W /= W.max()
    W_ = W[nodes][:, nodes]
    Ic = W[nodes][:, iconst].sum(axis=1)
    # build dataset
    tm = c_[T>100.0, T<450.0].all(axis=1)
    T = T[tm]
    print (ps.shape)
    ps = ps[picks][:, tm]
    print (ps.shape)
    data_simple = {
        'nn': W_.shape[0], 'ns': Vr_.shape[0], 'nt': len(T),
        'I1': 3.1, 'tau0': 3.0, 'dt': dt,
        'SC': W_, 'SC_var': 5.0, 'gain': Vr_, 'seeg_log_power': ps.T, 'Ic': Ic,
        'K_lo': 1.0, 'K_u': 5.0, 'K_v': 10.0,
        'x0_lo': -15.0, 'x0_hi': 5.0, 'eps_hi': 0.2, 'sig_hi': 0.025,
        'zlim': r_[0.0, 10.0],
        'siguv': r_[-1.0, 0.5],
        'epsuv': r_[-1.0, 0.5],
        'use_data': 1,
        'tt': 0.08
    }
    npz = {'nodes': nodes}
    npz.update(data_simple)
    np.savez('data.R.npz', **npz)
    rdump('data.R', data_simple, )


class SpecgramPreproc(BasePreproc):
    pass