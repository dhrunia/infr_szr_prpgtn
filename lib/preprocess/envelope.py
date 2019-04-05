from scipy import signal
import numpy as np


def bfilt(data, samp_rate, fs, mode, order=3, axis = -1):
    b, a = signal.butter(order, 2*fs/samp_rate, mode)
    return signal.filtfilt(b, a, data, axis)


def compute_envelope(bip, samp_rate, hp_freq=5.0, lp_freq=0.1,
                     benv_cut=100):
    hp_freq = 5.0
    lp_freq = 0.1
    start = int(samp_rate / lp_freq)
    skip = int(samp_rate / (lp_freq * 3))
    benv = bfilt(np.abs(bfilt(bip, samp_rate, hp_freq, 'high')), samp_rate, lp_freq, 'low')[:, start::skip]
    te = t[start::skip]
    fm = benv > 100  # bipolar 100, otherwise 300 (roughly)
    incl_names = "HH1-2 HH2-3".split()
    incl_idx = np.array([i for i, (name, *_) in enumerate(contacts_bip) if name in incl_names])
    incl = np.setxor1d(
        np.unique(np.r_[
                      incl_idx,
                      np.r_[:len(fm)][fm.any(axis=1)]
                  ])
        , afc_idx)
    isort = incl[np.argsort([te[fm[i]].mean() for i in incl])]
    iother = np.setxor1d(np.r_[:len(benv)], isort)
    lbenv = np.log(np.clip(benv[isort], benv[benv > 0].min(), None))
    lbenv_all = np.log(np.clip(benv, benv[benv > 0].min(), None))
    return te, isort, iother, lbenv, lbenv_all


def mov_avg(a, win_len, pad=True):
    a_mov_avg = np.empty_like(a)
    a_pad = np.pad(a, ((0,win_len),(0,0)), 'constant') # pad with zeros at the end to compute moving average of same length as the signal itself
    for i in range(a.shape[0]):
        a_mov_avg[i,:] = np.mean(a_pad[i:i+win_len,:],axis=0)
    return a_mov_avg


def seeg_log_power(a, win_len, pad=True):
    envlp = np.empty_like(a)
    # pad with zeros at the end to compute moving average of same length as the signal itself
    envlp_pad = np.pad(a, ((0, win_len), (0, 0)), 'constant')
    for i in range(a.shape[0]):
        envlp[i, :] = np.log(np.mean(envlp_pad[i:i+win_len, :]**2, axis=0))
    return envlp

def compute_fitting_target(data, samp_rate, fcut=5.0, logtransform=False):
    '''
    Extracts smoothed log. power of given SEEG
    '''
    # high pass filter to remove baseline shift
    data_hpf = bfilt(data, samp_rate, fcut, 'highpass', axis = 0) 
    # compute the log power over a sliding window
    data_lpwr = np.log(mov_avg(data_hpf**2, 50)) if logtransform else mov_avg(data_hpf**2, 50)
    # low pass filter the log power for smoothing
    data_lpwr = bfilt(data_lpwr, samp_rate, 2.0, 'lowpass', axis = 0) 
    return data_lpwr
