

from scipy import signal


def bfilt(data, samp_rate, fs, mode, order=3):
    b, a = signal.butter(order, 2*fs/samp_rate, mode)
    return signal.lfilter(b, a, data)


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