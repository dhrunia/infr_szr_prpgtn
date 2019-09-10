import numpy as np
from .dmeeg import EEG


def _maybe_bip(seeg, time, proj_bip=None):
    if proj_bip is not None:
        return seeg, time, proj_bip.dot(seeg)
    else:
        return seeg, time


def load_npy(fname, samp_rate, proj_bip=None):
    seeg = np.load(fname)[:-2]  # last two are ECG & SAT
    time = np.r_[:seeg.shape[1]] / samp_rate
    return _maybe_bip(seeg, time, proj_bip)


def load_eeg(fname, proj_bip=None):
    eeg = EEG(fname)
    seeg, time = eeg.read_data()
    seeg = seeg[:-2]
    return _maybe_bip(seeg, time, proj_bip)

def read_contatcs(cntcts_file, type='dict'):
    cntcts = zip(np.loadtxt(cntcts_file, usecols=[0], dtype='str'), np.loadtxt(cntcts_file, usecols=[1,2,3]))
    if(type == 'dict'):
        return dict(cntcts)
    elif(type == 'list'):
        return list(cntcts)

def read_roi_cntrs(cntrs_zipfile):
    zf = zipfile.ZipFile(cntrs_zipfile)
    roi_cntrs = []
    roi_lbls = []
    with zf.open('centres.txt') as t:
        for line in t:
            roi_cntrs.append(line.decode('utf-8').strip().split(' ')[1:])
            roi_lbls.append(line.decode('utf-8').strip().split(' ')[0])
    roi_cntrs = np.array(roi_cntrs, dtype=float)
    return roi_cntrs, roi_lbls
