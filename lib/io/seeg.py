import numpy as np
from .dmeeg import EEG
import mne
import os
import json


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


def read_contacts(cntcts_file, type='dict'):
    cntcts = zip(
        np.loadtxt(cntcts_file, usecols=[0], dtype='str'),
        np.loadtxt(cntcts_file, usecols=[1, 2, 3]))
    if (type == 'dict'):
        return dict(cntcts)
    elif (type == 'list'):
        return list(cntcts)


def find_picks(json_fnames):
    picks = []
    for fname in json_fnames:
        fname_wo_xtnsn = os.path.splitext(os.path.basename(fname))[0]
        with open(fname) as fd:
            meta_data = json.load(fd)
        if (meta_data['type'].lower() == "spontaneous seizure"):
            exclude = meta_data['bad_channels'] + meta_data['non_seeg_channels']
            ch_names = set(
                mne.io.read_info(
                    f'{os.path.dirname(fname)}/{fname_wo_xtnsn}.raw.fif')[
                        'ch_names'])
            picks.append(ch_names - set(exclude))
        else:
            raise Exception('Not a spontaneous seizure')
    return set.intersection(*picks)


def read_seeg(data_dir, meta_data_fname, raw_seeg_fname):
    picks = find_picks(
        [os.path.join(data_dir, 'seeg', 'fif', meta_data_fname)])
    seeg = dict()
    fname_wo_xtnsn = os.path.splitext(os.path.basename(raw_seeg_fname))[0]
    with open(os.path.join(data_dir, 'seeg', 'fif', meta_data_fname)) as fd:
        meta_data = json.load(fd)
    if (meta_data['type'].lower() == "spontaneous seizure"):
        raw = mne.io.Raw(
            os.path.join(data_dir, 'seeg', 'fif', raw_seeg_fname),
            verbose='WARNING',
            preload=True)
        assert meta_data['onset'] is not None and meta_data['termination'] is not None
        raw.pick_types(meg=False, eeg=True)
        raw.pick_channels(picks)
        raw.crop(
            tmin=meta_data['onset'] - 5, tmax=meta_data['termination'] + 5)
        raw.reorder_channels(picks)
        seeg['fname'] = f'{fname_wo_xtnsn}'
        seeg['time_series'] = raw.get_data().T
        seeg['sfreq'] = raw.info['sfreq']
        seeg['ch_names'] = raw.ch_names
    return seeg
