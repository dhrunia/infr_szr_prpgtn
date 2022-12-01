import numpy as np
from .dmeeg import EEG
import json
import mne
import os


class BadSeizure(Exception):
    '''
    Excpetion to raise when the seizure is not a spontaneous seizure
    '''

    def __init__(self):
        super().__init__('Not a spontaneous seizure')


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
    cntcts = zip(np.loadtxt(cntcts_file, usecols=[0], dtype='str'),
                 np.loadtxt(cntcts_file, usecols=[1, 2, 3]))
    if (type == 'dict'):
        return dict(cntcts)
    elif (type == 'list'):
        return list(cntcts)


def read_seeg_xyz(seeg_xyz_path):
    lines = []
    with open(seeg_xyz_path, 'r') as fd:
        for line in fd.readlines():
            name, *sxyz = line.strip().split()
            xyz = [float(_) for _ in sxyz]
            lines.append((name, xyz))
    return lines


def read_gain(data_dir, picks, parcellation='destrieux'):
    gain = np.loadtxt(f'{data_dir}/elec/gain_inv-square.{parcellation}.txt')
    seeg_xyz = read_seeg_xyz(data_dir)
    # Re-order the gain matrix to match with order of channels read using MNE
    gain_idxs = []
    for label in picks:
        for i, (gain_label, _) in enumerate(seeg_xyz):
            if (label == gain_label):
                gain_idxs.append(i)
    gain = gain[gain_idxs]
    return gain


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
                    f'{os.path.dirname(fname)}/{fname_wo_xtnsn}.raw.fif')
                ['ch_names'])
            picks.append(ch_names - set(exclude))
        else:
            raise BadSeizure()
    return sorted(tuple(set.intersection(*picks)))


def read_one_seeg(data_dir, meta_data_fname, raw_seeg_fname):
    picks = find_picks(
        [os.path.join(data_dir, 'seeg', 'fif', meta_data_fname)])
    seeg = dict()
    fname_wo_xtnsn = os.path.splitext(os.path.basename(raw_seeg_fname))[0]
    with open(os.path.join(data_dir, 'seeg', 'fif', meta_data_fname)) as fd:
        meta_data = json.load(fd)
    if (meta_data['type'].lower() == "spontaneous seizure"):
        raw = mne.io.Raw(os.path.join(data_dir, 'seeg', 'fif', raw_seeg_fname),
                         verbose='WARNING',
                         preload=True)
        assert meta_data['onset'] is not None and meta_data[
            'termination'] is not None
        raw.pick_types(meg=False, eeg=True)
        raw.pick_channels(picks)
        raw.reorder_channels(picks)
        seeg['fname'] = f'{fname_wo_xtnsn}'
        seeg['onset'] = meta_data['onset']
        seeg['offset'] = meta_data['termination']
        seeg['time_series'] = raw.get_data().T
        seeg['sfreq'] = raw.info['sfreq']
        seeg['picks'] = picks
    return seeg


def read_seeg(data_dir):
    # Reads all seizures(.fif files) in the given folder
    json_fnames = glob.glob(f'{data_dir}/seeg/fif/*.json')
    json_fnames_woxtnsn = [fname.split('.json')[0] for fname in json_fnames]
    fif_fnames = glob.glob(f'{data_dir}/seeg/fif/*.raw.fif')
    fif_fnames_woxtnsn = [fname.split('.raw.fif')[0] for fname in fif_fnames]
    json_fnames = list(
        set.intersection(set(json_fnames_woxtnsn), set(fif_fnames_woxtnsn)))
    json_fnames = [f'{el}.json' for el in json_fnames]
    picks = find_picks(json_fnames)
    data = {'seizures': [], 'picks': picks}
    for fname in json_fnames:
        t = {}
        fname_wo_xtnsn = ''.join(fname.split('/')[-1].split('.json'))
        with open(fname) as fd:
            meta_data = json.load(fd)
        if (meta_data['type'].lower() == "spontaneous seizure"):
            raw = mne.io.Raw(f'{data_dir}/seeg/fif/{fname_wo_xtnsn}.raw.fif',
                             verbose='WARNING',
                             preload=True)
            assert meta_data['onset'] is not None and meta_data[
                'termination'] is not None
            #         raw.crop(tmin=meta_data['onset'], tmax=meta_data['termination'])
            raw.pick_types(meg=False, eeg=True)
            raw.pick_channels(picks)
            raw.reorder_channels(picks)
            t['fname'] = f'{fname_wo_xtnsn}'
            t['onset'] = meta_data['onset']
            t['offset'] = meta_data['termination']
            t['seeg'] = raw.get_data().T
            t['sfreq'] = raw.info['sfreq']
            data['seizures'].append(t)
    return data


def find_szr_len(data_dir, szr_name):
    '''
    Returns the length of the given seizure in seconds
    '''
    seeg = read_one_seeg(data_dir=data_dir,
                         meta_data_fname=f'{szr_name}.json',
                         raw_seeg_fname=f'{szr_name}.raw.fif')
    start_idx = int(seeg['onset'] * seeg['sfreq']) - int(seeg['sfreq'])
    end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq'])
    seeg_trunc = seeg['time_series'][start_idx:end_idx]
    szr_len = seeg_trunc.shape[0] / seeg['sfreq']
    return szr_len
