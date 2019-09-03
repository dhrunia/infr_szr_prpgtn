#!/usr/bin/env python3

import numpy as np
import lib.io.stan
#import lib.utils.stan
import lib.preprocess.envelope
import glob
import os
import zipfile
import mne
import json
import re

def find_picks(json_fnames):
    picks = []
    for fname in json_fnames:
        fname_wo_xtnsn = os.path.splitext(os.path.basename(fname))[0]
        with open(fname) as fd:
            meta_data = json.load(fd)
        if (meta_data['type'].lower() == "spontaneous seizure"):
            exclude = meta_data['bad_channels'] + meta_data['non_seeg_channels']
            ch_names = set(
                mne.io.read_info(f'{os.path.dirname(fname)}/{fname_wo_xtnsn}.raw.fif')[
                    'ch_names'])
            picks.append(ch_names - set(exclude))
        else:
            raise Exception('Not a spontaneous seizure')
    return set.intersection(*picks)


def read_seeg_xyz(data_dir):
    lines = []
    fname = os.path.join(data_dir, 'elec/seeg.xyz')
    with open(fname, 'r') as fd:
        for line in fd.readlines():
            name, *sxyz = line.strip().split()
            xyz = [float(_) for _ in sxyz]
            lines.append((name, xyz))
    return lines


def read_gain(data_dir, picks):
    gain = np.loadtxt(f'{data_dir}/elec/gain_inv-square.vep.txt')
    seeg_xyz = read_seeg_xyz(data_dir)
    # Re-order the gain matrix to match with order of channels read using MNE
    gain_idxs = []
    for label in picks:
        for i, (gain_label, _) in enumerate(seeg_xyz):
            if (label == gain_label):
                gain_idxs.append(i)
    gain = gain[gain_idxs]
    return gain


def read_seeg(data_dir):
    json_fnames = glob.glob(f'{data_dir}/seeg/fif/*.json')
    json_fnames_woxtnsn = [fname.split('.json')[0] for fname in json_fnames]
    fif_fnames = glob.glob(f'{data_dir}/seeg/fif/*.raw.fif')
    fif_fnames_woxtnsn = [fname.split('.raw.fif')[0] for fname in fif_fnames]
    json_fnames = list(set.intersection(set(json_fnames_woxtnsn), set(fif_fnames_woxtnsn)))
    json_fnames = [f'{el}.json' for el in json_fnames]
    picks = find_picks(json_fnames)
    data = {'seizures': [], 'picks': picks}
    for fname in json_fnames:
        t = {}
        fname_wo_xtnsn = ''.join(fname.split('/')[-1].split('.json'))
        with open(fname) as fd:
            meta_data = json.load(fd)
        if (meta_data['type'].lower() == "spontaneous seizure"):
            raw = mne.io.Raw(
                f'{data_dir}/seeg/fif/{fname_wo_xtnsn}.raw.fif',
                verbose='WARNING', preload=True)
            assert meta_data['onset'] is not None and meta_data['termination'] is not None
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



def read_one_seeg(data_dir, meta_data_fname, raw_seeg_fname):
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
        raw.reorder_channels(picks)
        seeg['fname'] = f'{fname_wo_xtnsn}'
        seeg['onset'] = meta_data['onset']
        seeg['offset'] = meta_data['termination']
        seeg['time_series'] = raw.get_data().T
        seeg['sfreq'] = raw.info['sfreq']
        seeg['picks'] = picks
    return seeg

def read_one_bip_seeg(data_dir, meta_data_fname, raw_seeg_fname):
    
    with open(os.path.join(data_dir, 'seeg', 'fif', meta_data_fname)) as fd:
            meta_data = json.load(fd)

    raw = mne.io.Raw(
                os.path.join(data_dir, 'seeg', 'fif', raw_seeg_fname),
                verbose='WARNING',
                preload=True)
    assert meta_data['onset'] is not None and meta_data['termination'] is not None

    #seeg['fname'] = f'{fname_wo_xtnsn}'
    seeg = dict()
    seeg['onset'] = meta_data['onset']
    seeg['offset'] = meta_data['termination']
    seeg['time_series'] = raw.get_data().T
    seeg['sfreq'] = raw.info['sfreq']
    return seeg



# def compute_slp(szr_data,
#                 lpf=100.0,
#                 hpf=10.0,
#                 filter_order=5.0,
#                 exp_slp_len=100):
#     slp = []
#     for seeg in szr_data['seeg'].T:
#         start_idx = int(szr_data['onset'] * szr_data['sfreq']) - int(
#             szr_data['sfreq'])
#         end_idx = int(szr_data['offset'] * szr_data['sfreq']) + int(
#             szr_data['sfreq'])
#         seeg_crop = seeg[start_idx:end_idx]
#         nyq = szr_data['sfreq'] / 2.0
#         # High pass filter the data
#         b, a = signal.butter(filter_order, hpf / nyq, 'high')
#         seeg_crop = signal.filtfilt(b, a, seeg_crop)
#         # Low pass filter the data
#         b, a = signal.butter(filter_order, lpf / nyq, 'low')
#         seeg_crop = signal.filtfilt(b, a, seeg_crop)
#         nperseg = int(8 * seeg_crop.shape[0] / (7 * exp_slp_len))
#         F, T, S = signal.spectrogram(
#             seeg_crop,
#             fs=szr_data['sfreq'],
#             nperseg=nperseg,
#             noverlap=nperseg // 8)
#         slp.append(np.log(S.sum(0))[:-1])
#     slp = np.array(slp).T
#     return slp


def compute_slp(seeg, hpf=10.0, lpf=1.0, filter_order=5.0):
    start_idx = int(seeg['onset'] * seeg['sfreq']) - int(seeg['sfreq'])
    end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq'])
    slp = seeg['time_series'][start_idx:end_idx]
    # Remove outliers i.e data > 2*sd
    for i in range(slp.shape[1]):
        ts = slp[:, i]
        ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
    # High pass filter the data
    slp = lib.preprocess.envelope.bfilt(
        slp, seeg['sfreq'], hpf, 'highpass', axis=0)
    # Compute seeg log power
    slp = lib.preprocess.envelope.seeg_log_power(slp, 100)
    # Remove outliers i.e data > 2*sd
    for i in range(slp.shape[1]):
        ts = slp[:, i]
        ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
    # Low pass filter the data to smooth
    slp = lib.preprocess.envelope.bfilt(
        slp, seeg['sfreq'], lpf, 'lowpass', axis=0)
    return slp


# def prepare_data(data_dir, results_dir):
#     try:
#         with zipfile.ZipFile(
#                 f'{data_dir}/tvb/connectivity.destrieux.zip') as sczip:
#             with sczip.open('weights.txt') as weights:
#                 SC = np.loadtxt(weights)
#                 SC = SC/SC.max()
#     except FileNotFoundError as err:
#         print(f'Structural connectivity not found for {data_dir}')
#         return
#     # Read SEEG data from fif files
#     raw_data = read_seeg(data_dir)
#     gain = read_gain(data_dir, raw_data['picks'])
#     # Compute seeg log power
#     for i, szr in enumerate(raw_data['seizures']):
#         slp = compute_slp(szr)
#         snsr_pwr = np.sum(slp**2, axis=0)
#         nn = gain.shape[1]
#         ns = gain.shape[0]
#         nt = slp.shape[0]
#         I1 = 3.1
#         time_step = 0.1
#         x_init = -2.0 * np.ones(nn)
#         z_init = 3.5 * np.ones(nn)
#         epsilon_snsr_pwr = 5.0
#         epsilon_slp = 0.1
#         data = {
#             'nn': nn,
#             'ns': ns,
#             'nt': nt,
#             'I1': I1,
#             'time_step': time_step,
#             'SC': SC,
#             'gain': gain,
#             'x_init': x_init,
#             'z_init': z_init,
#             'slp': slp,
#             'snsr_pwr': snsr_pwr,
#             'epsilon_slp': epsilon_slp,
#             'epsilon_snsr_pwr': epsilon_snsr_pwr
#         }
#         lib.io.stan.rdump(f"{results_dir}/Rfiles/fit_data_{szr['fname']}.R",
#                           data)
#     x0_star_star = np.zeros(nn)
#     amplitude_star_star = 0.0
#     offset = 0.0
#     K_star_star = 0.0
#     tau0_star_star = 0.0
#     param_init = {
#         'x0_star_star': x0_star_star,
#         'amplitude_star_star': amplitude_star_star,
#         'offset': offset,
#         'K_star_star': K_star_star,
#         'tau0_star_star': tau0_star_star
#     }
#     lib.io.stan.rdump(f'{results_dir}/Rfiles/param_init.R', param_init)


# def prepare_data_pymc3(data_dir,
#                        results_dir,
#                        meta_data_fname,
#                        raw_seeg_fname,
#                        hpf=10.0,
#                        lpf=1.0):
#     try:
#         with zipfile.ZipFile(
#                 f'{data_dir}/tvb/connectivity.destrieux.zip') as sczip:
#             with sczip.open('weights.txt') as weights:
#                 SC = np.loadtxt(weights)
#                 SC = SC / SC.max()
#     except FileNotFoundError as err:
#         print(f'Structural connectivity not found for {data_dir}')
#         return
#     # Read SEEG data from fif files
#     raw_data = read_one_seeg(data_dir, meta_data_fname, raw_seeg_fname)
#     gain = read_gain(data_dir, raw_data['picks'])
#     # Compute seeg log power
#     slp = compute_slp(raw_data, hpf, lpf)
#     nn = gain.shape[1]
#     ns = gain.shape[0]
#     nt = slp.shape[0]
#     I1 = 3.1
#     time_step = 0.1
#     eps_slp = 0.1
#     eps_snsr_pwr = 0.1
#     x_init = -2.0 * np.ones(nn)
#     z_init = 3.5 * np.ones(nn)
#     consts = {
#         'nn': nn,
#         'ns': ns,
#         'nt': nt,
#         'I1': I1,
#         'time_step': time_step,
#         'SC': SC,
#         'gain': gain,
#         'x_init': x_init,
#         'z_init': z_init,
#         'eps_slp': eps_slp,
#         'eps_snsr_pwr': eps_snsr_pwr
#     }
#     obs = {
#         'slp': slp,
#     }
#     x0_star = np.zeros(nn)
#     amplitude_star = 0.0
#     offset_star = 0.0
#     K_star = 0.0
#     tau0_star = 0.0
#     x_init_star = np.zeros(nn)
#     z_init_star = np.zeros(nn)
#     params_init = {
#         'x0_star': x0_star,
#         'amplitude_star': amplitude_star,
#         'offset_star': offset_star,
#         'K_star': K_star,
#         'tau0_star': tau0_star,
#         'x_init_star': x_init_star,
#         'z_init_star': z_init_star
#     }
#     return obs, consts, params_init


def prepare_data(data_dir, meta_data_fname, raw_seeg_fname, fname_suffix, hpf=10.0, lpf=1.0):
    try:
        with zipfile.ZipFile(
                f'{data_dir}/tvb/connectivity.vep.zip') as sczip:
            with sczip.open('weights.txt') as weights:
                SC = np.loadtxt(weights)
                SC[np.diag_indices(SC.shape[0])] = 0
                SC = SC/SC.max()
    except FileNotFoundError as err:
        print(f'Structural connectivity not found for {data_dir}')
        return
    # Read SEEG data from fif files
    #raw_data = read_one_bip_seeg(data_dir, meta_data_fname, raw_seeg_fname)
    raw_data = read_one_seeg(data_dir, meta_data_fname, raw_seeg_fname)
    # if(raw_data == -1):
    #     return -1, -1
    gain = read_gain(data_dir, raw_data['picks'])
    #gain = np.load(f'{data_dir}/seeg/fif/{fname_suffix}.gain.npy')
    # Compute seeg log power
    slp = compute_slp(raw_data, hpf, lpf)
    data = {
        'SC': SC,
        'gain': gain,
        'slp': slp
    }
    return data

def prepare_data_bip(data_dir, meta_data_fname, raw_seeg_fname, fname_suffix, hpf=10.0, lpf=1.0):
    try:
        with zipfile.ZipFile(
                f'{data_dir}/tvb/connectivity.vep.zip') as sczip:
            with sczip.open('weights.txt') as weights:
                SC = np.loadtxt(weights)
                SC[np.diag_indices(SC.shape[0])] = 0
                SC = SC/SC.max()
    except FileNotFoundError as err:
        print(f'Structural connectivity not found for {data_dir}')
        return
    # Read SEEG data from fif files
    raw_data = read_one_bip_seeg(data_dir, meta_data_fname, raw_seeg_fname)
     # if(raw_data == -1):
    #     return -1, -1
    #gain = read_gain(data_dir, raw_data['picks'])
    gain = np.load(f'{data_dir}/seeg/fif/{fname_suffix}.gain.npy')
    # Compute seeg log power
    slp = compute_slp(raw_data, hpf, lpf)
    data = {
        'SC': SC,
        'gain': gain,
        'slp': slp
    }
    return data



def seeg_ch_name_split(nm):
    """
    Split an sEEG channel name into its electrode name and index

    >>> seeg_ch_name_split('GPH10')
    ('GPH', 10)

    """

    try:
        elec, idx = re.match(r"([A-Za-z']+)(\d+)", nm).groups()
    except AttributeError as exc:
        return None
    return elec, int(idx)



def bipify_raw(raw):
    split_names = [seeg_ch_name_split(_) for _ in raw.ch_names]
    bip_ch_names = []
    bip_ch_data = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                bip_ch_names.append("%s%d-%d" % (name, idx, next_idx))
                data, _ = raw[[i, i + 1]]
                bip_ch_data.append(data[1] - data[0])
        except Exception as exc:
            print(exc)
    info = mne.create_info(
        ch_names=bip_ch_names,
        sfreq=raw.info["sfreq"],
        ch_types=["eeg" for _ in bip_ch_names],
    )
    bip = mne.io.RawArray(np.array(bip_ch_data), info, verbose="WARNING")
    return bip


def bipolarize_gain(gain, seeg_xyz, seeg_xyz_names):
    split_names = [seeg_ch_name_split(_) for _ in seeg_xyz_names]
    bip_gain_rows = []
    bip_xyz = []
    bip_names = []
    for i in range(len(split_names) - 1):
        try:
            name, idx = split_names[i]
            next_name, next_idx = split_names[i + 1]
            if name == next_name:
                bip_gain_rows.append(gain[i + 1] - gain[i])
                bip_xyz.append(
                    [(p + q) / 2.0 for p, q in zip(seeg_xyz[i][1], seeg_xyz[i + 1][1])]
                )
                bip_names.append("%s%d-%d" % (name, idx, next_idx))
        except Exception as exc:
            print(exc)
    # abs val, envelope/power always postive
    bip_gain = np.abs(np.array(bip_gain_rows))
    bip_xyz = np.array(bip_xyz)
    return bip_gain, bip_xyz, bip_names


def read_gain(subj_proc_dir):
    np_fname = os.path.join(
        subj_proc_dir,
        "elec/gain_inv-square.vep.txt")
    return np.loadtxt(np_fname)


def read_seeg_xyz(subj_proc_dir):
    lines = []
    fname = os.path.join(subj_proc_dir, "elec/seeg.xyz")
    with open(fname, "r") as fd:
        for line in fd.readlines():
            name, *sxyz = line.strip().split()
            xyz = [float(_) for _ in sxyz]
            lines.append((name, xyz))
    return lines


def load_raw_fif(js):
    fif_fname = os.path.join(os.path.dirname(js["_source"]), js["filename"])
    raw = mne.io.Raw(fif_fname, preload=True, verbose='WARNING')
    drops = [_ for _ in (js["bad_channels"] + js["non_seeg_channels"]) if _ in raw.ch_names]
    raw = raw.drop_channels(drops)
    raw = bipify_raw(raw)
    raw = raw.crop(tmin=js["onset"], tmax=js["termination"])
    return raw


def load_js(js_fname):
    with open(js_fname, "r") as fd:
        js = json.load(fd)
    js["_source"] = js_fname
    return js


def bipolarize_js(js_fname):
    sd = js_fname.split('/seeg/fif/')[0]
    js = load_js(js_fname)
    raw = load_raw_fif(js)
    gain = read_gain(sd)
    seeg_xyz = read_seeg_xyz(sd)
    seeg_xyz_names = [label for label, _ in seeg_xyz]
    gain, seeg_xyz, seeg_xyz_names = bipolarize_gain(
        gain, seeg_xyz, seeg_xyz_names
    )
    gain_pick = []
    raw_drop = []
    for i, ch_name in enumerate(raw.ch_names):
        if ch_name in seeg_xyz_names:
            gain_pick.append(seeg_xyz_names.index(ch_name))
        else:
            raw_drop.append(ch_name)
    raw = raw.drop_channels(raw_drop)
    gain_pick = np.array(gain_pick)
    gain = gain[gain_pick]
    return raw, gain


def bipolarize_save(js_fname):
    js = load_js(js_fname)
    fif_fname = os.path.join(os.path.dirname(js["_source"]), js["filename"])
    bip_fif_fname = fif_fname.replace('.raw.fif', '.bip.raw.fif')
    raw, gain = bipolarize_js(js_fname)
    raw.save(bip_fif_fname, overwrite=True)
    gain_fname = fif_fname.replace('.raw.fif', '.gain.npy')
    np.save(gain_fname, gain)


def try_bipolarize_fname(js_fname):
    try:
        bipolarize_save(js_fname)
    except FileNotFoundError as exc:
        if not any(_ in exc.filename.lower()
                   for _ in ('stim', 'inter')):
            print(exc)





if (__name__ == '__main__'):
    data_root_dir = 'datasets/retro'
    results_root_dir = 'datasets/retro'
    patient_ids = [
        os.path.basename(path) for path in glob.glob(f'{data_root_dir}/id*')
    ]

    for id in patient_ids:
        data_dir = f'{data_root_dir}/{id}'
        results_dir = f'{results_root_dir}/{id}/fit_target'
        os.makedirs(results_dir, exist_ok=True)
        for json_path in glob.glob(f'{data_dir}/seeg/fif/*.json'):
            fname_wo_xtnsn = os.path.splitext(os.path.basename(json_path))[0]
            meta_data_fname = f'{fname_wo_xtnsn}.json'
            seeg_fname = f'{fname_wo_xtnsn}.raw.fif'
            print(f'Preparing {id} -> {meta_data_fname}')
            fif_exists = os.path.isfile(
                os.path.join(data_dir, 'seeg', 'fif', seeg_fname))
            sc_exists = os.path.isfile(
                os.path.join(data_dir, 'tvb', 'connectivity.destrieux.zip'))
            gain_exists = os.path.isfile(
                os.path.join(data_dir, 'elec',
                             'gain_inv-square.destrieux.txt'))
            with open(os.path.join(data_dir, 'seeg', 'fif',
                                   meta_data_fname)) as fd:
                meta_data = json.load(fd)
                is_spontaneous = (
                    meta_data['type'].lower() == "spontaneous seizure")

            if (fif_exists and sc_exists and gain_exists and is_spontaneous):
                data = prepare_data(data_dir, meta_data_fname, seeg_fname, 10,
                                    0.04)
                np.savez(
                    os.path.join(results_dir,
                                 f'obs_data_{fname_wo_xtnsn}.npz'), **data)
            else:
                if (not fif_exists):
                    print(f'\t ERROR: {seeg_fname} not found')
                elif (not sc_exists):
                    print('\t ERROR: SC not found')
                elif (not gain_exists):
                    print('\t ERROR: gain not found')
                elif (not is_spontaneous):
                    print('\t ERROR: Not a spontaneous seizure')
                else:
                    print('ERROR: Something went wrong')
