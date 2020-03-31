from scipy import signal
import numpy as np
import lib.io.stan
import lib.io.seeg
import glob
import os
import zipfile
import mne
import json


def bfilt(data, samp_rate, fs, mode, order=3, axis=-1):
    b, a = signal.butter(order, 2 * fs / samp_rate, mode)
    return signal.filtfilt(b, a, data, axis)


def mov_avg(a, win_len, pad=True):
    a_mov_avg = np.empty_like(a)
    a_pad = np.pad(
        a, ((0, win_len), (0, 0)), 'constant'
    )  # pad with zeros at the end to compute moving average of same length as the signal itself
    for i in range(a.shape[0]):
        a_mov_avg[i, :] = np.mean(a_pad[i:i + win_len, :], axis=0)
    return a_mov_avg


def seeg_log_power(a, win_len, pad=True):
    envlp = np.empty_like(a)
    # pad with zeros at the end to compute moving average of same length as the signal itself
    envlp_pad = np.pad(a, ((0, win_len), (0, 0)), 'constant')
    for i in range(a.shape[0]):
        envlp[i, :] = np.log(
            np.mean(envlp_pad[i:i + win_len, :]**2, axis=0) + 1)
    return envlp


def compute_slp_syn(data,
                    samp_rate,
                    win_len=100,
                    hpf=10.0,
                    lpf=1.0,
                    logtransform=False):
    '''
    Extracts smoothed log. power envelope of given SEEG for simulated data
    '''
    # high pass filter to remove baseline shift
    data_hpf = bfilt(data, samp_rate, hpf, 'highpass', axis=0)
    # compute the log power over a sliding window
    # data_lpwr = np.log(mov_avg(data_hpf**2, win_len) +
    #                    1) if logtransform else mov_avg(data_hpf**2, win_len)
    data_lpwr = np.log(mov_avg(data_hpf**2, win_len)) if logtransform else mov_avg(data_hpf**2, win_len)
    # low pass filter the log power for smoothing
    data_lpwr = bfilt(data_lpwr, samp_rate, lpf, 'lowpass', axis=0)
    return data_lpwr


def compute_slp(seeg, hpf=10.0, lpf=1.0, filter_order=5.0):
    start_idx = int(seeg['onset'] * seeg['sfreq']) - int(seeg['sfreq'])
    end_idx = int(seeg['offset'] * seeg['sfreq']) + int(seeg['sfreq'])
    slp = seeg['time_series'][start_idx:end_idx]
    # High pass filter the data
    slp = bfilt(slp, seeg['sfreq'], hpf, 'highpass', axis=0)
    # Compute seeg log power
    slp = seeg_log_power(slp, 100)
    # Low pass filter the data to smooth
    slp = bfilt(slp, seeg['sfreq'], lpf, 'lowpass', axis=0)
    return slp


def prepare_data(data_dir, meta_data_fname, raw_seeg_fname, hpf=10.0, lpf=1.0):
    try:
        with zipfile.ZipFile(
                f'{data_dir}/tvb/connectivity.destrieux.zip') as sczip:
            with sczip.open('weights.txt') as weights:
                SC = np.loadtxt(weights)
                SC[np.diag_indices(SC.shape[0])] = 0
                SC = SC / SC.max()
    except FileNotFoundError as err:
        print(f'Structural connectivity not found for {data_dir}')
        return
    # Read SEEG data from fif files
    raw_data = lib.io.seeg.read_one_seeg(data_dir, meta_data_fname,
                                         raw_seeg_fname)
    gain = lib.io.seeg.read_gain(data_dir, raw_data['picks'])
    # Compute seeg log power
    slp = compute_slp(raw_data, hpf, lpf)
    data = {'SC': SC, 'gain': gain, 'slp': slp}
    return data
