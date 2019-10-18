from scipy import signal
import numpy as np


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
        envlp[i, :] = np.log(np.mean(envlp_pad[i:i + win_len, :]**2, axis=0))
    return envlp


def compute_fitting_target(data,
                           samp_rate,
                           win_len=100,
                           hpf=10.0,
                           lpf=1.0,
                           logtransform=False):
    '''
    Extracts smoothed log. power of given SEEG
    '''
    # high pass filter to remove baseline shift
    data_hpf = bfilt(data, samp_rate, hpf, 'highpass', axis=0)
    # compute the log power over a sliding window
    data_lpwr = np.log(mov_avg(data_hpf**2, win_len) +
                       1) if logtransform else mov_avg(data_hpf**2, win_len)
    # low pass filter the log power for smoothing
    data_lpwr = bfilt(data_lpwr, samp_rate, lpf, 'lowpass', axis=0)
    return data_lpwr
