from lib.io.seeg import *
from lib.io.tvb import *
from scipy.spatial.distance import cdist
import numpy as np
import scipy.signal


def get_ez_from_epindx(ep_idx, ei_thrshld, cntcts_file, cntrs_zipfile,
                       dist_prcntl):
    '''
    Finds epileptogenic zone(EZ) in the ROI space given epileptogenicity index
    in sensor space
    
    Parameters:
    ep_idx - dict: A dictionary of EI values indexed by channel names
    ei_thrshld - float: A scalar threshold on EI values to identify contacts in EZ
    cntcts_file - string: Path to seeg.xyz file
    cntrs_zipfile - string: Path to connectivity*.zip file containing ROI centres
    dist_prcntl - float: Threshold percentile below which ROIs are considered as EZ
    
    Returns:
    ez_names - list: list of ROI names in EZ
    ez_idcs - list: list of ROI indexes in EZ
    '''
    ep_idx_keys, ep_idx_vals = zip(*ep_idx.items())
    ep_idx_vals = np.array(ep_idx_vals)
    ez_cntct_idcs = np.where(ep_idx_vals > ei_thrshld)[0]
    ez_cntct_names = [ep_idx_keys[_] for _ in ez_cntct_idcs]
    cntct_pos = read_contacts(cntcts_file)
    ez_cntct_coords = [cntct_pos[cntct] for cntct in ez_cntct_names]
    roi_cntrs, roi_lbls = read_roi_cntrs(cntrs_zipfile)
    dists = cdist(ez_cntct_coords, roi_cntrs)
    ez_idcs = []
    for row in dists:
        dist_thrshld = np.percentile(row, dist_prcntl)
        ez_idcs.extend(np.where(row < dist_thrshld)[0])
    # ez_idcs = np.unique(np.argmin(dists, axis=1))
    ez_idcs = list(set(ez_idcs))  # remove any duplicates
    ez_names = [roi_lbls[_] for _ in ez_idcs]
    return ez_names, ez_idcs


def comp_epindx(data_dir,
                raw_seeg_fname,
                meta_data_fname,
                bias=20,
                thrshld=200,
                tau=1,
                wndw_len=1024,
                wndw_ovrlp=512,
                ei_wndw_len=5):
    '''
    Computes Epileptogenicity Index of all electrodes

    Parameters:
    data_dir - string: directory containing patient folder, folder Hierarchy follows
    the structure used for retrospective patient data on cluster
    seeg_fname - string: .fif file name containing raw seeg time series
    meta_data_fname - string: .json file name corresponding to .fif file
    bias - float: EI algorithm parameter, parameter v in EI paper
    thrshld - float: EI algorithm parameter, parameter $\lambda$ in EI paper 
    tau - float: EI algorithm parameter 
    wndw_len - int: sliding window length
    wndw_ovrlp - int: overlap used for sliding window
    ei_wndw_len - int: Amount of time in seconds to use for computing EI, parameter 'H' in EI paper

    All EI algorithm parameters and need to be hand tuned for each patient

    Returns:
    ep_idx - dict: A dictionary of EI values indexed by channel names
    Nd - dict: A dictionary of rapid discharge onset times indexed by channel names
    N0 - tuple: A tuple of (channel name, Nd) of channel with earliest onset of rapid discharges
    '''
    seeg = read_seeg(data_dir, meta_data_fname, raw_seeg_fname)
    seeg_spcgrm = dict()
    ER = dict()
    Un = dict()
    Na = dict()
    Nd = dict()
    N0 = ('', 1e5)
    ep_idx = dict()
    for ch_name, ts in zip(seeg['ch_names'], seeg['time_series'].T):
        f, t, seeg_spcgrm[ch_name] = scipy.signal.spectrogram(
            ts, fs=seeg['sfreq'], nperseg=wndw_len, noverlap=wndw_ovrlp)
        lf_indcs = np.where(np.logical_and(f >= 4, f <= 8))[0]
        hf_indcs = np.where(np.logical_and(f >= 8, f <= 124))[0]
        E_lf = seeg_spcgrm[ch_name][lf_indcs].mean(axis=0)
        E_hf = seeg_spcgrm[ch_name][hf_indcs].mean(axis=0)
        ER[ch_name] = E_lf / E_hf
        Un[ch_name] = np.zeros(ER[ch_name].size)
        for i in range(ER[ch_name].size):
            for j in range(i + 1):
                Un[ch_name][
                    i] += ER[ch_name][j] - ER[ch_name][0:j + 1].mean() - bias
        for i in range(Un[ch_name].size):
            Nd[ch_name] = Un[ch_name][0:i + 1].argmin()
            Na[ch_name] = 0
            Un_min = Un[ch_name][Nd[ch_name]]
            if (abs(Un[ch_name][i] - Un_min) > thrshld):
                Na[ch_name] = i
                break
        if (Na[ch_name] == 0):
            Nd[ch_name] = Un[ch_name].size - 1
        if (Nd[ch_name] != 0 and Nd[ch_name] < N0[1]):
            N0 = (ch_name, Nd[ch_name])
    for ch_name in seeg['ch_names']:
        start_idx = Nd[ch_name]
        H = int(
            np.ceil((ei_wndw_len * seeg['sfreq']) / (wndw_len - wndw_ovrlp)))
        if (start_idx + H <= Un[ch_name].size):
            end_idx = start_idx + H
        else:
            end_idx = ER[ch_name].size
        ep_idx[ch_name] = (np.sum(
            ER[ch_name][start_idx:end_idx])) / (Nd[ch_name] - N0[1] + tau)
    # Normalze EI by maximum
    k, v = zip(*ep_idx.items())
    v = v / max(v)
    ep_idx = dict(zip(k, v))
    return ep_idx, Nd, N0
