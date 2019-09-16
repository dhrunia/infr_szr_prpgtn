from lib.io.seeg import *
from lib.io.tvb import *
from scipy.spatial.distance import cdist
import numpy as np
import scipy.signal

def get_ez_from_epindx(epindx, ei_cntcts, ei_thrshld, cntcts_file, cntrs_zipfile):
    ez_idx_ei_cntcts = epindx > 0.2
    ez_ei_cntcts = ei_cntcts[ez_idx_ei_cntcts]
    cntct_pos = read_contacts(cntcts_file)
    ez_cntct_coords = []
    for cntct in ez_ei_cntcts:
        c1,c2 = cntct.split('-')
        mdpnt = (cntct_pos[c1] + cntct_pos[c2])/2.0
        ez_cntct_coords.append(mdpnt)
    roi_cntrs, roi_lbls = read_roi_cntrs(cntrs_zipfile)
    dists = cdist(ez_cntct_coords, roi_cntrs)
    ez = np.unique(np.argmin(dists, axis=1))
    return ez


def comp_epindx(data_dir, seeg_fname, meta_data_fname, bias=20, thrshld=200, tau=1, wndw_len=1024, wndw_ovrlp=512, ei_wndw_len=5):
    # EI algorithm parameters
    seeg = read_seeg(data_dir, meta_data_fname, raw_seeg_fname)
    seeg_spcgrm = dict()
    ER = dict()
    Un = dict()
    Na = dict()
    Nd = dict()
    N0 = ('',1e5)
    EI= dict()
    for ch_name, ts in zip(seeg['ch_names'], seeg['time_series'].T):
        f,t,seeg_spcgrm[ch_name] = scipy.signal.spectrogram(ts, fs=seeg['sfreq'], nperseg=wndw_len, noverlap=wndw_ovrlp)
        lf_indcs = np.where(np.logical_and(f >= 4, f <= 8))[0]
        hf_indcs = np.where(np.logical_and(f >= 8, f <= 124))[0]
        E_lf = seeg_spcgrm[ch_name][lf_indcs].mean(axis=0)
        E_hf = seeg_spcgrm[ch_name][hf_indcs].mean(axis=0)
        ER[ch_name] = E_lf / E_hf
        Un[ch_name] = np.zeros(ER[ch_name].size)
        for i in range(ER[ch_name].size):
            for j in range(i+1):
                Un[ch_name][i] += ER[ch_name][j] - ER[ch_name][0:j+1].mean() - bias
        for i in range(Un[ch_name].size):
            Nd[ch_name] = Un[ch_name][0:i+1].argmin()
            Na[ch_name] = 0
            Un_min = Un[ch_name][Nd[ch_name]]
            if(abs(Un[ch_name][i] - Un_min) > thrshld):
                Na[ch_name] = i
                break
        if(Na[ch_name] == 0):
            Nd[ch_name] = Un[ch_name].size - 1
        if(Nd[ch_name] !=0 and Nd[ch_name] < N0[1]):
            N0 = (ch_name, Nd[ch_name])
    for ch_name in seeg['ch_names']:
        start_idx = Nd[ch_name]
        H = int(np.ceil((ei_wndw_len*seeg['sfreq'])/(wndw_len - wndw_ovrlp)))
        if(start_idx + H <= Un[ch_name].size):
            end_idx = start_idx + H
        else:
            end_idx = ER[ch_name].size
        EI[ch_name] = (np.sum(ER[ch_name][start_idx:end_idx]))/(Nd[ch_name] - N0[1] + tau)
    return EI, Nd, N0

bias = 20
thrshld = 200
tau = 1
wndw_len = 1024
wndw_ovrlp = 512
ei_wndw_len = 5
data_dir = 'datasets/retro/id030_bf'
meta_data_fname = 'BF_crise1P_110831B-CEX_0004.json'
raw_seeg_fname = 'BF_crise1P_110831B-CEX_0004.raw.fif'

ep_idx, Nd, N0 = comp_epindx(data_dir, raw_seeg_fname, meta_data_fname, bias=20, thrshld=200, tau=1, wndw_len=1024, wndw_ovrlp=512, ei_wndw_len=5)
import matplotlib.pyplot as plt
snsr_name = 'A1'
# plt.figure()
# plt.pcolormesh(t,f[0:250],np.log10(seeg_spcgrm[snsr_name][0:250,:]))
# plt.colorbar()
# plt.show(block=False)
# plt.figure()
# plt.plot(Un[snsr_name])
# plt.axvline(Na[snsr_name], color='red')
# plt.axvline(Nd[snsr_name], color='green')
# plt.show(block=False)
# plt.figure()
# plt.plot(ER[snsr_name])
# plt.show(block=False)
plt.figure()
plt.bar(np.r_[0:len(list(Nd.values()))], list(ep_idx.values()))
plt.title('EI', fontsize=18)
plt.show(block=False)
print(N0)

