"""
Workflow

"""
import os
import re
import subprocess as sp
import numpy as np
import pylab as pl
import nibabel as nib
import lib.io.dmeeg as dmeeg
import lib
from lib.io.base import maybe_unpack_tbz
from lib.io.implantation import load_contact_positions, load_ades_merge_positions
from lib.preprocess.envelope import bfilt, compute_envelope
from lib.io.network import load_weights, load_centers
from lib.preprocess.base import bipolar_info, simple_gain, exclude_idx
from lib.plots.seeg import source_sensors, plot_bip, plot_envelope
import lib.io.seeg

maybe_unpack_tbz('trec')

contact_positions = load_contact_positions('trec.d/elecs_name.txt')
contacts, cxyz = load_ades_merge_positions('trec.d/complex.ades', contact_positions)
contacts_bip, proj_bip, cxyz_bip = bipolar_info(contacts)

w = load_weights('trec.d/weights.txt')
regions, rxyz = load_centers('trec.d/centers.txt',
                             'trec.d/aparcaseg_2_diff_2.nii.gz')

gain, gain_bip = simple_gain(cxyz, rxyz, proj_bip)

# bipolar seeg data
seeg, t, bip = lib.io.seeg.load_npy('trec.d/complex.npy', 512.0, proj_bip)
# seeg, t, bip = lib.io.seeg.load_eeg('trec.d/110216B-CEX_0000.EEG')

# rm artifacted contacts
afc_names = "C'1-2 B9-10 B10-11 H'10-11 H'11-12 H'12-13 C'2-3".split()
afc_idx = exclude_idx(contacts_bip, afc_names)

# envelope preprocessing
te, isort, iother, lbenv, lbenv_all = compute_envelope(bip, 512.0)

# plots
plot_bip(t, bip, contacts_bip)
source_sensors(cxyz, rxyz)
plot_envelope(te, isort, iother, lbenv, lbenv_all, contacts_bip)

# patient-specific: hypothalamus
import lib.io.network

hypo_idx = lib.io.network.pick_by_name(regions, 'Hypothalamus')


node_pick_idx = np.r_[hypo_idx, np.unique(np.argsort(gain_bip[isort], axis=1)[:, -1:])]
node_pick_idx = np.array([i for i in node_pick_idx if 'unknown' not in regions[i][0]])
[regions[i][0] for i in node_pick_idx]
gain_bip_ = gain_bip[isort][:, node_pick_idx]; gain_bip_.shape

# save image with picked regions for viz on brain
pick_img_dat = seg.get_data().copy()
for i in np.unique(pick_img_dat):
    mask = pick_img_dat == i
    pick_img_dat[mask] = -1 if i in node_pick_idx else -10
pick_img = nib.nifti1.Nifti1Image(pick_img_dat, seg.affine)
nib.save(pick_img, 'trec.d/pick.nii.gz')

# coefficients for constant nodes
iconst = np.setxor1d(np.r_[:w.shape[0]], node_pick_idx)
for i in range(w.shape[0]):
    w[i, i] = 0.0
w_pick = w[node_pick_idx][:, node_pick_idx]
w_pick /= w_pick.max()
Ic = w[node_pick_idx][:, iconst].sum(axis=1)

# process interictal data for data dump
seeg_ii = dmeeg.EEG('trec.d/110216B-CEX_0000.EEG')
for l, (r,i,*_) in zip(seeg_ii.chnm, contacts):
    assert l.decode('ascii') == f'{r}{i}'
seeg_ii = seeg_ii.read_data()[0][:-2]
t_ii = np.r_[:seeg_ii.shape[1]] / samp_rate
bip_ii = proj_bip.dot(seeg_ii)
benv_ii = bfilt(np.abs(bfilt(bip_ii, samp_rate, hp_freq, 'high')), samp_rate, lp_freq, 'low')[:, start::skip]
te_ii = t_ii[start::skip]
lbenv_ii = np.log(np.clip(benv_ii[isort], benv_ii[benv_ii>0].min(), None))

# dump it all
lib.rdump('new-data.R', {
    'nn': w_pick.shape[0], 'ns': gain_bip_.shape[0], 'nt': lbenv.shape[1],
    'I1': 3.1, 'tau0': 3.0, 'dt': te[1] - te[0],
    'SC': w_pick, 'SC_var': 5.0, 'gain': gain_bip_, 'seeg_log_power': lbenv.T, 'Ic': Ic,
    'K_lo': 1.0, 'K_u': 5.0, 'K_v': 10.0,
    'x0_lo': -15.0, 'x0_hi': 5.0, 'eps_hi': 0.2, 'sig_hi': 0.025,
    'zlim': np.r_[0.0, 10.0],
    'siguv': np.r_[-1.0, 0.5],
    'epsuv': np.r_[-1.0, 0.5],
    'use_data': 1,
    'tt': 0.08,
    'seeg_log_power_ii': lbenv_ii,
    'nt_ii': lbenv_ii.shape[1]
})

# TODO pick nearby nodes, or aparc+aseg as direct observation

# TODO est fixed points; e.g. algebra_solver

# TODO can preprocess all EEG files in same way, and choose in data file


#This is the preprocessing of the time series I’ve settled on now, without normalization, so that different data sets are comparable.  in the lower part, I’ve plotted histograms of the values pre, post and during the seizure; these should better constrain priors
