"""
Workflow

"""
import os
import re
import subprocess as sp
import numpy as np
import pylab as pl
import nibabel as nib
from scipy import signal
import dmeeg
import lib



def bfilt(data, samp_rate, fs, mode, order=3):
    b, a = signal.butter(order, 2*fs/samp_rate, mode)
    return signal.lfilter(b, a, data)

# data
subj_name = 'trec'
dir_name = f'{subj_name}.d'
if not os.path.exists(dir_name):
    import tarfile
    tarfile.open(f'{subj_name}.tbz').extractall(dir_name)

# connectivity
w = np.loadtxt('trec.d/weights.txt')
np.fill_diagonal(w, 0.0)
pl.imshow(w**0.5); pl.title('W^0.1'); pl.show()

# parse seeg contacts as localized in brain w/ x y z positions
contact_positions = {}
with open('trec.d/elecs_name.txt', 'r') as fd:
    for line in fd.readlines():
        name_idx, x, y, z = line.strip().split()
        contact_positions[name_idx] = np.r_[float(x), float(y), float(z)]

# parse seeg contacts present in electrophysiological data
ades_contact_re = r"([A-Z]+[a-z]*[']*)([0-9]+)"
contacts = []
samp_rate = None
with open('trec.d/complex.ades', 'r') as fd:
    for line in fd.readlines():
        if line.strip().startswith('#'):
            continue
        key, val = [p.strip() for p in line.strip().split('=')]
        if key == 'samplingRate':
            samp_rate = float(val)
        if val == 'SEEG':
            name, idx = re.match(ades_contact_re, key).groups()
            idx = int(idx)
            contacts.append((name, idx, contact_positions[key.lower()]))
cxyz = np.array([xyz for _, _, xyz in contacts])

# construct bipolar contacts
contacts_bip = []
for i in range(len(contacts) - 1):
    (ln, li, lp), (rn, ri, rp) = contacts[i:i + 2]
    if ln != rn:
        continue
    contacts_bip.append(('%s%d-%d' % (ln, li, ri), (i, i + 1), (lp + rp) / 2))
proj_bip = np.zeros((len(contacts_bip), len(contacts)))
for i, (_, idx, _) in enumerate(contacts_bip):
    proj_bip[i, idx] = 1, -1

# load regions
regions = []
with open('trec.d/centers.txt', 'r') as fd:
    for i, line in enumerate(fd.readlines()):
        rname, x, y, z = line.split()
        regions.append((f'r{i}-{rname}', np.r_[float(x), float(y), float(z)]))
rxyz = np.array([xyz for _, xyz in regions])

# move to correct coord system
seg = nib.load('trec.d/aparcaseg_2_diff_2.nii.gz')  # this is aa in diff space
rxyz = seg.affine.dot(np.c_[rxyz, np.ones((rxyz.shape[0],))].T)[:3].T

# plot regions v contacts, brain regions as red x's
pl.figure()
for i, (j, k) in enumerate([(0, 1), (1, 2), (0, 2)]):
    pl.subplot(2, 2, i + 1)
    pl.plot(cxyz[:, j], cxyz[:, k], 'bo')
    pl.plot(rxyz[:, j], rxyz[:, k], 'rx')
pl.show()

# look at gain matrix
gain = 1/np.sqrt(np.sum((cxyz[:, None] - rxyz)**2, axis=2))
pl.imshow(gain, cmap='binary', interpolation='none'), pl.show()
# TODO compute voxelwise & avg to avoid peaks

# bipolar seeg data
seeg = np.load('trec.d/complex.npy')[:-2]  # last two are ECG & SAT
# seeg = dmeeg.EEG('trec.d/110216B-CEX_0000.EEG')
# for l, (r,i,*_) in zip(seeg.chnm, contacts):
#     assert l.decode('ascii') == f'{r}{i}'
# seeg = seeg.read_data()[0][:-2]
t = np.r_[:seeg.shape[1]] / samp_rate
bip = proj_bip.dot(seeg)

# rm artifacted contacts
afc_names = "C'1-2 B9-10 B10-11 H'10-11 H'11-12 H'12-13 C'2-3".split()
afc_idx = np.array([i for i, (name, *_) in enumerate(contacts_bip) if name in afc_names])

# plot time series for inspection
pl.figure(figsize=(10, 20))
tm = ((np.r_[:len(t)] % 10) == 0)
pl.plot(t[tm], bip[:, tm].T/800 + np.r_[:len(bip)], 'k', linewidth=0.05)
pl.axis('tight')
pl.yticks(np.r_[:len(bip)], [f'{contacts_bip[i][0]}' for i in np.r_[:len(bip)]])
pl.ylim((-1, len(bip) + 1)); pl.xlim((0, t[-1]))
pl.grid(1, linewidth=0.3, alpha=0.4)
pl.title('Bipolar sEEG'); pl.tight_layout(); pl.show()

# compute envelope, keep those w/ large amplitude
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
lbenv = np.log(np.clip(benv[isort], benv[benv>0].min(), None))
lbenv_all = np.log(np.clip(benv, benv[benv>0].min(), None));

# visualize kept envelopes & window statistics
pl.figure(figsize=(8, 6))
pl.subplot(211)
pl.plot(te, lbenv.T/lbenv.max() + np.r_[:len(lbenv)], 'k')
pl.yticks(np.r_[:len(lbenv)], [contacts_bip[i][0] for i in isort])
pl.xlabel('Time (s)'), pl.ylabel('Seizure Sensors')
pl.grid(1, linestyle='--', alpha=0.3)
pl.title('log bipolar sEEG envelope (`lbenv`)')
for i, (t0, t1) in enumerate([(0, 50), (60, 100), (120, 250)]):
    tm = (te > t0) * (te < t1)
    pl.subplot(2,3,i+4)
    pl.hist(lbenv[:,tm].flat[:], np.r_[0.0:6.0:30j], normed=True, orientation='horizontal')
    pl.hist(lbenv_all[iother[:, None],tm].flat[:], np.r_[0.0:6.0:30j], normed=True, orientation='horizontal')
    #pl.ylim([0, 1])
    pl.title(f'Time {t0} - {t1}s')
    pl.legend(('Seizure Sensors', 'Others'))
    pl.ylabel('p(lbenv)'), pl.xlabel('lbenv')
    pl.grid(1, linestyle='--', alpha=0.3)
pl.tight_layout()
pl.show()

# use gain matrix to pick n nodes per contact
gain_bip = proj_bip.dot(gain)
# patient-specific: hypothalamus
hypo_idx = np.array([i for i, (name, *_) in enumerate(regions) if 'Hypothalamus' in name])
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
