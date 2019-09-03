
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import lib.io.stan
import lib.plots.stan
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import retro_prepare
import matplotlib.colors


# In[2]:


#patient_id = 'id030_bf'
#patient_id = 'id017_mk'
patient_id = 'id001_bt'
data_dir = f'/home/hfw/Retro/datasets/{patient_id}'
results_dir = f'/home/hfw/Retro/results/{patient_id}'
#szr_name = 'BF_crise1P_110831B-CEX_0004'
#szr_name = 'BF_crise1P_110831B-CEX_0004'
#szr_name = "MK_crise1Part1_170328C-BEX_0004_MK_crise1Part2_170328C-BEX_0007"
#szr_name = "MK_crise2_PSG_170328C-CEX_0002"
szr_name = 'BTcrise2appportable_0013'
#szr_name = 'BTcrise1appportable_0006'
meta_data_fname = f'{szr_name}.json'
#raw_seeg_fname = f'{szr_name}.raw.fif'
fname_suffix = f'{szr_name}'
# if os.path.isdir(results_dir):
#     os.rmdir(results_dir)
os.makedirs(results_dir,exist_ok=True)
os.makedirs(f'{results_dir}/logs',exist_ok=True)
os.makedirs(f'{results_dir}/figures',exist_ok=True)
os.makedirs(f'{results_dir}/Rfiles', exist_ok=True)


# In[3]:


hpf = 10
lpf = 0.2
raw_seeg_fname = f'{szr_name}.raw.fif'
#data_mono = retro_prepare_data.prepare_data(data_dir, meta_data_fname, raw_seeg_fname, szr_name, hpf, lpf)
#raw_seeg_fname = f'{szr_name}.bip.raw.fif'
data = retro_prepare.prepare_data_bip(data_dir, meta_data_fname, raw_seeg_fname, szr_name, hpf, lpf)
#fname_suffix += f'_hpf{hpf}_lpf{lpf}'


# In[4]:


seeg, bip = retro_prepare.read_one_seeg(data_dir, meta_data_fname, raw_seeg_fname)
slp = retro_prepare.compute_slp(seeg, bip, hpf, lpf)


# In[5]:



import mne
import pandas as pd
import re




all_fb_d0=np.load(f'{results_dir}/EZdelay/ez_prior_{szr_name}.npy')
d0_prior=np.mean(all_fb_d0,axis=0)
ez_prior=np.where(d0_prior>0.5)


# In[9]:



ts = 5
base_length = int(seeg['sfreq']*ts)

start_idx = int(seeg['onset'] * seeg['sfreq']) - base_length
end_idx = int(seeg['offset'] * seeg['sfreq']) + base_length
slp = bip.get_data().T[start_idx:end_idx]
#slp = seeg['time_series'].copy()
# Remove outliers i.e data > 2*sd
'''
plt.figure(figsize=(20,20))
plt.subplot(411)
plt.plot(slp, color='black', alpha=0.3);
plt.plot(slp[:,ezh]+2000,color='red');
plt.plot(slp[:,ezh],color='blue', alpha = 0.3);
plt.axvline(int(seeg['sfreq']*ts))
plt.axvline(len(slp)-int(seeg['sfreq']*ts))
'''

for i in range(slp.shape[1]):
    ts = slp[:, i]
    ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
# High pass filter the data
slp = lib.preprocess.envelope.bfilt(
    slp, seeg['sfreq'], hpf, 'highpass', axis=0)
'''
plt.subplot(412)
plt.plot(slp, color='black', alpha=0.3);
plt.plot(slp[:,ezh],color='blue',alpha = 0.3);
plt.plot(slp[:,ezh]+1000,color='red');
'''
# Compute seeg log power
slp = lib.preprocess.envelope.seeg_log_power(slp, 100)
'''
plt.subplot(413)
plt.plot(slp, color='black', alpha=0.3);
plt.plot(slp[:,ezh],color='blue',alpha = 0.3);
plt.plot(slp[:,ezh]+10,color='red');
'''
# Remove outliers i.e data > 2*sd
for i in range(slp.shape[1]):
    ts = slp[:, i]
    ts[abs(ts - ts.mean()) > 2 * ts.std()] = ts.mean()
# Low pass filter the data to smooth
slp = lib.preprocess.envelope.bfilt(
    slp, seeg['sfreq'], lpf, 'lowpass', axis=0)

'''
plt.subplot(414)
plt.plot(slp, color='black', alpha=0.3);
plt.plot(slp[:,ezh],color='blue',alpha = 0.3);
plt.plot(slp[:,ezh]+10,color='red');
plt.axvline(base_length)
plt.axvline(len(slp)-base_length)
'''

# In[10]:


import json
with open(f'../ANSM/util/data/ei-final.json','r') as f:
    ezh_all = json.load(f)
#ind_ez = ezh_all[patient_id]['i_ez']
#ind_pz = ezh_all[patient_id]['i_pz']


# In[11]:


ds_freq = int(data['slp'].shape[0]/150)
data['slp'] = data['slp'][0:-1:ds_freq]
data['snsr_pwr'] = (data['slp']**2).mean(axis=0)
data['ns'], data['nn'] = data['gain'].shape
data['nt'] = data['slp'].shape[0]
ez_hyp = np.where(np.loadtxt(f'{data_dir}/tvb/ez_hypothesis.vep.txt') == 1)[0]
data['x0_mu'] = -3.0*np.ones(data['nn'])
fname_suffix += f'_hpf{hpf}_lpf{lpf}_ezdelay'
data['x0_mu'][ez_prior[0]] = -1.5


# In[12]:

'''
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.imshow(data['SC'],norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=data['SC'].max()));
plt.colorbar(fraction=0.046,pad=0.04);
plt.title('Normalized SC (log scale)',fontsize=12, fontweight='bold')

plt.subplot(122)
plt.imshow(data['gain'],norm=matplotlib.colors.LogNorm(vmin=data['gain'].min(), vmax=data['gain'].max()));
plt.colorbar(fraction=0.046,pad=0.04);
plt.xlabel('Region#', fontsize=12)
plt.ylabel('Channel#', fontsize=12)
plt.title('Gain Matrix (log scale)',fontsize=12, fontweight='bold')
plt.savefig(f'{results_dir}/figures/network.png')

plt.figure(figsize=(25,13))
plt.subplot(211)
plt.plot(data['slp'], color='black', alpha=0.3);
plt.xlabel('Time', fontsize=12)
plt.ylabel('SLP', fontsize=12)

plt.subplot(212)
plt.bar(np.r_[1:data['ns']+1],data['snsr_pwr'], color='black', alpha=0.3);
plt.xlabel('Time', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.title('SEEG channel power', fontweight='bold')
plt.savefig(f'{results_dir}/figures/fitting_target_{fname_suffix}.png')
# plt.tight_layout()


# In[13]:


plt.figure(figsize=(25,5))
plt.bar(np.r_[1:data['nn']+1],data['x0_mu'], color='black', alpha=0.3)
plt.xticks(np.r_[1:data['nn']+1:2], fontsize=8);
plt.xlabel('ROI#')
plt.ylabel(r'$x_0$', fontsize=10)
plt.savefig(f'{results_dir}/figures/ez_hyp.png')

'''
# In[14]:


#stan_fname = 'vep-snsrfit-ode-nointerp'
stan_fname = 'szr_prpgtn'

x0 = data['x0_mu']
amplitude = 1.0 
offset = 0
K = 1.0
tau0 = 20
eps_slp = 1.0
eps_snsr_pwr = 1.0
x_init = -2.0*np.ones(data['nn'])
z_init = 3.5*np.ones(data['nn'])

param_init = {'x0':x0, 'amplitude':amplitude,
              'offset':offset, 'K':K, 'tau0':tau0, 'x_init':x_init, 'z_init':z_init,
              'eps_slp':eps_slp, 'eps_snsr_pwr':eps_snsr_pwr}

param_init_file = 'param_init.R'
os.makedirs(f'{results_dir}/Rfiles',exist_ok=True)
lib.io.stan.rdump(f'{results_dir}/Rfiles/param_init.R',param_init)

input_Rfile = f'fit_data_{fname_suffix}.R'
os.makedirs(f'{results_dir}/Rfiles',exist_ok=True)
lib.io.stan.rdump(f'{results_dir}/Rfiles/{input_Rfile}',data)


os.system('bash', '-s "$stan_fname" "$results_dir" "$input_Rfile" "$fname_suffix"', '\nSTAN_FNAME=$1\nRESULTS_DIR=$2\nINPUT_RFILE=$3\nFNAME_SUFFIX=$4\n\n\nfor i in {1..4};\ndo\n./${STAN_FNAME} optimize algorithm=lbfgs tol_param=1e-4 iter=20000 save_iterations=0  \\\ndata file=${RESULTS_DIR}/Rfiles/${INPUT_RFILE} \\\ninit=${RESULTS_DIR}/Rfiles/param_init.R \\\noutput file=${RESULTS_DIR}/samples_${FNAME_SUFFIX}_chain${i}.csv refresh=10 \\\n&> ${RESULTS_DIR}/logs/snsrfit_ode_${FNAME_SUFFIX}_chain${i}.log &\ndone')

