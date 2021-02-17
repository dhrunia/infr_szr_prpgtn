# %% [markdown]
### Prepare data with different signal to noise ratio in observed SEEG
#%%
import numpy as np
import matplotlib.pyplot as plt
import lib.preprocess.envelope
import os
import lib.io.stan
# %%
data_dir = 'datasets/syn_data/id001_bt'
results_dir = 'results/exp10/exp10.88.1/snr0.1_5.0_step0.1/'
os.makedirs(results_dir, exist_ok=True)
tvb_syn_data = np.load(os.path.join(
    data_dir, 'syn_tvb_ez=48-79_pz=11-17-22-75.npz'))
start_idx = 800
end_idx = 2200
seeg = tvb_syn_data['seeg'][:, start_idx:end_idx].T
seeg_hpf = lib.preprocess.envelope.bfilt(seeg, samp_rate=256, fs=10.0, mode='highpass', axis=0)
avg_pwr = (seeg_hpf**2).mean(axis=0).mean()
snr = np.arange(0.1, 5.1, 0.1)
# %% [markdown]
#### Plot raw seeg
# %%
plt.figure(figsize=(20, 5))
plt.plot(seeg);
# %%
slp = lib.preprocess.envelope.compute_slp_syn(data=seeg,
                                              samp_rate=256, win_len=50, hpf=10.0, lpf=2.0, logtransform=True)
# %% [markdown]
#### Plot slp
# %%
plt.figure(figsize=(25,5))
plt.plot(slp);
# %%
network = np.load(f'{data_dir}/network.npz')
SC = network['SC']
K = np.max(SC)
SC = SC / K
SC[np.diag_indices(SC.shape[0])] = 0

gain_mat = network['gain_mat']

nn = SC.shape[0]
ns = gain_mat.shape[0]
nt = 300

x0_mu = -3.0*np.ones(nn)
ez_hyp = tvb_syn_data['ez']
pz_hyp = tvb_syn_data['pz']
x0_mu[ez_hyp] = -1.8
x0_mu[pz_hyp] = -2.3

x0 = x0_mu
alpha = 1.0
beta = 0
K = 1.0
tau0 = 20
eps_slp = 1.0
eps_snsr_pwr = 1.0
x_init = -2.0*np.ones(nn)
z_init = 3.5*np.ones(nn)

os.makedirs(os.path.join(results_dir, 'Rfiles'), exist_ok=True)
param_init = {'x0':x0, 'alpha':alpha,
              'beta':beta, 'K':K, 'tau0':tau0, 'x_init':x_init, 'z_init':z_init,
              'eps_slp':eps_slp, 'eps_snsr_pwr':eps_snsr_pwr}
param_init_file = 'param_init.R'
lib.io.stan.rdump(f'{results_dir}/Rfiles/param_init.R', param_init)

# %%
for el_snr in snr:
    seeg_noised = seeg + \
        np.random.normal(loc=0.0, scale=avg_pwr/el_snr, size=seeg.shape)
    slp = lib.preprocess.envelope.compute_slp_syn(data=seeg_noised,
                                                  samp_rate=256, win_len=50, hpf=10.0, lpf=2.0, logtransform=True)
    ds_freq = int(np.round(slp.shape[0]/nt))
    slp_ds = slp[0:-1:ds_freq, :]
    snsr_pwr = np.mean(slp_ds**2, axis=0)
    data = {'nn': nn, 'ns': ns, 'nt': slp_ds.shape[0], 'SC': SC, 'gain': gain_mat,
            'slp': slp_ds, 'snsr_pwr': snsr_pwr, 'x0_mu': x0_mu}
    input_Rfile = f'fit_data_snsrfit_ode_snr{el_snr:.1f}.R'
    lib.io.stan.rdump(os.path.join(results_dir, 'Rfiles', input_Rfile), data)

# %%
fit_data = lib.io.stan.rload(os.path.join(results_dir,
                                          'Rfiles',
                                          'fit_data_snsrfit_ode_snr4.5.R'))
plt.figure(figsize=(20,5))
plt.plot(fit_data['slp']);
