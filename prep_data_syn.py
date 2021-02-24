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
results_dir = 'results/exp10/exp10.88.1/snr0.1_5.0_step0.1_10samples_per_step/'
os.makedirs(os.path.join(results_dir,'Rfiles'), exist_ok=True)
os.makedirs(os.path.join(results_dir,'samples'), exist_ok=True)
os.makedirs(os.path.join(results_dir,'logs'), exist_ok=True)
os.makedirs(os.path.join(results_dir,'code'), exist_ok=True)
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

param_init = {'x0':x0, 'alpha':alpha,
              'beta':beta, 'K':K, 'tau0':tau0, 'x_init':x_init, 'z_init':z_init,
              'eps_slp':eps_slp, 'eps_snsr_pwr':eps_snsr_pwr}
param_init_file = 'param_init.R'
lib.io.stan.rdump(f'{results_dir}/Rfiles/param_init.R', param_init)

# %%
for el_snr in snr:
    for i in range(1,11):
        seeg_noised = seeg + \
            np.random.normal(loc=0.0, scale=avg_pwr/el_snr, size=seeg.shape)
        slp = lib.preprocess.envelope.compute_slp_syn(data=seeg_noised,
                                                    samp_rate=256, win_len=50, hpf=10.0, lpf=2.0, logtransform=True)
        ds_freq = int(np.round(slp.shape[0]/nt))
        slp_ds = slp[0:-1:ds_freq, :]
        snsr_pwr = np.mean(slp_ds**2, axis=0)
        data = {'nn': nn, 'ns': ns, 'nt': slp_ds.shape[0], 'SC': SC, 'gain': gain_mat,
                'slp': slp_ds, 'snsr_pwr': snsr_pwr, 'x0_mu': x0_mu}
        input_Rfile = f'fit_data_snsrfit_ode_snr{el_snr:.1f}_sample{i}.R'
        lib.io.stan.rdump(os.path.join(results_dir, 'Rfiles', input_Rfile), data)

# %%
fit_data = lib.io.stan.rload(os.path.join(results_dir,
                                          'Rfiles',
                                          'fit_data_snsrfit_ode_snr4.5.R'))
plt.figure(figsize=(20,5))
plt.plot(fit_data['slp']);
# %% [markdown]
### Prepare data for different initial conditions
# %%
import numpy as np
import matplotlib.pyplot as plt
import lib.preprocess.envelope
import os
import lib.io.stan
import scipy.stats as stats
# %%
data_dir = 'datasets/syn_data/id001_bt'
results_dir = 'results/exp10/exp10.88.2'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/Rfiles',exist_ok=True)
os.makedirs(f'{results_dir}/samples',exist_ok=True)
os.makedirs(f'{results_dir}/logs',exist_ok=True)
os.makedirs(f'{results_dir}/code',exist_ok=True)
os.makedirs(f'{results_dir}/code/slurm_logs',exist_ok=True)
network = np.load(f'{data_dir}/network.npz')
SC = network['SC']
K = np.max(SC)
SC = SC / K
SC[np.diag_indices(SC.shape[0])] = 0

gain_mat = network['gain_mat']

slp = np.load(f'{data_dir}/fit_trgt.npz')['fit_trgt']
npoints = 300
ds_freq = int(np.round(slp.shape[0]/npoints))
slp_ds = slp[0:-1:ds_freq,:]
snsr_pwr = np.mean(slp_ds**2, axis=0)
# %%
nn = SC.shape[0]
ns = gain_mat.shape[0]
nt = slp_ds.shape[0]
tvb_syn_data = np.load(os.path.join(
    data_dir, 'syn_tvb_ez=48-79_pz=11-17-22-75.npz'))

x0_mu = -3.0*np.ones(nn)
ez_hyp = np.array(tvb_syn_data['ez'])
pz_hyp = tvb_syn_data['pz']
x0_mu[ez_hyp] = -1.8
x0_mu[pz_hyp] = -2.3

sigma_prior = np.arange(0.1, 1.1, 0.1)
for el_sigma_prior in sigma_prior:
    for i in range(1,11):
        x0_clip_a, x0_clip_b = -5.0, 0.0
        x0_loc = x0_mu
        x0 = stats.truncnorm.rvs(a=(x0_clip_a - x0_loc)/el_sigma_prior, b=(
            x0_clip_b - x0_loc)/el_sigma_prior, loc=x0_loc, scale=el_sigma_prior)

        alpha_clip_a, alpha_clip_b = 0.0, 50.0
        alpha_loc = 1.0
        alpha = stats.truncnorm.rvs(a=(alpha_clip_a - alpha_loc)/el_sigma_prior, b=(
            alpha_clip_b - alpha_loc)/el_sigma_prior, loc=alpha_loc, scale=el_sigma_prior)[0]

        beta_clip_a, beta_clip_b = -10.0, 10.0
        beta_loc = 0.0
        beta = stats.truncnorm.rvs(a=(beta_clip_a - beta_loc)/el_sigma_prior, b=(
            beta_clip_b - beta_loc)/el_sigma_prior, loc=beta_loc, scale=el_sigma_prior)[0]

        K_clip_a, K_clip_b = 0.0, 5.0
        K_loc = 1.0
        K = stats.truncnorm.rvs(a=(K_clip_a - K_loc)/el_sigma_prior, b=(
            K_clip_b - K_loc)/el_sigma_prior, loc=K_loc, scale=el_sigma_prior)[0]

        tau0_clip_a, tau0_clip_b = 15.0, 100.0
        tau0_loc = 20.0
        tau0 = stats.truncnorm.rvs(a=(tau0_clip_a - tau0_loc)/el_sigma_prior, b=(
            tau0_clip_b - tau0_loc)/el_sigma_prior, loc=tau0_loc, scale=el_sigma_prior)[0]

        eps_slp_clip_a, eps_slp_clip_b = 0.5, 10.0
        eps_slp_loc = 1.0
        eps_slp = stats.truncnorm.rvs(a=(eps_slp_clip_a - eps_slp_loc)/el_sigma_prior, b=(
            eps_slp_clip_b - eps_slp_loc)/el_sigma_prior, loc=eps_slp_loc, scale=el_sigma_prior)[0]

        eps_snsr_pwr_clip_a, eps_snsr_pwr_clip_b = 0.5, 10.0
        eps_snsr_pwr_loc = 1.0
        eps_snsr_pwr = stats.truncnorm.rvs(a=(eps_snsr_pwr_clip_a - eps_snsr_pwr_loc)/el_sigma_prior, b=(
            eps_snsr_pwr_clip_b - eps_snsr_pwr_loc)/el_sigma_prior, loc=eps_snsr_pwr_loc, scale=el_sigma_prior)[0]

        x_init_clip_a, x_init_clip_b = -2.5, -1.5
        x_init_loc = -2.0*np.ones(nn)
        x_init = stats.truncnorm.rvs(a=(x_init_clip_a - x_init_loc)/el_sigma_prior, b=(
            x_init_clip_b - x_init_loc)/el_sigma_prior, loc=x_init_loc, scale=el_sigma_prior)

        z_init_clip_a, z_init_clip_b = 3.0, 5.0
        z_init_loc = 3.5*np.ones(nn)
        z_init = stats.truncnorm.rvs(a=(z_init_clip_a - z_init_loc)/el_sigma_prior, b=(
            z_init_clip_b - z_init_loc)/el_sigma_prior, loc=z_init_loc, scale=el_sigma_prior)

        param_init = {'x0': x0, 'alpha': alpha,
                      'beta': beta, 'K': K, 'tau0': tau0, 'x_init': x_init, 'z_init': z_init,
                      'eps_slp': eps_slp, 'eps_snsr_pwr': eps_snsr_pwr}
        lib.io.stan.rdump(
            f'{results_dir}/Rfiles/param_init_sigmaprior{el_sigma_prior:.1f}_sample{i}.R', param_init)

data = {'nn': nn, 'ns': ns, 'nt': nt, 'SC': SC, 'gain': gain_mat,
        'slp': slp_ds, 'snsr_pwr': snsr_pwr, 'x0_mu': x0_mu}
input_Rfile = f'fit_data_snsrfit_ode.R'
os.makedirs(f'{results_dir}/Rfiles', exist_ok=True)
lib.io.stan.rdump(f'{results_dir}/Rfiles/{input_Rfile}', data)
