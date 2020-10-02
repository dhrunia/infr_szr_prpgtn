# %%
# Genearate plots from MAP estimate:
# - x0 violin
# - pair plots
# - Inferred activity
# - Fit to slp
#%%
import lib.io.stan
import lib.plots.stan
import sys
import json
import os
import numpy as np


with open('datasets/retro/ez_hyp_destrieux.json') as fd:
    ez_hyp_all = json.load(fd)


ez_hyp = ez_hyp_all[snakemake.params['patient_id']]['i_ez']
data = lib.io.stan.rload(snakemake.input['fit_data'])
pstr_samples = lib.io.stan.read_samples([snakemake.input['csv']])
lib.plots.stan.x0_violin_patient(pstr_samples['x0'], ez_hyp, figsize=(25,5), figname=snakemake.output['x0_violin'])
lib.plots.stan.pair_plots(pstr_samples, ['tau0', 'alpha', 'beta', 'K', 'eps_slp', 'eps_snsr_pwr', 'x_init', 'z_init'], figname=snakemake.output['pair_plot'])
x_pp_mean = np.mean(pstr_samples['y'][:,:,0:data['nn']], axis=0)
z_pp_mean = np.mean(pstr_samples['y'][:,:,data['nn']:2*data['nn']], axis=0)
lib.plots.stan.plot_source(x_pp_mean, z_pp_mean, ez_hyp, [], figname=snakemake.output['pred_src'])
# lib.plots.stan.plot_phase(x_pp_mean, z_pp_mean, ez_hyp, [], non_szng_roi,
#                           figname=f'{results_dir}/figures/posterior_predicted_phase_plot_{fname_suffix}.png')
lib.plots.stan.plot_fit_target({'slp':pstr_samples['mu_slp'].mean(axis=0), 'snsr_pwr':pstr_samples['mu_snsr_pwr'].mean(axis=0)},
                                data, figname=snakemake.output['fit_to_slp'])