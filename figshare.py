# %% [markdown]
# Figure 2: Fitting on synthetic data with good hypothesis
# %% [markdown]
## Panel a
# %%
import numpy as np
import matplotlib.pyplot as plt
import lib.io.stan
import lib.io.tvb
import os
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/fit_syn_good_hyp'
os.makedirs(figs_dir, exist_ok=True)
_, roi_names = lib.io.tvb.read_roi_cntrs('datasets/syn_data/id001_bt/connectivity.destrieux.zip')
sim_data = np.load('datasets/syn_data/id001_bt/syn_tvb_ez=48-79_pz=11-17-22-75.npz')
start_idx = 500
end_idx = 2600
sim_x = sim_data['src_sig'][start_idx:end_idx,0,:,0] + sim_data['src_sig'][start_idx:end_idx,3,:,0]
sim_z = sim_data['src_sig'][start_idx:end_idx,2,:,0]
n_roi = sim_x.shape[1]
map_est = lib.io.stan.read_samples(['results/exp10/exp10.69/samples_syn_optim_run1.csv'])
map_x = map_est['y'][0, :, :n_roi]
map_z = map_est['y'][0, :, n_roi:]
ez = sim_data['ez']
pz = sim_data['pz']
hz = np.setdiff1d(np.arange(0, n_roi), np.concatenate((ez,pz)))
# %%
fig = plt.figure(figsize=(5,25))
gs = fig.add_gridspec(1, ez.shape[0])
ez_pz = np.concatenate((ez, pz))
for i, roi in enumerate(ez_pz):
    plt.subplot(ez_pz.shape[0]+1, 1, i+1)
    plt.plot(sim_x[:, roi], sim_z[:, roi], color='black', alpha=0.3);
    plt.plot(map_x[:, roi], map_z[:, roi], color='red' if roi in ez else 'orange')
    plt.xlabel('x', fontweight='bold', fontsize=20)
    plt.ylabel('z', fontweight='bold', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(roi_names[roi] + (' (EZ)' if roi in ez else ' (PZ)'), fontsize=15)
plt.subplot(ez_pz.shape[0]+1, 1,  7)
plt.plot(sim_x[0:-1:5, hz], sim_z[0:-1:5, hz], color='black', alpha=0.3);
plt.plot(map_x[:, hz], map_z[:, hz], color='xkcd:black')
plt.xlabel('x', fontweight='bold', fontsize=20)
plt.ylabel('z', fontweight='bold', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Healthy Nodes', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'phasespace.svg'))
# %% [markdown]
### Save plotted data in txt format
# %%
np.savetxt(os.path.join(figs_dir, 'ground truth x.txt'), sim_x)
np.savetxt(os.path.join(figs_dir, 'ground truth z.txt'), sim_z)
np.savetxt(os.path.join(figs_dir, 'inferred x.txt'), map_x)
np.savetxt(os.path.join(figs_dir, 'inferred z.txt'), map_z)
# %% [markdown]
### Panel b: Fit to observations
# %%
import numpy as np
import matplotlib.pyplot as plt
import lib.io.stan
import lib.io.seeg
import os
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/fit_syn_good_hyp'
map_estim = lib.io.stan.read_samples(
    ['results/exp10/exp10.69/samples_syn_optim_run1.csv'])
slp_pred = map_estim['mu_slp'][0]
snsr_pwr_pred = map_estim['mu_snsr_pwr'][0]

obs_dat = lib.io.stan.rload(
    'results/exp10/exp10.69/Rfiles/fit_data_snsrfit_ode_syn_optim.R')
slp_obs = obs_dat['slp']
snsr_pwr_obs = obs_dat['snsr_pwr']
nt, ns = slp_pred.shape

chnl_names = [el[0] for el in lib.io.seeg.read_contacts(
    'datasets/syn_data/id001_bt/seeg.xyz', type='list')]

# %%
chnls = np.argsort(snsr_pwr_obs)[::-1][0:10]
fig = plt.figure(figsize=(18,20))
gs = fig.add_gridspec(10,4)
for i, chnl in enumerate(chnls):
    fig.add_subplot(gs[i, 0:3])
    plt.plot(slp_obs[:, chnl], color='xkcd:black', alpha=0.3)
    plt.plot(slp_pred[:, chnl], color='xkcd:red')
    if(i == 0):
        plt.title('SEEG Log. Power', fontsize=30)
    plt.ylabel(f'{chnl_names[chnl]} ({chnl+1})', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks([0,2,4,6], fontsize=20)
fig.add_subplot(gs[:,3])
plt.barh(np.r_[1:ns+1], snsr_pwr_pred, color='xkcd:red', alpha=0.3)
plt.barh(np.r_[1:ns+1], snsr_pwr_obs, color='xkcd:black', alpha=0.3)
plt.title('Total Power', fontsize=30)
plt.ylabel('Sensor', fontsize=30)
plt.yticks(np.r_[1:ns+1:10], np.r_[1:ns+1:10], fontsize=20);
plt.xticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'fit_to_observations.svg'))
# %%
np.savetxt(os.path.join(figs_dir, 'seeg log power - observed.txt'), slp_obs)
np.savetxt(os.path.join(figs_dir, 'seeg log power - predicted.txt'), slp_pred)
np.savetxt(os.path.join(figs_dir, 'total sensor power - observed.txt'), snsr_pwr_obs)
np.savetxt(os.path.join(figs_dir, 'total sensor power - predicted.txt'), snsr_pwr_pred)
# %% [markdown]
## panel c: onset times
# %%
import numpy as np
import lib.io.stan
import lib.plots.stan
import matplotlib.pyplot as plt
from matplotlib import colors,cm
from matplotlib.lines import Line2D
import os
import sklearn.linear_model as lnr_mdl
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/fit_syn_good_hyp'
sim_data = np.load('datasets/syn_data/id001_bt/syn_tvb_ez=48-79_pz=11-17-22-75.npz')
start_idx = 800
end_idx = 2200
sim_src_sig = sim_data['src_sig'][start_idx:end_idx,0,:,0] + sim_data['src_sig'][start_idx:end_idx,3,:,0]
ds_freq = 5
sim_src_sig_ds = sim_src_sig[0:-1:ds_freq,:]
n_roi = sim_src_sig.shape[1]
map_est = lib.io.stan.read_samples(['results/exp10/exp10.69/samples_syn_optim_run1.csv'])
map_src_sig = map_est['y'][0, :, :n_roi]
ez = sim_data['ez']
pz = sim_data['pz']
szng_roi = np.concatenate((ez, pz))
# %%
sim_onsets = sim_src_sig_ds.shape[0]*np.zeros(sim_src_sig_ds.shape[1])
for i, sig in enumerate(sim_src_sig_ds.T):
    sig_cond = sig > 0
    if sig_cond.any():
        sim_onsets[i] = np.min(np.nonzero(sig_cond))

map_onsets = np.zeros(map_src_sig.shape[1])
for i, sig in enumerate(map_src_sig.T):
    sig_cond = sig > 0
    if sig_cond.any():
        map_onsets[i] = np.min(np.nonzero(sig_cond))
# %%
lnr_rgrsn_mdl = lnr_mdl.LinearRegression()
lnr_rgrsn_mdl.fit(sim_onsets[szng_roi].reshape((-1,1)), map_onsets[szng_roi].reshape((-1,1)))
slope, intercept = lnr_rgrsn_mdl.coef_, lnr_rgrsn_mdl.intercept_
# print(slope, intercept)
# %%
plt.figure(figsize=(8,5))
for roi in ez:
    plt.scatter(sim_onsets[roi], map_onsets[roi], color='red')
for roi in pz:
    plt.scatter(sim_onsets[roi], map_onsets[roi], color='darkorange')
plt.xlabel('Onset Time - Ground Truth', fontsize=15);

plt.ylabel('Onset Time - Model Prediction', fontsize=15);
plt.yticks(fontsize=15)

t = np.r_[65:150:1]
plt.plot(t, intercept + (slope*t)[0], linestyle='solid', color='black');
plt.plot([60,150], [60,150], color='black', linestyle=(0, (5, 5)))
plt.xticks(ticks=np.r_[65:150:10], fontsize=15);
plt.yticks(ticks=np.r_[65:150:10], fontsize=15);
plt.xlim([60,150])
plt.ylim([60,150])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(os.path.join(figs_dir, 'onsets_scatter.svg'))
# %%
np.savetxt(os.path.join(figs_dir, 'onsets ground truth.txt'), sim_onsets[szng_roi])
np.savetxt(os.path.join(figs_dir, 'onsets predicted.txt'), map_onsets[szng_roi])
# %% [markdown]
# Figure 3: MAP robustness analysis
# %% [markdown]
### Panel a: SNR sweep
# %%
import matplotlib.pyplot as plt
import numpy as np
import lib.io.stan
import lib.plots.stan
import lib.utils.stan
import lib.preprocess.envelope
import csv
import os
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/robustness_analysis_syn_data'
os.makedirs(figs_dir, exist_ok=True)
syn_data = np.load('datasets/syn_data/id001_bt/syn_tvb_ez=48-79_pz=11-17-22-75.npz')
snr = np.arange(0.1, 2.6, 0.1)

prcsn = dict()
rcl = dict()
for el_snr in snr:
    prcsn[f'snr{el_snr:.1f}'] = []
    rcl[f'snr{el_snr:.1f}'] = []
    for i in range(1,11):
        p, r = lib.utils.stan.precision_recall_single(src_thrshld=0.0,
                                                    t_eps=10.0,
                                                    csv_path=[f'results/exp10/exp10.88.1/snr0.1_5.0_step0.1_10samples_per_step/samples/samples_snr{el_snr:0.1f}_sample{i}.csv'],
                                                    ez_hyp_roi=syn_data['ez'])
        prcsn[f'snr{el_snr:.1f}'].append(p)
        rcl[f'snr{el_snr:.1f}'].append(r)
# %%
plt.figure(figsize=(20, 5))
plt.subplot(121)
parts = plt.violinplot([prcsn[f'snr{el_snr:.1f}']
                        for el_snr in snr], positions=snr, widths=0.08, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('teal')
    pc.set_edgecolor('teal')
    pc.set_linewidth(3.0)
    pc.set_alpha(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.r_[0.1:2.6:0.2], fontsize=20)
plt.yticks(np.r_[0:1.1:0.2], fontsize=20)
plt.xlabel('SNR', fontsize=30)
plt.ylabel('Precision', fontsize=30)
plt.subplot(122)
parts = plt.violinplot([rcl[f'snr{el_snr:.1f}'] for el_snr in snr],
                       positions=snr, widths=0.08, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('purple')
    pc.set_edgecolor('purple')
    pc.set_linewidth(3.0)
    pc.set_alpha(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.r_[0.1:2.6:0.2], fontsize=20)
plt.yticks(np.r_[0:1.1:0.2], fontsize=20)
plt.xlabel('SNR', fontsize=30)
plt.ylabel('Recall', fontsize=30)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'snr_sweep.svg'))

# %%
f = open(os.path.join(figs_dir, 'precision across snr.csv'), 'w')
w = csv.writer(f)
for key, val in prcsn.items():
    w.writerow([key,val])
f.close()
f = open(os.path.join(figs_dir, 'recall across snr.csv'), 'w')
w = csv.writer(f)
for key, val in rcl.items():
    w.writerow([key,val])
f.close()

# %% [markdown]
### Panel b: initial condition sweep
# %%
import matplotlib.pyplot as plt
import numpy as np
import lib.io.stan
import lib.plots.stan
import lib.utils.stan
import lib.preprocess.envelope
import os
import csv
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/robustness_analysis_syn_data'
syn_data = np.load('datasets/syn_data/id001_bt/syn_tvb_ez=48-79_pz=11-17-22-75.npz')
prcsn = dict()
rcl = dict()
sigmaprior = np.arange(0.1, 1.1, 0.1)

for el_sigmaprior in sigmaprior:
    prcsn[f'sigmaprior{el_sigmaprior:.1f}'] = []
    rcl[f'sigmaprior{el_sigmaprior:.1f}'] = []
    for i in range(1,11):
        p, r = lib.utils.stan.precision_recall_single(src_thrshld=0.0,
                                                    t_eps=10,
                                                    csv_path=[f'results/exp10/exp10.88.2/samples/samples_sigmaprior{el_sigmaprior:.1f}_sample{i}.csv'],
                                                    ez_hyp_roi=syn_data['ez'])
        prcsn[f'sigmaprior{el_sigmaprior:.1f}'].append(p)
        rcl[f'sigmaprior{el_sigmaprior:.1f}'].append(r)
# %%
plt.figure(figsize=(20,5))
plt.subplot(121)
parts = plt.violinplot([prcsn[f'sigmaprior{el_sigmaprior:.1f}'] for el_sigmaprior in sigmaprior], positions=sigmaprior, widths=0.08, showextrema=False);
for pc in parts['bodies']:
    pc.set_facecolor('teal')
    pc.set_edgecolor('teal')
    pc.set_linewidth(3.0)
    pc.set_alpha(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.r_[0.1:1.1:0.1],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Standard deviation', fontsize=30)
plt.ylabel('Precision', fontsize=30)
plt.subplot(122)
parts = plt.violinplot([rcl[f'sigmaprior{el_sigmaprior:.1f}'] for el_sigmaprior in sigmaprior], positions=sigmaprior, widths=0.08, showextrema=False);
for pc in parts['bodies']:
    pc.set_facecolor('purple')
    pc.set_edgecolor('purple')
    pc.set_linewidth(3.0)
    pc.set_alpha(0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(np.r_[0.1:1.1:0.1],fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Standard deviation', fontsize=30)
plt.ylabel('Recall', fontsize=30)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'init_cond_sweep.svg'))
# %%
f = open(os.path.join(figs_dir, 'precision across std.csv'), 'w')
w = csv.writer(f)
for key, val in prcsn.items():
    w.writerow([key,val])
f.close()
f = open(os.path.join(figs_dir, 'recall across std.csv'), 'w')
w = csv.writer(f)
for key, val in rcl.items():
    w.writerow([key,val])
f.close()
# %% [markdown]
# Figure 4: Precision-Recall on retrospective patient cohort
# %% [markdown]
### panel a: precision-recall groupwise
# %%
import numpy as np
import glob
import os
import lib.plots.stan
import lib.io.stan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lib.utils.stan
import csv
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/precision_recall_retro_patient_cohort'
os.makedirs(figs_dir, exist_ok=True)
root_dir = 'results/exp10/exp10.86'
engel_scores = ['engel1or2', 'engel3or4']
plt.figure(figsize=(8,5))
ax = plt.subplot(111)
t_eps = 10
for i,es in enumerate(engel_scores):
    precision = np.load(os.path.join(root_dir, 'stats', f'prcsn_vs_teps_{es}.npy'))
    precision = precision[t_eps-1]
    recall = np.load(os.path.join(root_dir, 'stats', f'rcl_vs_teps_{es}.npy'))
    recall = recall[t_eps-1]
    ax.bar([3*i + 1, 3*i + 2], [precision, recall], color=['teal', 'purple'], width=0.5, alpha=0.5)

ax.set_xticks([np.mean([3*i + 1, 3*i + 2]) for i in range(len(engel_scores))])
ax.set_xticklabels(['I and II', 'III and IV'], fontsize=20)
ax.set_xlabel('Engel Score', fontsize=30)
ax.set_yticks(np.r_[0:0.9:0.2])
# ax.set_yticklabels(np.r_[0:0.9:0.2])
ax.tick_params(axis='y', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 0.8)
legend_elements = [Line2D([0], [0], color='teal', lw=5, alpha=0.5, label='Precision'),
                    Line2D([0], [0], color='purple', lw=5, alpha=0.5, label='Recall')]
ax.legend(handles=legend_elements, fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(figs_dir,'precision_recall_pergroup.svg'))
# %%
data = {}
precision = np.load(os.path.join(root_dir, 'stats',
                                 f'prcsn_vs_teps_engel1or2.npy'))
data['precision_engel1and2'] = precision[t_eps-1]
recall = np.load(os.path.join(root_dir, 'stats',
                              f'rcl_vs_teps_engel1or2.npy'))
data['recall_engel1and2'] = recall[t_eps-1]
precision = np.load(os.path.join(root_dir, 'stats',
                                 f'prcsn_vs_teps_engel3or4.npy'))
data['precision_engel3and4'] = precision[t_eps-1]
recall = np.load(os.path.join(root_dir, 'stats',
                              f'rcl_vs_teps_engel3or4.npy'))
data['recall_engel3and4'] = recall[t_eps-1]
f = open(os.path.join(figs_dir, 'precision-recall per group.csv'), 'w')
w = csv.writer(f)
for key, val in data.items():
    w.writerow([key,val])
f.close()

# %% [markdown]
### Panel b: Precision-Recall per patient
# %%
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id017_mk', 'id020_lma', 'id022_te', 'id025_mc', 'id030_bf', 'id039_mra', 'id050_sx']
patient_ids['engel2'] = ['id014_vc', 'id021_jc', 'id027_sj', 'id040_ms']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id011_gr', 'id033_fc', 'id036_dm', 'id045_bc']

patient_ids['engel1or2'] = patient_ids['engel1'] + patient_ids['engel2']
patient_ids['engel3or4'] = patient_ids['engel3'] + patient_ids['engel4']


def find_xpos(delta=0.05, data_arr=[], bin_center=0):
    # delta : distance between points
    # d     : 1D data array
    # c     : bin center
    hist, bin_edges = np.histogram(data_arr)
    bin_idcs = np.digitize(data_arr, bin_edges)
    x_pos = np.zeros_like(data_arr)
    for bin_idx in np.unique(bin_idcs):
        flag = True
        x_pos_idcs = np.where(bin_idcs == bin_idx)[0]
        i = 0
        j = 1
        for idx in x_pos_idcs:
            if flag:
                x_pos[idx] = bin_center + i*delta
                i += 1
                flag = False
            else:
                x_pos[idx] = bin_center - j*delta
                j += 1
                flag = True
    return x_pos

engel_scores = ['engel1or2', 'engel3or4']
engel_labels = {'engel1or2': 'Engel I/II', 'engel3or4': 'Engel III/IV'}
precision = dict()
recall = dict()
cdf_prcsn = dict()
cdf_recall = dict()
fig = plt.figure(figsize=(8, 5))
ax_bxplt = fig.add_subplot(111)
# ax_cdf = fig.add_subplot(122)

for i, score in enumerate(engel_scores):
    src_thrshld = 0
    t_eps = 10
    precision[score] = []
    recall[score] = []
    # print(score)
    for subj_id in patient_ids[score]:
        p, r = lib.utils.stan.precision_recall(patient_ids=[subj_id], 
                                            root_dir=root_dir, 
                                            src_thrshld=src_thrshld, t_eps=t_eps, 
                                            parcellation='destrieux', 
                                            outfile_regex='samples_*.csv', 
                                            npoints=300)
        precision[score].append(p)
        recall[score].append(r)
box_prcsn = ax_bxplt.boxplot([precision[score] for score in engel_scores],
                            widths=0.8,
                            positions=[3*i+1 for i in range(len(engel_scores))], whis=1.5,
                            boxprops={'linewidth': 1, 'color': 'teal', 'alpha': 0.5},
                            patch_artist=True,
                            medianprops={'linewidth': 2, 'color':'red', 'zorder':10},
                            whiskerprops={'color': 'teal', 'alpha': 0.5},
                            capprops={'color': 'teal', 'alpha': 0.5},
                            showmeans=True,
                            meanprops={'marker':'*', 
                                    'markerfacecolor':'red',
                                    'markeredgecolor':'red'},
                        showfliers=False)
for patch_artist in box_prcsn['boxes']:
    patch_artist.set_facecolor('teal')
    patch_artist.set_alpha(0.5)

box_recall = ax_bxplt.boxplot([recall[score] for score in engel_scores],
                            widths=0.8,
                            positions=[3*i+2 for i in range(len(engel_scores))],
                            whis=1.5,
                            boxprops={'linewidth': 1, 
                                    'color': 'purple', 
                                    'alpha': 0.5},
                            patch_artist=True,
                            medianprops={'linewidth': 2, 'color':'red', 'zorder':10},
                            whiskerprops={'color': 'purple', 'alpha': 0.5},
                            capprops={'color': 'purple', 'alpha': 0.5},
                            showmeans=True,
                            meanprops={'marker':'*',
                                        'markerfacecolor':'red',
                                        'markeredgecolor':'red'},
                            showfliers=False)
for patch_artist in box_recall['boxes']:
    patch_artist.set_facecolor('purple')
    patch_artist.set_alpha(0.5)

ax_bxplt.set_xticks([np.mean([3*i + 1, 3*i + 2])
                        for i in range(len(engel_scores))])
ax_bxplt.set_xticklabels(['I and II', 'III and IV'], fontsize=20)
ax_bxplt.set_xlabel('Engel Score', fontsize=30)
ax_bxplt.tick_params(axis='y', labelsize=20)
ax_bxplt.tick_params(axis='x', labelsize=20)
ax_bxplt.spines['top'].set_visible(False)
ax_bxplt.spines['right'].set_visible(False)
# legend_elements = [Line2D([0], [0], color='black', alpha=0.5, lw=5, label='Precision'),
#                     Line2D([0], [0], color='gray', alpha=0.5, lw=5, label='Recall')]
# ax_bxplt.legend(handles=legend_elements, loc='best', fontsize=15)

# Overlay of the data points
for i, scr in enumerate(engel_scores):
    x_pos = find_xpos(delta=0.07, data_arr=precision[scr], bin_center=(3*i + 1))
    ax_bxplt.scatter(x_pos, precision[scr],
                     s=18, marker='o', color='black', zorder=3)
    x_pos = find_xpos(delta=0.07, data_arr=recall[scr], bin_center=(3*i + 2))
    ax_bxplt.scatter(x_pos, recall[scr],
                     s=18, marker='o', color='black', zorder=3)

# ax_cdf.legend()
plt.tight_layout()
plt.savefig(os.path.join(figs_dir,'precision_recall_perpatient.svg'))
# %%
f = open(os.path.join(figs_dir, 'precision per patient.csv'), 'w')
w = csv.writer(f)
for key, val in precision.items():
    w.writerow([key,val])
f.close()
f = open(os.path.join(figs_dir, 'recall per patient.csv'), 'w')
w = csv.writer(f)
for key, val in recall.items():
    w.writerow([key,val])
f.close()
# %% [markdown]
# Figure 7: Precision-Recall vs onset tolerance
# %%
import numpy as np
import glob
import os
import lib.plots.stan
import lib.io.stan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lib.utils.stan
import csv
# %%
figs_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/figures/final/precision_recall_vs_onset_tolerance'
os.makedirs(figs_dir, exist_ok=True)
root_dir = 'results/exp10/exp10.86'
engel_scores = ['engel1or2', 'engel3or4']
engel_scores_rmn = ['I and II', 'III and IV']
fig = plt.figure(figsize=(15, 5))
ax_prec = fig.add_subplot(121)
ax_rec = fig.add_subplot(122)
t_eps_min = 5
t_eps_max = 60
t_eps_step = 1
x_tick_start = 5
x_tick_end = 61
x_tick_step = 5


for i, es in enumerate(engel_scores):
    t_eps_range = np.arange(t_eps_min, t_eps_max + 1, t_eps_step, dtype=int)
    precision = np.load(os.path.join(
        root_dir, 'stats', f'prcsn_vs_teps_{es}.npy'))
    precision = precision[t_eps_min-1:t_eps_max]
    recall = np.load(os.path.join(root_dir, 'stats', f'rcl_vs_teps_{es}.npy'))
    recall = recall[t_eps_min-1:t_eps_max]
    ax_prec.plot(t_eps_range, precision, label=f"Engel score {engel_scores_rmn[i]}",
                 linewidth=5)
    ax_rec.plot(t_eps_range, recall, label=f"Engel score {engel_scores_rmn[i]}",
                linewidth=5)
ax_prec.axvline(10, color='red')
ax_prec.set_xlabel(r'Onset Tolerance ($t_{\epsilon}$)', fontsize=30)
ax_prec.set_ylabel('Precision', fontsize=30)
ax_prec.legend(frameon=False, loc='upper right', fontsize=15)
ax_prec.set_xticks(np.r_[x_tick_start:x_tick_end:x_tick_step])
ax_prec.set_xticklabels(
    map(str, np.r_[x_tick_start:x_tick_end:x_tick_step]), fontsize=20)
ax_prec.set_yticks(np.r_[0.1:0.9:0.1])
ax_prec.set_yticklabels(
    map(lambda x: round(x, 1), np.r_[0.1:0.9:0.1]), fontsize=20)
ax_prec.spines['top'].set_visible(False)
ax_prec.spines['right'].set_visible(False)
ax_prec.grid(True, alpha=0.5)
ax_prec.set_xlim(0, t_eps_max)
ax_prec.set_ylim(0.1, 0.9)
ax_rec.axvline(10, color='red')
ax_rec.set_xlabel(r'Onset Tolerance ($t_{\epsilon}$)', fontsize=30)
ax_rec.set_ylabel('Recall', fontsize=30)
ax_rec.set_xticks(np.r_[x_tick_start:x_tick_end:x_tick_step])
ax_rec.set_xticklabels(
    map(str, np.r_[x_tick_start:x_tick_end:x_tick_step]), fontsize=20)
ax_rec.set_yticks(np.r_[0.1:0.9:0.1])
ax_rec.set_yticklabels(
    map(lambda x: round(x, 1), np.r_[0.1:0.9:0.1]), fontsize=20)
ax_rec.spines['top'].set_visible(False)
ax_rec.spines['right'].set_visible(False)
ax_rec.grid(True, alpha=0.5)
ax_rec.set_xlim(0, t_eps_max)
ax_rec.set_ylim(0.1, 0.9)
plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'precision_recall_vs_onset_tolerance.svg'))
# %%
data = {}
precision = np.load(os.path.join(root_dir, 'stats', f'prcsn_vs_teps_engel1or2.npy'))
data['precision_engel1and2'] = precision[t_eps_min-1:t_eps_max]
recall = np.load(os.path.join(root_dir, 'stats', f'rcl_vs_teps_engel1or2.npy'))
data['recall_engel1and2'] = recall[t_eps_min-1:t_eps_max]
precision = np.load(os.path.join(root_dir, 'stats', f'prcsn_vs_teps_engel3or4.npy'))
data['precision_engel3and4'] = precision[t_eps_min-1:t_eps_max]
recall = np.load(os.path.join(root_dir, 'stats', f'rcl_vs_teps_engel3or4.npy'))
data['recall_engel3and4'] = recall[t_eps_min-1:t_eps_max]
f = open(os.path.join(figs_dir, 'precision recall across onset tolerance.csv'), 'w')
w = csv.writer(f)
for key, val in data.items():
    w.writerow([key,val])
f.close()
# %% [markdown]
# Save the synthetic dataset used in this study
# %%
import numpy as np
# %%
syn_data = np.load('datasets/syn_data/id001_bt/syn_tvb_ez=48-79_pz=11-17-22-75.npz')
network = np.load('datasets/syn_data/id001_bt/network.npz')
# %%
start_idx = 800
end_idx = 2200
sim_src_sig = syn_data['src_sig'][start_idx:end_idx, 0,
                                  :, 0] + syn_data['src_sig'][start_idx:end_idx, 3, :, 0]
# %%
data_dir = '/home/anirudh/Nextcloud/Academia/Papers/infr_szr_prpgtn_paper/comm_bio/Final revision/syn_data'
os.makedirs(data_dir, exist_ok=True)
np.savetxt(os.path.join(data_dir, 'source actvity.txt'), sim_src_sig)
np.savetxt(os.path.join(data_dir, 'seeg.txt'), syn_data['seeg'])
np.savetxt(os.path.join(data_dir, 'connectome.txt'), network['SC'])
np.savetxt(os.path.join(data_dir, 'gain matrix.txt'), network['gain_mat'])

# %%
