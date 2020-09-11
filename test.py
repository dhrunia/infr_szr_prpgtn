#%% Precision Recall bar plot

import numpy as np
import glob
import os
import lib.plots.stan
import lib.io.stan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lib.utils.stan

# figs_dir = '/home/anirudh/Academia/papers/infr_szr_prpgtn_paper/figures/Retro/'
root_dir = '/home/anirudh/Academia/projects/isp_paper_figures/results/exp10/exp10.85'
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id014_vc', 'id017_mk', 'id020_lma', 'id022_te', 'id025_mc']
patient_ids['engel2'] = ['id021_jc', 'id027_sj']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca']
patient_ids['engel4'] = ['id011_gr']

patient_ids['engel1or2'] = patient_ids['engel1'] + patient_ids['engel2']
patient_ids['engel3or4'] = patient_ids['engel3'] + patient_ids['engel4']
patient_ids['engel2or3or4'] = patient_ids['engel2'] + patient_ids['engel3'] + \
                              patient_ids['engel4']

precision = []
recall = []
engel_scores = ['engel1', 'engel2', 'engel3or4']
plt.figure(figsize=(8,5))
ax = plt.subplot(111)
for i,scr in enumerate(engel_scores):
    src_thrshld = 0
    onst_wndw_sz = 10
    p, r = lib.utils.stan.precision_recall(patient_ids[scr], root_dir,
                                           src_thrshld, onst_wndw_sz,
                                           parcellation='vep', outfile_regex="*run1.csv")
    precision.append(p)
    recall.append(r)
    ax.bar([3*i + 1, 3*i + 2], [precision[i], recall[i]], color=['teal', 'purple'], width=0.5, alpha=0.5)
    # print(scr, '\t', 'precision: ',p, '\t', 'recall: ', r)

# ax.bar([5,6], [precision[0],recall[0]], color=['black', 'grey'])
ax.set_xticks([np.mean([3*i + 1, 3*i + 2]) for i in range(len(engel_scores))])
ax.set_xticklabels(['I', 'II', 'III and IV'], fontsize=15)
ax.set_xlabel('Engel Score', fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [Line2D([0], [0], color='teal', lw=5, alpha=0.5, label='Precision'),
                    Line2D([0], [0], color='purple', lw=5, alpha=0.5, label='Recall')]
ax.legend(handles=legend_elements, fontsize=15)
plt.tight_layout()
plt.savefig('/home/anirudh/Documents/precision_recall.png', dpi=512)
# %% Precision Recall per patient
import numpy as np
import glob
import os
import lib.plots.stan
import lib.io.stan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lib.utils.stan

root_dir = '/home/anirudh/Academia/projects/isp_paper_figures/results/exp10/exp10.85'
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id014_vc', 'id017_mk', 'id020_lma', 'id022_te', 'id025_mc']
patient_ids['engel2'] = ['id021_jc', 'id027_sj']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca']
patient_ids['engel4'] = ['id011_gr']

patient_ids['engel1or2'] = patient_ids['engel1'] + patient_ids['engel2']
patient_ids['engel3or4'] = patient_ids['engel3'] + patient_ids['engel4']
patient_ids['engel2or3or4'] = patient_ids['engel2'] + patient_ids['engel3'] + \
                              patient_ids['engel4']

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

engel_scores = ['engel1', 'engel2', 'engel3or4']
engel_labels = {'engel1': 'Engel I', 'engel2': 'Engel II', 'engel3or4': 'Engel III/IV'}
precision = dict()
recall = dict()
cdf_prcsn = dict()
cdf_recall = dict()
fig = plt.figure(figsize=(8, 5))
ax_bxplt = fig.add_subplot(111)
# ax_cdf = fig.add_subplot(122)

for i, score in enumerate(engel_scores):
    src_thrshld = 0
    onst_wndw_sz = 10
    precision[score] = []
    recall[score] = []
    # print(score)
    for subj_id in patient_ids[score]:
        # print(f'\t {subj_id}')
        p, r = lib.utils.stan.precision_recall(
            [subj_id], root_dir, src_thrshld, onst_wndw_sz, parcellation='vep', outfile_regex='*run1.csv')
        precision[score].append(p)
        recall[score].append(r)
        print(f'\t precision = {p} \t recall = {r}')
    # cdf_prcsn[score] = []
    # cdf_recall[score] = []
    # prcsn_arr = np.array(precision[score])
    # recall_arr = np.array(recall[score])
    # for p in np.arange(0, 1.1, 0.1):
    #     cdf_prcsn[score].append(
    #         np.count_nonzero(prcsn_arr <= p)/prcsn_arr.size)
    #     cdf_recall[score].append(np.count_nonzero(
    #         recall_arr <= p)/recall_arr.size)
    # ax_cdf.plot(np.arange(0, 1.1, 0.1),
    #             cdf_prcsn[score], label=engel_labels[score])
box_prcsn = ax_bxplt.boxplot([precision[score] for score in engel_scores],
                             widths=0.8,
                             positions=[3*i+1 for i in range(3)], whis=1.5,
                             boxprops={'linewidth': 1, 'color': 'teal', 'alpha': 0.5},
                             patch_artist=True,
                             medianprops={'linewidth': 2, 'color':'red', 'zorder':10},
                             whiskerprops={'color': 'teal', 'alpha': 0.5},
                             capprops={'color': 'teal', 'alpha': 0.5},
                             showmeans=False,
                             meanprops={'marker':'*', 
                                        'markerfacecolor':'navy',
                                        'markeredgecolor':'navy'})
for patch_artist in box_prcsn['boxes']:
    patch_artist.set_facecolor('teal')
    patch_artist.set_alpha(0.5)
box_recall = ax_bxplt.boxplot([recall[score] for score in engel_scores],
                              widths=0.8,
                              positions=[3*i+2 for i in range(3)], whis=1.5,
                              boxprops={'linewidth': 1, 'color': 'purple', 'alpha': 0.5},
                              patch_artist=True,
                              medianprops={'linewidth': 2, 'color':'red', 'zorder':10},
                              whiskerprops={'color': 'purple', 'alpha': 0.5},
                              capprops={'color': 'purple', 'alpha': 0.5},
                              showmeans=False,
                              meanprops={'marker':'*',
                                         'markerfacecolor':'navy',
                                         'markeredgecolor':'navy'})
for patch_artist in box_recall['boxes']:
    patch_artist.set_facecolor('purple')
    patch_artist.set_alpha(0.5)

ax_bxplt.set_xticks([np.mean([3*i + 1, 3*i + 2])
                        for i in range(len(engel_scores))])
ax_bxplt.set_xticklabels(['I', 'II', 'III and IV'], fontsize=15)
ax_bxplt.set_xlabel('Engel Score', fontsize=15)
ax_bxplt.tick_params(axis='y', labelsize=12)
ax_bxplt.tick_params(axis='x', labelsize=12)
ax_bxplt.spines['top'].set_visible(False)
ax_bxplt.spines['right'].set_visible(False)
# legend_elements = [Line2D([0], [0], color='black', alpha=0.5, lw=5, label='Precision'),
#                     Line2D([0], [0], color='gray', alpha=0.5, lw=5, label='Recall')]
# ax_bxplt.legend(handles=legend_elements, loc='best', fontsize=15)

# Overlay of the data points
for i, scr in enumerate(engel_scores):
    x_pos = find_xpos(delta=0.09, data_arr=precision[scr], bin_center=(3*i + 1))
    ax_bxplt.scatter(x_pos, precision[scr],
                     s=18, marker='o', color='black', zorder=3)
    x_pos = find_xpos(delta=0.09, data_arr=recall[scr], bin_center=(3*i + 2))
    ax_bxplt.scatter(x_pos, recall[scr],
                     s=18, marker='o', color='black', zorder=3)

# ax_cdf.legend()
plt.tight_layout()
plt.savefig('/home/anirudh/Documents/precision_recall_per_patient.png', dpi=512)

