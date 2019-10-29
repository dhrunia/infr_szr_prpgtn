import numpy as np
# import lib.utils.stan
import glob
import os
import lib.plots.stan
import lib.io.stan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# def check_completed(patient_ids, nchains, fname_suffix, root_dir):
#     with open(os.path.join(root_dir, 'chains_report.csv'), 'w') as fd:
#         for patient_id in patient_ids:
#             # szr_names = [
#             #     os.path.splitext(os.path.basename(fname))[0] for fname in glob.glob(
#             #         os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R'))
#             # ]
#             # Read EZ hypothesis or skip patient if hypothesis doesn't exist
#             try:
#                 ez_hyp = np.where(
#                     np.loadtxt(
#                         f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt'
#                     ) == 1)[0]
#             except Exception as err:
#                 print(err)
#                 continue
#             print(patient_id)
#             fd.write(patient_id)
#             for szr_path in glob.glob(
#                     os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
#                 szr_name = os.path.splitext(
#                     os.path.basename(szr_path))[0].split('obs_data_')[-1]
#                 csv_paths = []
#                 for i in range(nchains):
#                     log_path = os.path.join(root_dir, patient_id, 'logs',
#                                             f'{szr_name}_{fname_suffix}{i+1}.log')
#                     csv_path = os.path.join(
#                         root_dir, patient_id, 'results',
#                         f'samples_{szr_name}_{fname_suffix}{i+1}.csv')
#                     if (lib.utils.stan.is_completed(log_path)):
#                         csv_paths.append(csv_path)
#                 fd.write(f',{szr_name},{len(csv_paths)}')
#             fd.write('\n')


# def make_plots():
#     for patient_id in patient_ids:
#         # szr_names = [
#         #     os.path.splitext(os.path.basename(fname))[0] for fname in glob.glob(
#         #         os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R'))
#         # ]
#         # Read EZ hypothesis or skip patient if hypothesis doesn't exist
#         try:
#             ez_hyp = np.where(
#                 np.loadtxt(
#                     f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt'
#                 ) == 1)[0]
#         except Exception as err:
#             print(err)
#             continue
#         print(patient_id)
#         for szr_path in glob.glob(
#                 os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
#             szr_name = os.path.splitext(
#                 os.path.basename(szr_path))[0].split('obs_data_')[-1]
#             print(szr_name)
#             csv_paths = []
#             for i in range(nchains):
#                 log_path = os.path.join(root_dir, patient_id, 'logs',
#                                         f'{szr_name}_{fname_suffix}{i+1}.log')
#                 csv_path = os.path.join(
#                     root_dir, patient_id, 'results',
#                     f'samples_{szr_name}_{fname_suffix}{i+1}.csv')
#                 if (lib.utils.stan.is_completed(log_path)):
#                     csv_paths.append(csv_path)
#             if (len(csv_paths) == 0):
#                 continue
#             fit_data = lib.io.stan.rload(szr_path)
#             samples = lib.io.stan.read_samples(
#                 csv_paths, nwarmup=0, nsampling=200)
#             lib.plots.stan.x0_violin_patient(
#                 samples['x0'],
#                 ez_hyp,
#                 figsize=(25, 5),
#                 figname=os.path.join(root_dir, patient_id, 'figures',
#                                      f'x0_violin_{szr_name}.png'),
#                 plt_close=True)
#             lib.plots.stan.nuts_diagnostics(
#                 samples,
#                 figsize=(15, 10),
#                 figname=os.path.join(root_dir, patient_id, 'figures',
#                                      f'nuts_diagnostics_{szr_name}.png'),
#                 plt_close=True)
#             lib.plots.stan.pair_plots(
#                 samples, ['K', 'tau0', 'amplitude', 'offset', 'alpha'],
#                 figname=os.path.join(root_dir, patient_id, 'figures',
#                                      f'params_pair_plots_{szr_name}.png'),
#                 sampler='HMC',
#                 plt_close=True)
#             lib.plots.stan.plot_source(
#                 samples['x'].mean(axis=0),
#                 samples['z'].mean(axis=0),
#                 ez_hyp, [],
#                 figname=os.path.join(
#                     root_dir, patient_id, 'figures',
#                     f'posterior_predicted_src_{szr_name}.png'),
#                 plt_close=True)
#             lib.plots.stan.plot_fit_target(
#                 {
#                     'slp': samples['mu_slp'].mean(axis=0),
#                     'snsr_pwr': samples['mu_snsr_pwr'].mean(axis=0)
#                 },
#                 fit_data,
#                 figname=os.path.join(
#                     root_dir, patient_id, 'figures',
#                     f'posterior_predicted_slp_{szr_name}.png'),
#                 plt_close=True)


# def extract_x0(patient_ids, nchains, fname_suffix, root_dir):
#     '''
#     Extracts x0 values from csv files and saves them in a numpy file
#     '''
#     for patient_id in patient_ids:
#         # Read EZ hypothesis or skip patient if hypothesis doesn't exist
#         try:
#             ez_hyp = np.where(
#                 np.loadtxt(
#                     f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt'
#                 ) == 1)[0]
#         except Exception as err:
#             print(err)
#             continue
#         print(patient_id)
#         first_szr = True
#         for szr_path in glob.glob(
#                 os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
#             # szr_name = os.path.splitext(
#             #     os.path.basename(szr_path))[0].split('obs_data_')[-1]
#             szr_name = os.path.splitext(
#                 os.path.basename(szr_path))[0]
#             print(szr_name)
#             csv_paths = []
#             for i in range(nchains):
#                 log_path = os.path.join(root_dir, patient_id, 'logs',
#                                         f'{szr_name}_{fname_suffix}{i+1}.log')
#                 csv_path = os.path.join(
#                     root_dir, patient_id, 'results',
#                     f'samples_{szr_name}_{fname_suffix}{i+1}.csv')
#                 if (lib.utils.stan.is_completed(log_path)):
#                     csv_paths.append(csv_path)
#             if (len(csv_paths) == 0):
#                 continue
#             fit_data = lib.io.stan.rload(szr_path)
#             samples = lib.io.stan.read_samples(
#                 csv_paths, nwarmup=0, nsampling=200, variables_of_interest=['x0'])
#             np.save(os.path.join(root_dir, patient_id, 'results', f'x0_{szr_name}.npy'), samples['x0'])
#             if(first_szr):
#                 x0_cumul = samples['x0']
#                 first_szr = False
#             else:
#                 x0_cumul = np.append(x0_cumul, samples['x0'], axis=0)
#         np.save(os.path.join(root_dir, patient_id, 'results', 'x0_cumul.npy'), x0_cumul)


def ez_pred(patient_ids, nchains, fname_suffix, root_dir, x0_threshold):
    for patient_id in patient_ids:
        # Read EZ hypothesis or skip patient if hypothesis doesn't exist
        try:
            ez_hyp = np.where(
                np.loadtxt(
                    f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt'
                ) == 1)[0]
        except Exception as err:
            print(err)
            continue
        first_szr = True
        for szr_path in glob.glob(
                os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
            # szr_name = os.path.splitext(
            #     os.path.basename(szr_path))[0].split('obs_data_')[-1]
            szr_name = os.path.splitext(
                os.path.basename(szr_path))[0]
            csv_paths = []
            for i in range(nchains):
                log_path = os.path.join(root_dir, patient_id, 'logs',
                                        f'{szr_name}_{fname_suffix}{i+1}.log')
                csv_path = os.path.join(
                    root_dir, patient_id, 'results',
                    f'samples_{szr_name}_{fname_suffix}{i+1}.csv')
                if (lib.utils.stan.is_completed(log_path)):
                    csv_paths.append(csv_path)
            if (len(csv_paths) == 0):
                continue
            # samples = lib.io.stan.read_samples(
            #     csv_paths, nwarmup=0, nsampling=200, variables_of_interest=['x0'])
            samples = np.load(os.path.join(root_dir, patient_id, 'results', f'x0_{szr_name}.npy'))
            if(first_szr):
                ez_pred = samples.mean(axis=0) > x0_threshold
                first_szr = False
            else:
                ez_pred = np.logical_or(ez_pred, samples.mean(axis=0) > x0_threshold)
        np.save(os.path.join(root_dir, patient_id, 'ez_pred.npy'), ez_pred)


def find_ez(onst_thrshld, bin_thrshld, nbins, patient_ids, root_dir):
    for patient_id in patient_ids:
        csv_path = glob.glob(os.path.join(root_dir, patient_id, '*chain1.csv'))
        optima = lib.io.stan.read_samples(csv_path)
        x = optima['x'][0]
        nn = x.shape[1]
        onsets = 200*np.ones(nn)
        for i in range(nn):
            xt = x[:,i] > onst_thrshld
            if(xt.any()):
                onsets[i] = np.where(x[:,i] > onst_thrshld)[0][0]
        a, b = np.histogram(onsets[onsets<150], bins=nbins)
        ez_pred = np.zeros(nn)
        ez_pred[np.where(onsets<b[bin_thrshld])] = 1
        np.save(os.path.join(root_dir, patient_id, 'ez_pred.npy'), ez_pred)
   

def find_ez_single(onst_thrshld, bin_thrshld, nbins, csv_path, szr_len):
    optima = lib.io.stan.read_samples([csv_path])
    x = optima['x'][0]
    nn = x.shape[1]
    onsets = (szr_len + 50)*np.ones(nn)
    for i in range(nn):
        xt = x[:,i] > onst_thrshld
        if(xt.any()):
            onsets[i] = np.where(x[:, i] > onst_thrshld)[0][0]
    a, b = np.histogram(onsets[onsets<szr_len], bins=nbins)
    ez = np.nonzero(onsets < b[bin_thrshld])[0]
    pz = np.nonzero(np.logical_and(onsets > b[bin_thrshld], onsets < szr_len))[0]
    return ez, pz


def precision_recall(patient_ids, root_dir):
    tp = fp = fn = 0
    for patient_id in patient_ids:
        # Read EZ hypothesis or skip patient if hypothesis doesn't exist
        try:
            ez_hyp = np.loadtxt(f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt')
        except Exception as err:
            print(err)
            continue
        ez_pred = np.load(os.path.join(root_dir, patient_id, 'ez_pred.npy')).astype(int)
        for a, b in zip(ez_hyp, ez_pred):
            if(a == 1 and b == 1):
                tp += 1
            elif(a == 1 and b == 0):
                fn += 1
            elif(a == 0 and b == 1):
                fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def tpr_and_fpr(patient_ids, root_dir):
    tp = fp = fn = tn = 0
    for patient_id in patient_ids:
        # Read EZ hypothesis or skip patient if hypothesis doesn't exist
        try:
            ez_hyp = np.loadtxt(f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt')
        except Exception as err:
            print(err)
            continue
        ez_pred = np.load(os.path.join(root_dir, patient_id, 'ez_pred.npy')).astype(int)
        for a, b in zip(ez_hyp, ez_pred):
            if(a == 1 and b == 1):
                tp += 1
            elif(a == 1 and b == 0):
                fn += 1
            elif(a == 0 and b == 1):
                fp += 1
            elif(a ==0 and b == 0):
                tn += 1
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return tpr, fpr


if (__name__ == '__main__'):
    root_dir = '/home/anirudh/Nextcloud/Academia/Projects/VEP/results/exp10.67'
    patient_ids = dict()
    patient_ids['engel1'] = ['id001_bt','id003_mg','id004_bj','id010_cmn','id013_lk','id014_vc','id017_mk','id020_lma','id022_te','id025_mc','id027_sj','id030_bf','id039_mra','id050_sx']
    patient_ids['engel2'] = ['id021_jc','id040_ms']
    patient_ids['engel3'] = ['id007_rd','id008_dmc','id023_br','id028_ca', 'id037_cg']
    patient_ids['engel4'] = ['id033_fc','id036_dm']
    patient_ids['engel2or3or4'] = patient_ids['engel2'] + patient_ids['engel3'] + patient_ids['engel4']

    # # Precision-Recall curves for bin thresholding
    # precision = []
    # recall = []
    # onst_thrshld = -0.05
    # nbins = 20
    # bin_thrshld =  np.arange(1, nbins, dtype=int)
    # for bin_thrshld_ in bin_thrshld:
    #     find_ez(onst_thrshld, bin_thrshld_, nbins, patient_ids['engel1'], root_dir)
    #     p, r = precision_recall(patient_ids['engel1'], root_dir)
    #     precision.append(p)
    #     recall.append(r)
    # plt.figure()
    # plt.plot(recall, precision, color='black')
    # plt.xlabel('Recall', fontsize=13)
    # plt.ylabel('Precision', fontsize=13)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Engel score I', fontsize=15)
    # plt.show(block=False)

    # precision = []
    # recall = []
    # onst_thrshld = -0.05
    # nbins = 20
    # bin_thrshld = np.arange(1, nbins, dtype=int)
    # for bin_thrshld_ in bin_thrshld:
    #     find_ez(onst_thrshld, bin_thrshld_, nbins, patient_ids['engel2or3or4'], root_dir)
    #     p, r = precision_recall(patient_ids['engel2or3or4'], root_dir)
    #     precision.append(p)
    #     recall.append(r)
    # plt.figure()
    # plt.plot(recall, precision, color='black')
    # plt.xlabel('Recall', fontsize=13)
    # plt.ylabel('Precision', fontsize=13)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Engel score II, III and IV', fontsize=15)
    # plt.show(block=False)

    # ## Precision-Recall curves for onset thresholding
    # precision = []
    # recall = []
    # nbins = 10
    # onst_thrshld = np.linspace(0, -2.0, 10)
    # bin_thrshld = 1
    # for onst_thrshld_ in onst_thrshld:
    #     find_ez(onst_thrshld_, bin_thrshld, nbins, patient_ids['engel1'], root_dir)
    #     p, r = precision_recall(patient_ids['engel1'], root_dir)
    #     precision.append(p)
    #     recall.append(r)
    #     print(onst_thrshld_, p, r)
    # plt.figure()
    # plt.plot(recall, precision)
    # plt.xlabel('Recall', fontsize=13)
    # plt.ylabel('Precision', fontsize=13)
    # plt.title('Engel score I', fontsize=15)
    # plt.show(block=False)

    # precision = []
    # recall = []
    # nbins = 10
    # onst_thrshld =  np.linspace(0, -0.5, 20)
    # bin_thrshld = 1
    # for onst_thrshld_ in onst_thrshld:
    #     find_ez(onst_thrshld_, bin_thrshld, nbins, patient_ids['engel2or3or4'], root_dir)
    #     p, r = precision_recall(patient_ids['engel2or3or4'], root_dir)
    #     precision.append(p)
    #     recall.append(r)
    # plt.figure()
    # plt.plot(recall, precision)
    # plt.xlabel('Recall', fontsize=13)
    # plt.ylabel('Precision', fontsize=13)
    # plt.title('Engel score II, III and IV', fontsize=15)
    # plt.show(block=False)

    ## Bar plot
    precision = []
    recall = []
    for i in range(4):
        onst_thrshld = -0.05
        bin_thrshld = 1
        find_ez(onst_thrshld, bin_thrshld, 10, patient_ids['engel' + str(i + 1)], root_dir)
        p, r = precision_recall(patient_ids['engel' + str(i + 1)], root_dir)
        precision.append(p)
        recall.append(r)

        ax = plt.subplot(111)
        ax.bar([5*i + 1, 5*i + 2], [precision[i], recall[i]], color=['black', 'grey'])

    # precision = []
    # recall = []
    # onst_thrshlds = [-0.05]
    # for threshold in onst_thrshlds:
    #     find_ez(threshold, patient_ids['engel2or3or4'], root_dir)
    #     p, r = precision_recall(patient_ids['engel2or3or4'], root_dir)
    #     precision.append(p)
    #     recall.append(r)

    # ax.bar([5,6], [precision[0],recall[0]], color=['black', 'grey'])
    ax.set_xticks([np.mean([5*i + 1, 5*i + 2]) for i in range(4)])
    ax.set_xticklabels(['Engel score I', 'Engel score II', 'Engel score III', 'Engel score IV'], fontsize=15)
    ax.tick_params(axis='y', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_elements = [Line2D([0], [0], color='black', lw=5, label='Precision'),
                       Line2D([0], [0], color='grey', lw=5, label='Recall')]
    ax.legend(handles=legend_elements)
    plt.show()


    # tpr = []
    # fpr = []
    # onst_thrshld = -0.05
    # nbins = 50
    # bin_thrshld =  np.arange(1, nbins)
    # for bin_thrshld_ in bin_thrshld:
    #     find_ez(onst_thrshld, bin_thrshld_, nbins, patient_ids['engel1'], root_dir)
    #     tpr_, fpr_ = tpr_and_fpr(patient_ids['engel1'], root_dir)
    #     tpr.append(tpr_)
    #     fpr.append(fpr_)
    # plt.figure()
    # plt.plot(fpr, tpr)
    # plt.xlabel('False Positive Rate', fontsize=15)
    # plt.ylabel('True Positive Rate', fontsize=15)
    # plt.show()
