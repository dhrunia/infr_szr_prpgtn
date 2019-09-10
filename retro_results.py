import numpy as np
import lib.utils.stan
import glob
import os
import lib.plots.stan
import lib.io.stan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def check_completed(patient_ids, nchains, fname_suffix, root_dir):
    with open(os.path.join(root_dir, 'chains_report.csv'), 'w') as fd:
        for patient_id in patient_ids:
            # szr_names = [
            #     os.path.splitext(os.path.basename(fname))[0] for fname in glob.glob(
            #         os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R'))
            # ]
            # Read EZ hypothesis or skip patient if hypothesis doesn't exist
            try:
                ez_hyp = np.where(
                    np.loadtxt(
                        f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt'
                    ) == 1)[0]
            except Exception as err:
                print(err)
                continue
            print(patient_id)
            fd.write(patient_id)
            for szr_path in glob.glob(
                    os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
                szr_name = os.path.splitext(
                    os.path.basename(szr_path))[0].split('obs_data_')[-1]
                csv_paths = []
                for i in range(nchains):
                    log_path = os.path.join(root_dir, patient_id, 'logs',
                                            f'{szr_name}_{fname_suffix}{i+1}.log')
                    csv_path = os.path.join(
                        root_dir, patient_id, 'results',
                        f'samples_{szr_name}_{fname_suffix}{i+1}.csv')
                    if (lib.utils.stan.is_completed(log_path)):
                        csv_paths.append(csv_path)
                fd.write(f',{szr_name},{len(csv_paths)}')
            fd.write('\n')


def make_plots():
    for patient_id in patient_ids:
        # szr_names = [
        #     os.path.splitext(os.path.basename(fname))[0] for fname in glob.glob(
        #         os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R'))
        # ]
        # Read EZ hypothesis or skip patient if hypothesis doesn't exist
        try:
            ez_hyp = np.where(
                np.loadtxt(
                    f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt'
                ) == 1)[0]
        except Exception as err:
            print(err)
            continue
        print(patient_id)
        for szr_path in glob.glob(
                os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
            szr_name = os.path.splitext(
                os.path.basename(szr_path))[0].split('obs_data_')[-1]
            print(szr_name)
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
            fit_data = lib.io.stan.rload(szr_path)
            samples = lib.io.stan.read_samples(
                csv_paths, nwarmup=0, nsampling=200)
            lib.plots.stan.x0_violin_patient(
                samples['x0'],
                ez_hyp,
                figsize=(25, 5),
                figname=os.path.join(root_dir, patient_id, 'figures',
                                     f'x0_violin_{szr_name}.png'),
                plt_close=True)
            lib.plots.stan.nuts_diagnostics(
                samples,
                figsize=(15, 10),
                figname=os.path.join(root_dir, patient_id, 'figures',
                                     f'nuts_diagnostics_{szr_name}.png'),
                plt_close=True)
            lib.plots.stan.pair_plots(
                samples, ['K', 'tau0', 'amplitude', 'offset', 'alpha'],
                figname=os.path.join(root_dir, patient_id, 'figures',
                                     f'params_pair_plots_{szr_name}.png'),
                sampler='HMC',
                plt_close=True)
            lib.plots.stan.plot_source(
                samples['x'].mean(axis=0),
                samples['z'].mean(axis=0),
                ez_hyp, [],
                figname=os.path.join(
                    root_dir, patient_id, 'figures',
                    f'posterior_predicted_src_{szr_name}.png'),
                plt_close=True)
            lib.plots.stan.plot_fit_target(
                {
                    'slp': samples['mu_slp'].mean(axis=0),
                    'snsr_pwr': samples['mu_snsr_pwr'].mean(axis=0)
                },
                fit_data,
                figname=os.path.join(
                    root_dir, patient_id, 'figures',
                    f'posterior_predicted_slp_{szr_name}.png'),
                plt_close=True)


def extract_x0(patient_ids, nchains, fname_suffix, root_dir):
    '''
    Extracts x0 values from csv files and saves them in a numpy file
    '''
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
        print(patient_id)
        first_szr = True
        for szr_path in glob.glob(
                os.path.join(root_dir, patient_id, 'Rfiles', 'obs_data*.R')):
            # szr_name = os.path.splitext(
            #     os.path.basename(szr_path))[0].split('obs_data_')[-1]
            szr_name = os.path.splitext(
                os.path.basename(szr_path))[0]
            print(szr_name)
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
            fit_data = lib.io.stan.rload(szr_path)
            samples = lib.io.stan.read_samples(
                csv_paths, nwarmup=0, nsampling=200, variables_of_interest=['x0'])
            np.save(os.path.join(root_dir, patient_id, 'results', f'x0_{szr_name}.npy'), samples['x0'])
            if(first_szr):
                x0_cumul = samples['x0']
                first_szr = False
            else:
                x0_cumul = np.append(x0_cumul, samples['x0'], axis=0)
        np.save(os.path.join(root_dir, patient_id, 'results', 'x0_cumul.npy'), x0_cumul)


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


def find_ez(onst_thrshld, patient_ids, root_dir):
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
        a, b = np.histogram(onsets[onsets<150])
        ez_pred = np.zeros(nn)
        ez_pred[np.where(onsets<b[1])] = 1
        np.save(os.path.join(root_dir, patient_id, 'ez_pred.npy'), ez_pred)
    
    
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


root_dir = 'results/exp10/exp10.67'
patient_ids = dict()
patient_ids['engel1or2'] = ['id001_bt','id003_mg','id004_bj','id010_cmn','id013_lk','id014_vc','id017_mk','id020_lma','id021_jc','id022_te','id025_mc','id027_sj','id030_bf','id039_mra','id040_ms', 'id050_sx']
patient_ids['engel3or4'] = ['id007_rd','id008_dmc','id023_br','id028_ca','id033_fc','id036_dm', 'id037_cg']
precision = []
recall = []
onst_thrshlds = [-0.05] #np.linspace(-0.5, 0.0, int(0.5/0.01)+1)
for threshold in onst_thrshlds:
    find_ez(threshold, patient_ids['engel1or2'], root_dir)
    p, r = precision_recall(patient_ids['engel1or2'], root_dir)
    precision.append(p)
    recall.append(r)
    # print(f"x0_threshold:{x0_threshold} => Precision:{p} \t Recall:{r} \n")
# plt.figure()
# plt.plot(onst_thrshlds, precision, 'rx', label='Precision')
# plt.plot(onst_thrshlds, recall, 'rx', label='Recall')
# plt.title('Engel score I or II')
# plt.legend()
# plt.show(block=False)

ax = plt.subplot(111)
ax.bar([1,2], [precision[0],recall[0]], color=['black', 'grey'])

precision = []
recall = []
onst_thrshlds = [-0.05] #np.linspace(-0.5, 0.0, int(0.5/0.01)+1)
for threshold in onst_thrshlds:
    find_ez(threshold, patient_ids['engel3or4'], root_dir)
    p, r = precision_recall(patient_ids['engel3or4'], root_dir)
    precision.append(p)
    recall.append(r)
    # print(f"onst_thrshlds:{onst_thrshlds} => Precision:{p} \t Recall:{r} \n")
# plt.figure()
# plt.plot(onst_thrshlds, precision, 'rx', label='Precision')
# plt.plot(onst_thrshlds, recall, 'rx', label='Recall')
# plt.title('Engel score III or IV')
# plt.legend()
# plt.show(block=False)

ax.bar([5,6], [precision[0],recall[0]], color=['black', 'grey'])
ax.set_xticks([1.5, 5.5])
ax.set_xticklabels(['Engel score I or II', 'Engel score III or IV'], fontsize=15)
# ax.set_yticks(ax.get_yticks())
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
legend_elements = [Line2D([0], [0], color='black', lw=5, label='Precision'),
                   Line2D([0], [0], color='grey', lw=5, label='Recall')]
ax.legend(handles=legend_elements)
plt.show()
