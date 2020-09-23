# %%
# Save the names of best seizure in a json file
# %% 
import json
import lib.preprocess.envelope
import os

root_data_dir = 'datasets/retro'
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id014_vc',
                         'id017_mk', 'id020_lma', 'id022_te', 'id025_mc', 'id030_bf', 'id039_mra', 'id050_sx']
patient_ids['engel2'] = ['id021_jc', 'id027_sj', 'id040_ms']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc',
                         'id009_ba', 'id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id011_gr', 'id033_fc', 'id036_dm', 'id045_bc']
patient_ids['all'] = patient_ids['engel1'] + patient_ids['engel2'] + \
    patient_ids['engel3'] + patient_ids['engel4']

with open(os.path.join(root_data_dir, 'param_preprocess.json')) as fd:
    param_preproc = json.load(fd)

szr_name = dict()
for pat_id in patient_ids['all']:
    hpf = 10
    lpf = param_preproc[pat_id]['lpf']
    szr_name[pat_id], _ = lib.preprocess.envelope.find_bst_szr_slp(
        os.path.join(root_data_dir, pat_id), hpf=hpf, lpf=lpf, npoints=300)
    print(f"{pat_id} \t {szr_name[pat_id]}\n")
with open(os.path.join(root_data_dir, 'bst_szr_name_meansub.json'), 'w') as fd:
    json.dump(szr_name, fd)
# %%
# Prepare data for only the best seizure
#%%
import lib.preprocess.envelope
import os
import glob
import re
import lib.io.stan
import numpy as np
import json
import lib.io.seeg
# %%
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id014_vc', 'id017_mk', 'id020_lma', 'id022_te', 'id025_mc', 'id030_bf', 'id039_mra', 'id050_sx']
patient_ids['engel2'] = ['id021_jc', 'id027_sj', 'id040_ms']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id011_gr', 'id033_fc', 'id036_dm', 'id045_bc']
patient_ids['all'] = patient_ids['engel1'] + patient_ids['engel2'] + patient_ids['engel3'] + patient_ids['engel4']

# %%
results_dir = 'results/exp10/exp10.87'
os.makedirs(results_dir, exist_ok=True)
raw_data_dir = 'datasets/retro/'

with open(os.path.join(raw_data_dir, 'param_preprocess.json')) as fd:
    param_preproc = json.load(fd)

with open(os.path.join(raw_data_dir, 'ez_hyp_destrieux.json')) as fd:
    ez_hyp_all = json.load(fd)

npoints = 300
for pat_id in patient_ids['all']:
    os.makedirs(os.path.join(results_dir, pat_id, 'Rfiles'), exist_ok=True)
    # Read seizure name, hpf, lpf from the previous run
    pat_data_dir = os.path.join(raw_data_dir, pat_id)
    pat_res_dir = os.path.join(results_dir, pat_id)
    lpf = param_preproc[pat_id]['lpf']
    hpf = 10
    ez_hyp = ez_hyp_all[pat_id]['i_ez']
    szr_name,_ = lib.preprocess.envelope.find_bst_szr_slp(pat_data_dir, hpf=hpf, lpf=lpf, npoints=npoints)
    raw_seeg_fname = szr_name + '.raw.fif'
    meta_data_fname = szr_name + '.json'
    print(f"{pat_id} Preprocessing {szr_name} with lpf = {lpf}, hpf = {hpf}")
    try:
        data = lib.preprocess.envelope.prepare_data(pat_data_dir, meta_data_fname, raw_seeg_fname, hpf, lpf, 'destrieux')
    except lib.io.seeg.BadSeizure as err:
        print(err)
        continue
    except FileNotFoundError as fn_err:
        print(fn_err)
        continue
    fname_suffix = f'{szr_name}_hpf{hpf}_lpf{lpf}'
    # Prepare data file in .R format
    ds_freq = int(data['slp'].shape[0]/npoints)
    data['slp'] = data['slp'][0:-1:ds_freq]
    data['snsr_pwr'] = (data['slp']**2).mean(axis=0)
    data['ns'], data['nn'] = data['gain'].shape
    data['nt'] = data['slp'].shape[0]
    # ez_hyp = np.where(np.loadtxt(f'{pat_data_dir}/tvb/ez_hypothesis.destrieux.txt') == 1)[0]
    data['x0_mu'] = -3.0*np.ones(data['nn'])
    data['x0_mu'][ez_hyp] = -1.5
    lib.io.stan.rdump(os.path.join(results_dir, pat_id, 'Rfiles', f'fit_data_{fname_suffix}.R'), data)
    # Prepare parameter initialization file in .R format
    param_init = dict()
    param_init['x0'] = data['x0_mu']
    param_init['alpha'] = 1
    param_init['beta'] = 1
    param_init['K'] = 1.0
    param_init['tau0'] = 20
    param_init['eps_slp'] = 1.0
    param_init['eps_snsr_pwr'] = 1.0
    param_init['x_init'] = -2.0*np.ones(data['nn'])
    param_init['z_init'] = 3.5*np.ones(data['nn'])
    lib.io.stan.rdump(os.path.join(pat_res_dir, 'Rfiles', 'param_init.R'), param_init)
# %%
# Prepare data from all seizures
#%% 
import lib.preprocess.envelope
import os
import glob
import re
import lib.io.stan
import numpy as np
import json
import lib.io.seeg
# %%
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id014_vc', 'id017_mk', 'id020_lma', 'id022_te', 'id025_mc', 'id030_bf', 'id039_mra', 'id050_sx']
patient_ids['engel2'] = ['id021_jc', 'id027_sj', 'id040_ms']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id011_gr', 'id033_fc', 'id036_dm', 'id045_bc']
patient_ids['all'] = patient_ids['engel1'] + patient_ids['engel2'] + patient_ids['engel3'] + patient_ids['engel4']

# %%
results_dir = 'results/exp10/exp10.86'
os.makedirs(results_dir, exist_ok=True)
raw_data_dir = 'datasets/retro/'

with open(os.path.join(raw_data_dir, 'param_preprocess.json')) as fd:
    param_preproc = json.load(fd)

with open(os.path.join(raw_data_dir, 'ez_hyp_destrieux.json')) as fd:
    ez_hyp_all = json.load(fd)

npoints = 300
for pat_id in patient_ids['all']:
    os.makedirs(os.path.join(results_dir, pat_id, 'Rfiles'), exist_ok=True)
    # Read seizure name, hpf, lpf from the previous run
    pat_data_dir = os.path.join(raw_data_dir, pat_id)
    pat_res_dir = os.path.join(results_dir, pat_id)
    lpf = param_preproc[pat_id]['lpf']
    hpf = 10
    ez_hyp = ez_hyp_all[pat_id]['i_ez']
    for fif_path in glob.glob(os.path.join(raw_data_dir, pat_id, 'seeg', 'fif', '*.raw.fif')):
    # szr_name = re.findall(r"_.*_hpf", szr_fname)[0][1:-4]
    # lpf = float(re.findall(r"(lpf\d+\.\d+)", szr_fname)[0][3:])
    # hpf = float(re.findall(r"(hpf\d+)", szr_fname)[0][3:])
        szr_name = os.path.basename(fif_path).split('.')[0]
        raw_seeg_fname = szr_name + '.raw.fif'
        meta_data_fname = szr_name + '.json'
        print(f"{pat_id} Preprocessing {szr_name} with lpf = {lpf}, hpf = {hpf}")
        try:
            data = lib.preprocess.envelope.prepare_data(pat_data_dir, meta_data_fname, raw_seeg_fname, hpf, lpf, 'destrieux')
        except lib.io.seeg.BadSeizure as err:
            print(err)
            continue
        except FileNotFoundError as fn_err:
            print(fn_err)
            continue
        fname_suffix = f'{szr_name}_hpf{hpf}_lpf{lpf}'
        # Prepare data file in .R format
        ds_freq = int(data['slp'].shape[0]/npoints)
        data['slp'] = data['slp'][0:-1:ds_freq]
        data['snsr_pwr'] = (data['slp']**2).mean(axis=0)
        data['ns'], data['nn'] = data['gain'].shape
        data['nt'] = data['slp'].shape[0]
        # ez_hyp = np.where(np.loadtxt(f'{pat_data_dir}/tvb/ez_hypothesis.destrieux.txt') == 1)[0]
        data['x0_mu'] = -3.0*np.ones(data['nn'])
        data['x0_mu'][ez_hyp] = -1.5
        lib.io.stan.rdump(os.path.join(results_dir, pat_id, 'Rfiles', f'fit_data_{fname_suffix}.R'), data)
        # Prepare parameter initialization file in .R format
        param_init = dict()
        param_init['x0'] = data['x0_mu']
        param_init['alpha'] = 1
        param_init['beta'] = 1
        param_init['K'] = 1.0
        param_init['tau0'] = 20
        param_init['eps_slp'] = 1.0
        param_init['eps_snsr_pwr'] = 1.0
        param_init['x_init'] = -2.0*np.ones(data['nn'])
        param_init['z_init'] = 3.5*np.ones(data['nn'])
        lib.io.stan.rdump(os.path.join(pat_res_dir, 'Rfiles', 'param_init.R'), param_init)