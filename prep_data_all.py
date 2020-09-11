#%%
import lib.preprocess.envelope
import os
import glob
import re
import lib.io.stan
import numpy as np
# %%
patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk', 'id014_vc', 'id017_mk', 'id020_lma', 'id022_te', 'id025_mc', 'id030_bf', 'id039_mra', 'id050_sx']
patient_ids['engel2'] = ['id021_jc', 'id027_sj', 'id040_ms']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id011_gr', 'id033_fc', 'id036_dm', 'id045_bc']
patient_ids['all'] = patient_ids['engel1'] + patient_ids['engel2'] + patient_ids['engel3'] + patient_ids['engel4']
# %%
results_dir = 'results/exp10/exp10.83'
raw_data_dir = 'datasets/retro/'
for pat_id in patient_ids['all']:
    os.makedirs(os.path.join(results_dir, pat_id, 'Rfiles'), exist_ok=True)
    # Read seizure name, hpf, lpf from the previous run
    szr_fname = glob.glob(f"results/exp10/retro_results/{pat_id}/*chain1.csv")[0].split('/')[-1]
    szr_name = re.findall(r"_.*_hpf", szr_fname)[0][1:-4]
    lpf = float(re.findall(r"(lpf\d+\.\d+)", szr_fname)[0][3:])
    hpf = float(re.findall(r"(hpf\d+)", szr_fname)[0][3:])
    raw_seeg_fname = szr_name + '.raw.fif'
    meta_data_fname = szr_name + '.json'
    pat_data_dir = os.path.join(raw_data_dir, pat_id)
    pat_res_dir = os.path.join(results_dir, pat_id)
    print(f"{pat_id} Preprocessing {szr_name} with lpf = {lpf}, hpf = {hpf}")
    data = lib.preprocess.envelope.prepare_data(pat_data_dir, meta_data_fname, raw_seeg_fname, hpf, lpf)
    fname_suffix = f'{szr_name}_hpf{hpf}_lpf{lpf}'
    # Prepare data file in .R format
    ds_freq = int(data['slp'].shape[0]/150)
    data['slp'] = data['slp'][0:-1:ds_freq]
    data['snsr_pwr'] = (data['slp']**2).mean(axis=0)
    data['ns'], data['nn'] = data['gain'].shape
    data['nt'] = data['slp'].shape[0]
    ez_hyp = np.where(np.loadtxt(f'{pat_data_dir}/tvb/ez_hypothesis.destrieux.txt') == 1)[0]
    data['x0_mu'] = -3.0*np.ones(data['nn'])
    data['x0_mu'][ez_hyp] = -1.5
    lib.io.stan.rdump(os.path.join(results_dir, pat_id, 'Rfiles', f'fit_data_{fname_suffix}.R'), data)
    # Prepare parameter initialization file in .R format
    param_init = dict()
    param_init['x0'] = data['x0_mu']
    param_init['amplitude'] = 1
    param_init['offset'] = 1
    param_init['K'] = 1.0
    param_init['tau0'] = 20
    param_init['eps_slp'] = 1.0
    param_init['eps_snsr_pwr'] = 1.0
    param_init['x_init'] = -2.0*np.ones(data['nn'])
    param_init['z_init'] = 3.5*np.ones(data['nn'])
    lib.io.stan.rdump(os.path.join(pat_res_dir, 'Rfiles', 'param_init.R'), param_init)