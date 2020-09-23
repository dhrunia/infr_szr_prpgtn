import glob
import os

patient_ids = dict()
patient_ids['engel1'] = ['id001_bt', 'id003_mg', 'id004_bj', 'id010_cmn', 'id013_lk',  'id017_mk', 'id020_lma', 'id022_te', 'id025_mc', 'id030_bf', 'id039_mra', 'id050_sx']
patient_ids['engel2'] = ['id014_vc','id021_jc', 'id027_sj', 'id040_ms']
patient_ids['engel3'] = ['id007_rd', 'id008_dmc', 'id009_ba', 'id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id011_gr', 'id033_fc', 'id036_dm', 'id045_bc']
patient_ids['all'] = patient_ids['engel1'] + patient_ids['engel2'] + patient_ids['engel3'] + patient_ids['engel4']
szr_name = dict()

stan_fname = 'vep-snsrfit-ode-rk4'
iters = '20000'


for pat_id in patient_ids['all']:
    szr_fname = glob.glob(f'results/exp10/exp10.87/{pat_id}/Rfiles/fit_data*.R')[0]
    szr_name[pat_id] = os.path.basename(szr_fname)[9:-2]


rule all:
    input:[f'results/exp10/exp10.87/{pat_id}/samples_{szr_name[pat_id]}.csv' for pat_id in patient_ids['all']]

rule map_fit:
    input:
        fit_data='results/exp10/exp10.87/{pat_id}/Rfiles/fit_data_{szr_name}.R',
        init_data='results/exp10/exp10.87/{pat_id}/Rfiles/param_init.R'
    output:
        csv='results/exp10/exp10.87/{pat_id}/samples_{szr_name}.csv'
    log:'results/exp10/exp10.87/{pat_id}/logs/snsrfit_ode_{szr_name}.log'
    shell:
        f"./{stan_fname} optimize algorithm=lbfgs iter={iters} save_iterations=0 "
        " data file={input.fit_data} "
        "init={input.init_data} "
        "output file={output.csv} refresh=10 &> {log}"