import numpy as np
import os
import lib.preprocess.envelope
import glob

patient_ids = dict()
patient_ids['engel1'] = ['id003_mg','id004_bj','id010_cmn','id013_lk','id014_vc','id017_mk','id020_lma','id022_te','id025_mc','id027_sj','id030_bf','id039_mra','id050_sx']
patient_ids['engel2'] = ['id001_bt','id021_jc','id040_ms']
patient_ids['engel3'] = ['id007_rd','id008_dmc','id023_br','id028_ca', 'id037_cg']
patient_ids['engel4'] = ['id033_fc','id036_dm']
patient_ids['engel3or4'] = patient_ids['engel3'] + patient_ids['engel4']
patient_ids['engel2or3or4'] = patient_ids['engel2'] + patient_ids['engel3'] + patient_ids['engel4']

data_dir = '/home/anirudh/hdisk/Academia/projects/vep.stan/datasets/retro'

for pat_id in patient_ids['engel1'] + patient_ids['engel2or3or4']:
    szr_max_var = ''
    max_snsr_pwr_var = 0
    pat_data_dir = os.path.join(data_dir, pat_id)
    print(pat_id)
    for fif_path in glob.glob(os.path.join(data_dir, pat_id, 'seeg/fif')+'/*.json'):
        szr_name = os.path.splitext(os.path.basename(fif_path))[0]
        raw_seeg_fname = f'{szr_name}.raw.fif'
        meta_data_fname = f'{szr_name}.json'
        try:
            data = lib.preprocess.envelope.prepare_data(pat_data_dir, meta_data_fname, raw_seeg_fname, 10, 0.05)
        except (FileNotFoundError, Exception):
            continue
        ds_freq = int(data['slp'].shape[0]/150)
        data['slp'] = data['slp'][0:-1:ds_freq]
        snsr_pwr = (data['slp']**2).mean(axis=0)
        snsr_pwr_var = snsr_pwr.var()
        if(snsr_pwr_var > max_snsr_pwr_var):
            szr_max_var = szr_name
            max_snsr_pwr_var = snsr_pwr_var
        print('\t', szr_name, snsr_pwr_var)
    print('\t\t', pat_id, szr_max_var)
    
