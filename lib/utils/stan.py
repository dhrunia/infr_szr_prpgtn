import lib.io.stan
import numpy as np
import glob
import os
import json
import lib.io.seeg

def find_onsets(ts, thrshld):
    nt, nn = ts.shape
    onsets = (nt + 50)*np.ones(nn)
    for i in range(nn):
        ts_i_thrshd = ts[:,i] > thrshld
        if(ts_i_thrshd.any()):
            onsets[i] = np.nonzero(ts_i_thrshd)[0][0]
        else:
            onsets[i] = np.inf
    return onsets

def teps_to_wndwsz(data_dir, szr_name, t_eps, npoints):
    szr_len = lib.io.seeg.find_szr_len(data_dir, szr_name)
    dt = szr_len/npoints
    wndw_sz = int(np.round(t_eps/dt))
    return wndw_sz

def find_ez(src_thrshld, onst_wndw_sz, csv_path):
    # print(csv_path)
    optima = lib.io.stan.read_samples(csv_path)
    # x = optima['x'][0]
    x = optima['y'][0][:, 0:int(optima['y'].shape[2]/2)]
    nt, nn = x.shape
    onsets = find_onsets(x, src_thrshld)
    nszng = np.size(np.nonzero(onsets < nt))
    assert nszng > 0, "No seizing regions found for: {}".format(csv_path) 
    # counts, edges = np.histogram(onsets[onsets<nt], range=(0,nt), bins=nbins)
    first_onset_time = onsets.min()
    ez_pred = np.zeros(nn)
    onst_thrshld = first_onset_time + onst_wndw_sz
    ez_pred[np.nonzero(onsets <= onst_thrshld)[0]] = 1
    pz_pred = np.zeros(nn)
    pz_pred[np.nonzero(np.logical_and(onsets > onst_thrshld, onsets < nt))[0]] = 1
    return ez_pred, pz_pred


def precision_recall(patient_ids, root_dir, src_thrshld, t_eps,
                    parcellation, outfile_regex, npoints):
    tp = fp = fn = 0
    for patient_id in patient_ids:
        # Read EZ hypothesis or skip patient if hypothesis doesn't exist
        # print(patient_id)
        try:
            if(parcellation == 'destrieux'):
                # ez_hyp = np.loadtxt(
                #     f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt')
                with open('datasets/retro/ez_hyp_destrieux.json') as fd:
                    ez_hyp_all = json.load(fd)
                    ez_hyp_roi = ez_hyp_all[patient_id]['i_ez']
            elif(parcellation == 'vep'):
                with open('datasets/retro/ei-vep_53.json') as fd:
                    ez_hyp_all = json.load(fd)
                    ez_hyp_roi = ez_hyp_all[patient_id]['i_ez']
        except Exception as err:
            print(err)
            continue
        # ez_pred = np.load(os.path.join(root_dir, patient_id, 'ez_pred.npy')).astype(int)
        csv_path = glob.glob(os.path.join(root_dir, patient_id, outfile_regex))
        szr_name = '_'.join(os.path.basename(csv_path[0]).split('_')[1:-2])
        szr_len = lib.io.seeg.find_szr_len(os.path.join('datasets/retro', patient_id), szr_name)
        dt = szr_len/npoints
        ez_pred, pz_pred = find_ez(src_thrshld, int(np.round(t_eps/dt)), csv_path)
        ez_hyp = np.zeros_like(ez_pred)
        ez_hyp[ez_hyp_roi] = 1
        # print(f'EZ hypothesis: {np.where(ez_hyp == 1)[0]}')
        # print(f'EZ prediction: {np.where(ez_pred == 1)[0]}')
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
