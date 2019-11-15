import lib.io.stan
import numpy as np
import glob
import os

def find_ez(src_thrshld, onst_wndw_sz, csv_path):
    optima = lib.io.stan.read_samples(csv_path)
    x = optima['x'][0]
    nt, nn = x.shape
    onsets = (nt + 50)*np.ones(nn)
    for i in range(nn):
        xt = x[:,i] > src_thrshld
        if(xt.any()):
            onsets[i] = np.nonzero(xt)[0][0]
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


def precision_recall(patient_ids, root_dir, src_thrshld, onst_wndw_sz):
    tp = fp = fn = 0
    for patient_id in patient_ids:
        # Read EZ hypothesis or skip patient if hypothesis doesn't exist
        try:
            ez_hyp = np.loadtxt(f'datasets/retro/{patient_id}/tvb/ez_hypothesis.destrieux.txt')
        except Exception as err:
            print(err)
            continue
        # ez_pred = np.load(os.path.join(root_dir, patient_id, 'ez_pred.npy')).astype(int)
        csv_path = glob.glob(os.path.join(root_dir, patient_id, '*chain1.csv'))
        ez_pred, pz_pred = find_ez(src_thrshld, onst_wndw_sz, csv_path)
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
