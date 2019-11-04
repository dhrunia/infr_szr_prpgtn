import lib.io.stan
import numpy as np

def find_ez(onst_thrshld, bin_thrshld, nbins, csv_path, enforce_thrshld=False):
    optima = lib.io.stan.read_samples([csv_path])
    x = optima['x'][0]
    nt, nn = x.shape
    onsets = (nt + 50)*np.ones(nn)
    for i in range(nn):
        xt = x[:,i] > onst_thrshld
        if(xt.any()):
            onsets[i] = np.nonzero(x[:,i] > onst_thrshld)[0][0]
    nszng = np.size(np.nonzero(onsets < nt))
    assert nszng > 0, "No seizing regions found for: {}".format(csv_path) 
    a, b = np.histogram(onsets[onsets<nt], bins=nbins)
    if(nszng <= 1 and not enforce_thrshld):
        bin_thrshld = nbins
    ez_pred = np.zeros(nn)
    ez_pred[np.nonzero(onsets<b[bin_thrshld])[0]] = 1
    pz_pred = np.zeros(nn)
    pz_pred[np.nonzero(np.logical_and(onsets > b[bin_thrshld], onsets < nt))[0]] =1
    return ez_pred, pz_pred
