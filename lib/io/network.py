"""
I/O for loading information about the brain network.

"""

import os
import numpy as np
import nibabel as nib


# TODO this is specific to trec dataset/timpx's work
def load_centers(centers: str, aparcaseg: str=None):
    regions = []
    with open(centers, 'r') as fd:
        for i, line in enumerate(fd.readlines()):
            rname, x, y, z = line.split()
            regions.append((f'r{i}-{rname}',
                            np.r_[float(x), float(y), float(z)]))
    rxyz = np.array([xyz for _, xyz in regions])
    if aparcaseg:
        # move to correct coord system
        seg = nib.load(aparcaseg)  # this is aa in diff space
        rxyz = seg.affine.dot(np.c_[rxyz, np.ones((rxyz.shape[0],))].T)[:3].T
    return regions, rxyz


def load_weights(fname, diag=0, norm='max', pow=1.0):
    try:
        weights = np.load(fname)
    except:
        weights = np.loadtxt(fname)
    np.fill_diagonal(weights, diag)
    weights /= weights.max()
    weights = weights ** pow
    return weights


def pick_by_name(regions, qname):
    idx = []
    for i, (name, *_) in enumerate(regions):
        if qname in name:
            idx.append(i)
    return np.array(idx)


def pick_by_gain(gain, isort, extra, n=1):
    idx = np.r_[extra,
                np.unique(
                    np.argsort(
                        gain[isort],
                        axis=1)[:, -n:])
                ]
    
