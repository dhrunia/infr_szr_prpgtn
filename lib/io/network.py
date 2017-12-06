"""
I/O for loading information about the brain network.

"""

import os
import numpy as np
import nibabel


# TODO this is specific to trec dataset/timpx's work
def load_centers(centers: os.PathLike, aparcaseg: os.PathLike):
    from numpy import c_, ones
    reg_xyz = np.loadtxt(centers, usecols=(1, 2, 3))
    reg_names = []
    with open('data/centers.txt', 'r') as fd:
        for i, line in enumerate(fd.readlines()):
            reg_names.append('r%03d-%s' % (i, line.strip().split()[0]))
    seg = nibabel.load(aparcaseg)
    reg_xyz = seg.affine.dot(c_[reg_xyz, ones((reg_xyz.shape[0], ))].T)[:3].T
    return reg_xyz
