
"""
Base routines for I/O

"""

import io
import tarfile
import numpy as np


def load_tf_npy(tf, name):
    bio = io.BytesIO()
    with tf.extractfile(name) as fd:
        bio.write(fd.read())
    bio.seek(0)
    return np.load(bio)


def load_tbz_npy(fname, name):
    with tarfile.open(fname) as tf:
        npy = load_tf_npy(tf, name)
    return npy