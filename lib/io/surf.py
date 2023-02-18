import numpy as np


def read_vep_parcellation(freesurfer_lut_fname, mrtrix_lut_fname):
    freesurf_lut_numbers = np.genfromtxt(freesurfer_lut_fname,
                                         usecols=0,
                                         dtype=int)
    freesurf_lut_names = np.genfromtxt(freesurfer_lut_fname,
                                       usecols=1,
                                       dtype=str)
    vep_numbers = np.genfromtxt(mrtrix_lut_fname, usecols=0, dtype=int)
    vep_labels = np.genfromtxt(mrtrix_lut_fname, usecols=1, dtype=str)
    vep_colors = np.genfromtxt(mrtrix_lut_fname, usecols=[2, 3, 4], dtype=int)
    vep_aseg_numbers = []
    for vep_label in vep_labels:
        aseg_index = np.nonzero(freesurf_lut_names == vep_label)[0]
        assert aseg_index.size == 1
        aseg_index = aseg_index[0]
        vep_aseg_numbers.append(freesurf_lut_numbers[aseg_index])
    return vep_numbers, vep_labels, vep_colors, vep_aseg_numbers
