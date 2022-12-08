import numpy as np


def find_xpos(data_arr, bin_center=0, delta=0.05):
    # data_arr     : 1D data array
    # bin_center     : position to center around
    # delta : distance between points
    _, bin_edges = np.histogram(data_arr)
    bin_idcs = np.digitize(data_arr, bin_edges)
    x_pos = np.zeros_like(data_arr)
    for bin_idx in np.unique(bin_idcs):
        flag = True
        x_pos_idcs = np.where(bin_idcs == bin_idx)[0]
        i = 0
        j = 1
        for idx in x_pos_idcs:
            if flag:
                x_pos[idx] = bin_center + i * delta
                i += 1
                flag = False
            else:
                x_pos[idx] = bin_center - j * delta
                j += 1
                flag = True
    return x_pos