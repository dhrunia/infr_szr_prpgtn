import scipy.spatial
import numpy as np
import tensorflow as tf


def find_rgn_map(N_LAT, N_LON, cos_theta, verts_irreg_fname,
                 rgn_map_irreg_fname):
    theta = np.arccos(cos_theta)
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / N_LON)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    verts_reg = np.zeros((N_LAT * N_LON, 3))
    verts_reg[:, 0] = (np.cos(phi_grid) * np.sin(theta_grid)).flatten()
    verts_reg[:, 1] = (np.sin(phi_grid) * np.sin(theta_grid)).flatten()
    verts_reg[:, 2] = np.cos(theta_grid).flatten()

    verts_irreg = np.loadtxt(verts_irreg_fname)
    verts_irreg -= verts_irreg.mean(axis=0)
    mean_radius = np.sqrt((verts_irreg**2).sum(axis=1)).mean()
    verts_irreg /= mean_radius

    # No.of vertices per hemisphere in the irregular sphere
    nvph_irreg = verts_irreg.shape[0] // 2
    kdtree = scipy.spatial.KDTree(verts_irreg[0:nvph_irreg, :])
    _, idcs_lh = kdtree.query(verts_reg)
    kdtree = scipy.spatial.KDTree(verts_irreg[nvph_irreg:, :])
    _, idcs_rh = kdtree.query(verts_reg)
    idcs_rh += nvph_irreg
    idcs = np.concatenate((idcs_lh, idcs_rh))

    rgn_map_irreg = np.loadtxt(rgn_map_irreg_fname)
    rgn_map_reg = tf.constant(rgn_map_irreg[idcs], dtype=tf.uint32)

    return rgn_map_reg
