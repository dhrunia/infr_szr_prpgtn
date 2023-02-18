import scipy.spatial
import numpy as np
import pyshtools as pysh


def find_rgn_map_reg(N_LAT, N_LON, cos_theta, verts_irreg_fname,
                     rgn_map_irreg_fname):
    theta = np.arccos(cos_theta)
    # NOTE: Indexed phi from [0:N_LON] because np.arange can return
    # N_LON + 1 values when the step is a float
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / N_LON)[0:N_LON]
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
    idcs = np.concatenate((idcs_lh, idcs_rh), dtype=np.int32)

    rgn_map_irreg = np.loadtxt(rgn_map_irreg_fname, dtype=np.int32)
    rgn_map_reg = rgn_map_irreg[idcs]

    return rgn_map_reg


def find_nbrs_irreg_sphere(N_LAT, N_LON, cos_theta, verts_irreg_fname):
    theta = np.arccos(cos_theta)
    # NOTE: Indexed phi from [0:N_LON] because np.arange can return
    # N_LON + 1 values when the step is a float
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / N_LON)[0:N_LON]
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
    return idcs


def find_nbrs_reg_sphere(N_LAT, N_LON, verts_irreg_fname, cos_theta=None):
    '''
    Finds the nearest neibhors of vertices on irregular grid (FreeSurfer)
    to vertices on regular grid (SHT)
    '''
    if cos_theta is None:
        cos_theta, _ = pysh.expand.SHGLQ(N_LAT - 1)
    theta = np.arccos(cos_theta)
    # NOTE: Indexed phi from [0:N_LON] because np.arange can return
    # N_LON + 1 values when the step is a float
    phi = np.arange(0, 2 * np.pi, 2 * np.pi / N_LON)[0:N_LON]
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
    # No. of vertices per hemisphere in the regular sphere
    nvph_reg = verts_reg.shape[0]
    kdtree = scipy.spatial.KDTree(verts_reg)
    _, idcs_lh = kdtree.query(verts_irreg[0:nvph_irreg, :])
    _, idcs_rh = kdtree.query(verts_irreg[nvph_irreg:, :])
    idcs_rh += nvph_reg
    idcs = np.concatenate((idcs_lh, idcs_rh))
    return idcs