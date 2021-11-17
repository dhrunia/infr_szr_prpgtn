import tensorflow as tf
import numpy as np
import pyshtools as pysh


def prep(L_MAX):
    N_LON = 2 * L_MAX + 1
    N_LAT = L_MAX + 1
    cos_theta, glq_wts = pysh.expand.SHGLQ(L_MAX)
    cos_theta = tf.constant(cos_theta, dtype=tf.float32)
    glq_wts = tf.constant(glq_wts, dtype=tf.complex64)

    P_l_m_costheta = np.zeros((L_MAX+1, L_MAX+1, N_LAT), dtype=np.complex64)
    t = np.zeros((L_MAX+1, L_MAX+1), dtype=np.complex64)
    idcs = np.tril_indices(L_MAX+1)
    for i in range(cos_theta.shape[0]):
        t[idcs] = pysh.legendre.PlmON(L_MAX, cos_theta[i], -1, 1)
        P_l_m_costheta[:, :, i] = t
    P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
    return N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta


def analys(N_LON, F_theta_phi, glq_wts, P_l_m_costheta):
    F_theta_m = tf.signal.rfft(F_theta_phi)
    F_theta_m_glq_wtd = F_theta_m * glq_wts[:, np.newaxis]
    F_l_m = tf.einsum('ijk,kj->ij', P_l_m_costheta,
                      F_theta_m_glq_wtd) * (2*np.math.pi/N_LON)
    return F_l_m


def synth(N_LON, F_l_m, P_l_m_costheta):
    F_theta_m = tf.einsum('ij,ijk->kj', F_l_m, P_l_m_costheta)
    F_theta_phi = N_LON * \
        tf.signal.irfft(F_theta_m, fft_length=tf.constant(
            N_LON, dtype=tf.int32, shape=[1]))
    return F_theta_phi
