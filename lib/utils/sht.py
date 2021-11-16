import scipy.special as sp
import tensorflow as tf
import numpy as np
import pyshtools as pysh

def prep(L_MAX, N_LAT):
    cos_theta, glq_wts = sp.roots_legendre(N_LAT)
    cos_theta = tf.constant(cos_theta, dtype=tf.float32)
    glq_wts = tf.constant(glq_wts, dtype=tf.complex64)

    P_l_m_costheta = np.zeros((L_MAX+1, L_MAX+1, N_LAT), dtype=np.complex64)
    t = np.zeros((L_MAX+1, L_MAX+1), dtype=np.complex64)
    idcs = np.tril_indices(L_MAX+1)
    for i in range(cos_theta.shape[0]):
        t[idcs] = pysh.legendre.PlmON(L_MAX, cos_theta[i], -1, 1)
        P_l_m_costheta[:,:,i] = t
    P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
    return cos_theta, glq_wts, P_l_m_costheta

def analys(L_MAX, N_LON, F_theta_phi, glq_wts, P_l_m_costheta):
    F_theta_m = tf.signal.rfft(F_theta_phi)
    F_theta_m = F_theta_m[:, 0:L_MAX+1]
    F_m_theta_glq_wtd = F_theta_m * glq_wts[:, np.newaxis]
    F_l_m = tf.einsum('ijk,kj->ij', P_l_m_costheta,
                    F_m_theta_glq_wtd) * (2*np.math.pi/N_LON)
    return F_l_m

def synth(L_MAX, N_LON, F_l_m, P_l_m_costheta):
    F_theta_m = tf.einsum('ij,ijk->kj', F_l_m , P_l_m_costheta)
    F_theta_m_pad = tf.pad(F_theta_m, [[0, 0], [0, (N_LON//2)- L_MAX]])
    F_theta_phi = N_LON * tf.signal.irfft(F_theta_m_pad)
    return F_theta_phi