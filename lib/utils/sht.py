import tensorflow as tf
import numpy as np
import pyshtools as pysh


def prep(L_MAX, N_LAT=None, N_LON=None):
    if (N_LON is None):
        N_LON = tf.constant(2 * L_MAX + 1, dtype=tf.int32)
    else:
        assert N_LON >= 2 * L_MAX + 1, "N_LON must be >= 2 * L_MAX + 1"
        N_LON = tf.constant(N_LON, dtype=tf.int32)
    if(N_LAT is None):
        N_LAT = tf.constant(L_MAX + 1, dtype=tf.int32)
    else:
        assert N_LAT >= L_MAX + 1, "N_LAT must be >= L_MAX + 1"
        N_LAT = tf.constant(N_LAT, dtype=tf.int32)
    cos_theta, glq_wts = pysh.expand.SHGLQ(N_LAT.numpy() - 1)
    cos_theta = tf.constant(cos_theta, dtype=tf.float32)
    glq_wts = tf.constant(glq_wts, dtype=tf.complex64)

    P_l_m_costheta = np.zeros((L_MAX + 1, L_MAX + 1, N_LAT.numpy()),
                              dtype=np.complex64)
    t = np.zeros((L_MAX + 1, L_MAX + 1), dtype=np.complex64)
    idcs = np.tril_indices(L_MAX + 1)
    for i in range(cos_theta.shape[0]):
        t[idcs] = pysh.legendre.PlmON(L_MAX, cos_theta[i], -1, 1)
        P_l_m_costheta[:, :, i] = t
    P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
    return N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta


@tf.function(jit_compile=True)
def analys(L_MAX, N_LON, F_theta_phi, glq_wts, P_l_m_costheta):
    pi = tf.constant(np.math.pi, dtype=tf.complex64)
    F_theta_m = tf.signal.rfft(F_theta_phi)
    F_theta_m_glq_wtd = F_theta_m[:, 0:L_MAX + 1] * glq_wts[:, tf.newaxis]
    F_l_m = tf.einsum('ijk,kj->ij', P_l_m_costheta, F_theta_m_glq_wtd) * (
        2 * pi / tf.cast(N_LON, tf.complex64))
    return F_l_m


@tf.function(jit_compile=True)
def synth(N_LON, F_l_m, P_l_m_costheta):
    F_theta_m = tf.einsum('ij,ijk->kj', F_l_m, P_l_m_costheta)
    F_theta_phi = tf.cast(N_LON, tf.float32) * \
        tf.signal.irfft(F_theta_m, fft_length=tf.reshape(N_LON, shape=[1]))
    return F_theta_phi
