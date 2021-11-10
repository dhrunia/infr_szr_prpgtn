# %%
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
# %%
L_MAX = 64
N_LON = 140 # 4 * L_MAX
N_LAT = 72 # N_LON // 2

cos_theta, glq_wts = np.polynomial.legendre.leggauss(deg=N_LAT)

theta = tf.math.acos(cos_theta)
phi = tf.linspace(0.0, 2*np.math.pi, N_LON)

P_l_m_costheta = np.zeros((L_MAX+1, L_MAX+1, N_LAT), dtype=np.complex64)
for n in range(L_MAX + 1):
    for m in range(n + 1):
        for i in range(theta.shape[0]):
            P_l_m_costheta[n, m, i] = sp.sph_harm(m, n, 0.0, theta[i])

# t1 = tf.math.cos(phi[:, tf.newaxis] *
#                  tf.range(0, L_MAX + 1, dtype=tf.float32)[tf.newaxis, :])
# t2 = -tf.math.sin(phi[:, tf.newaxis] *
#                   tf.range(0, L_MAX+1, dtype=tf.float32)[tf.newaxis, :])
# E_m_phi = tf.complex(t1, t2)

# %%
f_theta_phi = np.zeros(shape=(N_LAT, N_LON), dtype=np.complex64)

for i, lat in enumerate(theta.numpy()):
    for j, lon in enumerate(phi.numpy()):
        f_theta_phi[i, j] = sp.sph_harm(10, 20, lon, lat).real

plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(f_theta_phi.real)
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(f_theta_phi.imag)
plt.colorbar(fraction=0.025)
plt.tight_layout()
f_theta_phi = tf.constant(f_theta_phi, dtype=tf.complex64)

# %%
# delta_phi = phi[1] - phi[0]
# F_m_theta = tf.linalg.matmul(f_theta_phi, E_m_phi) * \
#     tf.cast(delta_phi, tf.complex64)
F_m_theta = tf.signal.fft(f_theta_phi)
F_m_theta = F_m_theta[:,0:L_MAX+1]
F_m_theta_glq_wtd = F_m_theta * glq_wts[:, np.newaxis]
P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
f_l_m = tf.einsum('ijk,kj->ij',P_l_m_costheta, F_m_theta_glq_wtd)
# %%
# F_m_theta = tf.signal.rfft(tf.math.real(f_theta_phi))[:, 0:L_MAX+1]
# F_m_theta_glq_wtd = F_m_theta * glq_wts[:, np.newaxis]
# P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
# f_l_m_fft = tf.einsum('ijk,kj->ij',P_l_m_costheta, F_m_theta_glq_wtd)
# %%
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(tf.math.real(f_l_m))
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(tf.math.imag(f_l_m))
plt.colorbar(fraction=0.025)
plt.tight_layout()

# %%
import shtns
# %%
lmax = L_MAX
mmax = L_MAX
sh = shtns.sht(lmax, mmax)
nlat, nphi = sh.set_grid()
flm_flat = sh.analys(np.array(f_theta_phi.numpy().real, dtype=np.float64))
flm = np.zeros((lmax+1,mmax+1), dtype=np.complex128)
flm[np.tril_indices_from(flm)] = flm_flat
# %%
plt.figure(figsize=(10,6))
plt.subplot(221)
plt.imshow(tf.math.real(f_l_m)*(np.math.pi*2/N_LON))
plt.colorbar(fraction=0.025)
plt.subplot(222)
plt.imshow(tf.math.imag(f_l_m)*(np.math.pi*2/(N_LON)))
plt.colorbar(fraction=0.025)
plt.tight_layout()

plt.figure(figsize=(10,6))
plt.subplot(223)
plt.imshow(flm.real)
plt.colorbar(fraction=0.025)
plt.subplot(224)
plt.imshow(flm.imag)
plt.colorbar(fraction=0.025)
plt.tight_layout()
# %%
sh.gauss_wts