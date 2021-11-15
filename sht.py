# %%
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
# %%
L_MAX = 128
N_LON = 270 # 2*L_MAX + 1
N_LAT = 136 # N_LON // 2

# cos_theta, glq_wts = np.polynomial.legendre.leggauss(deg=N_LAT)
cos_theta, glq_wts = sp.roots_legendre(N_LAT)

theta = tf.math.acos(cos_theta)
phi = tf.linspace(0.0, 2*np.math.pi, N_LON)

# P_l_m_costheta = np.zeros((L_MAX+1, L_MAX+1, N_LAT), dtype=np.complex64)
# for l in range(L_MAX + 1):
#     for m in range(l + 1):
#         for i in range(theta.shape[0]):
#             P_l_m_costheta[l, m, i] = sp.sph_harm(m, l, 0.0, theta[i])

P_l_m_costheta = np.zeros((L_MAX+1, L_MAX+1, N_LAT), dtype=np.complex64)
t = np.zeros((L_MAX+1, L_MAX+1), dtype=np.complex64)
for i in range(theta.shape[0]):
    t[np.tril_indices(L_MAX+1)] = pysh.legendre.PlmON(L_MAX, cos_theta[i], -1, 1)
    P_l_m_costheta[:,:,i] = t

# %%
F_theta_phi = np.zeros(shape=(N_LAT, N_LON), dtype=np.complex64)

for i, lat in enumerate(theta.numpy()):
    for j, lon in enumerate(phi.numpy()):
        F_theta_phi[i, j] = sp.sph_harm(10, 20, lon, lat).real

plt.figure(figsize=(5,7))
plt.imshow(F_theta_phi.real)
plt.tight_layout()
F_theta_phi = tf.constant(F_theta_phi, dtype=tf.complex64)

# %%
F_m_theta = tf.signal.fft(F_theta_phi)
F_m_theta = F_m_theta[:, 0:L_MAX+1]
F_m_theta_glq_wtd = F_m_theta * glq_wts[:, np.newaxis]
P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
F_l_m = tf.einsum('ijk,kj->ij', P_l_m_costheta,
                  F_m_theta_glq_wtd) * (2*np.math.pi/N_LON)
# %%
# F_m_theta = tf.signal.rfft(tf.math.real(f_theta_phi))[:, 0:L_MAX+1]
# F_m_theta_glq_wtd = F_m_theta * glq_wts[:, np.newaxis]
# P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
# f_l_m_fft = tf.einsum('ijk,kj->ij',P_l_m_costheta, F_m_theta_glq_wtd)
# %%
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(tf.math.real(F_l_m))
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(tf.math.imag(F_l_m))
plt.colorbar(fraction=0.025)
plt.tight_layout()

# %%
import shtns
# %%
lmax = L_MAX
mmax = L_MAX
sh = shtns.sht(lmax, mmax)
nlat, nphi = sh.set_grid()
flm_flat = sh.analys(np.array(F_theta_phi.numpy().real, dtype=np.float64))
flm = np.zeros((lmax+1,mmax+1), dtype=np.complex128)
flm[np.tril_indices_from(flm)] = flm_flat

# %%
plt.figure(figsize=(10,6))
plt.subplot(221)
plt.imshow(tf.math.real(F_l_m))
plt.colorbar(fraction=0.025)
plt.subplot(222)
plt.imshow(tf.math.imag(F_l_m))
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
iF_m_theta = tf.einsum('ij,ijk->kj', F_l_m , P_l_m_costheta)
iF_m_theta_pad = tf.pad(iF_m_theta, [[0, 0], [0, N_LON-(L_MAX+1)]])
f_theta_phi_recon = 2 * N_LON * \
    tf.math.real(tf.signal.ifft(iF_m_theta_pad)) - \
    tf.math.real(iF_m_theta_pad[:, 0, tf.newaxis])
# %%
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(tf.math.real(F_theta_phi))
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(tf.math.imag(F_theta_phi))
plt.colorbar(fraction=0.025)
plt.tight_layout()
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(tf.math.real(f_theta_phi_recon))
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(tf.math.imag(f_theta_phi_recon))
plt.colorbar(fraction=0.025)
plt.tight_layout()
# %%
t = tf.math.real(F_theta_phi) - f_theta_phi_recon
plt.figure(figsize=(5,7))
plt.imshow(t)
plt.colorbar()
print(tf.reduce_max(t))

# %%
ftp = sh.synth(flm_flat)
# %%
plt.figure(figsize=(10,6))
plt.subplot(121)
plt.imshow(ftp)
plt.colorbar(fraction=0.025)
# %%
t = ftp - f_theta_phi_recon
plt.figure(figsize=(5,7))
plt.imshow(t)
plt.colorbar()
print(tf.reduce_max(t))
# %%
