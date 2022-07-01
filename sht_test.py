# %%
import pyshtools as pysh
import lib.utils.sht as tfsht
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# %%
L_MAX = 128
N_LON = 2 * 128 + 1
N_LAT = 128 + 1

N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta = tfsht.prep(
    L_MAX, N_LAT, N_LON)
# %%
# F_theta_phi = np.random.randn(N_LAT, N_LON)
theta, phi = np.meshgrid(np.arange(0, 2 * np.pi, 2 * np.pi / N_LON.numpy()),
                         np.arccos(cos_theta.numpy()))
F_theta_phi = pysh.expand.spharm_lm(20, 10, theta, phi, 'ortho', 'real', -1,
                                    False)
# F_l_m_gt = pysh.SHCoeffs.from_random(power=np.random.randn(L_MAX+1)**2, lmax=L_MAX,
#             kind='real', normalization='ortho', csphase=-1, exact_power=True)
# F_theta_phi_gt = F_l_m_gt.expand(grid='GLQ')

plt.figure(figsize=(5, 7))
plt.imshow(F_theta_phi)
plt.title('Synthetic Ground truth')
plt.tight_layout()
plt.colorbar(fraction=0.025)
F_theta_phi = tf.constant(F_theta_phi, dtype=tf.float32)
# %%
F_l_m = tfsht.analys(L_MAX, N_LON, F_theta_phi, glq_wts, P_l_m_costheta)
# %%
# import shtns
# %%
# lmax = L_MAX
# mmax = L_MAX
# sh = shtns.sht(lmax, mmax)
# nlat, nphi = sh.set_grid()
# flm_flat = sh.analys(np.array(F_theta_phi.numpy(), dtype=np.float64))
# flm = np.zeros((lmax+1,mmax+1), dtype=np.complex128)
# for l in range(L_MAX + 1):
#     for m in range(0, l + 1):
#         flm[l,m] = flm_flat[sh.idx(l,m)]
# %%
flmr = pysh.expand.SHExpandGLQ(F_theta_phi.numpy(),
                               glq_wts.numpy().real,
                               cos_theta.numpy(),
                               norm=4,
                               csphase=-1,
                               lmax_calc=L_MAX)
flmc = pysh.shio.SHrtoc(flmr)
# %%
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(tf.math.real(F_l_m))
plt.title(r'TFSHT - $f_l^m$ real')
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(tf.math.imag(F_l_m))
plt.title(r'TFSHT - $f_l^m$ imaginary')
plt.colorbar(fraction=0.025)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(flmc[0])
plt.title(r'PYSH - $f_l^m$ real')
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(flmc[1])
plt.title(r'PYSH - $f_l^m$ imaginary')
plt.colorbar(fraction=0.025)
plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.imshow(tf.math.abs(flmc[0] - tf.math.real(F_l_m)))
plt.title(r'Absolute Difference')
plt.colorbar(fraction=0.025)
plt.subplot(122)
plt.imshow(tf.math.abs(flmc[1] - tf.math.imag(F_l_m)))
plt.title(r'Absolute Difference')
plt.colorbar(fraction=0.025)
plt.tight_layout()
# %%
F_theta_phi_recon = tfsht.synth(N_LON, F_l_m, P_l_m_costheta)
# %%
# ftp = sh.synth(flm_flat)
# %%
ftp = pysh.expand.MakeGridGLQ(pysh.shio.SHctor(flmc),
                              cos_theta.numpy(),
                              lmax=L_MAX,
                              norm=4,
                              csphase=-1)
# %%
plt.figure(figsize=(15, 10))
plt.subplot(331)
plt.imshow(F_theta_phi)
plt.title('Ground truth')
plt.colorbar(fraction=0.025)
plt.subplot(332)
plt.imshow(F_theta_phi_recon)
plt.title('TFSHT - Reconstruction')
plt.colorbar(fraction=0.025)
plt.subplot(333)
diff1 = tf.math.abs(F_theta_phi_recon - F_theta_phi)
plt.imshow(diff1)
plt.title('Absolute Difference')
plt.colorbar(fraction=0.025)

plt.subplot(334)
plt.imshow(F_theta_phi)
plt.title('Ground truth')
plt.colorbar(fraction=0.025)
plt.subplot(335)
plt.imshow(ftp)
plt.title('PYSH - Reconstruction')
plt.colorbar(fraction=0.025)
plt.subplot(336)
diff2 = tf.math.abs(ftp - F_theta_phi)
plt.imshow(diff2)
plt.title('Absolute Difference')
plt.colorbar(fraction=0.025)

plt.subplot(337)
plt.imshow(tf.math.abs(F_theta_phi - F_theta_phi))
plt.title('Absolute Difference')
plt.colorbar(fraction=0.025)
plt.subplot(338)
plt.imshow(tf.math.abs(F_theta_phi_recon - ftp))
plt.title('Absolute Difference')
plt.colorbar(fraction=0.025)
plt.subplot(339)
plt.imshow(tf.math.abs(diff1 - diff2))
plt.title('Absolute Difference')
plt.colorbar(fraction=0.025)
plt.tight_layout()

# %%
D = tf.constant(-0.01, dtype=tf.float32)
l = tf.range(0, L_MAX + 1, dtype=tf.float32)
Dll = tf.cast(D * l * (l + 1), dtype=tf.complex64)
F_l_m_hat = Dll[:, tf.newaxis] * F_l_m
F_theta_phi_recon_hat = tfsht.synth(N_LON, F_l_m_hat, P_l_m_costheta)
# %%
# flm_flat_hat = (-0.01*sh.l*(sh.l+1)) * flm_flat
# flm_hat = np.zeros((lmax+1,mmax+1), dtype=np.complex128)
# for l in range(L_MAX + 1):
#     for m in range(0, l + 1):
#         flm_hat[l,m] = flm_flat_hat[sh.idx(l,m)]
# ftp_hat = sh.synth(flm_flat_hat)
# %%
flmc_hat = np.zeros_like(flmc)
flmc_hat[0] = Dll[:, tf.newaxis].numpy().real * flmc[0]
flmc_hat[1] = Dll[:, tf.newaxis].numpy().real * flmc[1]
ftp_hat = pysh.expand.MakeGridGLQ(pysh.shio.SHctor(flmc_hat),
                                  cos_theta.numpy(),
                                  lmax=L_MAX,
                                  norm=4,
                                  csphase=-1)
# %%
plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(F_theta_phi_recon_hat)
plt.title('TFSHT')
plt.colorbar(fraction=0.025)
plt.subplot(132)
plt.imshow(ftp_hat)
plt.title('PYSH')
plt.colorbar(fraction=0.025)
plt.subplot(133)
plt.imshow(tf.math.abs(ftp_hat - F_theta_phi_recon_hat))
plt.title('Absolute difference')
plt.colorbar(fraction=0.025)
plt.tight_layout()

# %%
plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(tf.math.real(F_l_m_hat))
plt.title('TFSHT')
plt.colorbar(fraction=0.025)
plt.subplot(132)
plt.imshow(flmc_hat[0])
plt.title('PYSH')
plt.colorbar(fraction=0.025)
plt.subplot(133)
plt.imshow(tf.math.abs(flmc_hat[0] - tf.math.real(F_l_m_hat)))
plt.title('Absolute difference')
plt.colorbar(fraction=0.025)
plt.tight_layout()

plt.figure(figsize=(12, 8))
plt.subplot(131)
plt.imshow(tf.math.imag(F_l_m_hat))
plt.title('TFSHT')
plt.colorbar(fraction=0.025)
plt.subplot(132)
plt.imshow(flmc_hat[1])
plt.title('PYSH')
plt.colorbar(fraction=0.025)
plt.subplot(133)
plt.imshow(tf.math.abs(flmc_hat[1] - tf.math.imag(F_l_m_hat)))
plt.title('Absolute difference')
plt.colorbar(fraction=0.025)
plt.tight_layout()

# %%
y1 = F_theta_phi_recon_hat.numpy()
np.testing.assert_allclose(y1, ftp_hat, rtol=1e-3, atol=1e-3)
