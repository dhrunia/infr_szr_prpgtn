# %%
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import sph_harm, lpmn
import numpy as np
import matplotlib.pyplot as plt
# %%
N_LAT = 100 # N_LAT must be >= L_MAX + 1
N_LON = 2 * N_LAT
L_MAX = 80

cos_theta, glq_wts = np.polynomial.legendre.leggauss(deg=N_LAT)

theta = tf.math.acos(cos_theta)
phi = tf.linspace(0.0, 2*np.math.pi, N_LON)
delta_phi = phi[1] - phi[0]

P_l_m_costheta = np.zeros((L_MAX+1, L_MAX+1, N_LAT), dtype=np.complex64)
for n in range(L_MAX + 1):
    for m in range(n + 1):
        for i in range(theta.shape[0]):
            P_l_m_costheta[n,m,i] = sph_harm(m, n, 0.0, theta[i])

t1 = tf.math.cos(phi[:, tf.newaxis] *
                 tf.range(0, L_MAX + 1, dtype=tf.float32)[tf.newaxis, :])
t2 = -tf.math.sin(phi[:, tf.newaxis] *
                  tf.range(0, L_MAX+1, dtype=tf.float32)[tf.newaxis, :])
E_m_phi = tf.complex(t1, t2)

# %%
f_theta_phi = np.zeros(shape=(N_LAT, N_LON))

for i, lat in enumerate(theta.numpy()):
    for j, lon in enumerate(phi.numpy()):
        f_theta_phi[i, j] = sph_harm(10, 20, lat, lon).real

plt.imshow(f_theta_phi)
plt.colorbar(fraction=0.025)
f_theta_phi = tf.constant(f_theta_phi, dtype=tf.complex64)

# %%
F_m_theta = tf.linalg.matmul(f_theta_phi, E_m_phi) * \
    tf.cast(delta_phi, tf.complex64)
F_m_theta_glq_wtd = F_m_theta * glq_wts[:, np.newaxis]
P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
f_l_m = tf.einsum('ijk,kj->ij',P_l_m_costheta, F_m_theta_glq_wtd)
# %%
# F_m_theta = tf.signal.rfft(tf.math.real(f_theta_phi))[:, 0:L_MAX+1]
# F_m_theta_glq_wtd = F_m_theta * glq_wts[:, np.newaxis]
# P_l_m_costheta = tf.constant(P_l_m_costheta, dtype=tf.complex64)
# f_l_m_fft = tf.einsum('ijk,kj->ij',P_l_m_costheta, F_m_theta_glq_wtd)
# %%
plt.imshow(tf.math.real(f_l_m))
plt.colorbar()
plt.imshow(tf.math.imag(f_l_m))
plt.colorbar()

# %%
def sht_analysis(f, l_max):
    n_lat, n_lon = f.shape
    

# %%
N_theta = 5
L = 2
f = np.arange(N_theta*(L+1)).reshape(N_theta, L+1)
# p = np.array(['-']*((L+1)*(L+1)*N_theta), dtype=object).reshape(L+1, L+1, N_theta)
p = np.arange((L+1)*(L+1)*N_theta).reshape(L+1, L+1, N_theta)
print(f)
print(p)
c1 = np.einsum('ijk,kj->ij', p, f)
c2 = np.zeros((L+1, L+1))
c3 = np.tensordot(p, f, ((1,2), (1,0)))
for i in range(L+1):
    for j in range(L+1):
        for k in range(N_theta):
            c2[i,j] += p[i,j,k]*f[k,j]

print(np.all(c1 == c2))
print(np.all(c3 == c2))
# %%
t = sph_harm(np.broadcast_to(m, (N_LAT, L_MAX+1, L_MAX+1)), np.broadcast_to(n, (N_LAT, L_MAX+1, L_MAX+1)), np.broadcast_to(theta, (N_LAT, L_MAX+1, L_MAX+1)))