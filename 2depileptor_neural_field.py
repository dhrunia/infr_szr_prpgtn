# %%
import numpy as np
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import lib.utils.sht as tfsht
import time
# %%
L_MAX = 128
N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta = tfsht.prep(L_MAX)
D = tf.constant(-0.01, dtype=tf.float32)
l = tf.range(0, L_MAX+1, dtype=tf.float32)
Dll = tf.cast(D * l * (l+1), dtype=tf.complex64)
# %%
def epileptor2D_nf_ode_fn(t, y, x0, tau):
    nn = N_LAT * N_LON
    x = y[0:nn]
    z = y[nn:2*nn]
    I1 = tf.constant(4.1, dtype=tf.float32)
    x_lm = tfsht.analys(N_LON, tf.reshape(tf.math.sigmoid(x), (N_LAT, N_LON)), 
                        glq_wts, P_l_m_costheta)
    x_lm_hat = Dll[:, tf.newaxis] * x_lm
    local_cplng = tf.reshape(tfsht.synth(N_LON, x_lm_hat, P_l_m_costheta), [-1])
    dx = 1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - z + I1 + local_cplng
    # gx = tf.reduce_sum(K * SC * (x[tf.newaxis, :] - x[:, tf.newaxis]), axis=1)
    dz = (1.0/tau)*(4*(x - x0) - z)
    return tf.concat((dx, dz), axis=0)
# %%
def euler_integrator(ode_fn, nsteps, y_init, x0, tau):
    y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
    y_next = y_init
    time_step = tf.constant(0.01, dtype=tf.float32)
    for i in tf.range(nsteps, dtype=tf.int32):
        for j in tf.range(0.1/time_step):
            y_next = y_next + time_step * epileptor2D_nf_ode_fn(0.0, y_next, x0, tau)
        y = y.write(i, y_next)
    return y.stack()

def rk4_integrator(ode_fn, nsteps, y_init, x0, tau):
    y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
    y_next = y_init
    h = 0.01
    for i in tf.range(nsteps, dtype=tf.int32):
        for j in tf.range(0.1/h):
            k1 = epileptor2D_nf_ode_fn(0.0, y_next, x0, tau)
            k2 = epileptor2D_nf_ode_fn(0.0, y_next + h*(k1/2), x0, tau)
            k3 = epileptor2D_nf_ode_fn(0.0, y_next + h*(k2/2), x0, tau)
            k4 = epileptor2D_nf_ode_fn(0.0, y_next + h*k3, x0, tau)
            y_next = y_next + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        y = y.write(i, y_next)
    return y.stack()

def bdf_integrator(ode_fn, t_init, y_init, solution_times, constants):
    return tfp.math.ode.BDF().solve(ode_fn=ode_fn,
                             initial_time=t_init,
                             initial_state=y_init,
                             solution_times=solution_times,
                             constants=constants)

def dormandprince_integrator(ode_fn, t_init, y_init, solution_times, constants):
    return tfp.math.ode.DormandPrince(atol=1e-3, rtol=1e-2).solve(ode_fn=ode_fn,
                             initial_time=t_init,
                             initial_state=y_init,
                             solution_times=solution_times,
                             constants=constants)
# %%
t_init = tf.constant(0.0, dtype=tf.float32)
x_init_true = tf.constant(-2.0, dtype=tf.float32) * tf.ones(N_LAT*N_LON, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * tf.ones(N_LAT*N_LON, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32)
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = tf.constant(-1.8, dtype=tf.float32) * tf.ones(N_LAT*N_LON, dtype=tf.float32)
t_init = tf.constant(0.0, dtype=tf.float32)

# %%
nsteps = 300
start_time = time.time()
y_true = euler_integrator(epileptor2D_nf_ode_fn, nsteps, y_init_true,
                    x0_true, tau_true)
print(f"Time elapsed: {time.time() - start_time} seconds")

x_true = y_true[:, :N_LAT*N_LON]
z_true = y_true[:, N_LAT*N_LON:]
idx = 10
plt.figure(figsize=(7,5), dpi=150, constrained_layout=True)
plt.subplot(211)
plt.ylabel(r'$x$', fontsize=15)
plt.plot(x_true[:, idx])
plt.subplot(212)
plt.plot(z_true[:, idx])
plt.ylabel(r'$z$', fontsize=15)
plt.suptitle(f"Euler Integration - dt={0.01}")

# %%
nsteps = 300
start_time = time.time()
y_true = rk4_integrator(epileptor2D_nf_ode_fn, nsteps, y_init_true,
                    x0_true, tau_true)
print(f"Time elapsed: {time.time() - start_time} seconds")

x_true = y_true[:, :N_LAT*N_LON]
z_true = y_true[:, N_LAT*N_LON:]
idx = 10
plt.figure(figsize=(7,5), dpi=150, constrained_layout=True)
plt.subplot(211)
plt.ylabel(r'$x$', fontsize=15)
plt.plot(x_true[:, idx])
plt.subplot(212)
plt.plot(z_true[:, idx])
plt.ylabel(r'$z$', fontsize=15)
plt.suptitle(f"RK4 Integration - dt={0.1}")
# %%
time_step = tf.constant(0.1, dtype=tf.float32)
solution_times = time_step * tf.range(0, 300, dtype=tf.float32)
start_time = time.time()
y_true = dormandprince_integrator(epileptor2D_nf_ode_fn, t_init, y_init_true, 
                        solution_times, {'x0':x0_true, 'tau':tau_true})
print(f"Time elapsed: {time.time() - start_time} seconds")

x_true = y_true.states[:, :N_LAT*N_LON]
z_true = y_true.states[:, N_LAT*N_LON:]
idx = 10
plt.figure(figsize=(7,5), dpi=150, constrained_layout=True)
plt.subplot(211)
plt.ylabel(r'$x$', fontsize=15)
plt.plot(x_true[:, idx])
plt.subplot(212)
plt.plot(z_true[:, idx])
plt.ylabel(r'$z$', fontsize=15)
plt.suptitle(f"DormandPrince Integration - dt={time_step:.1f}")