# %%
import numpy as np
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import lib.utils.sht as tfsht
import lib.utils.projector
import time
# %%
L_MAX = 128
N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta = tfsht.prep(L_MAX)
D = tf.constant(-0.01, dtype=tf.float32)
l = tf.range(0, L_MAX + 1, dtype=tf.float32)
Dll = tf.cast(D * l * (l + 1), dtype=tf.complex64)
nv = tf.constant(2 * N_LAT * N_LON, dtype=tf.int32)  # Total no. of vertices
nvph = tf.math.floordiv(nv, 2)  # No.of vertices per hemisphere

verts_irreg_fname = 'datasets/id004_bj_jd/tvb/ico7/vertices.txt'
rgn_map_irreg_fname = 'datasets/id004_bj_jd/tvb/Cortex_region_map_ico7.txt'
rgn_map_reg = lib.utils.projector.find_rgn_map(
    N_LAT=N_LAT,
    N_LON=N_LON,
    cos_theta=cos_theta,
    verts_irreg_fname=verts_irreg_fname,
    rgn_map_irreg_fname=rgn_map_irreg_fname)
unkown_roi_mask = np.ones(nv)
unkown_roi_mask[np.nonzero(rgn_map_reg == 0)[0]] = 0
unkown_roi_mask = tf.constant(unkown_roi_mask, dtype=tf.float32)

rgn_map_reg_sorted = tf.gather(rgn_map_reg, tf.argsort(rgn_map_reg))
low_idcs = []
high_idcs = []
for roi in tf.unique(rgn_map_reg_sorted)[0]:
    roi_idcs = tf.squeeze(tf.where(rgn_map_reg_sorted == roi))
    low_idcs.append(roi_idcs[0])
    high_idcs.append(roi_idcs[-1] + 1)

# Compute a region mapping such that all cortical rois are contiguous
# NOTE: This shouldn't be necessary once subcortical regions are also included in the simulation
tmp = rgn_map_reg.numpy()
tmp[tmp > 81] = tmp[tmp > 81] - 9
vrtx_roi_map = tf.constant(tmp, dtype=tf.int32)
# SC = tf.random.normal((145, 145), mean=0, stddev=0.2)


# %%
@tf.function
def epileptor2D_nf_ode_fn(t, y, x0, tau, K):
    x = y[0:nv]
    z = y[nv:2 * nv]
    I1 = tf.constant(4.1, dtype=tf.float32)
    # NOTE: alpha > 7.0 is causing DormandPrince integrator to diverge
    alpha = tf.constant(1.0, dtype=tf.float32)
    theta = tf.constant(-1.0, dtype=tf.float32)
    gamma_lc = tf.constant(1.0, dtype=tf.float32)
    x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
    x_hat_lh = tf.reshape(x_hat[0:nvph], (N_LAT, N_LON))
    x_hat_rh = tf.reshape(x_hat[nvph:], (N_LAT, N_LON))
    x_lm_lh = tfsht.analys(N_LON, x_hat_lh, glq_wts, P_l_m_costheta)
    x_lm_hat_lh = Dll[:, tf.newaxis] * x_lm_lh
    x_lm_rh = tfsht.analys(N_LON, x_hat_rh, glq_wts, P_l_m_costheta)
    x_lm_hat_rh = Dll[:, tf.newaxis] * x_lm_rh
    local_cplng_lh = tf.reshape(
        tfsht.synth(N_LON, x_lm_hat_lh, P_l_m_costheta), [-1])
    local_cplng_rh = tf.reshape(
        tfsht.synth(N_LON, x_lm_hat_rh, P_l_m_costheta), [-1])
    local_cplng = tf.concat((local_cplng_lh, local_cplng_rh), axis=0)
    x_sorted = tf.gather(x, rgn_map_reg_sorted)
    x_roi = tfp.stats.windowed_mean(x_sorted, low_idcs, high_idcs)
    # tf.print(x_hat_roi.shape)
    global_cplng_roi = tf.reduce_sum(
        K * SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]), axis=1)
    # global_cplng_roi = tf.reduce_sum(
    #     (x_hat_roi[tf.newaxis, :]), axis=1)
    global_cplng_vrtcs = tf.gather(global_cplng_roi, vrtx_roi_map)
    dx = 1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - \
        z + I1 + gamma_lc * local_cplng - global_cplng_vrtcs
    dz = (1.0 / tau) * (4 * (x - x0) - z)
    return tf.concat((dx, dz), axis=0)


# %%


@tf.function
def euler_integrator(ode_fn, nsteps, sampling_period, time_step, y_init, x0,
                     tau, K):
    y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
    y_next = y_init
    for i in tf.range(nsteps, dtype=tf.int32):
        for j in tf.range(sampling_period / time_step):
            y_next = y_next + time_step * ode_fn(0.0, y_next, x0, tau, K)
        y = y.write(i, y_next)
    return y.stack()


@tf.function
def rk4_integrator(ode_fn, nsteps, sampling_period, time_step, y_init, x0, tau,
                   K):
    y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
    y_next = y_init
    h = time_step
    for i in tf.range(nsteps, dtype=tf.int32):
        for j in tf.range(sampling_period / h):
            k1 = ode_fn(0.0, y_next, x0, tau, K)
            k2 = ode_fn(0.0, y_next + h * (k1 / 2), x0, tau, K)
            k3 = ode_fn(0.0, y_next + h * (k2 / 2), x0, tau, K)
            k4 = ode_fn(0.0, y_next + h * k3, x0, tau, K)
            y_next = y_next + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        y = y.write(i, y_next)
    return y.stack()


@tf.function
def bdf_integrator(ode_fn, t_init, y_init, solution_times, constants):
    return tfp.math.ode.BDF(atol=1e-4, rtol=1e-3, validate_args=True).solve(
        ode_fn=ode_fn,
        initial_time=t_init,
        initial_state=y_init,
        solution_times=solution_times,
        constants=constants)


@tf.function
def dormandprince_integrator(ode_fn, t_init, y_init, solution_times,
                             constants):
    return tfp.math.ode.DormandPrince(atol=1e-4, rtol=1e-3,
                                      validate_args=True).solve(
                                          ode_fn=ode_fn,
                                          initial_time=t_init,
                                          initial_state=y_init,
                                          solution_times=solution_times,
                                          constants=constants)


# %%
t_init = tf.constant(0.0, dtype=tf.float32)
x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(2*N_LAT*N_LON, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(2*N_LAT*N_LON, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32)
K_true = tf.constant(1.0, dtype=tf.float32)
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(2 * N_LAT * N_LON)
ez_hyp_roi = [10, 15, 23]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == rgn_map_reg)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32)
SC = np.loadtxt('datasets/id004_bj_jd/tvb/vep_conn/weights.txt')

# remove subcortical regions
# NOTE: not required once subcortical regions are included in simulation
idcs1, idcs2 = np.meshgrid(np.unique(rgn_map_reg),
                           np.unique(rgn_map_reg),
                           indexing='ij')
SC = SC[idcs1, idcs2]

SC = SC / np.max(SC)
SC[np.diag_indices_from(SC)] = 0.0
SC = tf.constant(SC, dtype=tf.float32)

t_init = tf.constant(0.0, dtype=tf.float32)

# %%
time_step = tf.constant(0.1, dtype=tf.float32)
solution_times = time_step * tf.range(0, 300, dtype=tf.float32)
start_time = time.time()
y_true = dormandprince_integrator(epileptor2D_nf_ode_fn, t_init, y_init_true,
                                  solution_times, {
                                      'x0': x0_true,
                                      'tau': tau_true,
                                      'K': K_true
                                  })
print(f"Time elapsed: {time.time() - start_time} seconds")
x_true = y_true.states[:, :nv].numpy()
z_true = y_true.states[:, nv:].numpy()
ode_solver = "DormandPrince Integration"
# %%
# time_step = tf.constant(0.1, dtype=tf.float32)
# solution_times = time_step * tf.range(0, 300, dtype=tf.float32)
# start_time = time.time()
# y_true = bdf_integrator(epileptor2D_nf_ode_fn, t_init, y_init_true,
#                         solution_times, {
#                             'x0': x0_true,
#                             'tau': tau_true,
#                             'K': K_true
#                         })
# print(f"Time elapsed: {time.time() - start_time} seconds")
# x_true = y_true.states[:, :nv].numpy()
# z_true = y_true.states[:, nv:].numpy()
# %%
nsteps = 300
sampling_period = 0.1
time_step = 0.01
start_time = time.time()
y_true = euler_integrator(epileptor2D_nf_ode_fn, nsteps, sampling_period,
                          time_step, y_init_true, x0_true, tau_true, K_true)
print(f"Time elapsed: {time.time() - start_time} seconds")

x_true = y_true[:, :N_LAT * N_LON].numpy()
z_true = y_true[:, N_LAT * N_LON:].numpy()
ode_solver = "Euler Integartion"
# %%
nsteps = 300
sampling_period = 0.1
time_step = 0.01
start_time = time.time()
y_true = rk4_integrator(epileptor2D_nf_ode_fn, nsteps, sampling_period,
                        time_step, y_init_true, x0_true, tau_true, K_true)
print(f"Time elapsed: {time.time() - start_time} seconds")

x_true = y_true[:, :N_LAT * N_LON].numpy()
z_true = y_true[:, N_LAT * N_LON:].numpy()
ode_solver = "RK4 Integration"
# %%
plt.figure(figsize=(7, 5), dpi=150, constrained_layout=True)
plt.subplot(211)
plt.ylabel(r'$x$', fontsize=15)
plt.plot(x_true[:, ez_hyp_vrtcs])
plt.subplot(212)
plt.plot(z_true[:, ez_hyp_vrtcs])
plt.ylabel(r'$z$', fontsize=15)
plt.suptitle(f"{ode_solver} - dt={time_step:.1f}")
plt.show()
