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
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.autograph.set_verbosity(10)
# %%
L_MAX = 128
N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta = tfsht.prep(L_MAX)
P_l_1tom_costheta = tf.constant(tf.math.real(P_l_m_costheta)[:, 1:, :],
                                dtype=tf.float32)
P_l_0_costheta = tf.constant(tf.math.real(P_l_m_costheta)[:, 0, :],
                             dtype=tf.float32)
glq_wts_real = tf.constant(tf.math.real(glq_wts), dtype=tf.float32)
D = tf.constant(-0.01, dtype=tf.float32)
l = tf.range(0, L_MAX + 1, dtype=tf.float32)
Dll = tf.cast(D * l * (l + 1), dtype=tf.complex64)
nv = tf.constant(2 * N_LAT * N_LON, dtype=tf.int32)  # Total no. of vertices
nvph = tf.math.floordiv(nv, 2)  # No.of vertices per hemisphere

verts_irreg_fname = 'datasets/id004_bj_jd/tvb/ico7/vertices.txt'
rgn_map_irreg_fname = 'datasets/id004_bj_jd/tvb/Cortex_region_map_ico7.txt'
rgn_map_reg = lib.utils.projector.find_rgn_map(
    N_LAT=N_LAT.numpy(),
    N_LON=N_LON.numpy(),
    cos_theta=cos_theta,
    verts_irreg_fname=verts_irreg_fname,
    rgn_map_irreg_fname=rgn_map_irreg_fname)
unkown_roi_mask = np.ones(nv)
unkown_roi_mask[np.nonzero(rgn_map_reg == 0)[0]] = 0
unkown_roi_mask = tf.constant(unkown_roi_mask, dtype=tf.float32)

# Constants cached for computing gradients of local coupling
delta_phi = tf.constant(2.0 * np.pi / N_LON.numpy(), dtype=tf.float32)
phi = tf.range(0, 2.0 * np.pi, delta_phi, dtype=tf.float32)
phi_db = phi[:, tf.newaxis] - phi[tf.newaxis, :]
m = tf.range(0, L_MAX + 1, dtype=tf.float32)
cos_m_phidb = 2.0 * tf.math.cos(tf.einsum("m,db->mdb", m, phi_db))
cos_1tom_phidb = tf.constant(cos_m_phidb[1:, :, :], dtype=tf.float32)
cos_0_phidb = tf.constant(cos_m_phidb[0, :, :] * 0.5, dtype=tf.float32)
P_l_m_Dll = delta_phi * tf.math.real(
    Dll)[:, tf.newaxis, tf.newaxis] * tf.math.real(P_l_m_costheta)
P_l_1tom_Dll = tf.constant(P_l_m_Dll[:, 1:, :], dtype=tf.float32)
P_l_0_Dll = tf.constant(P_l_m_Dll[:, 0, :], dtype=tf.float32)

rgn_map_reg_sorted = tf.gather(rgn_map_reg, tf.argsort(rgn_map_reg))
low_idcs = []
high_idcs = []
for roi in tf.unique(rgn_map_reg_sorted)[0]:
    roi_idcs = tf.squeeze(tf.where(rgn_map_reg_sorted == roi))
    low_idcs.append(roi_idcs[0])
    high_idcs.append(roi_idcs[-1] + 1)

# Compute a region mapping such that all cortical rois are contiguous
# NOTE: This shouldn't be necessary once subcortical regions are also
# included in the simulation
tmp = rgn_map_reg.numpy()
tmp[tmp > 81] = tmp[tmp > 81] - 9
vrtx_roi_map = tf.constant(tmp, dtype=tf.int32)
# SC = tf.random.normal((145, 145), mean=0, stddev=0.2)

# jacobian = -1.0 * tf.einsum("a,lmc,lma,mdb->cdab",
#                             tf.math.real(glq_wts),
#                             P_l_m_Dll[:, 1:, :],
#                             tf.math.real(P_l_m_costheta[:, 1:, :]),
#                             cos_m_phidb[1:, :, :],
#                             optimize="optimal") - tf.einsum(
#                                 "a,lc,la,db->cdab",
#                                 tf.math.real(glq_wts),
#                                 P_l_m_Dll[:, 0, :],
#                                 tf.math.real(P_l_m_costheta)[:, 0, :],
#                                 cos_m_phidb[0, :, :],
#                                 optimize="optimal")

# %%


@tf.custom_gradient
def local_coupling(
    x,
    glq_wts,
    P_l_m_costheta,
    Dll,
    N_LAT,
    N_LON,
    glq_wts_real,
    P_l_1tom_Dll,
    P_l_1tom_costheta,
    cos_1tom_phidb,
    P_l_0_Dll,
    P_l_0_costheta,
    cos_0_phidb,
):
    # print("local_coupling()")
    x_hat_lh = tf.stop_gradient(tf.reshape(x[0:N_LAT * N_LON], (N_LAT, N_LON)))
    x_hat_rh = tf.stop_gradient(tf.reshape(x[N_LAT * N_LON:], (N_LAT, N_LON)))
    x_lm_lh = tf.stop_gradient(
        tfsht.analys(N_LON, x_hat_lh, glq_wts, P_l_m_costheta))
    x_lm_hat_lh = tf.stop_gradient(Dll[:, tf.newaxis] * x_lm_lh)
    x_lm_rh = tf.stop_gradient(
        tfsht.analys(N_LON, x_hat_rh, glq_wts, P_l_m_costheta))
    x_lm_hat_rh = tf.stop_gradient(Dll[:, tf.newaxis] * x_lm_rh)
    local_cplng_lh = tf.reshape(
        tfsht.synth(N_LON, x_lm_hat_lh, P_l_m_costheta), [-1])
    local_cplng_rh = tf.reshape(
        tfsht.synth(N_LON, x_lm_hat_rh, P_l_m_costheta), [-1])
    local_cplng = tf.stop_gradient(
        tf.concat((local_cplng_lh, local_cplng_rh), axis=0))

    def grad(upstream):
        upstream_lh = tf.reshape(upstream[0:N_LAT * N_LON], (N_LAT, N_LON))
        upstream_rh = tf.reshape(upstream[N_LAT * N_LON:], (N_LAT, N_LON))
        glq_wts_grad = None  #tf.zeros_like(glq_wts, dtype=tf.complex64)
        P_l_m_costheta_grad = None  #tf.zeros_like(P_l_m_costheta, dtype=tf.complex64)
        Dll_grad = None  #tf.zeros_like(Dll, dtype=tf.complex64)
        N_LAT_grad = None  #tf.zeros_like(N_LAT, dtype=tf.complex64)
        N_LON_grad = None  #tf.zeros_like(N_LON, dtype=tf.complex64)
        glq_wts_real_grad = None  #tf.zeros_like(glq_wts_real, dtype=tf.float32)
        P_l_1tom_Dll_grad = None  #tf.zeros_like(P_l_1tom_Dll, dtype=tf.float32)
        P_l_1tom_costheta_grad = None  #tf.zeros_like(P_l_1tom_costheta,
        #        dtype=tf.float32)
        cos_1tom_phidb_grad = None  #tf.zeros_like(cos_1tom_phidb, dtype=tf.float32)
        P_l_0_Dll_grad = None  #tf.zeros_like(P_l_0_Dll, dtype=tf.float32)
        P_l_0_costheta_grad = None  #tf.zeros_like(P_l_0_costheta, dtype=tf.float32)
        cos_0_phidb_grad = None  #tf.zeros_like(cos_0_phidb, dtype=tf.float32)
        # print(upstream_lh.dtype, upstream_rh.dtype)
        # print(upstream_lh)
        # print(type(upstream_lh), type(glq_wts_real), type(P_l_1tom_Dll),
        #       type(P_l_1tom_costheta), type(cos_1tom_phidb))
        g_lh = tf.einsum("cd,a,lmc,lma,mdb->ab",
                         upstream_lh,
                         glq_wts_real,
                         P_l_1tom_Dll,
                         P_l_1tom_costheta,
                         cos_1tom_phidb,
                         optimize="optimal") + tf.einsum("cd,a,lc,la,db->ab",
                                                         upstream_lh,
                                                         glq_wts_real,
                                                         P_l_0_Dll,
                                                         P_l_0_costheta,
                                                         cos_0_phidb,
                                                         optimize="optimal")
        g_rh = tf.einsum("cd,a,lmc,lma,mdb->ab",
                         upstream_rh,
                         glq_wts_real,
                         P_l_1tom_Dll,
                         P_l_1tom_costheta,
                         cos_1tom_phidb,
                         optimize="optimal") + tf.einsum("cd,a,lc,la,db->ab",
                                                         upstream_rh,
                                                         glq_wts_real,
                                                         P_l_0_Dll,
                                                         P_l_0_costheta,
                                                         cos_0_phidb,
                                                         optimize="optimal")
        g = tf.clip_by_norm(
            tf.concat((tf.reshape(g_lh, [-1]), tf.reshape(g_rh, [-1])),
                      axis=0), 100)
        return [
            g, glq_wts_grad, P_l_m_costheta_grad, Dll_grad, N_LAT_grad,
            N_LON_grad, glq_wts_real_grad, P_l_1tom_Dll_grad,
            P_l_1tom_costheta_grad, cos_1tom_phidb_grad, P_l_0_Dll_grad,
            P_l_0_costheta_grad, cos_0_phidb_grad
        ]

    return local_cplng, grad


# @tf.function
def epileptor2D_nf_ode_fn(y, x0, tau, K, SC, glq_wts, P_l_m_costheta, Dll,
                          N_LAT, N_LON, unkown_roi_mask, rgn_map_reg_sorted,
                          low_idcs, high_idcs, vrtx_roi_map, glq_wts_real,
                          P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb,
                          P_l_0_Dll, P_l_0_costheta, cos_0_phidb):
    print("epileptor2d_nf_ode_fn()")
    x = y[0:nv]
    z = y[nv:2 * nv]
    I1 = tf.constant(4.1, dtype=tf.float32)
    # NOTE: alpha > 7.0 is causing DormandPrince integrator to diverge
    alpha = tf.constant(1.0, dtype=tf.float32)
    theta = tf.constant(-1.0, dtype=tf.float32)
    gamma_lc = tf.constant(1.0, dtype=tf.float32)
    x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
    local_cplng = local_coupling(x_hat, glq_wts, P_l_m_costheta, Dll, N_LAT,
                                 N_LON, glq_wts_real, P_l_1tom_Dll,
                                 P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
                                 P_l_0_costheta, cos_0_phidb)
    x_sorted = tf.gather(x, rgn_map_reg_sorted)
    x_roi = tfp.stats.windowed_mean(x_sorted, low_idcs, high_idcs)
    # tf.print(x_hat_roi.shape)
    global_cplng_roi = tf.reduce_sum(
        K[0] * SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]), axis=1)
    global_cplng_vrtcs = tf.gather(global_cplng_roi, vrtx_roi_map)
    dx = 1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - \
        z + I1 + gamma_lc * local_cplng
    dz = (1.0 / tau[0]) * (4 * (x - x0) - z - global_cplng_vrtcs)
    return tf.concat((dx, dz), axis=0)
    # return y * tf.constant(-0.1, dtype=tf.float32)


# %%
# Test custom gradients

def local_coupling_wocg(
    x,
    glq_wts,
    P_l_m_costheta,
    Dll,
    N_LAT,
    N_LON,
):
    # print("local_coupling()")
    x_hat_lh = tf.reshape(x[0:N_LAT * N_LON], (N_LAT, N_LON))
    x_hat_rh = tf.reshape(x[N_LAT * N_LON:], (N_LAT, N_LON))
    x_lm_lh = tfsht.analys(N_LON, x_hat_lh, glq_wts, P_l_m_costheta)
    x_lm_hat_lh = Dll[:, tf.newaxis] * x_lm_lh
    x_lm_rh = tfsht.analys(N_LON, x_hat_rh, glq_wts, P_l_m_costheta)
    x_lm_hat_rh = Dll[:, tf.newaxis] * x_lm_rh
    local_cplng_lh = tf.reshape(
        tfsht.synth(N_LON, x_lm_hat_lh, P_l_m_costheta), [-1])
    local_cplng_rh = tf.reshape(
        tfsht.synth(N_LON, x_lm_hat_rh, P_l_m_costheta), [-1])
    local_cplng = tf.concat((local_cplng_lh, local_cplng_rh), axis=0)

    return local_cplng

def find_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)

        alpha = tf.constant(1.0, dtype=tf.float32)
        theta = tf.constant(-1.0, dtype=tf.float32)
        x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
        lc = local_coupling(x_hat, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON,
                            glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
                            cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
                            cos_0_phidb)
        return lc, tape.gradient(lc, x)

def find_grad_wocg(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        alpha = tf.constant(1.0, dtype=tf.float32)
        theta = tf.constant(-1.0, dtype=tf.float32)
        x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
        lc = local_coupling_wocg(x_hat, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON)
        return lc, tape.gradient(lc, x)

x = tf.constant(-2.0, dtype=tf.float32) * \
tf.ones(2*N_LAT*N_LON, dtype=tf.float32)

# thrtcl, nmrcl = tf.test.compute_gradient(local_coupling, [
#     x_hat, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON, glq_wts_real,
#     P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
#     cos_0_phidb
# ])
lc1, x_hat_grad_cg = find_grad(x)
lc2, x_hat_grad_wocg = find_grad_wocg(x)
print(tf.reduce_max(tf.math.abs(x_hat_grad_cg - x_hat_grad_wocg)))

# %%


# @tf.function(input_signature=[
#     tf.TensorSpec(shape=(), dtype=tf.int32),
#     tf.TensorSpec(shape=(), dtype=tf.float32),
#     tf.TensorSpec(shape=(), dtype=tf.float32),
#     tf.TensorSpec(shape=(2 * 2 * N_LAT * N_LON, ), dtype=tf.float32),
#     tf.TensorSpec(shape=(2 * N_LAT * N_LON, ), dtype=tf.float32),
#     tf.TensorSpec(shape=(1, ), dtype=tf.float32),
#     tf.TensorSpec(shape=(1, ), dtype=tf.float32)
# ], jit_compile=True)
def euler_integrator(nsteps, sampling_period, time_step, y_init, x0, tau, K,
                     SC, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON,
                     unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
                     vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                     P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
                     P_l_0_costheta, cos_0_phidb):
    print("euler_integrator()")
    y = tf.TensorArray(dtype=tf.float32, size=nsteps)
    y_next = y_init
    for i in tf.range(nsteps, dtype=tf.int32):
        for _ in tf.range(sampling_period / time_step):
            y_next = y_next + time_step * epileptor2D_nf_ode_fn(
                y_next, x0, tau, K, SC, glq_wts, P_l_m_costheta, Dll, N_LAT,
                N_LON, unkown_roi_mask, rgn_map_reg_sorted, low_idcs,
                high_idcs, vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
                cos_0_phidb)
        y = y.write(i, y_next)
    return y.stack()


# # @tf.function
# def rk4_integrator(ode_fn, nsteps, sampling_period, time_step, y_init, x0, tau,
#                    K):
#     y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
#     y_next = y_init
#     h = time_step
#     for i in tf.range(nsteps, dtype=tf.int32):
#         for j in tf.range(sampling_period / h):
#             k1 = ode_fn(0.0, y_next, x0, tau, K)
#             k2 = ode_fn(0.0, y_next + h * (k1 / 2), x0, tau, K)
#             k3 = ode_fn(0.0, y_next + h * (k2 / 2), x0, tau, K)
#             k4 = ode_fn(0.0, y_next + h * k3, x0, tau, K)
#             y_next = y_next + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
#         y = y.write(i, y_next)
#     return y.stack()

# # @tf.function
# def bdf_integrator(ode_fn, t_init, y_init, solution_times, constants):
#     return tfp.math.ode.BDF(atol=1e-4, rtol=1e-3, validate_args=True).solve(
#         ode_fn=ode_fn,
#         initial_time=t_init,
#         initial_state=y_init,
#         solution_times=solution_times,
#         constants=constants)

# # @tf.function
# def dormandprince_integrator(ode_fn, t_init, y_init, solution_times,
#                              constants):
#     return tfp.math.ode.DormandPrince(atol=1e-4, rtol=1e-3,
#                                       validate_args=True).solve(
#                                           ode_fn=ode_fn,
#                                           initial_time=t_init,
#                                           initial_state=y_init,
#                                           solution_times=solution_times,
#                                           constants=constants)

# %%
t_init = tf.constant(0.0, dtype=tf.float32)
x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(2*N_LAT*N_LON, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(2*N_LAT*N_LON, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=(1, ))
K_true = tf.constant(1.0, dtype=tf.float32, shape=(1, ))
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
# time_step = tf.constant(0.1, dtype=tf.float32)
# solution_times = time_step * tf.range(0, 300, dtype=tf.float32)
# start_time = time.time()
# y_true = dormandprince_integrator(epileptor2D_nf_ode_fn, t_init, y_init_true,
#                                   solution_times, {
#                                       'x0': x0_true,
#                                       'tau': tau_true,
#                                       'K': K_true
#                                   })
# print(f"Time elapsed: {time.time() - start_time} seconds")
# x_true = y_true.states[:, :nv].numpy()
# z_true = y_true.states[:, nv:].numpy()
# ode_solver = "DormandPrince Integration"
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
# nsteps = 300
# sampling_period = 0.1
# time_step = 0.01
# start_time = time.time()
# y_true = euler_integrator(epileptor2D_nf_ode_fn, nsteps, sampling_period,
#                           time_step, y_init_true, x0_true, tau_true, K_true)
# print(f"Time elapsed: {time.time() - start_time} seconds")

# x_true = y_true[:, :nv].numpy()
# z_true = y_true[:, nv:].numpy()
# ode_solver = "Euler Integartion"
# %%
nsteps = tf.constant(150, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.01, dtype=tf.float32)

# # @tf.function
# def test_ode_fn(x):
#     alpha = tf.constant(1.0, dtype=tf.float32)
#     theta = tf.constant(-1.0, dtype=tf.float32)
#     gamma_lc = tf.constant(1.0, dtype=tf.float32)
#     x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
#     local_cplng = local_coupling(x_hat)
#     # x_sorted = tf.gather(x, rgn_map_reg_sorted)
#     # x_roi = tfp.stats.windowed_mean(x_sorted, low_idcs, high_idcs)
#     # # tf.print(x_hat_roi.shape)
#     # global_cplng_roi = tf.reduce_sum(
#     #     K * SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]), axis=1)
#     # global_cplng_vrtcs = tf.gather(global_cplng_roi, vrtx_roi_map)
#     dx = 1 + gamma_lc * local_cplng
#     return dx

# # @tf.function
# def test_integrator(ode_fn, nsteps, sampling_period, time_step, y_init, x0,
#                     tau, K):
#     y = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
#     y_next = y_init
#     # for i in tf.range(nsteps, dtype=tf.int32):
#     #     for j in tf.range(sampling_period / time_step):
#     #         y_next = y_next + time_step * ode_fn(0.0, y_next, x0, tau, K)
#     #     y = y.write(i, y_next)
#     # return y.stack()
#     for i in tf.range(nsteps):
#         dydt = ode_fn(0.0, y_next, x0, tau, K)
#         # y_next = y_next + time_step * ode_fn(0.0, y_next, x0, tau, K)
#         y = y.write(i, y_next + dydt)
#     # i = tf.constant(0)

#     # @tf.function
#     # def cond(i, y_next, y):
#     #     return tf.less(i, nsteps)

#     # @tf.function
#     # def body(i, y_next, y):
#     #     y_next = y_next + time_step * ode_fn(0.0, y_next, x0, tau, K)
#     #     return (i + 1, y_next, y.write(i, y_next))

#     # i, y_next, y = tf.while_loop(cond, body, [i, y_next, y])

#     # y_next = y_init + time_step * ode_fn(0.0, y_init, x0, tau, K)
#     # y = y.write(0, y_next)
#     r = y.stack()
#     return r

# integrator_tf_graph = tf.autograph.to_graph(euler_integrator, recursive=True)
# integrator_tf_fn = tf.function(euler_integrator, jit_compile=True)
# input_signature=[
#     tf.TensorSpec(shape=(), dtype=tf.int32),
#     tf.TensorSpec(shape=(), dtype=tf.float32),
#     tf.TensorSpec(shape=(), dtype=tf.float32),
#     tf.TensorSpec(shape=(2 * 2 * N_LAT * N_LON, ), dtype=tf.float32),
#     tf.TensorSpec(shape=(2 * N_LAT * N_LON, ), dtype=tf.float32),
#     tf.TensorSpec(shape=(1, ), dtype=tf.float32),
#     tf.TensorSpec(shape=(1, ), dtype=tf.float32)
# ])
# integrator_conc_fn = integrator_tf_fn.get_concrete_function(
#     nsteps, sampling_period, time_step, y_init_true, x0_true, tau_true, K_true,
#     SC, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON, unkown_roi_mask,
#     rgn_map_reg_sorted, low_idcs, high_idcs, vrtx_roi_map, glq_wts_real,
#     P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
#     cos_0_phidb)

@tf.function(jit_compile=True)
def get_sim_and_gradients(nsteps, sampling_period, time_step, y_init_true,
                          x0_true, tau_true, K_true, SC, glq_wts,
                          P_l_m_costheta, Dll, N_LAT, N_LON, unkown_roi_mask,
                          rgn_map_reg_sorted, low_idcs, high_idcs,
                          vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                          P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
                          P_l_0_costheta, cos_0_phidb):
    with tf.GradientTape() as tape:
        with tf.device("GPU:0"):
            tape.watch([x0_true, K_true, tau_true, y_init_true])
            y = euler_integrator(nsteps, sampling_period, time_step,
                                 y_init_true, x0_true, tau_true, K_true, SC,
                                 glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON,
                                 unkown_roi_mask, rgn_map_reg_sorted, low_idcs,
                                 high_idcs, vrtx_roi_map, glq_wts_real,
                                 P_l_1tom_Dll, P_l_1tom_costheta,
                                 cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
                                 cos_0_phidb)
            return y, tape.gradient(y, tau_true)
        # tape.watch([y_init_true])
        # y_true = test_integrator(epileptor2D_nf_ode_fn, nsteps,
        #                          sampling_period, time_step, y_init_true,
        #                          x0_true, tau_true, K_true)
        # return y_true, tape.gradient(y_true, [y_init_true])


start_time = time.time()
# get_sim_and_gradients_tf_fn = tf.function(get_sim_and_gradients,
#                                           jit_compile=True)
# get_sim_and_gradients_conc_fn = get_sim_and_gradients_tf_fn.get_concrete_function(
#     nsteps, sampling_period, time_step, y_init_true, x0_true, tau_true, K_true,
#     SC, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON, unkown_roi_mask,
#     rgn_map_reg_sorted, low_idcs, high_idcs, vrtx_roi_map, glq_wts_real,
#     P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
#     cos_0_phidb)

y_true, y_tau_grad = get_sim_and_gradients(
    nsteps, sampling_period, time_step, y_init_true, x0_true, tau_true, K_true,
    SC, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON, unkown_roi_mask,
    rgn_map_reg_sorted, low_idcs, high_idcs, vrtx_roi_map, glq_wts_real,
    P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
    cos_0_phidb)
print(f"Time elapsed: {time.time() - start_time} seconds")
x_true = y_true[:, :nv].numpy()
z_true = y_true[:, nv:].numpy()
ode_solver = "Euler Integartion"
tf.print(y_true, y_tau_grad)
# %%
# nsteps = 300
# sampling_period = 0.1
# time_step = 0.01
# start_time = time.time()
# y_true = rk4_integrator(epileptor2D_nf_ode_fn, nsteps, sampling_period,
#                         time_step, y_init_true, x0_true, tau_true, K_true)
# print(f"Time elapsed: {time.time() - start_time} seconds")

# x_true = y_true[:, :N_LAT * N_LON].numpy()
# z_true = y_true[:, N_LAT * N_LON:].numpy()
# ode_solver = "RK4 Integration"
# # %%

# time_step = tf.constant(0.1, dtype=tf.float32)
# solution_times = time_step * tf.range(0, 300, dtype=tf.float32)
# start_time = time.time()
# with tf.GradientTape() as tape:
#     tape.watch([x0_true, K_true, tau_true])
#     y_true = tfp.math.ode.DormandPrince(atol=1e-4,
#                                         rtol=1e-3,
#                                         validate_args=True).solve(
#                                             ode_fn=epileptor2D_nf_ode_fn,
#                                             initial_time=t_init,
#                                             initial_state=y_init_true,
#                                             solution_times=solution_times,
#                                             constants={
#                                                 'x0': x0_true,
#                                                 'tau': tau_true,
#                                                 'K': K_true
#                                             })

# # y_true = tfp.math.ode.DormandPrince(atol=1e-4,
# #                                     rtol=1e-3,
# #                                     validate_args=True).solve(
# #                                         ode_fn=epileptor2D_nf_ode_fn,
# #                                         initial_time=t_init,
# #                                         initial_state=y_init_true,
# #                                         solution_times=solution_times,
# #                                         constants={
# #                                             'x0': x0_true,
# #                                             'tau': tau_true,
# #                                             'K': K_true
# #                                         })
# print(f"Time elapsed: {time.time() - start_time} seconds")
# # x0_grads = tape.gradient(y_true.states, x0_true)
# x_true = y_true.states[:, :nv].numpy()
# z_true = y_true.states[:, nv:].numpy()
# ode_solver = "DormandPrince Integration"

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
