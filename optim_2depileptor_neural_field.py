# %%
import numpy as np
import tensorflow as tf

# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import lib.utils.sht as tfsht
import lib.utils.projector
import time
from lib.plots.neuralfield import create_video
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
D = tf.constant(0.01, dtype=tf.float32)
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

# %%


@tf.custom_gradient
def local_coupling(
    x,
    glq_wts,
    P_l_m_costheta,
    Dll,
    L_MAX,
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
    print("local_coupling()...")
    x_hat_lh = tf.stop_gradient(tf.reshape(x[0:N_LAT * N_LON], (N_LAT, N_LON)))
    x_hat_rh = tf.stop_gradient(tf.reshape(x[N_LAT * N_LON:], (N_LAT, N_LON)))
    x_lm_lh = tf.stop_gradient(
        tfsht.analys(L_MAX, N_LON, x_hat_lh, glq_wts, P_l_m_costheta))
    x_lm_hat_lh = tf.stop_gradient(-1.0 * Dll[:, tf.newaxis] * x_lm_lh)
    x_lm_rh = tf.stop_gradient(
        tfsht.analys(L_MAX, N_LON, x_hat_rh, glq_wts, P_l_m_costheta))
    x_lm_hat_rh = tf.stop_gradient(-1.0 * Dll[:, tf.newaxis] * x_lm_rh)
    local_cplng_lh = tf.stop_gradient(
        tf.reshape(tfsht.synth(N_LON, x_lm_hat_lh, P_l_m_costheta), [-1]))
    local_cplng_rh = tf.stop_gradient(
        tf.reshape(tfsht.synth(N_LON, x_lm_hat_rh, P_l_m_costheta), [-1]))
    local_cplng = tf.stop_gradient(
        tf.concat((local_cplng_lh, local_cplng_rh), axis=0))

    def grad(upstream):
        upstream_lh = tf.reshape(upstream[0:N_LAT * N_LON], (N_LAT, N_LON))
        upstream_rh = tf.reshape(upstream[N_LAT * N_LON:], (N_LAT, N_LON))
        glq_wts_grad = None
        P_l_m_costheta_grad = None
        Dll_grad = None
        L_MAX_grad = None
        N_LAT_grad = None
        N_LON_grad = None
        glq_wts_real_grad = None
        P_l_1tom_Dll_grad = None
        P_l_1tom_costheta_grad = None

        cos_1tom_phidb_grad = None
        P_l_0_Dll_grad = None
        P_l_0_costheta_grad = None
        cos_0_phidb_grad = None

        g_lh = -1.0 * tf.einsum("cd,a,lmc,lma,mdb->ab",
                                upstream_lh,
                                glq_wts_real,
                                P_l_1tom_Dll,
                                P_l_1tom_costheta,
                                cos_1tom_phidb,
                                optimize="optimal") - tf.einsum(
                                    "cd,a,lc,la,db->ab",
                                    upstream_lh,
                                    glq_wts_real,
                                    P_l_0_Dll,
                                    P_l_0_costheta,
                                    cos_0_phidb,
                                    optimize="optimal")
        g_rh = -1.0 * tf.einsum("cd,a,lmc,lma,mdb->ab",
                                upstream_rh,
                                glq_wts_real,
                                P_l_1tom_Dll,
                                P_l_1tom_costheta,
                                cos_1tom_phidb,
                                optimize="optimal") - tf.einsum(
                                    "cd,a,lc,la,db->ab",
                                    upstream_rh,
                                    glq_wts_real,
                                    P_l_0_Dll,
                                    P_l_0_costheta,
                                    cos_0_phidb,
                                    optimize="optimal")
        # g = tf.clip_by_norm(
        #     tf.concat((tf.reshape(g_lh, [-1]), tf.reshape(g_rh, [-1])),
        #               axis=0), 100)
        g = tf.concat((tf.reshape(g_lh, [-1]), tf.reshape(g_rh, [-1])), axis=0)
        return [
            g, glq_wts_grad, P_l_m_costheta_grad, Dll_grad, L_MAX_grad,
            N_LAT_grad, N_LON_grad, glq_wts_real_grad, P_l_1tom_Dll_grad,
            P_l_1tom_costheta_grad, cos_1tom_phidb_grad, P_l_0_Dll_grad,
            P_l_0_costheta_grad, cos_0_phidb_grad
        ]

    return local_cplng, grad


# @tf.function
def epileptor2D_nf_ode_fn(y, x0, tau, K, SC, glq_wts, P_l_m_costheta, Dll,
                          L_MAX, N_LAT, N_LON, unkown_roi_mask,
                          rgn_map_reg_sorted, low_idcs, high_idcs,
                          vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                          P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
                          P_l_0_costheta, cos_0_phidb):
    print("epileptor2d_nf_ode_fn()...")
    nv = 2 * N_LAT * N_LON
    x = y[0:nv]
    z = y[nv:2 * nv]
    I1 = tf.constant(4.1, dtype=tf.float32)
    # NOTE: alpha > 7.0 is causing DormandPrince integrator to diverge
    alpha = tf.constant(1.0, dtype=tf.float32)
    theta = tf.constant(-1.0, dtype=tf.float32)
    gamma_lc = tf.constant(1.0, dtype=tf.float32)
    x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
    local_cplng = local_coupling(x_hat, glq_wts, P_l_m_costheta, Dll, L_MAX,
                                 N_LAT, N_LON, glq_wts_real, P_l_1tom_Dll,
                                 P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
                                 P_l_0_costheta, cos_0_phidb)
    x_sorted = tf.gather(x, rgn_map_reg_sorted)
    x_roi = tfp.stats.windowed_mean(x_sorted, low_idcs, high_idcs)
    # tf.print(x_hat_roi.shape)
    global_cplng_roi = tf.reduce_sum(
        K[0] * SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]), axis=1)
    global_cplng_vrtcs = tf.gather(global_cplng_roi, vrtx_roi_map)
    dx = (1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - z +
          I1) * unkown_roi_mask
    dz = ((1.0 / tau[0]) * (4 * (x - x0) - z - global_cplng_vrtcs -
                            gamma_lc * local_cplng)) * unkown_roi_mask
    return tf.concat((dx, dz), axis=0)
    # return y * tf.constant(-0.1, dtype=tf.float32)


# %%
# # Test custom gradients

# def local_coupling_wocg(
#     x,
#     glq_wts,
#     P_l_m_costheta,
#     Dll,
#     L_MAX,
#     N_LAT,
#     N_LON,
# ):
#     # print("local_coupling()")
#     x_hat_lh = tf.reshape(x[0:N_LAT * N_LON], (N_LAT, N_LON))
#     x_hat_rh = tf.reshape(x[N_LAT * N_LON:], (N_LAT, N_LON))
#     x_lm_lh = tfsht.analys(L_MAX, N_LON, x_hat_lh, glq_wts, P_l_m_costheta)
#     x_lm_hat_lh = Dll[:, tf.newaxis] * x_lm_lh
#     x_lm_rh = tfsht.analys(L_MAX, N_LON, x_hat_rh, glq_wts, P_l_m_costheta)
#     x_lm_hat_rh = Dll[:, tf.newaxis] * x_lm_rh
#     local_cplng_lh = tf.reshape(
#         tfsht.synth(N_LON, x_lm_hat_lh, P_l_m_costheta), [-1])
#     local_cplng_rh = tf.reshape(
#         tfsht.synth(N_LON, x_lm_hat_rh, P_l_m_costheta), [-1])
#     local_cplng = tf.concat((local_cplng_lh, local_cplng_rh), axis=0)

#     return local_cplng

# def find_grad(x):
#     with tf.GradientTape() as tape:
#         tape.watch(x)

#         alpha = tf.constant(1.0, dtype=tf.float32)
#         theta = tf.constant(-1.0, dtype=tf.float32)
#         x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
#         lc = local_coupling(x_hat, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
#                             glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
#                             cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
#                             cos_0_phidb)
#         return lc, tape.gradient(lc, x)

# def find_grad_wocg(x):
#     with tf.GradientTape() as tape:
#         tape.watch(x)
#         alpha = tf.constant(1.0, dtype=tf.float32)
#         theta = tf.constant(-1.0, dtype=tf.float32)
#         x_hat = tf.math.sigmoid(alpha * (x - theta)) * unkown_roi_mask
#         lc = local_coupling_wocg(x_hat, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT,
#                                  N_LON)
#         return lc, tape.gradient(lc, x)

# x = tf.constant(-2.0, dtype=tf.float32) * \
# tf.ones(2*N_LAT*N_LON, dtype=tf.float32)

# # thrtcl, nmrcl = tf.test.compute_gradient(local_coupling, [
# #     x_hat, glq_wts, P_l_m_costheta, Dll, N_LAT, N_LON, glq_wts_real,
# #     P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
# #     cos_0_phidb
# # ])
# lc1, x_hat_grad_cg = find_grad(x)
# lc2, x_hat_grad_wocg = find_grad_wocg(x)
# print(tf.reduce_max(tf.math.abs(x_hat_grad_cg - x_hat_grad_wocg)))

# %%


# NOTE: setting jit_compile=True is causing OOM
# @tf.function
def euler_integrator(nsteps, nsubsteps, time_step, y_init, x0, tau, K, SC,
                     glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
                     unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
                     vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                     P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
                     P_l_0_costheta, cos_0_phidb):
    print("euler_integrator()...")
    y = tf.TensorArray(dtype=tf.float32, size=nsteps)
    y_next = y_init
    cond1 = lambda i, y, y_next: tf.less(i, nsteps)

    def body1(i, y, y_next):
        j = tf.constant(0)
        cond2 = lambda j, y_next: tf.less(j, nsubsteps)

        def body2(j, y_next):
            y_next = y_next + time_step * epileptor2D_nf_ode_fn(
                y_next, x0, tau, K, SC, glq_wts, P_l_m_costheta, Dll, L_MAX,
                N_LAT, N_LON, unkown_roi_mask, rgn_map_reg_sorted, low_idcs,
                high_idcs, vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
                cos_0_phidb)
            return j + 1, y_next

        j, y_next = tf.while_loop(cond2,
                                  body2, [j, y_next],
                                  maximum_iterations=nsubsteps)

        y = y.write(i, y_next)
        return i + 1, y, y_next

    i = tf.constant(0)
    i, y, y_next = tf.while_loop(cond1,
                                 body1, [i, y, y_next],
                                 maximum_iterations=nsteps)
    return y.stack()

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
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)


@tf.function
def run_sim(nsteps, nsubsteps, time_step, y_init, x0, tau,
            K, SC, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
            unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
            vrtx_roi_map, glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
            cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta, cos_0_phidb):
    y = euler_integrator(nsteps, nsubsteps, time_step, y_init, x0,
                         tau, K, SC, glq_wts, P_l_m_costheta, Dll,
                         L_MAX, N_LAT, N_LON, unkown_roi_mask,
                         rgn_map_reg_sorted, low_idcs, high_idcs, vrtx_roi_map,
                         glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
                         cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
                         cos_0_phidb)
    return y


y_obs = run_sim(nsteps, nsubsteps, time_step, y_init_true, x0_true, tau_true,
                K_true, SC, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
                unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
                vrtx_roi_map, glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
                cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta, cos_0_phidb)


# %%
@tf.function
def get_loss_and_gradients(nsteps, nsubsteps, time_step, y_init, x0_lm_lh_real,
                           x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag, tau, K,
                           SC, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT,
                           N_LON, unkown_roi_mask, rgn_map_reg_sorted,
                           low_idcs, high_idcs, vrtx_roi_map, glq_wts_real,
                           P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb,
                           P_l_0_Dll, P_l_0_costheta, cos_0_phidb,
                           P_l_m_costheta_32, y_obs):
    with tf.GradientTape() as tape:
        with tf.device("GPU:0"):
            x0_lh = tf.reshape(
                tfsht.synth(N_LON, tf.complex(x0_lm_lh_real, x0_lm_lh_imag),
                            P_l_m_costheta_32), [-1])
            x0_rh = tf.reshape(
                tfsht.synth(N_LON, tf.complex(x0_lm_rh_real, x0_lm_rh_imag),
                            P_l_m_costheta_32), [-1])
            x0 = tf.concat([x0_lh, x0_rh], axis=0)
            y_pred = euler_integrator(nsteps, nsubsteps, time_step, y_init, x0,
                                      tau, K, SC, glq_wts, P_l_m_costheta, Dll,
                                      L_MAX, N_LAT, N_LON, unkown_roi_mask,
                                      rgn_map_reg_sorted, low_idcs, high_idcs,
                                      vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
                                      P_l_1tom_costheta, cos_1tom_phidb,
                                      P_l_0_Dll, P_l_0_costheta, cos_0_phidb)
            nv = 2 * N_LAT * N_LON
            x_pred = y_pred[:, 0:nv]
            x_obs = y_obs[:, 0:nv]
            loss = tf.reduce_sum(tf.pow(x_obs - x_pred, 4))
            return loss, tape.gradient(
                loss,
                [x0_lm_lh_real, x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag])


# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# %%
_, _, cos_theta_32, glq_wts_32, P_l_m_costheta_32 = tfsht.prep(
    32, N_LAT, N_LON)

x0_lh = -3.0 * tf.ones(nvph, dtype=tf.float32)
x0_rh = -3.0 * tf.ones(nvph, dtype=tf.float32)
x0_lm_lh = tfsht.analys(32, N_LON, tf.reshape(x0_lh, [N_LAT, N_LON]),
                        glq_wts_32, P_l_m_costheta_32)
x0_lm_rh = tfsht.analys(32, N_LON, tf.reshape(x0_rh, [N_LAT, N_LON]),
                        glq_wts_32, P_l_m_costheta_32)
x0_lm_lh_real = tf.Variable(initial_value=tf.math.real(x0_lm_lh),
                            dtype=tf.float32)
x0_lm_lh_imag = tf.Variable(initial_value=tf.math.imag(x0_lm_lh),
                            dtype=tf.float32)
x0_lm_rh_real = tf.Variable(initial_value=tf.math.real(x0_lm_rh),
                            dtype=tf.float32)
x0_lm_rh_imag = tf.Variable(initial_value=tf.math.imag(x0_lm_rh),
                            dtype=tf.float32)

# %%


@tf.function
def train_loop(nsteps, nsubsteps, time_step, y_init, x0_lm_lh_real,
               x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag, tau, K, SC,
               glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
               unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
               vrtx_roi_map, glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
               cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta, cos_0_phidb,
               P_l_m_costheta_32, y_obs, niters):
    for i in range(niters):
        loss, x0_lm_grads = get_loss_and_gradients(
            nsteps, nsubsteps, time_step, y_init, x0_lm_lh_real, x0_lm_lh_imag,
            x0_lm_rh_real, x0_lm_rh_imag, tau, K, SC, glq_wts, P_l_m_costheta,
            Dll, L_MAX, N_LAT, N_LON, unkown_roi_mask, rgn_map_reg_sorted,
            low_idcs, high_idcs, vrtx_roi_map, glq_wts_real, P_l_1tom_Dll,
            P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta,
            cos_0_phidb, P_l_m_costheta_32, y_obs)
        tf.print(loss)
        optimizer.apply_gradients(
            zip(x0_lm_grads,
                [x0_lm_lh_real, x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag]))

# %%

start_time = time.time()
niters = tf.constant(5000, dtype=tf.int32)
train_loop(nsteps, nsubsteps, time_step, y_init_true, x0_lm_lh_real,
           x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag, tau_true, K_true, SC,
           glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON, unkown_roi_mask,
           rgn_map_reg_sorted, low_idcs, high_idcs, vrtx_roi_map, glq_wts_real,
           P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb, P_l_0_Dll,
           P_l_0_costheta, cos_0_phidb, P_l_m_costheta_32, y_obs, niters)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
# %%
x0_lh = tf.reshape(
    tfsht.synth(N_LON, tf.complex(x0_lm_lh_real, x0_lm_lh_imag),
                P_l_m_costheta_32), [-1])
x0_rh = tf.reshape(
    tfsht.synth(N_LON, tf.complex(x0_lm_rh_real, x0_lm_rh_imag),
                P_l_m_costheta_32), [-1])
x0 = tf.concat([x0_lh, x0_rh], axis=0)
y_test = run_sim(nsteps, nsubsteps, time_step, y_init_true, x0, tau_true,
                 K_true, SC, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
                 unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
                 vrtx_roi_map, glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
                 cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta, cos_0_phidb)
x_test = y_test[:, :nv].numpy()
out_dir = 'results/exp28/figures/infer'
create_video(x_test, N_LAT.numpy(), N_LON.numpy(), out_dir)

# %%
x_obs = y_obs[:, 0:nv]
out_dir = 'results/exp28/figures/ground_truth'
create_video(x_obs, N_LAT.numpy(), N_LON.numpy(), out_dir)

# %%
out_dir = 'results/exp28/figures'
fig_fname = 'x0_infer_vs_gt.png'
fs_small = 5
fs_med = 7
plt.figure(dpi=200, figsize=(7, 4))
x0_lh_gt = np.reshape(x0_true[0:nvph], (N_LAT, N_LON))
x0_rh_gt = np.reshape(x0_true[nvph:], (N_LAT, N_LON))
x0_lh_infr = np.reshape(x0.numpy()[0:nvph], (N_LAT, N_LON))
x0_rh_infr = np.reshape(x0.numpy()[nvph:], (N_LAT, N_LON))
clim_min = np.min([np.min(x0_true), np.min(x0)])
clim_max = np.max([np.max(x0_true), np.max(x0)])
plt.subplot(221)
plt.imshow(x0_lh_gt, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Ground Truth - Left hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.subplot(222)
plt.imshow(x0_rh_gt, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Ground Truh - Right hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)

plt.subplot(223)
plt.imshow(x0_lh_infr, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Inferred - Left hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.subplot(224)
plt.imshow(x0_rh_infr, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Inferred - Right hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.tight_layout()
plt.savefig(f"{out_dir}/{fig_fname}", facecolor='white')
# %%
