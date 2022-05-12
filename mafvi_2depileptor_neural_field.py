# %%
import numpy as np
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import lib.utils.sht as tfsht
import lib.utils.projector
import time
import lib.utils.tnsrflw
from lib.plots.neuralfield import create_video
import os
# %%
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.autograph.set_verbosity(10)
# %%
results_dir = 'exp33'
os.makedirs(results_dir, exist_ok=True)
figs_dir = f'{results_dir}/figures'
os.makedirs(figs_dir, exist_ok=True)
L_MAX = 32
N_LAT = 129
N_LON = 257
N_LAT, N_LON, cos_theta, glq_wts, P_l_m_costheta = tfsht.prep(
    L_MAX, N_LAT, N_LON)
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
rgn_map_reg = lib.utils.projector.find_rgn_map_reg(
    N_LAT=N_LAT.numpy(),
    N_LON=N_LON.numpy(),
    cos_theta=cos_theta,
    verts_irreg_fname=verts_irreg_fname,
    rgn_map_irreg_fname=rgn_map_irreg_fname)
unkown_roi_idcs = np.nonzero(rgn_map_reg == 0)[0]
unkown_roi_mask = np.ones(nv)
unkown_roi_mask[unkown_roi_idcs] = 0
unkown_roi_mask = tf.constant(unkown_roi_mask, dtype=tf.float32)

idcs_nbrs_irreg = lib.utils.projector.find_nbrs_irreg_sphere(
    N_LAT=N_LAT.numpy(),
    N_LON=N_LON.numpy(),
    cos_theta=cos_theta,
    verts_irreg_fname=verts_irreg_fname)

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
    low_idcs.append(roi_idcs[0] if roi_idcs.ndim > 0 else roi_idcs)
    high_idcs.append(roi_idcs[-1] + 1 if roi_idcs.ndim > 0 else roi_idcs + 1)

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
    # tf.print("tau = ", tau)
    global_cplng_roi = tf.reduce_sum(
        K * SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]), axis=1)
    global_cplng_vrtcs = tf.gather(global_cplng_roi, vrtx_roi_map)
    dx = (1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - z +
          I1) * unkown_roi_mask
    dz = ((1.0 / tau) * (4 * (x - x0) - z - global_cplng_vrtcs -
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


x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(2*N_LAT*N_LON, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(2*N_LAT*N_LON, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(2 * N_LAT * N_LON)
ez_hyp_roi = [116, 127, 151]
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
gain_irreg = np.load(
    'datasets/id004_bj_jd/tvb/gain_inv_square_ico7.npz')['gain_inv_square']
# Remove subcortical vertices
# NOTE: This won't be necessary once subcortical regions are
# included in simulation
num_verts_irreg = np.loadtxt(verts_irreg_fname).shape[0]
gain_irreg = gain_irreg[:, 0:num_verts_irreg]
gain_reg = tf.constant(gain_irreg[:, idcs_nbrs_irreg].T, dtype=tf.float32)
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
x_obs = y_obs[:, 0:nv] * unkown_roi_mask
slp_obs = tf.math.log(tf.matmul(tf.math.exp(x_obs), gain_reg))

# %%
plt.figure(figsize=(7, 6), dpi=200)
plt.imshow(tf.transpose(slp_obs), interpolation=None, aspect='auto', cmap='inferno')
plt.xlabel('Time')
plt.ylabel('Sensor')
# plt.plot(slp_obs, color='black', alpha=0.3);
plt.colorbar(fraction=0.02)
plt.savefig(f'{figs_dir}/obs_slp.png')
# %%
L_MAX_params = tf.constant(10, dtype=tf.int32)
_, _, cos_theta_params, glq_wts_params, P_l_m_costheta_params = tfsht.prep(
    L_MAX_params, N_LAT, N_LON)

# x0_lh = -3.0 * tf.ones(nvph, dtype=tf.float32)
# x0_rh = -3.0 * tf.ones(nvph, dtype=tf.float32)
# x0_lm_lh = tfsht.analys(32, N_LON, tf.reshape(x0_lh, [N_LAT, N_LON]),
#                         glq_wts_params, P_l_m_costheta_params)
# x0_lm_rh = tfsht.analys(32, N_LON, tf.reshape(x0_rh, [N_LAT, N_LON]),
#                         glq_wts_params, P_l_m_costheta_params)
# x0_lm_lh_real = tf.reshape(tf.math.real(x0_lm_lh), [-1])
# x0_lm_rh_real = tf.reshape(tf.math.real(x0_lm_rh), [-1])
# x0_lm_lh_imag = tf.reshape(tf.math.imag(x0_lm_lh), [-1])
# x0_lm_rh_imag = tf.reshape(tf.math.imag(x0_lm_rh), [-1])
# tau = tf.constant(50, dtype=tf.float32, shape=(1,))
# theta = tf.Variable(initial_value=tf.concat([
#     x0_lm_lh_real, x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag, tau],
#                                             axis=0),
#                     dtype=tf.float32)
# theta = tf.Variable(initial_value=tau, dtype=tf.float32)
# theta = tf.Variable(initial_value=tf.zeros(4*((L_MAX_params + 1)**2) + 1))

# %%
num_bijectors = 4
nparams = 4 * ((L_MAX_params + 1)**2)
num_hidden = 2 * nparams

bijectors = []


for i in range(num_bijectors - 1):
    made = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=[num_hidden, num_hidden],
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(0.1),
        validate_args=True)
    maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made,
                                       validate_args=True)
    bijectors.append(maf)
    bijectors.append(
        tfb.Permute(permutation=tf.random.shuffle(tf.range(nparams))))
    bijectors.append(tfb.BatchNormalization())

made = tfb.AutoregressiveNetwork(
    params=2,
    hidden_units=[num_hidden],
    activation='linear',
    kernel_initializer=tf.keras.initializers.VarianceScaling(0.1),
    validate_args=True)
maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made,
                                   validate_args=True)
bijectors.append(maf)
chained_maf = tfb.Chain(list(reversed(bijectors)))
base_dist = tfd.Independent(tfd.Normal(loc=tf.zeros(nparams, dtype=tf.float32),
                                       scale=1.0 *
                                       tf.ones(nparams, dtype=tf.float32),
                                       name='Base_Distribution'),
                            reinterpreted_batch_ndims=1)
flow_dist = tfd.TransformedDistribution(distribution=base_dist,
                                        bijector=chained_maf,
                                        name='Variational_Posterior')


# %%
x0_prior_mu = -3.0 * tf.ones(nv)


@tf.function
def log_prob(theta, slp_obs):
    nv = 2 * N_LAT * N_LON
    nmodes = tf.pow(L_MAX_params + 1, 2)
    eps = tf.constant(0.1, dtype=tf.float32)
    tf.print("nan in theta", tf.reduce_any(tf.math.is_nan(theta)))
    x0_lh = tf.reshape(
        tfsht.synth(
            N_LON,
            tf.reshape(tf.complex(theta[0:nmodes], theta[nmodes:2 * nmodes]),
                       [L_MAX_params + 1, L_MAX_params + 1]),
            P_l_m_costheta_params), [-1])
    x0_rh = tf.reshape(
        tfsht.synth(
            N_LON,
            tf.reshape(
                tf.complex(theta[2 * nmodes:3 * nmodes],
                           theta[3 * nmodes:4 * nmodes]),
                [L_MAX_params + 1, L_MAX_params + 1]), P_l_m_costheta_params),
        [-1])
    x0 = tf.concat([x0_lh, x0_rh], axis=0)
    tf.print("nan in x0", tf.reduce_any(tf.math.is_nan(x0)))
    x0_trans = lib.utils.tnsrflw.sigmoid_transform(
        x0, tf.constant(-5.0, dtype=tf.float32),
        tf.constant(-1.0, dtype=tf.float32)) * unkown_roi_mask
    tf.print("nan in x0_trans", tf.reduce_any(tf.math.is_nan(x0_trans)))
    # tau = theta[4 * nmodes]
    # tau_trans = lib.utils.tnsrflw.sigmoid_transform(
    #     tau, tf.constant(15, dtype=tf.float32),
    #     tf.constant(100, dtype=tf.float32))
    y_pred = euler_integrator(nsteps, nsubsteps, time_step, y_init_true,
                              x0_trans, tau_true, K_true, SC, glq_wts,
                              P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
                              unkown_roi_mask, rgn_map_reg_sorted, low_idcs,
                              high_idcs, vrtx_roi_map, glq_wts_real,
                              P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb,
                              P_l_0_Dll, P_l_0_costheta, cos_0_phidb)
    x_pred = y_pred[:, 0:nv] * unkown_roi_mask
    tf.print("nan in x_pred", tf.reduce_any(tf.math.is_nan(x_pred)))
    slp_mu = tf.math.log(tf.matmul(tf.math.exp(x_pred), gain_reg))
    # tf.print("Nan in x_mu", tf.reduce_any(tf.math.is_nan(x_mu)))
    likelihood = tf.reduce_sum(
        tfd.Normal(loc=slp_mu, scale=eps).log_prob(slp_obs))
    # x0_prior = tf.reduce_sum(tfd.Normal(loc=0.0, scale=5.0).log_prob(theta))
    x0_prior = tf.reduce_sum(
        tfd.Normal(loc=x0_prior_mu, scale=0.5).log_prob(x0_trans))
    lp = likelihood + x0_prior
    tf.print("likelihood = ", lp, "prior = ", x0_prior)
    return lp


@tf.function
def loss(slp_obs):
    nsamples = 1
    posterior_samples = flow_dist.sample(nsamples)
    # tf.print(posterior_samples)
    # tf.print(posterior_samples)
    # nsamples = base_dist_samples.shape[0]
    # loss_val = tf.reduce_sum(
    #     flow_dist.log_prob(posterior_samples) / nsamples)
    loss_val = tf.constant(0.0, shape=(1,), dtype=tf.float32)
    for theta in posterior_samples:
        # tf.print("theta: ", theta, summarize=-1)
        gm_log_prob = log_prob(theta, slp_obs)
        posterior_approx_log_prob = flow_dist.log_prob(theta[tf.newaxis, :])
        tf.print("gm_log_prob:", gm_log_prob, "\tposterior_approx_log_prob:",
                 posterior_approx_log_prob)
        loss_val += (posterior_approx_log_prob - gm_log_prob) / nsamples
        # tf.print("loss_val: ", loss_val)
    return loss_val


@tf.function
def get_loss_and_gradients(slp_obs):
    with tf.GradientTape() as tape:
        loss_val = loss(slp_obs)
        return loss_val, tape.gradient(loss_val,
                                       flow_dist.trainable_variables)


# %%
# initial_learning_rate = 1e-3
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100,
#     decay_rate=0.96,
#     staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


# %%
# @tf.function
def train_loop(num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        # base_dist_samples = base_dist.sample(batch_size)
        loss_value, grads = get_loss_and_gradients(slp_obs)
        nan_in_grads = tf.reduce_any(
            [tf.reduce_any(tf.math.is_nan(el)) for el in grads])
        tf.print("nan in grads", nan_in_grads)
        # grads = [tf.divide(el, batch_size) for el in grads]
        grads = [tf.clip_by_norm(el, 1000) for el in grads]
        # tf.print("gradient norm = ", [tf.norm(el) for el in grads], \
        # output_stream="file://debug.log")
        tf.print("Epoch ", epoch, "loss: ", loss_value)
        # training_loss.append(loss_value)
        optimizer.apply_gradients(zip(grads, flow_dist.trainable_variables))


# %%

# batch_size = 1
# base_dist_samples = tf.data.Dataset.from_tensor_slices(
#     base_dist.sample((1))).batch(batch_size)
# %%
num_epochs = 200
start_time = time.time()
train_loop(num_epochs)
print(f"Elapsed {time.time() - start_time} seconds for {num_epochs} Epochs")
# %%
x0_lh = lib.utils.tnsrflw.inv_sigmoid_transform(
    x0_true[0:nvph], tf.constant(-5.0, dtype=tf.float32),
    tf.constant(-1.0, dtype=tf.float32))
x0_rh = lib.utils.tnsrflw.inv_sigmoid_transform(
    x0_true[nvph:], tf.constant(-5.0, dtype=tf.float32),
    tf.constant(-1.0, dtype=tf.float32))
x0_lm_lh = tfsht.analys(L_MAX_params, N_LON, tf.reshape(x0_lh, [N_LAT, N_LON]),
                        glq_wts_params, P_l_m_costheta_params)
x0_lm_rh = tfsht.analys(L_MAX_params, N_LON, tf.reshape(x0_rh, [N_LAT, N_LON]),
                        glq_wts_params, P_l_m_costheta_params)
x0_lm_lh_real = tf.reshape(tf.math.real(x0_lm_lh), [-1])
x0_lm_rh_real = tf.reshape(tf.math.real(x0_lm_rh), [-1])
x0_lm_lh_imag = tf.reshape(tf.math.imag(x0_lm_lh), [-1])
x0_lm_rh_imag = tf.reshape(tf.math.imag(x0_lm_rh), [-1])

# tau = lib.utils.tnsrflw.inv_sigmoid_transform(
#     tau_true, tf.constant(15, dtype=tf.float32),
#     tf.constant(100, dtype=tf.float32))

theta_true = tf.concat([
    x0_lm_lh_real, x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag], axis=0)


@tf.function
def get_loss(theta, y_obs):
    nv = 2 * N_LAT * N_LON
    nmodes = tf.pow(L_MAX_params + 1, 2)
    eps = tf.constant(0.1, dtype=tf.float32)
    x0_lh = tf.reshape(
        tfsht.synth(
            N_LON,
            tf.reshape(tf.complex(theta[0:nmodes], theta[nmodes:2 * nmodes]),
                       [L_MAX_params + 1, L_MAX_params + 1]),
            P_l_m_costheta_params), [-1])
    x0_rh = tf.reshape(
        tfsht.synth(
            N_LON,
            tf.reshape(
                tf.complex(theta[2 * nmodes:3 * nmodes],
                           theta[3 * nmodes:4 * nmodes]),
                [L_MAX_params + 1, L_MAX_params + 1]), P_l_m_costheta_params),
        [-1])
    x0 = tf.concat([x0_lh, x0_rh], axis=0)
    x0_trans = lib.utils.tnsrflw.sigmoid_transform(
        x0, tf.constant(-5.0, dtype=tf.float32),
        tf.constant(-1.0, dtype=tf.float32)) * unkown_roi_mask
    # tau = theta[4 * nmodes]
    # tau_trans = lib.utils.tnsrflw.sigmoid_transform(
    #     tau, tf.constant(15, dtype=tf.float32),
    #     tf.constant(100, dtype=tf.float32))
    y_pred = euler_integrator(nsteps, nsubsteps, time_step, y_init_true,
                              x0_trans, tau_true, K_true, SC, glq_wts,
                              P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
                              unkown_roi_mask, rgn_map_reg_sorted, low_idcs,
                              high_idcs, vrtx_roi_map, glq_wts_real,
                              P_l_1tom_Dll, P_l_1tom_costheta, cos_1tom_phidb,
                              P_l_0_Dll, P_l_0_costheta, cos_0_phidb)
    x_mu = y_pred[:, 0:nv] * unkown_roi_mask
    x_obs = y_obs[:, 0:nv] * unkown_roi_mask
    likelihood = tf.reduce_sum(tfd.Normal(loc=x_mu, scale=eps).log_prob(x_obs))
    prior = tf.reduce_sum(tfd.Normal(loc=0.0, scale=5.0).log_prob(theta))
    lp = likelihood + prior
    return x_mu, lp


x_pred, lp = get_loss(theta_true, y_obs)
print(lp)
out_dir = 'tmp1'
create_video(x_pred, N_LAT.numpy(), N_LON.numpy(), out_dir)
# %%
nsamples = 100
posterior_samples = flow_dist.sample(nsamples)
x0_samples = tf.TensorArray(dtype=tf.float32,
                            size=nsamples,
                            clear_after_read=False)
for i, theta in enumerate(posterior_samples.numpy()):
    nmodes = tf.pow(L_MAX_params + 1, 2)
    x0_lh_i = tf.reshape(
        tfsht.synth(
            N_LON,
            tf.reshape(tf.complex(theta[0:nmodes], theta[nmodes:2 * nmodes]),
                       [L_MAX_params + 1, L_MAX_params + 1]),
            P_l_m_costheta_params), [-1])
    x0_rh_i = tf.reshape(
        tfsht.synth(
            N_LON,
            tf.reshape(
                tf.complex(theta[2 * nmodes:3 * nmodes],
                           theta[3 * nmodes:4 * nmodes]),
                [L_MAX_params + 1, L_MAX_params + 1]), P_l_m_costheta_params),
        [-1])
    x0_i = tf.concat([x0_lh_i, x0_rh_i], axis=0)
    x0_trans_i = lib.utils.tnsrflw.sigmoid_transform(
        x0_i, tf.constant(-5.0, dtype=tf.float32),
        tf.constant(-1.0, dtype=tf.float32)) * unkown_roi_mask
    x0_samples = x0_samples.write(i, x0_trans_i)
x0_samples = x0_samples.stack()
x0_mean = tf.reduce_mean(x0_samples, axis=0).numpy()
x0_std = tf.math.reduce_std(x0_samples, axis=0).numpy()
# %%

y_ppc = run_sim(nsteps, nsubsteps, time_step, y_init_true, x0_mean, tau_true,
                K_true, SC, glq_wts, P_l_m_costheta, Dll, L_MAX, N_LAT, N_LON,
                unkown_roi_mask, rgn_map_reg_sorted, low_idcs, high_idcs,
                vrtx_roi_map, glq_wts_real, P_l_1tom_Dll, P_l_1tom_costheta,
                cos_1tom_phidb, P_l_0_Dll, P_l_0_costheta, cos_0_phidb)
x_ppc = y_ppc[:, :nv].numpy()
slp_ppc = tf.math.log(tf.matmul(tf.math.exp(x_ppc), gain_reg))
out_dir = f'{figs_dir}/infer'
create_video(x_ppc, N_LAT.numpy(), N_LON.numpy(), out_dir)

# %%
x_obs = y_obs[:, 0:nv]
out_dir = f'{figs_dir}/ground_truth'
create_video(x_obs, N_LAT.numpy(), N_LON.numpy(), out_dir)

# %%
plt.figure(figsize=(7, 6), dpi=200)
plt.imshow(tf.transpose(slp_ppc),
           interpolation=None,
           aspect='auto',
           cmap='inferno')
plt.xlabel('Time')
plt.ylabel('Sensor')
# plt.plot(slp_obs, color='black', alpha=0.3);
plt.colorbar(fraction=0.02)
plt.savefig(f'{figs_dir}/slp_ppc.png')
# %%
fig_fname = 'x0_gt_vs_inferred_mean_and_std.png'
fs_small = 5
fs_med = 7
x0_mean[np.where(unkown_roi_mask == 0)[0]] = -3.0
# x0_std[np.where(unkown_roi_mask == 0)[0]] = -3.0
# x0_hat[np.where(unkown_roi_mask == 0)[0]] = -3.0
plt.figure(dpi=200, figsize=(7, 4))
x0_lh_gt = np.reshape(x0_true[0:nvph], (N_LAT, N_LON))
x0_rh_gt = np.reshape(x0_true[nvph:], (N_LAT, N_LON))
x0_mean_lh_infr = np.reshape(x0_mean[0:nvph], (N_LAT, N_LON))
x0_mean_rh_infr = np.reshape(x0_mean[nvph:], (N_LAT, N_LON))
x0_std_lh_infr = np.reshape(x0_std[0:nvph], (N_LAT, N_LON))
x0_std_rh_infr = np.reshape(x0_std[nvph:], (N_LAT, N_LON))
clim_min = np.min([np.min(x0_true), np.min(x0_mean)])
clim_max = np.max([np.max(x0_true), np.max(x0_mean)])
plt.subplot(321)
plt.imshow(x0_lh_gt, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Ground Truth - Left hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.subplot(322)
plt.imshow(x0_rh_gt, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Ground Truh - Right hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)

plt.subplot(323)
plt.imshow(x0_mean_lh_infr, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Inferred - Left hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.subplot(324)
plt.imshow(x0_mean_rh_infr, interpolation=None)
plt.clim(clim_min, clim_max)
plt.title("Inferred - Right hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.tight_layout()

plt.subplot(325)
plt.imshow(x0_std_lh_infr, interpolation=None)
# plt.clim(clim_min, clim_max)
plt.title("Standard Deviation - Left hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.subplot(326)
plt.imshow(x0_std_rh_infr, interpolation=None)
# plt.clim(clim_min, clim_max)
plt.title("Standard Deviation - Right hemisphere", fontsize=fs_small)
plt.xlabel("Longitude", fontsize=fs_med)
plt.ylabel("Latitude", fontsize=fs_med)
plt.xticks(fontsize=fs_med)
plt.yticks(fontsize=fs_med)
plt.colorbar(fraction=0.02)
plt.tight_layout()
plt.savefig(f"{figs_dir}/{fig_fname}", facecolor='white')