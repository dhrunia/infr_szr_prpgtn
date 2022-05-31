# %%
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import lib.utils.sht as tfsht
import lib.utils.projector
import time
import lib.utils.tnsrflw
import lib.plots.neuralfield
import os
import lib.model.neuralfield

tfd = tfp.distributions
tfb = tfp.bijectors
# %%
results_dir = 'tmp'
os.makedirs(results_dir, exist_ok=True)
figs_dir = f'{results_dir}/figures'
os.makedirs(figs_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=32,
    N_LAT=129,
    N_LON=257,
    verts_irreg_fname='datasets/id004_bj_jd/tvb/ico7/vertices.txt',
    rgn_map_irreg_fname='datasets/id004_bj_jd/tvb/Cortex_region_map_ico7.txt',
    SC_path='datasets/id004_bj_jd/tvb/vep_conn/weights.txt',
    gain_irreg_path='datasets/id004_bj_jd/tvb/gain_inv_square_ico7.npz',
    L_MAX_PARAMS=10)

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
x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(dyn_mdl.nv)
ez_hyp_roi = [116, 127, 151]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map_reg)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32)
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)


@tf.function
def run_sim(nsteps, nsubsteps, time_step, y_init, x0, tau, K):
    y = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init, x0, tau, K)
    return y


y_obs = run_sim(nsteps, nsubsteps, time_step, y_init_true, x0_true, tau_true,
                K_true)
x_obs = y_obs[:, 0:dyn_mdl.nv] * dyn_mdl.unkown_roi_mask
slp_obs = dyn_mdl.project_sensor_space(x_obs)

# %%
plt.figure(figsize=(7, 6), dpi=200)
plt.imshow(tf.transpose(slp_obs),
           interpolation=None,
           aspect='auto',
           cmap='inferno')
plt.xlabel('Time')
plt.ylabel('Sensor')
# plt.plot(slp_obs, color='black', alpha=0.3);
plt.colorbar(fraction=0.02)
plt.savefig(f'{figs_dir}/slp_obs.png')
# %%
num_bijectors = 4
nparams = 4 * ((dyn_mdl.L_MAX_PARAMS + 1)**2)
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


@tf.function
def get_loss_and_gradients(nsamples):
    loss = tf.constant(0, dtype=tf.float32, shape=(1, 1))
    grads = [tf.zeros_like(var) for var in flow_dist.trainable_variables]
    i = tf.constant(0, dtype=tf.uint32)

    def cond(i, grads, loss):
        return tf.less(i, nsamples)

    def body(i, grads, loss):
        with tf.GradientTape() as tape:
            theta = flow_dist.sample(1)
            # loss_val = tf.constant(0.0, shape=(1, ), dtype=tf.float32)
            gm_log_prob = tf.reduce_sum(dyn_mdl.log_prob(theta))
            posterior_approx_log_prob = flow_dist.log_prob(theta)
            tf.print("\tgm_log_prob:", gm_log_prob,
                     "\tposterior_approx_log_prob:", posterior_approx_log_prob)
            loss_i = posterior_approx_log_prob - gm_log_prob
        grad_i = tape.gradient(loss_i, flow_dist.trainable_variables)
        grads = [(g1 + g2) / tf.cast(nsamples, dtype=tf.float32)
                 for g1, g2 in zip(grads, grad_i)]
        loss += loss_i / tf.cast(nsamples, dtype=tf.float32)
        return i + 1, grads, loss

    i, grads, loss = tf.while_loop(cond=cond,
                                   body=body,
                                   loop_vars=(i, grads, loss),
                                   parallel_iterations=1)
    return loss, grads


# %%
# initial_learning_rate = 1e-3
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100,
#     decay_rate=0.96,
#     staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


# %%
@tf.function
def train_loop(num_epochs, nsamples):
    def cond(i):
        return tf.less(i, num_epochs)

    def body(i):
        loss_value, grads = get_loss_and_gradients(nsamples)
        # nan_in_grads = tf.reduce_any(
        #     [tf.reduce_any(tf.math.is_nan(el)) for el in grads])
        # tf.print("nan in grads", nan_in_grads)
        # grads = [tf.divide(el, batch_size) for el in grads]
        grads = [tf.clip_by_norm(el, 1000) for el in grads]
        # tf.print("gradient norm = ", [tf.norm(el) for el in grads], \
        # output_stream="file://debug.log")
        tf.print("Epoch ", i, "loss: ", loss_value)
        # training_loss.append(loss_value)
        optimizer.apply_gradients(zip(grads, flow_dist.trainable_variables))
        return i + 1

    i = tf.constant(0, dtype=tf.uint32)
    i = tf.while_loop(cond=cond,
                      body=body,
                      loop_vars=(i, ),
                      parallel_iterations=1)


# %%

x0_prior_mu = -3.0 * tf.ones(dyn_mdl.nv)
dyn_mdl.setup_inference(slp_obs=slp_obs,
                        nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        y_init=y_init_true,
                        tau=tau_true,
                        K=K_true,
                        x0_prior_mu=x0_prior_mu)
# %%

# %%
num_epochs = tf.constant(500, dtype=tf.uint32)
nsamples = tf.constant(10, dtype=tf.uint32)
start_time = time.time()
train_loop(num_epochs, nsamples)
print(f"Elapsed {time.time() - start_time} seconds for {num_epochs} Epochs")
# %%
# x0_lh = lib.utils.tnsrflw.inv_sigmoid_transform(x0_true[0:dyn_mdl.nvph],
#                                                 dyn_mdl.x0_lb, dyn_mdl.x0_ub)
# x0_rh = lib.utils.tnsrflw.inv_sigmoid_transform(x0_true[dyn_mdl.nvph:],
#                                                 dyn_mdl.x0_lb, dyn_mdl.x0_ub)
# x0_lm_lh = tfsht.analys(dyn_mdl.L_MAX_PARAMS, dyn_mdl.N_LAT, dyn_mdl.N_LON,
#                         x0_lh, dyn_mdl.glq_wts_params,
#                         dyn_mdl.P_l_m_costheta_params)
# x0_lm_rh = tfsht.analys(dyn_mdl.L_MAX_PARAMS, dyn_mdl.N_LAT, dyn_mdl.N_LON,
#                         x0_rh, dyn_mdl.glq_wts_params,
#                         dyn_mdl.P_l_m_costheta_params)
# x0_lm_lh_real = tf.reshape(tf.math.real(x0_lm_lh), [-1])
# x0_lm_rh_real = tf.reshape(tf.math.real(x0_lm_rh), [-1])
# x0_lm_lh_imag = tf.reshape(tf.math.imag(x0_lm_lh), [-1])
# x0_lm_rh_imag = tf.reshape(tf.math.imag(x0_lm_rh), [-1])

# # tau = lib.utils.tnsrflw.inv_sigmoid_transform(
# #     tau_true, tf.constant(15, dtype=tf.float32),
# #     tf.constant(100, dtype=tf.float32))

# theta_true = tf.concat(
#     [x0_lm_lh_real, x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag], axis=0)

# @tf.function
# def get_loss(theta, y_obs):
#     eps = tf.constant(0.1, dtype=tf.float32)
#     x0_lm = theta[0:dyn_mdl.nmodes_params]
#     x0 = dyn_mdl.x0_trans_to_vrtx_space(x0_lm)
#     x0_trans = dyn_mdl.x0_bounded_trnsform(x0) * dyn_mdl.unkown_roi_mask
#     # tau = theta[4 * nmodes]
#     # tau_trans = lib.utils.tnsrflw.sigmoid_transform(
#     #     tau, tf.constant(15, dtype=tf.float32),
#     #     tf.constant(100, dtype=tf.float32))
#     y_pred = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true,
#                               x0_trans, tau_true, K_true)
#     x_mu = y_pred[:, 0:dyn_mdl.nv] * dyn_mdl.unkown_roi_mask
#     x_obs = y_obs[:, 0:dyn_mdl.nv] * dyn_mdl.unkown_roi_mask
#     likelihood = tf.reduce_sum(tfd.Normal(loc=x_mu, scale=eps).log_prob(x_obs))
#     prior = tf.reduce_sum(tfd.Normal(loc=0.0, scale=5.0).log_prob(theta))
#     lp = likelihood + prior
#     return x_mu, lp

# x_pred, lp = get_loss(theta_true, y_obs)
# print(lp)
# out_dir = 'tmp1'
# create_video(x_pred, dyn_mdl.N_LAT.numpy(), dyn_mdl.N_LON.numpy(), out_dir)
# %%
nsamples = 100
posterior_samples = flow_dist.sample(nsamples)
x0_samples = tf.TensorArray(dtype=tf.float32,
                            size=nsamples,
                            clear_after_read=False)
for i, theta in enumerate(posterior_samples.numpy()):
    x0_lm_i = theta[0:4 * dyn_mdl.nmodes_params]
    x0_i = dyn_mdl.x0_trans_to_vrtx_space(x0_lm_i)
    x0_trans_i = dyn_mdl.x0_bounded_trnsform(x0_i) * dyn_mdl.unkown_roi_mask
    x0_samples = x0_samples.write(i, x0_trans_i)
x0_samples = x0_samples.stack()
x0_mean = tf.reduce_mean(x0_samples, axis=0).numpy()
x0_std = tf.math.reduce_std(x0_samples, axis=0).numpy()
# %%
y_ppc = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_mean,
                         tau_true, K_true)
x_ppc = y_ppc[:, :dyn_mdl.nv]
slp_ppc = dyn_mdl.project_sensor_space(x_ppc)
out_dir = f'{figs_dir}/infer'
lib.plots.neuralfield.create_video(x_ppc.numpy(), dyn_mdl.N_LAT.numpy(),
                                   dyn_mdl.N_LON.numpy(), out_dir, 'movie.mp4')

# %%
x_obs = y_obs[:, 0:dyn_mdl.nv]
out_dir = f'{figs_dir}/ground_truth'
lib.plots.neuralfield.create_video(x_obs, dyn_mdl.N_LAT.numpy(),
                                   dyn_mdl.N_LON.numpy(), out_dir, 'movie.mp4')

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
fig_name = 'x0_gt_vs_inferred_mean_and_std.png'
lib.plots.neuralfield.x0_gt_vs_infer(x0_true, x0_mean, x0_std,
                                     dyn_mdl.N_LAT.numpy(),
                                     dyn_mdl.N_LON.numpy(),
                                     dyn_mdl.unkown_roi_mask, figs_dir,
                                     fig_name)
