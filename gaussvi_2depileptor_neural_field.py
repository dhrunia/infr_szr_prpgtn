# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import lib.utils.sht as tfsht
import lib.utils.projector
import time
import lib.utils.tnsrflw
from lib.plots.neuralfield import create_video
tfd = tfp.distributions
tfb = tfp.bijectors
# tf.config.set_visible_devices([], 'GPU')

# %%
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.autograph.set_verbosity(10)
# %%
results_dir = 'exp34'
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
plt.imshow(tf.transpose(slp_obs), interpolation=None, aspect='auto', cmap='inferno')
plt.xlabel('Time')
plt.ylabel('Sensor')
# plt.plot(slp_obs, color='black', alpha=0.3);
plt.colorbar(fraction=0.02)
plt.savefig(f'{figs_dir}/obs_slp.png')
# %%
nparams = 4 * ((dyn_mdl.L_MAX_params + 1)**2)
loc = tf.Variable(initial_value=tf.zeros(nparams))
log_scale_diag = tf.Variable(initial_value=-2.3 * tf.ones(nparams))
# scale_diag = tf.exp(log_scale_diag)
# var_dist = tfd.MultivariateNormalDiag(
#     loc=loc, scale_diag=scale_diag, name="variational_posterior")

# %%
x0_prior_mu = -3.0 * tf.ones(dyn_mdl.nv)


@tf.function
def loss(y_obs):
    scale_diag = tf.exp(log_scale_diag)
    # scale_diag = 0.1 * tf.ones(nparams)
    nsamples = 1
    posterior_samples = tfd.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag).sample(nsamples)
    loss_val = tf.constant(0.0, shape=(1, ), dtype=tf.float32)
    for theta in posterior_samples:
        # tf.print("theta: ", theta, summarize=-1)
        gm_log_prob = dyn_mdl.log_prob(theta, slp_obs, nsteps, nsubsteps,
                                       time_step, y_init_true, tau_true,
                                       K_true, x0_prior_mu)
        posterior_approx_log_prob = tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag).log_prob(theta[tf.newaxis, :])
        tf.print("gm_log_prob:", gm_log_prob, "\tposterior_approx_log_prob:",
                 posterior_approx_log_prob)
        loss_val += (posterior_approx_log_prob - gm_log_prob) / nsamples
        # tf.print("loss_val: ", loss_val)
    return loss_val


@tf.function
def get_loss_and_gradients(y_obs):
    with tf.GradientTape() as tape:
        loss_val = loss(y_obs)
        return loss_val, tape.gradient(loss_val, [loc, log_scale_diag])


# %%
initial_learning_rate = 1e-2
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=200,
    decay_rate=0.96,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# %%
# @tf.function
def train_loop(num_epochs):
    for epoch in range(num_epochs):
        loss_value, grads = get_loss_and_gradients(slp_obs)
        # grads = [tf.divide(el, batch_size) for el in grads]
        # grads = [tf.clip_by_norm(el, 1000) for el in grads]
        # tf.print("gradient norm = ", [tf.norm(el) for el in grads], \
        # output_stream="file://debug.log")
        tf.print("Epoch ", epoch, "loss: ", loss_value)
        # training_loss.append(loss_value)
        optimizer.apply_gradients(zip(grads, [loc, log_scale_diag]))


# %%
num_epochs = 2000
start_time = time.time()
train_loop(num_epochs)
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

# theta_true = tf.concat([
#     x0_lm_lh_real, x0_lm_lh_imag, x0_lm_rh_real, x0_lm_rh_imag], axis=0)


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


# x_pred, slp_pred, lp = get_loss(theta_true, slp_obs)
# print(lp)
# # out_dir = 'tmp1'
# # create_video(x_pred, N_LAT.numpy(), N_LON.numpy(), out_dir)
# %%
scale_diag = tf.exp(log_scale_diag)
# scale_diag = 0.1 * tf.ones(nparams)
nsamples = 100
posterior_samples = tfd.MultivariateNormalDiag(
    loc=loc, scale_diag=scale_diag).sample(nsamples)
x0_samples = tf.TensorArray(dtype=tf.float32,
                            size=nsamples,
                            clear_after_read=False)
for i, theta in enumerate(posterior_samples.numpy()):
    x0_lm_i = theta[0:4*dyn_mdl.nmodes_params]
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
fig_name = 'x0_gt_vs_inferred_mean_and_std.png'
lib.plots.neuralfield.x0_gt_vs_infer(x0_true, x0_mean, x0_std,
                                     dyn_mdl.N_LAT.numpy(),
                                     dyn_mdl.N_LON.numpy(),
                                     dyn_mdl.unkown_roi_mask, figs_dir,
                                     fig_name)
# %%
plt.figure(figsize=(10, 6), dpi=200)
plt.subplot(121)
plt.imshow(tf.transpose(slp_obs),
           interpolation=None,
           aspect='auto',
           cmap='inferno')
plt.xlabel('Time')
plt.ylabel('Sensor')
plt.title("Observed")
plt.colorbar(fraction=0.02)
plt.subplot(122)
plt.imshow(tf.transpose(slp_ppc),
           interpolation=None,
           aspect='auto',
           cmap='inferno')
plt.xlabel('Time')
plt.ylabel('Sensor')
plt.title("Predicted")
plt.colorbar(fraction=0.02)
plt.tight_layout()
plt.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
