# %%
import lib.model.neuralfield
import lib.plots.neuralfield as nfplot
import lib.plots.epileptor_2d as epplot
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import lib.postprocess.accuracy as acrcy

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
results_dir = "results/exp113.3"
data_dir = 'datasets/data_jd/id004_bj'
os.makedirs(results_dir, exist_ok=True)
figs_dir = f"{results_dir}/figures"
os.makedirs(figs_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=32,
    N_LAT=128,
    N_LON=256,
    verts_irreg_fname=f"{data_dir}/tvb/ico7/vertices.txt",
    rgn_map_irreg_fname=f"{data_dir}/tvb/Cortex_region_map_ico7.txt",
    conn_zip_path=f"{data_dir}/tvb/connectivity.vep.zip",
    gain_irreg_path=f"{data_dir}/tvb/gain_inv_square_ico7.npz",
    gain_irreg_rgn_map_path=f"{data_dir}/tvb/gain_region_map_ico7.txt",
    L_MAX_PARAMS=16)

# %%
x_init_true = tf.constant(-2.0, dtype=tf.float32) * tf.ones(
    dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32) * \
        dyn_mdl.unkown_roi_mask
z_init_true = tf.constant(5.0, dtype=tf.float32) * tf.ones(
    dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32) * \
        dyn_mdl.unkown_roi_mask
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
x0_true = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
ez_true_roi_tvb = [116, 127, 157]
ez_true_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_true_roi_tvb]
ez_true_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_true_roi])
x0_true[ez_true_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32) * dyn_mdl.unkown_roi_mask
amp_true = dyn_mdl.amp_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
offset_true = dyn_mdl.offset_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
eps_true = 0.1
# %%
nfplot.spherical_spat_map(x0_true.numpy(),
                          N_LAT=dyn_mdl.N_LAT.numpy(),
                          N_LON=dyn_mdl.N_LON.numpy(),
                          clim={
                              "min": -5.0,
                              "max": 0.0
                          },
                          unkown_roi_mask=dyn_mdl.unkown_roi_mask,
                          fig_dir=figs_dir,
                          fig_name='x0_gt.png',
                          dpi=100)
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)

y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                         tau_true, K_true)
x_obs = y_obs[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_true = amp_true * dyn_mdl.project_sensor_space(x_obs) + offset_true
_slp = slp_true - tf.reduce_mean(slp_true, axis=0, keepdims=True)
avg_pwr = tf.reduce_max(tf.reduce_mean(_slp**2, axis=0))
SNR_DB = 50.0
snr = 10**(SNR_DB / 10)
noise_std = tf.sqrt(avg_pwr / snr)

slp_noised = slp_true + tfd.Normal(
    loc=tf.zeros_like(slp_true),
    scale=noise_std * tf.ones_like(slp_true)).sample()
# %%
epplot.plot_slp(slp_noised.numpy(),
                save_dir=figs_dir,
                fig_name="slp_noised.png")
# %%

ez_hyp_roi = ez_true_roi
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
x0_prior_mu = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
x0_prior_mu[ez_hyp_vrtcs] = -1.5
x0_prior_mu = tf.constant(x0_prior_mu, dtype=tf.float32)
x0_prior_std = 0.5 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)
x_init_prior_mu = -3.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)
z_init_prior_mu = 5.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)

prior_mean = {
    'x0': x0_prior_mu,
    'x_init': x_init_prior_mu,
    'z_init': z_init_prior_mu,
    'eps': 0.1,
    'K': 1.0,
}
prior_std = {
    'x0': x0_prior_std,
    'x_init': 0.5,
    'z_init': 0.5,
    'eps': 0.1,
    'K': 5,
}

x0 = tfd.TruncatedNormal(loc=prior_mean['x0'],
                         scale=prior_std['x0'],
                         low=dyn_mdl.x0_lb,
                         high=dyn_mdl.x0_ub).sample()
x_init = tfd.TruncatedNormal(loc=prior_mean['x_init'],
                             scale=prior_std['x_init'],
                             low=dyn_mdl.x_init_lb,
                             high=dyn_mdl.x_init_ub).sample()
z_init = tfd.TruncatedNormal(loc=prior_mean['z_init'],
                             scale=prior_std['z_init'],
                             low=dyn_mdl.z_init_lb,
                             high=dyn_mdl.z_init_ub).sample()
eps = tf.constant(0.3, dtype=tf.float32)
K = tf.constant(1.0, dtype=tf.float32)
tau = tf.constant(50.0, dtype=tf.float32)
# amp = tf.constant(1.0, dtype=tf.float32)
amp = dyn_mdl.amp_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
# offset = tf.constant(0.0, dtype=tf.float32)
offset = dyn_mdl.offset_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())

theta_init_val = dyn_mdl.inv_transformed_parameters(x0,
                                                    x_init,
                                                    z_init,
                                                    eps,
                                                    K,
                                                    tau,
                                                    amp,
                                                    offset,
                                                    param_space='mode')
theta = tf.Variable(initial_value=theta_init_val, dtype=tf.float32)
# %%


@tf.function
def get_loss_and_gradients():
    with tf.GradientTape() as tape:
        loss = -1.0 * dyn_mdl.log_prob(theta, 1)
    return loss, tape.gradient(loss, [theta])


# %%


# @tf.function
def train_loop(num_iters, optimizer):
    loss_at = tf.TensorArray(size=num_iters, dtype=tf.float32)

    def cond(i, loss_at):
        return tf.less(i, num_iters)

    def body(i, loss_at):
        loss_value, grads = get_loss_and_gradients()
        # tf.print("NAN in grads: ", tf.reduce_any(tf.math.is_nan(grads)), output_stream='file:///workspaces/isp_neural_fields/debug.txt')
        loss_at = loss_at.write(i, loss_value)
        tf.print("Iter ", i + 1, "loss: ", loss_value)
        optimizer.apply_gradients(zip(grads, [theta]))
        return i + 1, loss_at

    i = tf.constant(0, dtype=tf.int32)
    i, loss_at = tf.while_loop(cond=cond,
                               body=body,
                               loop_vars=(i, loss_at),
                               parallel_iterations=1)
    return loss_at.stack()


# %%
n_sample_aug = 50
obs_data_aug = tf.TensorArray(dtype=tf.float32, size=n_sample_aug)
for j in range(n_sample_aug):
    data_noised = slp_noised + \
        tf.random.normal(shape=slp_noised.shape, mean=0, stddev=eps_true)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()
# %%

dyn_mdl.setup_inference(nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        mean=prior_mean,
                        std=prior_std,
                        obs_data=obs_data_aug,
                        param_space='mode',
                        obs_space='sensor')
# %%
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2, decay_steps=50, decay_rate=0.96, staircase=True)
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate, decay_steps=100, decay_rate=0.5)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=10)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9)
# %%
start_time = time.time()
niters = tf.constant(500, dtype=tf.int32)
# lr = tf.constant(1e-4, dtype=tf.float32)
losses = train_loop(niters, optimizer)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
np.save(os.path.join(results_dir, 'theta.npy'), theta.numpy())
# %%
(x0_hat_l_m, x_init_hat_l_m, z_init_hat_l_m, eps_hat, K_hat, tau_hat, amp_hat,
 offset_hat) = dyn_mdl.split_params(theta)
(x0_pred, x_init_pred, z_init_pred, eps_pred, K_pred, tau_pred, amp_pred,
 offset_pred) = dyn_mdl.transformed_parameters(x0_hat_l_m,
                                               x_init_hat_l_m,
                                               z_init_hat_l_m,
                                               eps_hat,
                                               K_hat,
                                               tau_hat,
                                               amp_hat,
                                               offset_hat,
                                               param_space='mode')
x0_pred_masked = x0_pred * dyn_mdl.unkown_roi_mask
y_init_pred = tf.concat((x_init_pred * dyn_mdl.unkown_roi_mask,
                         z_init_pred * dyn_mdl.unkown_roi_mask),
                        axis=0)

np.save(f"{results_dir}/x0_pred_lmax={dyn_mdl.L_MAX_PARAMS}.npy",
        x0_pred.numpy())
y_pred = dyn_mdl.simulate(dyn_mdl.nsteps, dyn_mdl.nsubsteps, dyn_mdl.time_step,
                          y_init_pred, x0_pred_masked, tau_pred, K_pred)
x_pred = y_pred[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_pred = amp_pred * dyn_mdl.project_sensor_space(x_pred) + offset_pred
print(f"Param \tGround Truth \tPrediction\n\
K \t{K_true:.2f} \t{K_pred:.2f}\n\
eps \t{eps_true:.2f} \t{eps_pred:.2f}\n\
tau \t{tau_true:.2f} \t{tau_pred:.2f}\n\
amp \t{amp_true:.2f} \t{amp_pred:.2f}\n\
offset \t{offset_true:.2f} \t{offset_pred:.2f}")
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
epplot.plot_slp(slp_true.numpy(), ax=axs[0], title='Observed')
epplot.plot_slp(slp_pred.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
# %%
nfplot.spat_map_hyp_vs_pred(x0_true.numpy(),
                            x0_pred.numpy(),
                            dyn_mdl.N_LAT.numpy(),
                            dyn_mdl.N_LON.numpy(),
                            clim={
                                'min': dyn_mdl.x0_lb,
                                'max': dyn_mdl.x0_ub
                            },
                            dpi=100,
                            fig_dir=figs_dir,
                            fig_name='x0_gt_vs_infr.png')
# nfplot.spat_map_hyp_vs_pred(x_init_true.numpy(),
#                             x_init_pred.numpy(),
#                             dyn_mdl.N_LAT.numpy(),
#                             dyn_mdl.N_LON.numpy(),
#                             clim={
#                                 'min': dyn_mdl.x_init_lb,
#                                 'max': dyn_mdl.x_init_ub,
#                             },
#                             dpi=100,
#                             fig_dir=figs_dir,
#                             fig_name='x_init_gt_vs_infr.png')
# nfplot.spat_map_hyp_vs_pred(z_init_true.numpy(),
#                             z_init_pred.numpy(),
#                             dyn_mdl.N_LAT.numpy(),
#                             dyn_mdl.N_LON.numpy(),
#                             clim={
#                                 'min': dyn_mdl.z_init_lb,
#                                 'max': dyn_mdl.z_init_ub,
#                             },
#                             dpi=100,
#                             fig_dir=figs_dir,
#                             fig_name='z_init_gt_vs_infr.png')

# %%
t_obs = x_obs.numpy()
t_obs[:, dyn_mdl.unkown_roi_idcs] = -3.0
t_pred = x_pred.numpy()
t_pred[:, dyn_mdl.unkown_roi_idcs] = -3.0
ows = 30
ez_pred, pz_pred = acrcy.find_ez(t_pred, src_thrshld=0.0, onst_wndw_sz=ows)
ez_obs, pz_obs = acrcy.find_ez(t_obs, src_thrshld=0.0, onst_wndw_sz=ows)
nfplot.spat_map_hyp_vs_pred(ez_obs,
                            ez_pred,
                            dyn_mdl.N_LAT.numpy(),
                            dyn_mdl.N_LON.numpy(),
                            dpi=100)
p, r = acrcy.precision_recall(ez_hyp=ez_obs, ez_pred=ez_pred)
print(f"Precision: {p}\t Recall: {r}")
np.savez(f"{results_dir}/ez_pz_pred.npz", ez=ez_pred, pz=pz_pred)
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200, layout='tight')
epplot.plot_src(t_obs, ax=axs[0], title='Observed')
epplot.plot_src(t_pred, ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/src_obs_vs_pred.png', facecolor='white')
# %%
# lib.plots.neuralfield.create_video(
#     x_pred.numpy(),
#     N_LAT=dyn_mdl.N_LAT.numpy(),
#     N_LON=dyn_mdl.N_LON.numpy(),
#     out_dir=figs_dir,
#     movie_name="source_activity_infr.mp4",
#     unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
#     vis_type='spherical',
#     dpi=100,
#     ds_freq=3)
# %% loss at ground truth
# tmp_x0 = x0_true.numpy()
# tmp_x0[dyn_mdl.unkown_roi_idcs] = -3.0
# tmp_x0 = tf.constant(tmp_x0)
# tmp_x_init = x_init_true.numpy()
# tmp_x_init[dyn_mdl.unkown_roi_idcs] = -2.0
# tmp_x_init = tf.constant(tmp_x_init)
# tmp_z_init = z_init_true.numpy()
# tmp_z_init[dyn_mdl.unkown_roi_idcs] = 5.0
# tmp_z_init = tf.constant(tmp_z_init)
# theta_true = dyn_mdl.inv_transformed_parameters(tmp_x0,
#                                                 tmp_x_init,
#                                                 tmp_z_init,
#                                                 tf.constant(0.1,
#                                                             dtype=tf.float32),
#                                                 K_true,
#                                                 tau_true,
#                                                 param_space='mode')
# loss_gt = -1.0 * dyn_mdl.log_prob(theta_true)
# print(f"loss_gt = {loss_gt}")
# %%
# lib.plots.neuralfield.create_video(
#     x_obs.numpy(),
#     N_LAT=dyn_mdl.N_LAT.numpy(),
#     N_LON=dyn_mdl.N_LON.numpy(),
#     out_dir=f"{figs_dir}/ground_truth",
#     movie_name="source_activity.mp4",
#     unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
#     vis_type='spherical',
#     dpi=100,
#     ds_freq=3)
