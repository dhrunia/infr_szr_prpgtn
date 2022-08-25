# %%
import lib.model.neuralfield
import lib.plots.seeg
import lib.plots.neuralfield
import lib.utils.tnsrflw
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
results_dir = "results/exp69"
os.makedirs(results_dir, exist_ok=True)
figs_dir = f"{results_dir}/figures"
os.makedirs(figs_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=100,
    N_LAT=128,
    N_LON=256,
    verts_irreg_fname="datasets/data_jd/id004_bj/tvb/ico7/vertices.txt",
    rgn_map_irreg_fname=
    "datasets/data_jd/id004_bj/tvb/Cortex_region_map_ico7.txt",
    conn_zip_path="datasets/data_jd/id004_bj/tvb/connectivity.vep.zip",
    gain_irreg_path="datasets/data_jd/id004_bj/tvb/gain_inv_square_ico7.npz",
    gain_irreg_rgn_map_path=
    "datasets/data_jd/id004_bj/tvb/gain_region_map_ico7.txt",
    L_MAX_PARAMS=16,
    diff_coeff=0.01,
    alpha=1.0)

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
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
ez_hyp_roi_tvb = [116, 127]
ez_hyp_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_hyp_roi_tvb]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32) * dyn_mdl.unkown_roi_mask
# t = dyn_mdl.SC.numpy()
# t[dyn_mdl.roi_map_tvb_to_tfnf[140], dyn_mdl.roi_map_tvb_to_tfnf[116]] = 5.0
# dyn_mdl.SC = tf.constant(t, dtype=tf.float32)
# %%
lib.plots.neuralfield.spherical_spat_map(
    x0_true.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    clim={
        "min": -5.0,
        "max": 0.0
    },
    unkown_roi_mask=dyn_mdl.unkown_roi_mask,
    fig_dir=f'{figs_dir}/ground_truth',
    fig_name='x0_gt.png',
    dpi=100)
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)
gamma_lc = 0.3

y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                         tau_true, K_true, gamma_lc)
x_obs = y_obs[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_true = dyn_mdl.project_sensor_space(x_obs)
# %%
lib.plots.seeg.plot_slp(slp_true.numpy(),
                        save_dir=f"{figs_dir}/ground_truth",
                        fig_name="slp_obs.png")
# %%
# x0 = tf.random.uniform(shape=[dyn_mdl.nv + dyn_mdl.ns],
#                        minval=dyn_mdl.x0_lb,
#                        maxval=dyn_mdl.x0_ub)
x0 = -4.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
# x0 = tf.constant(np.load(f'{results_dir}/x0_pred_lmax=5.npy'),
#                  dtype=tf.float32)
# x0 = x0_true
eps = tf.constant(0.3, dtype=tf.float32, shape=(1, ))
theta_init_val = dyn_mdl.inv_transformed_parameters(x0,
                                                    eps,
                                                    param_space="mode")
theta = tf.Variable(initial_value=theta_init_val, dtype=tf.float32)
# %%


@tf.function
def get_loss_and_gradients():
    with tf.GradientTape() as tape:
        loss = -1.0 * dyn_mdl.log_prob(theta)
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
    data_noised = slp_true + \
        tf.random.normal(shape=slp_true.shape, mean=0, stddev=0.1)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()
# %%
x0_prior_mu = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
# x0_prior_mu[ez_hyp_vrtcs] = -1.5
x0_prior_mu = tf.constant(x0_prior_mu,
                          dtype=tf.float32) * dyn_mdl.unkown_roi_mask

dyn_mdl.setup_inference(nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        y_init=y_init_true,
                        tau=tau_true,
                        K=K_true,
                        gamma_lc=gamma_lc,
                        x0_prior_mu=x0_prior_mu,
                        obs_data=obs_data_aug,
                        param_space='mode',
                        obs_space='sensor',
                        prior_roi_weighted=False)
# %%
# initial_learning_rate = 1e-1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=15, decay_rate=0.96, staircase=True)
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate, decay_steps=100, decay_rate=0.5)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=10)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9)
# %%
start_time = time.time()
niters = tf.constant(500, dtype=tf.int32)
# lr = tf.constant(1e-4, dtype=tf.float32)
losses = train_loop(niters, optimizer)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
# %%
x0_pred, eps_pred = dyn_mdl.transformed_parameters(theta, param_space="mode")
# x0_pred = x0_pred * dyn_mdl.unkown_roi_mask
np.save(f"{results_dir}/x0_pred_lmax={dyn_mdl.L_MAX_PARAMS}.npy",
        x0_pred.numpy())
y_pred = dyn_mdl.simulate(dyn_mdl.nsteps, dyn_mdl.nsubsteps, dyn_mdl.time_step,
                          dyn_mdl._y_init, x0_pred, dyn_mdl._tau, dyn_mdl._K,
                          dyn_mdl._gamma_lc)
x_pred = y_pred[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_pred = dyn_mdl.project_sensor_space(x_pred)
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
lib.plots.seeg.plot_slp(slp_true.numpy(), ax=axs[0], title='Observed')
lib.plots.seeg.plot_slp(slp_pred.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
# %%
lib.plots.neuralfield.spherical_spat_map(x0_pred.numpy(),
                                         N_LAT=dyn_mdl.N_LAT.numpy(),
                                         N_LON=dyn_mdl.N_LON.numpy(),
                                         fig_dir=f'{figs_dir}/infer',
                                         fig_name='x0_map_estim.png')
# %%
lib.plots.neuralfield.create_video(
    x_pred.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f"{figs_dir}/infer",
    movie_name="source_activity.mp4",
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    vis_type='spherical',
    dpi=100)
# %% loss at ground truth
theta_pred = dyn_mdl.inv_transformed_parameters(x0_pred,
                                                tf.reshape(eps_pred,
                                                           shape=(1, )),
                                                param_space="mode")
theta_true = dyn_mdl.inv_transformed_parameters(x0_true,
                                                tf.reshape(eps_pred,
                                                           shape=(1, )),
                                                param_space="mode")
loss_gt = -1.0 * dyn_mdl.log_prob(theta_true)
loss_pred = -1.0 * dyn_mdl.log_prob(theta_pred)
print(f"loss_gt = {loss_gt}, loss_pred = {loss_pred}")
# %%
x0, _ = dyn_mdl.transformed_parameters(theta_true, param_space="mode")
y_test = dyn_mdl.simulate(dyn_mdl.nsteps, dyn_mdl.nsubsteps, dyn_mdl.time_step,
                          dyn_mdl._y_init, x0, dyn_mdl._tau, dyn_mdl._K,
                          dyn_mdl._gamma_lc)
x_test = y_test[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
lib.plots.neuralfield.create_video(
    x_test.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f"{figs_dir}/ground_truth",
    movie_name="source_activity.mp4",
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    vis_type='spherical')
# %%
lib.plots.neuralfield.create_video(
    x_obs.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f"{figs_dir}/ground_truth",
    movie_name="source_activity.mp4",
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    vis_type='spherical',
    dpi=100)
