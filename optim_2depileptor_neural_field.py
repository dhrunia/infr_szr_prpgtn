# %%
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import time
import lib.utils.tnsrflw
import lib.plots.neuralfield
import lib.plots.seeg
import os
import lib.model.neuralfield
import lib.utils.sht as tfsht

tfd = tfp.distributions
tfb = tfp.bijectors
import os

# %%
results_dir = 'tmp1'
os.makedirs(results_dir, exist_ok=True)
figs_dir = f'{results_dir}/figures'
os.makedirs(figs_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=32,
    N_LAT=129,
    N_LON=257,
    verts_irreg_fname='datasets/id004_bj_jd/tvb/ico7/vertices.txt',
    rgn_map_irreg_fname='datasets/id004_bj_jd/tvb/Cortex_region_map_ico7.txt',
    conn_zip_path='datasets/id004_bj_jd/tvb/connectivity.vep.zip',
    gain_irreg_path='datasets/id004_bj_jd/tvb/gain_inv_square_ico7.npz',
    gain_irreg_rgn_map_path='datasets/id004_bj_jd/tvb/gain_region_map_ico7.txt',
    L_MAX_PARAMS=10)

# %%
x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
ez_hyp_roi_tvb = [116, 127, 157]
ez_hyp_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_hyp_roi_tvb]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32)
# t = dyn_mdl.SC.numpy()
# t[dyn_mdl.roi_map_tvb_to_tfnf[140], dyn_mdl.roi_map_tvb_to_tfnf[116]] = 5.0
# dyn_mdl.SC = tf.constant(t, dtype=tf.float32)
# %%
lib.plots.neuralfield.spatial_map(
    x0_true.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    clim={
        'min': -3.5,
        'max': -1.0
    },
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)

y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                         tau_true, K_true)
x_obs = y_obs[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_obs = dyn_mdl.project_sensor_space(x_obs)
# %%
lib.plots.seeg.plot_slp(slp_obs.numpy(),
                        save_dir=f'{figs_dir}/ground_truth',
                        fig_name='slp_obs.png')
# %%

x0_crtx_lh = -3.0 * tf.ones(dyn_mdl.nvph, dtype=tf.float32)
x0_crtx_rh = -3.0 * tf.ones(dyn_mdl.nvph, dtype=tf.float32)
x0_crtx_lm_lh = tfsht.analys(dyn_mdl.L_MAX_PARAMS, dyn_mdl.N_LAT,
                             dyn_mdl.N_LON, x0_crtx_lh, dyn_mdl.glq_wts_params,
                             dyn_mdl.P_l_m_costheta_params)
x0_crtx_lm_rh = tfsht.analys(dyn_mdl.L_MAX_PARAMS, dyn_mdl.N_LAT,
                             dyn_mdl.N_LON, x0_crtx_rh, dyn_mdl.glq_wts_params,
                             dyn_mdl.P_l_m_costheta_params)
x0_subcrtx = -3.0 * tf.ones(dyn_mdl.ns, dtype=tf.float32)
eps = tf.constant(0.3, dtype=tf.float32, shape=(1,))
theta = tf.Variable(initial_value=tf.concat(
    (tf.math.real(x0_crtx_lm_lh), tf.math.imag(x0_crtx_lm_lh),
     tf.math.real(x0_crtx_lm_rh), tf.math.imag(x0_crtx_lm_rh), x0_subcrtx,
     eps),
    axis=0),
                    dtype=tf.float32)
# %%


@tf.function
def get_loss_and_gradients():
    with tf.GradientTape() as tape:
        loss = -1.0 * dyn_mdl.log_prob(theta[tf.newaxis, :])
        return loss, tape.gradient(loss, [theta])


# %%


# @tf.function
def train_loop(num_iters, optimizer):
    loss_at = tf.TensorArray(size=num_iters, dtype=tf.float32)

    def cond(i, loss_at):
        return tf.less(i, num_iters)

    def body(i, loss_at):
        loss_value, grads = get_loss_and_gradients()
        loss_at = loss_at.write(i, loss_value)
        # grads = [tf.divide(el, batch_size) for el in grads]
        # grads = [tf.clip_by_norm(el, 1000) for el in grads]
        # tf.print("gradient norm = ", [tf.norm(el) for el in grads], \
        # output_stream="file://debug.log")
        tf.print("Iter ", i + 1, "loss: ", loss_value)
        # training_loss.append(loss_value)
        optimizer.apply_gradients(zip(grads, [theta]))
        return i + 1, loss_at

    i = tf.constant(0, dtype=tf.int32)
    i, loss_at = tf.while_loop(cond=cond,
                               body=body,
                               loop_vars=(i, loss_at),
                               parallel_iterations=1)
    return loss_at.stack()


# %%

x0_prior_mu = -3.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)
dyn_mdl.setup_inference(obs_data=x_obs,
                        nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        y_init=y_init_true,
                        tau=tau_true,
                        K=K_true,
                        x0_prior_mu=x0_prior_mu)
# %%
# initial_learning_rate = 1e-1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=15, decay_rate=0.96, staircase=True)
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate, decay_steps=100, decay_rate=0.5)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1, clipnorm=10)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9)
# %%
start_time = time.time()
niters = tf.constant(500, dtype=tf.int32)
# lr = tf.constant(1e-4, dtype=tf.float32)
train_loop(niters, optimizer)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
# %% loss at ground truth
eps_true = tf.constant(0.22, shape=(1,), dtype=tf.float32)
theta_true = dyn_mdl.inv_transformed_parameters(x0_true, eps_true)

loss = -1.0 * dyn_mdl.log_prob(theta_true[tf.newaxis, :])
print(f"loss = {loss}")
# %%
# x0_crtx = dyn_mdl.x0_trans_to_vrtx_space(theta[0:4 * dyn_mdl.nmodes_params])
# x0_subcrtx = theta[4 * dyn_mdl.nmodes_params:4 * dyn_mdl.nmodes_params +
#                    dyn_mdl.ns]
# x0_hat = tf.concat([x0_crtx, x0_subcrtx], axis=0)
# x0 = dyn_mdl.x0_bounded(x0_hat) * dyn_mdl.unkown_roi_mask
x0, eps = dyn_mdl.transformed_parameters(theta)

y_pred = dyn_mdl.simulate(dyn_mdl._nsteps, dyn_mdl._nsubsteps,
                          dyn_mdl._time_step, dyn_mdl._y_init, x0,
                          dyn_mdl._tau, dyn_mdl._K)
x_pred = y_pred[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
# %%
lib.plots.neuralfield.create_video(
    x_pred.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f'{figs_dir}/infer',
    movie_name='movie.mp4',
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())
# %%
lib.plots.neuralfield.create_video(
    x_obs.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f'{figs_dir}/ground_truth',
    movie_name='movie.mp4',
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())
# %%
fig_name = 'x0_infer_vs_gt.png'
clim = {}
clim['min'] = np.min([np.min(x0_true.numpy()), np.min(x0.numpy())])
clim['max'] = np.max([np.max(x0_true.numpy()), np.max(x0.numpy())])

fig = plt.figure(figsize=(8, 7), dpi=200, constrained_layout=True)
sub_figs = fig.subfigures(2, 1)
gs_gt = sub_figs[0].add_gridspec(2,
                                 3,
                                 height_ratios=[0.9, 0.1],
                                 width_ratios=[0.49, 0.49, 0.02])
ax_gt = {}
ax_gt['crtx_lh'] = fig.add_subplot(gs_gt[0, 0])
ax_gt['crtx_rh'] = fig.add_subplot(gs_gt[0, 1])
ax_gt['subcrtx_lh'] = fig.add_subplot(gs_gt[1, 0])
ax_gt['subcrtx_rh'] = fig.add_subplot(gs_gt[1, 1])
ax_gt['clr_bar'] = fig.add_subplot(gs_gt[:, 2])
gs_infr = sub_figs[1].add_gridspec(2,
                                   3,
                                   height_ratios=[0.9, 0.1],
                                   width_ratios=[0.49, 0.49, 0.02])
ax_infr = {}
ax_infr['crtx_lh'] = fig.add_subplot(gs_infr[0, 0])
ax_infr['crtx_rh'] = fig.add_subplot(gs_infr[0, 1])
ax_infr['subcrtx_lh'] = fig.add_subplot(gs_infr[1, 0])
ax_infr['subcrtx_rh'] = fig.add_subplot(gs_infr[1, 1])
ax_infr['clr_bar'] = fig.add_subplot(gs_infr[:, 2])
lib.plots.neuralfield.spatial_map(
    x0_true.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    ax=ax_gt,
    clim=clim)
lib.plots.neuralfield.spatial_map(
    x0.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    ax=ax_infr,
    clim=clim)
plt.savefig(f'{figs_dir}/{fig_name}', facecolor='white', bbox_inches='tight')
