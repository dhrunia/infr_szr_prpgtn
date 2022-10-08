# %%
import lib.model.neuralfield
import lib.plots.seeg
import lib.plots.neuralfield
import lib.preprocess.envelope
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import json

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
pat_id = 'id001_bt'
data_dir = f"datasets/data_jd/{pat_id}"
results_dir = "results/exp89"
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
    L_MAX_PARAMS=16,
    diff_coeff=0.00047108,
    alpha=2.0,
    theta=-1.0)

# %%
lpf = 0.05
hpf = 10
npoints = 300
szr_name, _ = lib.preprocess.envelope.find_bst_szr_slp(data_dir=data_dir,
                                                       hpf=hpf,
                                                       lpf=lpf,
                                                       npoints=npoints)
raw_seeg_fname = szr_name + '.raw.fif'
meta_data_fname = szr_name + '.json'
data = lib.preprocess.envelope.prepare_data(data_dir, meta_data_fname,
                                            raw_seeg_fname, hpf, lpf)

# Downsample the seeg to ~npoints
ds_freq = int(data['slp'].shape[0] / npoints)
slp_obs = data['slp'][0:-1:ds_freq]

# Update the order of sensors in the gain matrix s.t. it matches with the
# order of seeg data read with MNE
dyn_mdl.update_gain(data_dir, data['snsr_picks'])

with open('datasets/data_jd/ei-vep_53.json', encoding='utf-8') as fd:
    ez_hyp_roi_names = json.load(fd)[pat_id]['ez']
ez_hyp_roi = [
    dyn_mdl.roi_names.index(roi_name) for roi_name in ez_hyp_roi_names
]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])

x0_hyp = np.zeros(dyn_mdl.nv + dyn_mdl.ns)
x0_hyp[ez_hyp_vrtcs] = 1
# %%
lib.plots.seeg.plot_slp(slp_obs, save_dir=figs_dir, fig_name="slp_obs.png")
# %%
lib.plots.neuralfield.spherical_spat_map(x0_hyp,
                                         N_LAT=dyn_mdl.N_LAT.numpy(),
                                         N_LON=dyn_mdl.N_LON.numpy(),
                                         fig_dir=figs_dir,
                                         fig_name='x0_hyp.png',
                                         dpi=100)
# %%

x0 = -4.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
x_init = -3.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
z_init = 4.5 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
eps = tf.constant(0.3, dtype=tf.float32)
K = tf.constant(1.0, dtype=tf.float32)
tau = tf.constant(50.0, dtype=tf.float32)
amp = tf.constant(1.0, dtype=tf.float32)
offset = tf.constant(0.0, dtype=tf.float32)
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
    data_noised = slp_obs + \
        tf.random.normal(shape=slp_obs.shape, mean=0, stddev=0.1)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()
# %%
x0_prior_mu = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
x0_prior_mu[ez_hyp_vrtcs] = -1.5
x0_prior_mu = tf.constant(x0_prior_mu, dtype=tf.float32)
x0_prior_std = 0.5 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)
x_init_prior_mu = -3.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)
z_init_prior_mu = 5.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns)

mean = {
    'x0': x0_prior_mu,
    'x_init': x_init_prior_mu,
    'z_init': z_init_prior_mu,
    'eps': 0.1,
    'K': 1.0,
}
std = {
    'x0': x0_prior_std,
    'x_init': 0.5,
    'z_init': 0.5,
    'eps': 0.1,
    'K': 5,
}

dyn_mdl.setup_inference(nsteps=slp_obs.shape[0],
                        nsubsteps=2,
                        time_step=0.05,
                        mean=mean,
                        std=std,
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
niters = tf.constant(1000, dtype=tf.int32)
# lr = tf.constant(1e-4, dtype=tf.float32)
losses = train_loop(niters, optimizer)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
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
y_pred = dyn_mdl.simulate(dyn_mdl.nsteps, dyn_mdl.nsubsteps, dyn_mdl.time_step,
                          y_init_pred, x0_pred_masked, tau_pred, K_pred)
x_pred = y_pred[:, 0:dyn_mdl.nv + dyn_mdl.ns]
z_pred = y_pred[:, dyn_mdl.nv + dyn_mdl.ns:2 * (dyn_mdl.nv + dyn_mdl.ns)]

slp_pred = amp_pred * dyn_mdl.project_sensor_space(
    x_pred * dyn_mdl.unkown_roi_mask) + offset_pred
print(f"Param \tPrediction\n\
K {K_pred:.2f}\n\
eps \t{eps_pred:.2f}\n\
tau \t{tau_pred:.2f}\n\
amp \t{amp_pred:.2f}\n\
offset \t{offset_pred:.2f}")
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
lib.plots.seeg.plot_slp(slp_obs, ax=axs[0], title='Observed')
lib.plots.seeg.plot_slp(slp_pred.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
# %%
lib.plots.neuralfield.spat_map_infr_vs_pred(x0_hyp,
                                            x0_pred.numpy(),
                                            dyn_mdl.N_LAT.numpy(),
                                            dyn_mdl.N_LON.numpy(),
                                            dpi=200,
                                            fig_dir=figs_dir,
                                            fig_name='x0_hyp_vs_infr.png')

# %%
lib.plots.neuralfield.create_video(
    x_pred.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=figs_dir,
    movie_name="source_activity_infr.mp4",
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    vis_type='spherical',
    dpi=100,
    ds_freq=3)
# %%
np.savez(f'{results_dir}/res_{pat_id}.npz',
         theta=theta.numpy(),
         losses=losses,
         x0=x0_pred.numpy(),
         x_init=x_init_pred.numpy(),
         z_init=z_init_pred.numpy(),
         eps=eps_pred.numpy(),
         K=K_pred.numpy(),
         tau=tau_pred.numpy(),
         amp=amp_pred.numpy(),
         offset=offset_pred.numpy(),
         x=x_pred.numpy(),
         z=z_pred.numpy(),
         slp=slp_pred.numpy())
