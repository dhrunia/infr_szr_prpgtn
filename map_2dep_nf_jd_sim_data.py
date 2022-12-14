# %%
import lib.model.neuralfield
import lib.plots.neuralfield
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from lib.preprocess.envelope import compute_slp_syn
import lib.plots.epileptor_2d as epplot
import lib.postprocess.accuracy as acrcy

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
data_dir = 'datasets/syn_data_jd'
results_dir = "results/exp98"
os.makedirs(results_dir, exist_ok=True)
figs_dir = f"{results_dir}/figures"
os.makedirs(figs_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=32,
    N_LAT=128,
    N_LON=256,
    verts_irreg_fname=f"{data_dir}/tvb/ico6/vertices.txt",
    rgn_map_irreg_fname=f"{data_dir}/tvb/Cortex_region_map_ico6.txt",
    conn_zip_path=f"{data_dir}/tvb/connectivity.vep.zip",
    gain_irreg_path=f"{data_dir}/tvb/gain_inv_square_ico6.npz",
    gain_irreg_rgn_map_path=f"{data_dir}/tvb/gain_region_map_ico6.txt",
    L_MAX_PARAMS=16,
    diff_coeff=0.00047108,
    alpha=2.0,
    theta=-1.0,
    gamma_lc=0.0)

# %%
sim_src = np.load(f"{data_dir}/simu_src.npz")['data_tavg']
start_idx = 100
end_idx = -1
x_obs_crtx = sim_src[start_idx:end_idx, 0, dyn_mdl.idcs_nbrs_irreg, 0]
x_obs_subcrtx = sim_src[start_idx:end_idx, 0, -18:, 0]
x_obs = tf.concat((x_obs_crtx, x_obs_subcrtx), axis=1)
x_obs = tf.cast(x_obs, dtype=tf.float32)

x_obs = x_obs * dyn_mdl.unkown_roi_mask
seeg_obs = tf.matmul(x_obs, dyn_mdl.gain)
slp_obs = compute_slp_syn(seeg_obs, 256, logtransform=True)
ds_freq = int(slp_obs.shape[0] / 300)
slp_obs_ds = tf.constant(slp_obs[0:-1:ds_freq, :], dtype=tf.float32)
# %%
epplot.plot_slp(slp_obs_ds.numpy(),
                save_dir=figs_dir,
                fig_name="slp_obs.png",
                title='Observed SEEG log power')
# %%

x0_prior_mu = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
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

x0 = tfd.TruncatedNormal(loc=mean['x0'],
                         scale=std['x0'],
                         low=dyn_mdl.x0_lb,
                         high=dyn_mdl.x0_ub).sample()
x_init = tfd.TruncatedNormal(loc=mean['x_init'],
                             scale=std['x_init'],
                             low=dyn_mdl.x_init_lb,
                             high=dyn_mdl.x_init_ub).sample()
z_init = tfd.TruncatedNormal(loc=mean['z_init'],
                             scale=std['z_init'],
                             low=dyn_mdl.z_init_lb,
                             high=dyn_mdl.z_init_ub).sample()

# x0 = -4.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
# x_init = -3.0 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
# z_init = 4.5 * tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
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
    data_noised = slp_obs_ds + \
        tf.random.normal(shape=slp_obs_ds.shape, mean=0, stddev=0.1)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()
# %%
dyn_mdl.setup_inference(nsteps=obs_data_aug.shape[1],
                        nsubsteps=2,
                        time_step=0.05,
                        mean=mean,
                        std=std,
                        obs_data=obs_data_aug,
                        param_space='mode',
                        obs_space='sensor')
# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=10)
# %%
start_time = time.time()
niters = tf.constant(500, dtype=tf.int32)
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
x_init_pred_masked = x_init_pred * dyn_mdl.unkown_roi_mask
z_init_pred_masked = z_init_pred * dyn_mdl.unkown_roi_mask

y_init_pred = tf.concat((x_init_pred_masked, z_init_pred_masked), axis=0)

np.save(f"{results_dir}/x0_pred_lmax={dyn_mdl.L_MAX_PARAMS}.npy",
        x0_pred.numpy())
y_pred = dyn_mdl.simulate(dyn_mdl.nsteps, dyn_mdl.nsubsteps, dyn_mdl.time_step,
                          y_init_pred, x0_pred_masked, tau_pred, K_pred)
x_pred = y_pred[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_pred = dyn_mdl.project_sensor_space(x_pred)
print(f"Scalar Parameters\n\
K \t{K_pred:.2f}\n\
eps \t{eps_pred:.2f}\n\
tau \t{tau_pred:.2f}\n\
amp \t{amp_pred:.2f}\n\
offset \t{offset_pred:.2f}")
# %%
np.savez(f"{results_dir}/map_estim_run1.npz",
         x0=x0_pred.numpy(),
         x_init=x_init_pred.numpy(),
         z_init=z_init_pred,
         x=x_pred.numpy(),
         slp=slp_pred.numpy(),
         theta=theta.numpy())
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
epplot.plot_slp(slp_obs_ds.numpy(), ax=axs[0], title='Observed')
epplot.plot_slp(slp_pred.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
# %%
_x_obs = x_obs.numpy()[::ds_freq, :]
_x_pred = x_pred.numpy()
# Set the activity of unkown roi idcs to -2.0 to avoid confusing zero activity
# as high activity in visualisations
_x_obs[:, dyn_mdl.unkown_roi_idcs] = -2.0
_x_pred[:, dyn_mdl.unkown_roi_idcs] = -2.0
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200, layout='tight')
epplot.plot_src(_x_obs, ax=axs[0], title='Observed')
epplot.plot_src(_x_pred, ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/src_obs_vs_pred.png', facecolor='white')
# %%
lib.plots.neuralfield.spherical_spat_map(x0_pred_masked.numpy(),
                                         dyn_mdl.N_LAT.numpy(),
                                         dyn_mdl.N_LON.numpy())
lib.plots.neuralfield.spherical_spat_map(x_init_pred_masked.numpy(),
                                         dyn_mdl.N_LAT.numpy(),
                                         dyn_mdl.N_LON.numpy())
lib.plots.neuralfield.spherical_spat_map(z_init_pred_masked.numpy(),
                                         dyn_mdl.N_LAT.numpy(),
                                         dyn_mdl.N_LON.numpy())

# %%
_x_pred = x_pred.numpy()
_x_pred[:, dyn_mdl.unkown_roi_idcs] = -2.0
ez_pred, pz_pred = acrcy.find_ez(_x_pred, 0.0, 10)

ez_pred_vrtx_idcs = np.nonzero(ez_pred)[0]
ez_pred_roi_idcs = np.unique(dyn_mdl.rgn_map.numpy()[ez_pred_vrtx_idcs])
ez_pred_roi_lbls = [dyn_mdl.roi_names[idx] for idx in ez_pred_roi_idcs]

pz_pred_vrtx_idcs = np.nonzero(pz_pred)[0]
pz_pred_roi_idcs = np.unique(dyn_mdl.rgn_map.numpy()[pz_pred_vrtx_idcs])
pz_pred_roi_lbls = [dyn_mdl.roi_names[idx] for idx in pz_pred_roi_idcs]

print(f"EZ : {ez_pred_roi_lbls}\nPZ:{pz_pred_roi_lbls}")
# %%
_x_obs = sim_src[start_idx:end_idx:ds_freq, 0, :-18, 0]
ez_irreg, pz_irreg = acrcy.find_ez(_x_obs, 0.0, 10.0)

rgn_map_irreg = np.loadtxt(f"{data_dir}/tvb/Cortex_region_map_ico6.txt")
ez_irreg_vrtx_idcs = np.nonzero(ez_irreg)[0]
ez_irreg_roi_idcs = np.unique(rgn_map_irreg[ez_irreg_vrtx_idcs])
ez_irreg_roi_lbls = [
    dyn_mdl.roi_names[dyn_mdl.roi_map_tvb_to_tfnf[int(idx)]]
    for idx in ez_irreg_roi_idcs
]

pz_irreg_vrtx_idcs = np.nonzero(pz_irreg)[0]
pz_irreg_roi_idcs = np.unique(rgn_map_irreg[pz_irreg_vrtx_idcs])
pz_irreg_roi_lbls = [
    dyn_mdl.roi_names[dyn_mdl.roi_map_tvb_to_tfnf[int(idx)]]
    for idx in pz_irreg_roi_idcs
]

print(
    f"EZ ground truth: {ez_irreg_roi_lbls}\nPZ ground truth: {pz_irreg_roi_lbls}"
)
# %%
t = np.load('results/exp96/map_estim_run1.npz')
vrtx_idcs = np.where(dyn_mdl.rgn_map == 45)[0]
fig = plt.figure(figsize=(7,4), dpi=200)
ax = fig.add_subplot(1,1,1)
ax.hist(x0_pred.numpy()[vrtx_idcs], color='blue', alpha=0.5, label='without local coupling')
ax.hist(t['x0'][vrtx_idcs], color='red', alpha=0.5, label='with local coupling')
ax.legend()
