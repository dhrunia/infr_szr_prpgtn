# %%
from lib.model import nmm
import lib.preprocess.envelope as envelope
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import tensorflow as tf
from lib.postprocess.accuracy import find_ez, teps_to_wndwsz, precision_recall
from lib.io.seeg import find_szr_len
import lib.plots.epileptor_2d as epplot

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
pat_id = 'id004_bj'

results_dir = f"results/exp92/{pat_id}"
os.makedirs(results_dir, exist_ok=True)
figs_dir = f"{results_dir}/figures"
os.makedirs(figs_dir, exist_ok=True)
data_dir = 'datasets/data_jd'

dyn_mdl = nmm.Epileptor2D(
    conn_path=f'{data_dir}/{pat_id}/tvb/connectivity.vep.zip',
    gain_path=f'{data_dir}/{pat_id}/tvb/gain_inv_square_ico7.npz',
    gain_rgn_map_path=f'{data_dir}/{pat_id}/tvb/gain_region_map_ico7.txt',
    seeg_xyz_path=f'{data_dir}/{pat_id}/elec/seeg.xyz',
    gain_mat_res='high')
# %%
lpf = 0.2
hpf = 10.0
npoints = 300
# szr_name, _ = envelope.find_bst_szr_slp(data_dir=f'{data_dir}/{pat_id}',
#                                         hpf=hpf,
#                                         lpf=lpf,
#                                         npoints=npoints)
szr_name = 'BJcrise1le161128B-BEX_0002_lpf0.05_hpf10.0'.split("_lpf")[0]
raw_seeg_fname = szr_name + '.raw.fif'
meta_data_fname = szr_name + '.json'
data = envelope.prepare_data(f'{data_dir}/{pat_id}', meta_data_fname,
                             raw_seeg_fname, hpf, lpf)

# Downsample the seeg to ~npoints
ds_freq = int(data['slp'].shape[0] / npoints)
slp_obs = data['slp'][0:-1:ds_freq]

with open(f'{data_dir}/ei-vep_53.json') as fd:
    ez_hyp_roi_lbls = json.load(fd)[pat_id]['ez']
ez_hyp_roi_idcs = [
    dyn_mdl.roi_names.index(roi_name) for roi_name in ez_hyp_roi_lbls
]
ez_hyp_bnry = np.zeros(dyn_mdl.num_roi)
ez_hyp_bnry[ez_hyp_roi_idcs] = 1

# Update the order of sensors in the gain matrix s.t. it matches with the
# order of seeg data read with MNE
dyn_mdl.update_gain(data['snsr_picks'])

# %%
epplot.plot_ez_hyp(ez_hyp=ez_hyp_bnry,
                   roi_names=dyn_mdl.roi_names,
                   save_dir=figs_dir,
                   fig_name='ez_hypothesis.png',
                   dpi=500)
# %%
epplot.plot_slp(slp_obs,
                snsr_lbls=dyn_mdl.snsr_lbls_picks,
                save_dir=figs_dir,
                fig_name=f"slp_obs_{szr_name}.png",
                dpi=500)
# %%
x0_prior_mu = -3.0 * np.ones(dyn_mdl.num_roi)
x0_prior_mu[ez_hyp_roi_idcs] = -1.5
x0_prior_mu = tf.constant(x0_prior_mu, dtype=tf.float32)
x0_prior_std = 0.1 * np.ones(dyn_mdl.num_roi)
x0_prior_std[ez_hyp_roi_idcs] = 0.5
x0_prior_std = tf.constant(x0_prior_std, dtype=tf.float32)
x_init_prior_mu = -3.0 * tf.ones(dyn_mdl.num_roi, dtype=tf.float32)
z_init_prior_mu = 5.0 * tf.ones(dyn_mdl.num_roi, dtype=tf.float32)
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
init_cond_mean = prior_mean
# init_cond_std = {
#     'x0': 0.5,
#     'x_init': 0.5,
#     'z_init': 0.5,
#     'eps': 0.1,
#     'K': 5,
# }
init_cond_std = prior_std
x0 = tfd.TruncatedNormal(loc=init_cond_mean['x0'],
                         scale=init_cond_std['x0'],
                         low=dyn_mdl.x0_lb,
                         high=dyn_mdl.x0_ub).sample()
x_init = tfd.TruncatedNormal(loc=init_cond_mean['x_init'],
                             scale=init_cond_std['x_init'],
                             low=dyn_mdl.x_init_lb,
                             high=dyn_mdl.x_init_ub).sample()
z_init = tfd.TruncatedNormal(loc=init_cond_mean['z_init'],
                             scale=init_cond_std['z_init'],
                             low=dyn_mdl.z_init_lb,
                             high=dyn_mdl.z_init_ub).sample()

tau = tf.constant(50.0, dtype=tf.float32)
K = tf.constant(1.0, dtype=tf.float32)
amp = dyn_mdl.amp_bounded(tfd.Normal(
    loc=0.0, scale=1.0).sample())  #tf.constant(1.0, dtype=tf.float32)
offset = dyn_mdl.offset_bounded(tfd.Normal(
    loc=0.0, scale=1.0).sample())  #tf.constant(0.0, dtype=tf.float32)
eps = tf.constant(0.3, dtype=tf.float32)
theta_init_val = dyn_mdl.join_params(*dyn_mdl.inv_transformed_parameters(
    x0, x_init, z_init, tau, K, amp, offset, eps))
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

dyn_mdl.setup_inference(nsteps=slp_obs.shape[0],
                        nsubsteps=2,
                        time_step=0.05,
                        mean=prior_mean,
                        std=prior_std,
                        obs_data=obs_data_aug,
                        obs_space='sensor')
# %%
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2, decay_steps=50, decay_rate=0.96, staircase=True)
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate, decay_steps=100, decay_rate=0.5)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=10)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9)
# %%
start_time = time.time()
niters = tf.constant(500, dtype=tf.int32)
# lr = tf.constant(1e-4, dtype=tf.float32)
losses = train_loop(niters, optimizer)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
# %%
(x0_pred, x_init_pred, z_init_pred, tau_pred, K_pred, amp_pred, offset_pred,
 eps_pred) = dyn_mdl.transformed_parameters(*dyn_mdl.split_params(theta))

y_init_pred = tf.concat((x_init_pred, z_init_pred), axis=0)

y_pred = dyn_mdl.simulate(dyn_mdl._nsteps, dyn_mdl._nsubsteps,
                          dyn_mdl._time_step, y_init_pred, x0_pred, tau_pred,
                          K_pred)
x_pred = y_pred[:, 0:dyn_mdl.num_roi]
slp_pred = dyn_mdl.project_sensor_space(x_pred, amp_pred, offset_pred)
print(f"Scalar Params Prediction\n\
K \t{K_pred:.2f}\n\
tau \t{tau_pred:.2f}\n\
amp \t{amp_pred:.2f}\n\
offset \t{offset_pred:.2f}\n\
eps \t{eps_pred:.2f}\n")
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200, layout='tight')
clim = {
    'min': np.min([slp_obs.min(), slp_pred.numpy().min()]),
    'max': np.max([slp_obs.max(), slp_pred.numpy().max()])
}
epplot.plot_slp(slp_obs, dyn_mdl.snsr_lbls_picks, ax=axs[0], title='Observed', clim=clim)
epplot.plot_slp(slp_pred.numpy(),
                dyn_mdl.snsr_lbls_picks,
                ax=axs[1],
                title='Predicted',
                clim=clim)
fig.savefig(f'{figs_dir}/slp_obs_vs_pred_{szr_name}.png', facecolor='white')
# %%
epplot.plot_src(x_pred.numpy(),
                roi_names=dyn_mdl.roi_names,
                title='Predicted Source Activity',
                dpi=500)
fig.savefig(f'{figs_dir}/src_pred_{szr_name}.png', facecolor='white')
# %%
szr_len = find_szr_len(f'{data_dir}/{pat_id}', szr_name)
onst_wndw_sz = teps_to_wndwsz(szr_len, 10, slp_obs.shape[0])
ez_pred_bnry, pz_pred_bnry = find_ez(x_pred.numpy(), 0.0, onst_wndw_sz)
ez_pred_lbls = [
    dyn_mdl.roi_names[roi_idx] for roi_idx in np.where(ez_pred_bnry == 1)[0]
]
pz_pred_lbls = [
    dyn_mdl.roi_names[roi_idx] for roi_idx in np.where(pz_pred_bnry == 1)[0]
]

p, r = precision_recall(ez_hyp_bnry, ez_pred_bnry)
print(
    f"\n\nCompared to clinical Hypothesis\n \t Precision: {p:.2f} \tRecall: {r:.2f}"
)

with open(f'{data_dir}/surgeries_ch.vep.json') as fd:
    ez_rsctn_roi_lbls = list(json.load(fd)['data'][pat_id]['resection'].keys())
ez_rsctn_roi_idcs = [
    dyn_mdl.roi_names.index(roi_name) for roi_name in ez_rsctn_roi_lbls
]
ez_rsctn_bnry = np.zeros(dyn_mdl.num_roi)
ez_rsctn_bnry[ez_rsctn_roi_idcs] = 1
p, r = precision_recall(ez_rsctn_bnry, ez_pred_bnry)
print(
    f"\nCompared to post surgical MRI\n \t Precision: {p:.2f} \tRecall: {r:.2f}\n"
)

print(f"EZ hypothesis: {ez_hyp_roi_lbls}\n")
print(f"EZ resection: {ez_rsctn_roi_lbls}\n")
print(f"EZ predicted: {ez_pred_lbls}\n")
print(f"PZ predicted: {pz_pred_lbls}\n")
