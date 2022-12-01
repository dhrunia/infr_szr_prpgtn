# %%
from lib.model import nmm
import lib.preprocess.envelope as envelope
import time
import os
import numpy as np
import json
import tensorflow as tf
import tensorflow_probability as tfp
import sys

tfd = tfp.distributions

# %%
PAT_ID = sys.argv[1]
SZR_NAME = sys.argv[2]
DATA_DIR = sys.argv[3]
RESULTS_DIR = sys.argv[4]
RUN_ID = int(sys.argv[5])

PAT_DATA_DIR = os.path.join(DATA_DIR, PAT_ID)
# FIT_DATA_PATH = os.path.join(PAT_DATA_DIR, 'fit', f'data_{SZR_NAME}.npz')
PAT_RESULTS_DIR = os.path.join(RESULTS_DIR, PAT_ID)
# FIGS_DIR = f"{PAT_RESULTS_DIR}/figures"

os.makedirs(PAT_RESULTS_DIR, exist_ok=True)
# os.makedirs(FIGS_DIR, exist_ok=True)

# %%
dyn_mdl = nmm.Epileptor2D(
    conn_path=f'{PAT_DATA_DIR}/tvb/connectivity.vep.zip',
    gain_path=f'{PAT_DATA_DIR}/tvb/gain_inv_square_ico7.npz',
    gain_rgn_map_path=f'{PAT_DATA_DIR}/tvb/gain_region_map_ico7.txt',
    seeg_xyz_path=f'{PAT_DATA_DIR}/elec/seeg.xyz',
    gain_mat_res='high')
# %%
# data = np.load(FIT_DATA_PATH)
lpf = 0.2
hpf = 10.0
npoints = 300
SZR_NAME = SZR_NAME.split("_lpf")[0]
raw_seeg_fname = SZR_NAME + '.raw.fif'
meta_data_fname = SZR_NAME + '.json'
data = envelope.prepare_data(PAT_DATA_DIR, meta_data_fname, raw_seeg_fname,
                             hpf, lpf)
# Downsample the seeg to ~npoints
ds_freq = int(data['slp'].shape[0] / npoints)
slp_obs = data['slp'][0:-1:ds_freq]

# Data Augmentation
n_sample_aug = 50
obs_data_aug = tf.TensorArray(dtype=tf.float32, size=n_sample_aug)
for j in range(n_sample_aug):
    data_noised = slp_obs + \
        tf.random.normal(shape=slp_obs.shape, mean=0, stddev=0.1)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()

with open(f'{DATA_DIR}/ei-vep_53.json') as fd:
    ez_hyp_roi_lbls = json.load(fd)[PAT_ID]['ez']
ez_hyp_roi_idcs = [
    dyn_mdl.roi_names.index(roi_name) for roi_name in ez_hyp_roi_lbls
]
ez_hyp_bnry = np.zeros(dyn_mdl.num_roi)
ez_hyp_bnry[ez_hyp_roi_idcs] = 1

# Update the order of sensors in the gain matrix s.t. it matches with the
# order of seeg data read with MNE
dyn_mdl.update_gain(data['snsr_picks'])

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
amp = dyn_mdl.amp_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
offset = dyn_mdl.offset_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
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

dyn_mdl.setup_inference(nsteps=obs_data_aug.shape[1],
                        nsubsteps=2,
                        time_step=0.05,
                        mean=prior_mean,
                        std=prior_std,
                        obs_data=obs_data_aug,
                        obs_space='sensor')
# %%
boundaries = [500, 8000]
values = [1e-2, 1e-3, 1e-4]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10)
# %%
start_time = time.time()
niters = tf.constant(10000, dtype=tf.int32)
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
z_pred = y_pred[:, dyn_mdl.num_roi:2 * dyn_mdl.num_roi]
slp_pred = dyn_mdl.project_sensor_space(x_pred, amp_pred, offset_pred)
# %%
np.savez(f'{PAT_RESULTS_DIR}/map_estimate_{SZR_NAME}_run{RUN_ID:d}.npz',
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
         slp=slp_pred.numpy(),
         ez_hyp_bnry=ez_hyp_bnry,
         ez_hyp_roi_lbls=ez_hyp_roi_lbls)