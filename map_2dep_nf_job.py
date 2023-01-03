import lib.model.neuralfield
import time
import tensorflow_probability as tfp
import os
import sys
import numpy as np
import tensorflow as tf

tfd = tfp.distributions
tfb = tfp.bijectors

DATA_DIR = sys.argv[1]
RESULTS_DIR = sys.argv[2]
N_LAT = int(sys.argv[3])
L_MAX = int(sys.argv[4])
L_MAX_PARAMS = int(sys.argv[5])
SNR_DB = float(sys.argv[6])
RUNID = int(sys.argv[7])

N_LON = 2 * N_LAT
os.makedirs(RESULTS_DIR, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=L_MAX,
    N_LAT=N_LAT,
    N_LON=N_LON,
    verts_irreg_fname=f"{DATA_DIR}/vertices.txt",
    rgn_map_irreg_fname=f"{DATA_DIR}/Cortex_region_map_ico7.txt",
    conn_zip_path=f"{DATA_DIR}/connectivity.vep.zip",
    gain_irreg_path=f"{DATA_DIR}/gain_inv_square_ico7.npz",
    gain_irreg_rgn_map_path=f"{DATA_DIR}/gain_region_map_ico7.txt",
    L_MAX_PARAMS=L_MAX_PARAMS)

sim_data = np.load(f'{DATA_DIR}/sim_N_LAT{N_LAT}.npz')
ez_true_roi_tvb = sim_data['ez_roi_tvb']
ez_true_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_true_roi_tvb]
dyn_mdl.SC = tf.constant(sim_data['SC'], dtype=tf.float32)

slp = tf.constant(sim_data['slp'], dtype=tf.float32)
_slp = slp - tf.reduce_mean(slp, axis=0, keepdims=True)
avg_pwr = tf.reduce_max(tf.reduce_mean(_slp**2, axis=0))
# Convert SNR from Decibels
snr = 10**(SNR_DB / 10)
noise_std = tf.sqrt(avg_pwr / snr)
slp_noised = slp + tfd.Normal(loc=tf.zeros_like(slp),
                              scale=noise_std * tf.ones_like(slp)).sample()

N_SAMPLE_AUG = 50
EPS_AUG = 0.1
obs_data_aug = tf.TensorArray(dtype=tf.float32, size=N_SAMPLE_AUG)
for j in range(N_SAMPLE_AUG):
    _slp = slp_noised + \
        tf.random.normal(shape=slp_noised.shape, mean=0, stddev=EPS_AUG)
    obs_data_aug = obs_data_aug.write(j, _slp)
obs_data_aug = obs_data_aug.stack()

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
    'K': 1.0
}
prior_std = {
    'x0': x0_prior_std,
    'x_init': 0.5,
    'z_init': 0.5,
    'eps': 0.1,
    'K': 5
}

dyn_mdl.setup_inference(nsteps=300,
                        nsubsteps=2,
                        time_step=tf.constant(0.05, dtype=tf.float32),
                        mean=prior_mean,
                        std=prior_std,
                        obs_data=obs_data_aug,
                        param_space='mode',
                        obs_space='sensor')

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


@tf.function
def get_loss_and_gradients():
    with tf.GradientTape() as tape:
        loss = -1.0 * dyn_mdl.log_prob(theta, 1)
    grads = tape.gradient(loss, [theta])
    return loss, grads


def train_loop(num_iters, optimizer):
    loss_at = tf.TensorArray(size=num_iters, dtype=tf.float32)

    def cond(i, loss_at):
        return tf.less(i, num_iters)

    def body(i, loss_at):
        loss_value, grads = get_loss_and_gradients()
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


boundaries = [500, 8000]
values = [1e-2, 1e-3, 1e-4]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
optmzr = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10)
start_time = time.time()
niters = tf.constant(10000, dtype=tf.int32)
losses = train_loop(niters, optmzr)

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

np.savez(
    f'{RESULTS_DIR}/res_N_LAT{N_LAT:d}_L_MAX{L_MAX:d}_L_MAX_PARAMS{L_MAX_PARAMS:d}_SNR{SNR_DB:.1f}_run{RUNID:d}.npz',
    theta=theta.numpy(),
    losses=losses,
    ez_hyp_roi=ez_hyp_roi)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
