# %%
from lib.model import nmm
import lib.plots.seeg
import time
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
results_dir = "results/exp91"
os.makedirs(results_dir, exist_ok=True)
figs_dir = f"{results_dir}/figures"
os.makedirs(figs_dir, exist_ok=True)

pat_id = 'id001_bt'
dyn_mdl = nmm.Epileptor2D(
    conn_path=f'datasets_low_res/retro/{pat_id}/tvb/connectivity.vep.zip',
    gain_path=f'datasets_low_res/retro/{pat_id}/elec/gain_inv-square.vep.txt')
# %%
with open('datasets_low_res/retro/ei-vep_53.json') as fd:
    ez_hyp_roi_lbls = json.load(fd)[pat_id]['ez']
ez_hyp_roi_idcs = []
for roi_name in ez_hyp_roi_lbls:
    ez_hyp_roi_idcs.append(dyn_mdl.roi_names.index(roi_name))

x0_true = -3.0 * np.ones(dyn_mdl.num_roi)
x0_true[ez_hyp_roi_idcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32)
x_init_true = tf.constant(-2.0, dtype=tf.float32) * tf.ones(dyn_mdl.num_roi,
                                                            dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * tf.ones(dyn_mdl.num_roi,
                                                           dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32)

t = dyn_mdl.SC.numpy()
t[116, 153] = 2.0
t[102, 153] = 2.0
t[80, 134] = 2.0
K_true = t.max()
t = t / t.max()
dyn_mdl.SC = tf.constant(t, dtype=tf.float32)

amp_true = dyn_mdl.amp_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
offset_true = dyn_mdl.offset_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
eps_obs_true = 0.1

# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.1, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)
y = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                     tau_true, K_true)
x_true = y[:, 0:dyn_mdl.num_roi]
z_true = y[:, dyn_mdl.num_roi:2 * dyn_mdl.num_roi]
slp_true = dyn_mdl.project_sensor_space(x_true, amp_true, offset_true)

# %%
lib.plots.seeg.plot_slp(slp_true.numpy(),
                        save_dir=figs_dir,
                        fig_name="slp_obs.png")
# %%
x0_prior_mu = -3.0 * np.ones(dyn_mdl.num_roi)
x0_prior_mu[ez_hyp_roi_idcs] = -1.5
x0_prior_mu = tf.constant(x0_prior_mu, dtype=tf.float32)
x_init_prior_mu = -3.0 * tf.ones(dyn_mdl.num_roi, dtype=tf.float32)
z_init_prior_mu = 5.0 * tf.ones(dyn_mdl.num_roi, dtype=tf.float32)
mean = {
    'x0': x0_prior_mu,
    'x_init': x_init_prior_mu,
    'z_init': z_init_prior_mu,
    'eps': 0.1,
    'K': 1.0,
}
std = {
    'x0': 0.1,
    'x_init': 0.1,
    'z_init': 0.1,
    'eps': 0.01,
    'K': 5,
}

#%%
nparams = 3 * dyn_mdl.num_roi + 5
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

tau = tf.constant(50.0, dtype=tf.float32)
K = tf.constant(1.0, dtype=tf.float32)
amp = tf.constant(1.0, dtype=tf.float32)
offset = tf.constant(0.0, dtype=tf.float32)
eps = tf.constant(0.3, dtype=tf.float32)
loc_init_val = dyn_mdl.join_params(*dyn_mdl.inv_transformed_parameters(
    x0, x_init, z_init, tau, K, amp, offset, eps))
loc = tf.Variable(initial_value=loc_init_val)
log_scale_diag = tf.Variable(initial_value=-1.0 * tf.ones(nparams))

# %%


@tf.function
def get_loss_and_gradients(nsamples):
    with tf.GradientTape() as tape:
        scale_diag = tf.exp(log_scale_diag)
        theta = tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag).sample(nsamples)
        gm_log_prob = dyn_mdl.log_prob(theta, nsamples)
        posterior_approx_log_prob = tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=scale_diag).log_prob(theta)
        tf.print("\tgm_log_prob:", tf.reduce_mean(gm_log_prob),
                 "\tposterior_approx_log_prob:",
                 tf.reduce_mean(posterior_approx_log_prob))
        loss = tf.reduce_mean(posterior_approx_log_prob - gm_log_prob, axis=0)
    grads = tape.gradient(loss, [loc, log_scale_diag])
    return loss, grads


# %%


# @tf.function
def train_loop(num_iters, optimizer):
    loss_at = tf.TensorArray(size=num_iters, dtype=tf.float32)

    def cond(i, loss_at):
        return tf.less(i, num_iters)

    def body(i, loss_at):
        loss_value, grads = get_loss_and_gradients(1)
        # tf.print("NAN in grads: ", tf.reduce_any(tf.math.is_nan(grads)), output_stream='file:///workspaces/isp_neural_fields/debug.txt')
        loss_at = loss_at.write(i, loss_value)
        tf.print("Iter ", i + 1, "loss: ", loss_value)
        optimizer.apply_gradients(zip(grads, [loc, log_scale_diag]))
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
        tf.random.normal(shape=slp_true.shape, mean=0, stddev=eps_obs_true)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()
# %%

dyn_mdl.setup_inference(nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        mean=mean,
                        std=std,
                        obs_data=obs_data_aug,
                        obs_space='sensor')
# %%
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2, decay_steps=50, decay_rate=0.96, staircase=True)
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     initial_learning_rate, decay_steps=100, decay_rate=0.5)

# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=10)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7, momentum=0.9)
# %%
start_time = time.time()
niters = tf.constant(1000, dtype=tf.int32)
# lr = tf.constant(1e-4, dtype=tf.float32)
losses = train_loop(niters, optimizer)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")
# %%
scale_diag = tf.exp(log_scale_diag)
theta = tf.reduce_mean(tfd.MultivariateNormalDiag(
    loc=loc, scale_diag=scale_diag).sample(100),
                       axis=0)
(x0_pred, x_init_pred, z_init_pred, tau_pred, K_pred, amp_pred, offset_pred,
 eps_pred) = dyn_mdl.transformed_parameters(*dyn_mdl.split_params(theta))

y_init_pred = tf.concat((x_init_pred, z_init_pred), axis=0)

y_pred = dyn_mdl.simulate(dyn_mdl._nsteps, dyn_mdl._nsubsteps,
                          dyn_mdl._time_step, y_init_pred, x0_pred, tau_pred,
                          K_pred)
x_pred = y_pred[:, 0:dyn_mdl.num_roi]
slp_pred = dyn_mdl.project_sensor_space(x_pred, amp_pred, offset_pred)
print(f"Param \tGround Truth \tPrediction\n\
K \t{K_true:.2f} \t{K_pred:.2f}\n\
eps \t{eps_obs_true:.2f} \t{eps_pred:.2f}\n\
tau \t{tau_true:.2f} \t{tau_pred:.2f}\n\
amp \t{amp_true:.2f} \t{amp_pred:.2f}\n\
offset \t{offset_true:.2f} \t{offset_pred:.2f}")
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
lib.plots.seeg.plot_slp(slp_true.numpy(), ax=axs[0], title='Observed')
lib.plots.seeg.plot_slp(slp_pred.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
lib.plots.seeg.plot_slp(x_true.numpy(), ax=axs[0], title='Observed')
lib.plots.seeg.plot_slp(x_pred.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
# %%
theta_true = theta = dyn_mdl.join_params(*dyn_mdl.inv_transformed_parameters(
    x0_true, x_init_true, z_init_true, tau_true, K_true, amp_true, offset_true,
    eps))
loss_gt = -1.0 * dyn_mdl.log_prob(theta_true, 1)
print(f"loss_gt = {loss_gt}")