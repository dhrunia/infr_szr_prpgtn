# %%
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import lib.utils.sht as tfsht
import time
import lib.utils.tnsrflw
import lib.plots.neuralfield
import lib.plots.seeg
import os
import lib.model.neuralfield

tfd = tfp.distributions
tfb = tfp.bijectors
import os

# %%
results_dir = 'tmp'
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
    tf.ones(dyn_mdl.nv, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(dyn_mdl.nv)
ez_hyp_roi = [116, 127, 151]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map_reg)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32)
t = dyn_mdl.SC.numpy()
t[140 - 9, 116 - 9] = 5.0
dyn_mdl.SC = tf.constant(t, dtype=tf.float32)
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)

# @tf.function
# def run_sim(nsteps, nsubsteps, time_step, y_init, x0, tau, K):
#     y = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init, x0, tau, K)
#     return y

y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                         tau_true, K_true)
x_obs = y_obs[:, 0:dyn_mdl.nv] * dyn_mdl.unkown_roi_mask
slp_obs = dyn_mdl.project_sensor_space(x_obs)
# %%
lib.plots.seeg.plot_slp(slp_obs.numpy(),
                        save_dir=figs_dir,
                        fig_name='slp_obs.png')
# %%
nparams = 4 * ((dyn_mdl.L_MAX_PARAMS + 1)**2)
loc = tf.Variable(initial_value=tf.zeros(nparams))
scale_diag = 0.1 * tf.ones(nparams)
# var_dist = tfd.MultivariateNormalDiag(
#     loc=loc, scale_diag=scale_diag, name="variational_posterior")

# %%


@tf.function
def get_loss_and_gradients(nsamples):
    loss = tf.constant(0, dtype=tf.float32, shape=(1, 1))
    loc_grad = tf.zeros_like(loc)

    def cond(i, loc_grad, loss):
        return tf.less(i, nsamples)

    def body(i, loc_grad, loss):
        with tf.GradientTape() as tape:
            theta = tfd.MultivariateNormalDiag(loc=loc,
                                               scale_diag=scale_diag).sample(1)
            # loss_val = tf.constant(0.0, shape=(1, ), dtype=tf.float32)
            gm_log_prob = tf.reduce_sum(dyn_mdl.log_prob(theta))
            posterior_approx_log_prob = tfd.MultivariateNormalDiag(
                loc=loc, scale_diag=scale_diag).log_prob(theta)
            tf.print("\tgm_log_prob:", gm_log_prob,
                     "\tposterior_approx_log_prob:", posterior_approx_log_prob)
            loss_i = posterior_approx_log_prob - gm_log_prob
        grads = tape.gradient(loss_i, [loc])
        loc_grad += grads[0] / tf.cast(nsamples, dtype=tf.float32)
        loss += loss_i / tf.cast(nsamples, dtype=tf.float32)
        return i + 1, loc_grad, loss

    i = tf.constant(0, dtype=tf.uint32)
    i, loc_grad, loss = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=(i, loc_grad, loss),
        parallel_iterations=1)
    return loss, [loc_grad]


# %%
@tf.function
def train_loop(num_epochs, nsamples):
    def cond(i):
        return tf.less(i, num_epochs)

    def body(i):
        loss_value, grads = get_loss_and_gradients(nsamples)
        # grads = [tf.divide(el, batch_size) for el in grads]
        # grads = [tf.clip_by_norm(el, 1000) for el in grads]
        # tf.print("gradient norm = ", [tf.norm(el) for el in grads], \
        # output_stream="file://debug.log")
        tf.print("Epoch ", i, "loss: ", loss_value)
        # training_loss.append(loss_value)
        optimizer.apply_gradients(zip(grads, [loc]))
        return i + 1

    i = tf.constant(0, dtype=tf.uint32)
    i = tf.while_loop(cond=cond,
                      body=body,
                      loop_vars=(i, ),
                      parallel_iterations=1)


# %%
x0_prior_mu = -3.0 * tf.ones(dyn_mdl.nv)
dyn_mdl.setup_inference(slp_obs=slp_obs,
                        nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        y_init=y_init_true,
                        tau=tau_true,
                        K=K_true,
                        x0_prior_mu=x0_prior_mu)
# %%
# initial_learning_rate = 1e-2
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=200, decay_rate=0.96, staircase=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# %%
num_epochs = tf.constant(500, dtype=tf.uint32)
nsamples = tf.constant(1, dtype=tf.uint32)
# %%
start_time = time.time()
train_loop(num_epochs, nsamples)
print(f"Elapsed {time.time() - start_time} seconds for {num_epochs} Epochs")
# %%
nsamples = 100
posterior_samples = tfd.MultivariateNormalDiag(
    loc=loc, scale_diag=scale_diag).sample(nsamples)
x0_samples = tf.TensorArray(dtype=tf.float32,
                            size=nsamples,
                            clear_after_read=False)
for i, theta in enumerate(posterior_samples.numpy()):
    x0_lm_i = theta[0:4 * dyn_mdl.nmodes_params]
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
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
lib.plots.seeg.plot_slp(slp_obs.numpy(), ax=axs[0], title='Observed')
lib.plots.seeg.plot_slp(slp_ppc.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
