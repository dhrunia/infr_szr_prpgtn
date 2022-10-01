import lib.model.neuralfield
import time
import tensorflow_probability as tfp
import os
import sys
import numpy as np
import tensorflow as tf

tfd = tfp.distributions
tfb = tfp.bijectors

data_dir = sys.argv[1]
results_dir = sys.argv[2]
N_LAT = int(sys.argv[3])
RUNID = int(sys.argv[4])
N_LON = 2 * N_LAT
os.makedirs(results_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=32,
    N_LAT=N_LAT,
    N_LON=N_LON,
    verts_irreg_fname=f"{data_dir}/vertices.txt",
    rgn_map_irreg_fname=f"{data_dir}/Cortex_region_map_ico7.txt",
    conn_zip_path=f"{data_dir}/connectivity.vep.zip",
    gain_irreg_path=f"{data_dir}/gain_inv_square_ico7.npz",
    gain_irreg_rgn_map_path=f"{data_dir}/gain_region_map_ico7.txt",
    L_MAX_PARAMS=16,
    diff_coeff=0.00047108,
    alpha=2.0,
    theta=-1.0)

sim_data = np.load(f'{data_dir}/sim_N_LAT{N_LAT}.npz')
ez_hyp_roi = [
    dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in sim_data['ez_roi_tvb']
]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
dyn_mdl.SC = tf.constant(sim_data['SC'], dtype=tf.float32)

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
    'K': 1.0
}
std = {'x0': x0_prior_std, 'x_init': 0.5, 'z_init': 0.5, 'eps': 0.1, 'K': 5}

dyn_mdl.setup_inference(nsteps=300,
                        nsubsteps=2,
                        time_step=tf.constant(0.05, dtype=tf.float32),
                        mean=mean,
                        std=std,
                        obs_data=sim_data['slp_aug'],
                        param_space='mode',
                        obs_space='sensor')

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

boundaries = [500, 8000]
values = [1e-2, 1e-3, 1e-4]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=10)
start_time = time.time()
niters = tf.constant(10000, dtype=tf.int32)
losses = train_loop(niters, optimizer)
np.savez(f'{results_dir}/res_N_LAT{N_LAT}_run{RUNID}.npz',
         theta=theta.numpy(),
         losses=losses)
print(f"Elapsed {time.time() - start_time} seconds for {niters} iterations")