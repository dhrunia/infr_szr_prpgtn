# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import lib.preprocess.envelope
step_module = tf.load_op_library('./eulerstep_2d_epileptor.so')
# %%
@tf.function
def integrator(nsteps, theta, y_init, SC):
    y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
    y_next = y_init
    for i in tf.range(nsteps, dtype=tf.int32):
        y_next = step_module.euler_step2d_epileptor(theta, y_next, SC)
        y = y.write(i, y_next)
    return y.stack()
# %% Run a simulation
tvb_syn_data = np.load("datasets/syn_data/id001_bt/syn_tvb_ez=48-79_pz=11-17-22-75.npz")
SC = np.load(f'datasets/syn_data/id001_bt/network.npz')['SC']
K_true = tf.constant(np.max(SC), dtype=tf.float32, shape=(1,))
SC = SC / K_true.numpy()
SC[np.diag_indices(SC.shape[0])] = 0
SC = tf.constant(SC, dtype=tf.float32)
gain = tf.constant(
    np.loadtxt('datasets/syn_data/id001_bt/gain_inv-square.destrieux.txt'), 
    dtype=tf.float32)
nn = SC.shape[0]
x_init_true = tf.constant(-2.0, dtype=tf.float32) * tf.ones(nn, dtype=tf.float32)
z_init_true = tf.constant(3.8, dtype=tf.float32) * tf.ones(nn, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=(1,))
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = tf.constant(-2.16*np.ones(nn), dtype=tf.float32)
theta_true = tf.concat((x0_true, tau_true, K_true), axis=0)
# time_step = tf.constant(0.1, dtype=tf.float32)
nsteps = tf.constant(280, dtype=tf.int32)
y_true = integrator(nsteps, theta_true, y_init_true, SC)
x_true = y_true[:, 0:nn]
z_true = y_true[:, nn:2*nn]
alpha = 1
beta = 0
slp_true = alpha * \
        tf.math.log(tf.matmul(tf.math.exp(x_true), gain, transpose_b=True)) + \
        beta
seeg = tvb_syn_data['seeg']
start_idx = 800
end_idx = 2200
obs_data = dict()
slp = lib.preprocess.envelope.compute_slp_syn(seeg[:, start_idx:end_idx].T, samp_rate=256, win_len=50, hpf=10.0, lpf=2.0, logtransform=True)
obs_data['slp'] = tf.constant(slp[::5, :], dtype=tf.float32)
# %%
plt.figure(figsize=(25,10))
plt.subplot(211)
plt.plot(y_true[:,0:nn])
plt.ylabel('x')
plt.subplot(212)
plt.plot(y_true[:, nn:2*nn])
plt.ylabel('z')
plt.figure(figsize=(25,5))
plt.plot(slp_true, 'k');
plt.plot(obs_data['slp'], 'r', alpha=0.5)
plt.ylabel('slp')
# %%
plt.plot(y_true[:,79])