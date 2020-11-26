# %% [markdown] 
# NFVI - Epileptor with RK4 integrator.ipynb

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import time
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

# %% [markdown]
#### Define dynamical model

# %% 
@tf.function
def epileptor2D_ode_fn(y, x0, tau):
    x = y[0]
    z = y[1]
    I1 = tf.constant(4.1, dtype=tf.float32)
    dx = 1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - z + I1
    dz = (1.0/tau)*(4*(x - x0) - z)
    return tf.stack([dx, dz], axis=0)

@tf.function
def integrator(ode_fn, nsteps, time_step, y_init, x0, tau):
    y = tf.TensorArray(dtype=tf.float32, size=nsteps, clear_after_read=False)
    y_next = y_init
    h = time_step/100
    for i in tf.range(nsteps, dtype=tf.int32):
        for j in tf.range(100):
            k1 = epileptor2D_ode_fn(y_next, x0, tau)
            k2 = epileptor2D_ode_fn(y_next + h*(k1/2), x0, tau)
            k3 = epileptor2D_ode_fn(y_next + h*(k2/2), x0, tau)
            k4 = epileptor2D_ode_fn(y_next + h*k3, x0, tau)
            y_next = y_next + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        y = y.write(i, y_next)
    return y.stack()

# %%
x_init_true = -2.0
z_init_true = 5.0
y_init_true = tf.constant([x_init_true, z_init_true], dtype=tf.float32)

tau_true = tf.constant(25, dtype=tf.float32)
x0_true = tf.constant(-1.8, dtype=tf.float32)
time_step = tf.constant(0.1, dtype=tf.float32)
nsteps = tf.constant(300, dtype=tf.int32)

# %%
y_true = integrator(epileptor2D_ode_fn, nsteps, time_step, y_init_true, x0_true, tau_true)
obs_data = dict()
obs_data['x'] = y_true[:,0].numpy() + tfd.Normal(loc=0, scale=0.1, ).sample(y_true.shape[0])
obs_data['z'] = y_true[:,1]

#%%
plt.figure(figsize=(15,7))
plt.subplot(2,1,1)
plt.plot(obs_data['x'])
plt.xlabel('Time', fontsize=15)
plt.ylabel('x', fontsize=15)

plt.subplot(2,1,2)
plt.plot(obs_data['z'])
plt.xlabel('Time', fontsize=15)
plt.ylabel('z', fontsize=15)
plt.tight_layout()

plt.figure()
plt.title("Phase space plot", fontsize=15)
plt.plot(obs_data['x'], obs_data['z'])
plt.xlabel('x', fontsize=15)
plt.ylabel('z', fontsize=15)

# %% [markdown]
#### Define Generative Model

# %%  
@tf.function
def epileptor2D_log_prob(theta, x_obs):
    time_step = tf.constant(0.1)
    nsteps = tf.constant(300, dtype=tf.int32)
    eps = tf.constant(0.1)
    # y_init = tf.constant([-2.0, 5.0], dtype=tf.float32)
    x0 = theta[0]
    tau = theta[1]
    tau_trans = tf.constant(10.0, dtype=tf.float32) + tf.math.exp(tau)
    y_init = theta[2:4]
    # tf.print('x0=', x0, '\t tau=', tau_trans, '\t y_init=', y_init)
    # log_prob = 0.0
    # Compute Likelihood
    y_pred = integrator(epileptor2D_ode_fn, nsteps, time_step, y_init, x0, tau_trans)
    x_mu = y_pred[:,0]
    likelihood = tf.reduce_sum(tfd.Normal(loc=x_mu, scale=eps).log_prob(x_obs))
    # Compute Prior probability
    prior_x0 = tfd.Normal(loc=-3.0, scale=5.0).log_prob(x0)
    prior_tau = tfd.Normal(loc=0, scale=5.0).log_prob(tau)
    prior_y_init = tfd.Independent(tfd.Normal(loc=[0.0, 0.0], scale=[10.0, 10.0]), reinterpreted_batch_ndims=1).log_prob(y_init)
    return likelihood + prior_x0 + prior_tau + prior_y_init

# %%
# @tf.function
# def find_log_prob(theta):
#     return gm.log_prob(theta)
import time
start_time = time.time()
theta = tf.constant([-1.73248208, -0.970714927, 4.35790968, 1.16606486])
print(epileptor2D_log_prob(theta, obs_data['x']))
print("Elapsed: %s seconds" % (time.time()-start_time))

#%%
x0_range = np.linspace(-3.0,0.0,10)
tau_range = np.linspace(10,30.0,10)
gm_log_prob = np.zeros((x0_range.size, tau_range.size))
for i, x0_val in enumerate(x0_range):
    for j, tau_val in enumerate(tau_range):
        gm_log_prob[j,i] = epileptor2D_log_prob([tf.constant(x0_val, dtype=tf.float32), tf.constant(np.log(tau_val), dtype=tf.float32)], obs_data['x'])

#%%
x0_mesh, tau_mesh = np.meshgrid(x0_range, tau_range)
tau_mesh = tau_mesh + 10.0
plt.contour(x0_mesh, tau_mesh, gm_log_prob, levels=1000)
plt.xlabel('x0', fontsize=15)
plt.ylabel('tau', fontsize=15)
plt.colorbar()
plt.title('True unnormalized Posterior')
# find_log_prob([-3.0, tf.math.log(20.0)])

# %% [markdown]
#### Define the variational posterior using Normalizing flows

#%% 
num_bijectors = 5
tf.random.set_seed(1234567)

bijectors = []
for i in range(num_bijectors-1):
    made = tfb.AutoregressiveNetwork(
        params=2, hidden_units=[256, 256], activation='relu')
    maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
    bijectors.append(maf)
    bijectors.append(tfb.Permute(permutation=tf.random.shuffle(tf.range(4))))

made = tfb.AutoregressiveNetwork(
    params=2, hidden_units=[256, 256], activation='relu')
maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
bijectors.append(maf)
chained_maf = tfb.Chain(list(reversed(bijectors)))
base_dist = tfd.Independent(tfd.Normal(loc=tf.ones(4, dtype=tf.float32),
                                       scale=tf.ones(4, dtype=tf.float32),
                                       name='Base Distribution'),
                            reinterpreted_batch_ndims=1)
flow_dist = tfd.TransformedDistribution(distribution=base_dist,
                                        bijector=chained_maf,
                                        name='Variational Posterior')

#%%
flow_log_prob = np.zeros((x0_range.size, tau_range.size))
for i, x0_val in enumerate(x0_range):
    for j, tau_val in enumerate(tau_range):
        flow_log_prob[j,i] = flow_dist.log_prob([x0_val, tau_val])

#%%
plt.contour(x0_mesh, tau_mesh, flow_log_prob, levels=1000)
plt.colorbar()
plt.title('Variational Posterior before training')

# %% [markdown]
#### Define functions to compute loss and gradients

#%%
@tf.function
def loss(posterior_approx, base_dist_samples, x_obs):
    posterior_samples = posterior_approx.bijector.forward(base_dist_samples)
    # tf.print(posterior_samples)
    # tf.print(posterior_samples)
    nsamples = base_dist_samples.shape[0]
    loss_val = 0.0
    for theta in posterior_samples:
        gm_log_prob = epileptor2D_log_prob(theta, x_obs)
        posterior_approx_log_prob = posterior_approx.log_prob(theta)
        loss_val += (posterior_approx_log_prob - gm_log_prob)/nsamples
        # tf.print("theta: ", theta)
        # tf.print("gm_log_prob:",gm_log_prob, "\t posterior_approx_log_prob:", posterior_approx_log_prob)
        # tf.print("loss_val: ", loss_val)
    return loss_val

# y_init = tf.constant([-2.0, 5.0])
# dt = tf.constant(0.1)
# solution_times = dt * np.arange(0, 300)
# eps = tf.constant(0.1)

@tf.function
def get_loss_and_gradients(posterior_approx, base_dist_samples, x_obs):
    # nsamples = base_dist_samples.shape[0]
    with tf.GradientTape() as tape:
        loss_val = loss(flow_dist, base_dist_samples, x_obs)
        return loss_val, tape.gradient(loss_val, posterior_approx.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# %%
# @tf.function
# def find_loss():
#     loss_val = loss(flow_dist, tf.constant([[-0.29448965, -0.95975673],[ 0.17711619,  0.222046]], dtype=tf.float32))
#     return loss_val
base_dist_samples = base_dist.sample((5))
for i in range(10):
    loss_val, grads = get_loss_and_gradients(flow_dist, base_dist_samples)
    tf.print('Final loss_val = ', loss_val)
    # tf.print(grads)
    optimizer.apply_gradients(zip(grads, flow_dist.trainable_variables))

# %%
# base_dist_samples = tf.constant([[-0.29448965, -0.95975673],[ 0.17711619,  0.222046]], dtype=tf.float32)
# posterior_samples = flow_dist.bijector.forward(base_dist_samples)
# for theta in posterior_samples:
#     gm_log_prob = gm.log_prob(theta)
#     print(gm_log_prob)


# %% [markdown]
#### Training
# %%
batch_size = 5
base_dist_samples = tf.data.Dataset.from_tensor_slices(
    base_dist.sample((batch_size*500))).batch(batch_size)

# %%
num_epochs = 1

training_loss = []
start_time = time.time()

for epoch in range(num_epochs):
    for batch_base_dist_samples in base_dist_samples.as_numpy_iterator():
        loss_value, grads = get_loss_and_gradients(flow_dist, tf.constant(
            batch_base_dist_samples, dtype=tf.float32), obs_data['x'])
        tf.print("loss: ", loss_value)
        training_loss.append(loss_value)
        optimizer.apply_gradients(
            zip([tf.divide(el, batch_size) for el in grads], flow_dist.trainable_variables))
    # if ((epoch+1) % 10 == 0):
    #     print(f'Epoch {epoch + 1}:\n Training Loss: {loss_value}')
    print(f"Elapsed: {time.time()-start_time} seconds for {num_epochs} Epochs")

# %% [markdown]
#### Results

# %%
# samples = flow_dist.sample((1000))
samples = np.zeros([batch_size*500, 4])
i = 0
for batch_base_dist_samples in base_dist_samples.as_numpy_iterator():
    samples[i*batch_size:(i+1)*batch_size] = flow_dist.bijector.forward(batch_base_dist_samples)
    i += 1

plt.figure(figsize=(25,5))
plt.subplot(1,4,1)
plt.hist(samples[:,0], density=True, color='black')
plt.axvline(x0_true, color='red', label='Ground truth')
plt.legend()
plt.xlabel('x0')

plt.subplot(1,4,2)
plt.hist(10 + np.exp(samples[:,1]), density=True, color='black')
plt.axvline(tau_true, color='red', label='Ground truth')
plt.legend()
plt.xlabel('tau')

plt.subplot(1,4,3)
plt.hist(samples[:,2], density=True, color='black')
plt.axvline(x_init_true, color='red', label='Ground truth')
plt.legend()

plt.xlabel('x_init')
plt.subplot(1,4,4)
plt.hist(samples[:,3], density=True, color='black')
plt.axvline(z_init_true, color='red', label='Ground truth')
plt.legend()
plt.xlabel('z_init')

# %%
x0 = tf.reduce_mean(samples[:,0], axis=0)
tau = tf.reduce_mean(samples[:,1], axis=0)
tau_trans = 10.0 + tf.math.exp(tau)
y_init = tf.reduce_mean(samples[:,2:4], axis=0)
t_init = tf.constant(0.0, dtype=tf.float32)
time_step = tf.constant(0.1, dtype=tf.float32)
solution_times = time_step * tf.range(0, 300, dtype=tf.float32)
y = integrator(epileptor2D_ode_fn, nsteps, time_step, y_init_true, x0_true, tau_true)

# %%
plt.figure(figsize=(15,7))
plt.subplot(2,1,1)
plt.plot(y[:,0], label='Prediction at Posterior Mean', lw=5.0, color='black')
plt.plot(obs_data['x'], color='red', label='Ground truth', alpha=0.8)
plt.xlabel('Time', fontsize=15)
plt.ylabel('x', fontsize=15)
plt.legend()

plt.subplot(2,1,2)
plt.plot(y[:,1], label='Prediction at Posterior Mean', lw=5.0, color='black')
plt.plot(obs_data['z'], color='red', label='Ground truth', alpha=0.8)
plt.xlabel('Time', fontsize=15)
plt.ylabel('z', fontsize=15)
plt.tight_layout()
plt.legend()

plt.figure()
plt.title("Phase space plot", fontsize=15)
plt.plot(y[:,0], y[:,1], color='black', lw=5.0)
plt.plot(obs_data['x'], obs_data['z'], color='red', alpha=0.8)
plt.xlabel('x', fontsize=15)
plt.ylabel('z', fontsize=15)
