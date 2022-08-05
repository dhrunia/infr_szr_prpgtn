# %%
import tensorflow as tf
from svgd import SVGD
import tensorflow_probability as tfp
import time
import matplotlib.pyplot as plt
# %%
prior_dist = tfp.distributions.MultivariateNormalDiag(loc=[0., 0.],
                                                      scale_diag=[5., 5.])
samples = prior_dist.sample([10])
# %%
class GMM():

    @tf.function
    def log_prob(self, theta):
        dist = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=[0.3, 0.7]),
            components_distribution=tfp.distributions.MultivariateNormalDiag(
                loc=[[-5., -6.],
                     [7.0, 5.0]],  # One for each component.
                scale_diag=[[0.1, 0.1],
                            [0.1, 0.1]]))  # And same here.
        lp = dist.log_prob(theta)
        # tf.print(lp)
        return lp
# %%
svgd_sampler = SVGD(dyn_mdl=GMM())
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# %%
start_time = time.time()
n_iters = 2000
samples = svgd_sampler.update(samples, n_iters)
print(f"Elapse {time.time() - start_time} seconds")
# %%
x = tf.cast(tf.linspace(-10, 10, 500), dtype=tf.float32)
y = tf.cast(tf.linspace(-10, 10, 500), dtype=tf.float32)
X, Y = tf.meshgrid(x, y)
m = GMM()
t = tf.transpose(tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)]))
Z = m.log_prob(t)
Z = tf.reshape(Z, X.shape)
# %%
fig, ax = plt.subplots()
cs = ax.contour(X, Y, Z, 50)
ax.clabel(cs, inline=True)
ax.scatter(samples[:, 1], samples[:, 0])