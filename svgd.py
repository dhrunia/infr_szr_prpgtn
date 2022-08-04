import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.spatial.distance import pdist, squareform

# class SVGD():

#     def __init__(self, dyn_mdl):
#         self._dyn_mdl = dyn_mdl

#     def svgd_kernel(self, theta, h=-1):
#         sq_dist = pdist(theta)
#         pairwise_dists = squareform(sq_dist)**2
#         if h < 0:  # if h < 0, using median trick
#             h = np.median(pairwise_dists)
#             h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

#         # compute the rbf kernel
#         Kxy = np.exp(-pairwise_dists / h**2 / 2)

#         dxkxy = -np.matmul(Kxy, theta)
#         sumkxy = np.sum(Kxy, axis=1)
#         for i in range(theta.shape[1]):
#             dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
#         dxkxy = dxkxy / (h**2)
#         return (Kxy, dxkxy)

#     def update(self,
#                x0,
#                lnprob,
#                n_iter=1000,
#                stepsize=1e-3,
#                bandwidth=-1,
#                alpha=0.9,
#                debug=False):
#         # Check input
#         if x0 is None or lnprob is None:
#             raise ValueError('x0 or lnprob cannot be None!')

#         theta = np.copy(x0)

#         # adagrad with momentum
#         fudge_factor = 1e-6
#         historical_grad = 0
#         for i in range(n_iter):
#             if debug and (i + 1) % 1000 == 0:
#                 print('iter ' + str(i + 1))

#             lnpgrad = self.compute_lp_grad(theta)
#             # calculating the kernel matrix
#             kxy, dxkxy = self.svgd_kernel(theta, h=-1)
#             grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

#             # adagrad
#             if i == 0:
#                 historical_grad = historical_grad + grad_theta**2
#             else:
#                 historical_grad = alpha * historical_grad + (1 - alpha) * (
#                     grad_theta**2)
#             adj_grad = np.divide(grad_theta,
#                                  fudge_factor + np.sqrt(historical_grad))
#             theta = theta + stepsize * adj_grad

#         return theta

#     def compute_lp_grad(theta):

# def svgd_kernel(theta, h=-1):
#     sq_dist = pdist(theta)
#     pairwise_dists = squareform(sq_dist)**2
#     if h < 0:  # if h < 0, using median trick
#         h = np.median(pairwise_dists)
#         h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

#     # compute the rbf kernel
#     Kxy = np.exp(-pairwise_dists / h**2 / 2)

#     dxkxy = -np.matmul(Kxy, theta)
#     sumkxy = np.sum(Kxy, axis=1)
#     for i in range(theta.shape[1]):
#         dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
#     dxkxy = dxkxy / (h**2)
#     return (Kxy, dxkxy)


@tf.function
def rbf_kernel(theta, h=-1.0):
    squared_dists = sq_dist(theta)

    if h < 0:
        h = tfp.stats.percentile(squared_dists, q=50, interpolation='midpoint')
        h = tf.sqrt(0.5 * h / tf.math.log(theta.shape[0] + 1.0))

    Kxy = tf.math.exp(-squared_dists / (2.0 * h**2))

    dxkxy = -tf.matmul(Kxy, theta)
    sumkxy = tf.reduce_sum(Kxy, axis=1)
    tmp = tf.TensorArray(size=theta.shape[1], dtype=tf.float32)
    for i in range(theta.shape[1]):
        tmp = tmp.write(i, dxkxy[:, i] + theta[:, i] * sumkxy)
    dxkxy = tf.transpose(tmp.stack()) / h**2

    return Kxy, dxkxy


@tf.function
def sq_dist(theta):
    # Squared sum of each point
    sq_sum = tf.reduce_sum(theta**2, axis=1, keepdims=True)
    # Pairwise dot product
    pdp = tf.matmul(theta, theta, transpose_b=True)
    dists = sq_sum + tf.transpose(sq_sum) - 2.0 * pdp
    return dists


@tf.function
def find_opt_prtrbtn(theta, dyn_mdl):
    lnpgrad = tf.TensorArray(dtype=tf.float32, size=theta.shape[0])

    def cond(i, lnpgrad, lnp_sum):
        return tf.less(i, theta.shape[0])

    def body(i, lnpgrad, lnp_sum):
        with tf.GradientTape() as tape:
            theta_i = theta[i]
            tape.watch(theta_i)
            lnp = dyn_mdl.log_prob(theta_i)
            lnpgrad = lnpgrad.write(i, tape.gradient(lnp, theta_i))
        return i + 1, lnpgrad, lnp_sum + lnp
    i = tf.constant(0, dtype=tf.int32)
    lnp_sum = tf.constant(0.0, dtype=tf.float32, shape=(1,))
    i, lnpgrad, lnp_sum = tf.while_loop(cond=cond,
                               body=body,
                               loop_vars=[i, lnpgrad, lnp_sum],
                               parallel_iterations=1)
    tf.print(lnp_sum)
    lnpgrad = lnpgrad.stack()
    # calculating the kernel matrix
    kxy, dxkxy = rbf_kernel(theta)
    phi_star = (tf.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0]
    return phi_star


# @tf.function
def update(dyn_mdl,
           theta,
           n_iters,
           alpha=0.9,
           stepsize=1e-3,
           fudge_factor=1e-6):
    for i in range(n_iters):
        print(f"Iter {i}:")
        grad_theta = find_opt_prtrbtn(theta, dyn_mdl)
        historical_grad = tf.zeros_like(grad_theta)
        # adagrad
        if i == 0:
            historical_grad = historical_grad + grad_theta**2
        else:
            historical_grad = alpha * historical_grad + \
                              (1.0 - alpha) * (grad_theta**2)
        adj_grad = tf.divide(grad_theta,
                             fudge_factor + tf.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad
    return theta


class GMM():

    @tf.function
    def log_prob(self, theta):
        c1 = tfp.distributions.Normal(-2.0, 0.2)
        c2 = tfp.distributions.Normal(2.0, 0.2)
        lp = tf.math.log(0.3 * c1.prob(theta) + 0.7 * c2.prob(theta))
        tf.print(lp)
        return lp
