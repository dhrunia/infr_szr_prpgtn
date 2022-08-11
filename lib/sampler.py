import tensorflow as tf
import tensorflow_probability as tfp


class SVGD():

    def __init__(self, dyn_mdl):
        self._dyn_mdl = dyn_mdl

    @tf.function
    def _sq_dist(self, theta):
        # Squared sum of each point
        sq_sum = tf.reduce_sum(theta**2, axis=1, keepdims=True)
        # Pairwise dot product
        pdp = tf.matmul(theta, theta, transpose_b=True)
        dists = sq_sum + tf.transpose(sq_sum) - 2.0 * pdp
        return dists

    @tf.function
    def _rbf_kernel(self, theta, h=-1.0):
        squared_dists = self._sq_dist(theta)

        if h < 0:
            h = tfp.stats.percentile(squared_dists,
                                     q=50,
                                     interpolation='midpoint')
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
    def _find_opt_prtrbtn(self, theta):
        lnpgrad = tf.TensorArray(dtype=tf.float32, size=theta.shape[0])
        lnp = tf.TensorArray(dtype=tf.float32, size=theta.shape[0])

        def cond(i, lnpgrad, lnp):
            return tf.less(i, theta.shape[0])

        def body(i, lnpgrad, lnp):
            with tf.GradientTape() as tape:
                theta_i = theta[i]
                tape.watch(theta_i)
                lnp_i = self._dyn_mdl.log_prob(theta_i)
                lnpgrad = lnpgrad.write(i, tape.gradient(lnp_i, theta_i))
                lnp = lnp.write(i, lnp_i)
            return i + 1, lnpgrad, lnp

        i = tf.constant(0, dtype=tf.int32)

        i, lnpgrad, lnp = tf.while_loop(cond=cond,
                                        body=body,
                                        loop_vars=[i, lnpgrad, lnp],
                                        parallel_iterations=1)
        lnpgrad = lnpgrad.stack()
        lnp = lnp.stack()
        # tf.print("lnpgrad=", lnpgrad)
        # calculating the kernel matrix
        kxy, dxkxy = self._rbf_kernel(theta)
        # tf.print("kxy=", kxy, "dxkxy=", dxkxy)
        phi_star = (tf.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0]
        return phi_star, lnp

    # @tf.function
    def update(self,
               theta,
               n_iters,
               alpha=0.9,
               stepsize=1e-3,
               fudge_factor=1e-6):
        for i in range(n_iters):
            grad_theta, lnp = self._find_opt_prtrbtn(theta)
            lnp_sum = tf.reduce_sum(lnp)
            if i % 1 == 0:
                print(f"Iter {i}: {lnp_sum}")
            # print("grad_theta=", grad_theta)
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


# @tf.function
# def _find_opt_prtrbtn(self, theta):
#     with tf.GradientTape() as tape:
#         tape.watch(theta)
#         lnp = tf.TensorArray(dtype=tf.float32, size=theta.shape[0])
#         def cond(i, lnp):
#             return tf.less(i, theta.shape[0])

#         def body(i, lnp):
#             with tf.GradientTape() as tape:
#                 theta_i = theta[i]
#                 tape.watch(theta_i)
#                 lnp_i = self._dyn_mdl.log_prob(theta_i)
#                 lnp = lnp.write(i, lnp_i)
#             return i + 1, lnp

#         i = tf.constant(0, dtype=tf.int32)

#         i, lnp = tf.while_loop(cond=cond,
#                             body=body,
#                             loop_vars=[i, lnp],
#                             parallel_iterations=5)
#         lnp = lnp.stack()
#     lnpgrad = tape.gradient(lnp, theta)
#     # tf.print("lnpgrad=", lnpgrad)
#     # calculating the kernel matrix
#     kxy, dxkxy = self._rbf_kernel(theta)
#     # tf.print("kxy=", kxy, "dxkxy=", dxkxy)
#     phi_star = (tf.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0]
#     return phi_star, lnp