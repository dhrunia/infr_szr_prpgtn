import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import lib.io.tvb
import lib.utils.tnsrflw
import lib.io.seeg


tfd = tfp.distributions


class Epileptor2D():

    def __init__(self,
                 conn_path,
                 gain_path,
                 gain_rgn_map_path,
                 seeg_xyz_path,
                 gain_mat_res='high',
                 param_bounds=None):
        # Read structural connectivity
        self._SC, self._num_roi, self._roi_names = self.read_sc(
            conn_path=conn_path)
        # Read electrode names
        seeg_xyz = lib.io.seeg.read_seeg_xyz(seeg_xyz_path)
        self._snsr_lbls_all = [lbl for lbl, _ in seeg_xyz]
        self._num_snsrs_all = len(self._snsr_lbls_all)

        # Read Gain matrix
        self._gain = self.read_gain(gain_path,
                                    gain_mat_res,
                                    gain_rgn_map_path=gain_rgn_map_path)
        # Bounds for parameters
        if param_bounds is None:
            param_bounds = dict()
        if 'x0' not in param_bounds.keys():
            param_bounds['x0'] = dict()
            param_bounds['x0']['lb'] = tf.constant(-5.0, dtype=tf.float32)
            param_bounds['x0']['ub'] = tf.constant(0.0, dtype=tf.float32)
        if 'eps' not in param_bounds.keys():
            param_bounds['eps'] = dict()
            param_bounds['eps']['lb'] = tf.constant(0.0, dtype=tf.float32)
            param_bounds['eps']['ub'] = tf.constant(1.0, dtype=tf.float32)
        if 'K' not in param_bounds.keys():
            param_bounds['K'] = dict()
            param_bounds['K']['lb'] = tf.constant(0.0, dtype=tf.float32)
            param_bounds['K']['ub'] = tf.constant(10.0, dtype=tf.float32)
        if 'x_init' not in param_bounds.keys():
            param_bounds['x_init'] = dict()
            param_bounds['x_init']['lb'] = tf.constant(-5.0, dtype=tf.float32)
            param_bounds['x_init']['ub'] = tf.constant(-1.5, dtype=tf.float32)
        if 'z_init' not in param_bounds.keys():
            param_bounds['z_init'] = dict()
            param_bounds['z_init']['lb'] = tf.constant(4.0, dtype=tf.float32)
            param_bounds['z_init']['ub'] = tf.constant(6.0, dtype=tf.float32)
        if 'tau' not in param_bounds.keys():
            param_bounds['tau'] = dict()
            param_bounds['tau']['lb'] = tf.constant(20.0, dtype=tf.float32)
            param_bounds['tau']['ub'] = tf.constant(100.0, dtype=tf.float32)
        if 'amp' not in param_bounds.keys():
            param_bounds['amp'] = dict()
            param_bounds['amp']['lb'] = tf.constant(0.0, dtype=tf.float32)
            param_bounds['amp']['ub'] = tf.constant(10.0, dtype=tf.float32)
        if 'offset' not in param_bounds.keys():
            param_bounds['offset'] = dict()
            param_bounds['offset']['lb'] = tf.constant(-10.0, dtype=tf.float32)
            param_bounds['offset']['ub'] = tf.constant(10.0, dtype=tf.float32)

        self._x0_lb = param_bounds['x0']['lb']
        self._x0_ub = param_bounds['x0']['ub']
        self._eps_lb = param_bounds['eps']['lb']
        self._eps_ub = param_bounds['eps']['ub']
        self._K_lb = param_bounds['K']['lb']
        self._K_ub = param_bounds['K']['ub']
        self._x_init_lb = param_bounds['x_init']['lb']
        self._x_init_ub = param_bounds['x_init']['ub']
        self._z_init_lb = param_bounds['z_init']['lb']
        self._z_init_ub = param_bounds['z_init']['ub']
        self._tau_lb = param_bounds['tau']['lb']
        self._tau_ub = param_bounds['tau']['ub']
        self._amp_lb = param_bounds['amp']['lb']
        self._amp_ub = param_bounds['amp']['ub']
        self._offset_lb = param_bounds['offset']['lb']
        self._offset_ub = param_bounds['offset']['ub']

    @property
    def num_roi(self):
        return self._num_roi

    @property
    def SC(self):
        return self._SC

    @SC.setter
    def SC(self, conn):
        self._SC = conn

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain_mat):
        self._gain = gain_mat

    @property
    def roi_names(self):
        return self._roi_names

    @property
    def snsr_lbls_all(self):
        return self._snsr_lbls_all

    @property
    def snsr_lbls_picks(self):
        return self._snsr_lbls_picks

    @property
    def x0_lb(self):
        return self._x0_lb

    @property
    def x0_ub(self):
        return self._x0_ub

    @property
    def x_init_lb(self):
        return self._x_init_lb

    @property
    def x_init_ub(self):
        return self._x_init_ub

    @property
    def z_init_lb(self):
        return self._z_init_lb

    @property
    def z_init_ub(self):
        return self._z_init_ub

    @tf.function(jit_compile=True)
    def x0_bounded(self, x0_hat):
        return lib.utils.tnsrflw.sigmoid_transform(x0_hat, self._x0_lb,
                                                   self._x0_ub)

    @tf.function(jit_compile=True)
    def x0_unbounded(self, x0):
        return lib.utils.tnsrflw.inv_sigmoid_transform(x0, self._x0_lb,
                                                       self._x0_ub)

    @tf.function(jit_compile=True)
    def x_init_bounded(self, x_init_hat):
        return lib.utils.tnsrflw.sigmoid_transform(x_init_hat, self._x_init_lb,
                                                   self._x_init_ub)

    @tf.function(jit_compile=True)
    def x_init_unbounded(self, x_init):
        return lib.utils.tnsrflw.inv_sigmoid_transform(x_init, self._x_init_lb,
                                                       self._x_init_ub)

    @tf.function(jit_compile=True)
    def z_init_bounded(self, z_init_hat):
        return lib.utils.tnsrflw.sigmoid_transform(z_init_hat, self._z_init_lb,
                                                   self._z_init_ub)

    @tf.function(jit_compile=True)
    def z_init_unbounded(self, z_init):
        return lib.utils.tnsrflw.inv_sigmoid_transform(z_init, self._z_init_lb,
                                                       self._z_init_ub)

    @tf.function(jit_compile=True)
    def eps_bounded(self, eps_hat):
        return lib.utils.tnsrflw.sigmoid_transform(eps_hat, self._eps_lb,
                                                   self._eps_ub)

    @tf.function(jit_compile=True)
    def eps_unbounded(self, eps):
        return lib.utils.tnsrflw.inv_sigmoid_transform(eps, self._eps_lb,
                                                       self._eps_ub)

    @tf.function(jit_compile=True)
    def K_bounded(self, K_hat):
        return lib.utils.tnsrflw.sigmoid_transform(K_hat, self._K_lb,
                                                   self._K_ub)

    @tf.function(jit_compile=True)
    def K_unbounded(self, K):
        return lib.utils.tnsrflw.inv_sigmoid_transform(K, self._K_lb,
                                                       self._K_ub)

    @tf.function(jit_compile=True)
    def tau_bounded(self, tau_hat):
        return lib.utils.tnsrflw.sigmoid_transform(tau_hat, self._tau_lb,
                                                   self._tau_ub)

    @tf.function(jit_compile=True)
    def tau_unbounded(self, tau):
        return lib.utils.tnsrflw.inv_sigmoid_transform(tau, self._tau_lb,
                                                       self._tau_ub)

    @tf.function(jit_compile=True)
    def amp_bounded(self, amp_hat):
        return lib.utils.tnsrflw.sigmoid_transform(amp_hat, self._amp_lb,
                                                   self._amp_ub)

    @tf.function(jit_compile=True)
    def amp_unbounded(self, amp):
        return lib.utils.tnsrflw.inv_sigmoid_transform(amp, self._amp_lb,
                                                       self._amp_ub)

    @tf.function(jit_compile=True)
    def offset_bounded(self, offset_hat):
        return lib.utils.tnsrflw.sigmoid_transform(offset_hat, self._offset_lb,
                                                   self._offset_ub)

    @tf.function(jit_compile=True)
    def offset_unbounded(self, offset):
        return lib.utils.tnsrflw.inv_sigmoid_transform(offset, self._offset_lb,
                                                       self._offset_ub)

    def read_sc(self, conn_path):
        SC, _, roi_names = lib.io.tvb.read_conn(conn_path)
        # NOTE: Indexing from 1 to remove the unkown ROI
        SC = tf.constant(SC[1:, 1:], dtype=tf.float32)
        num_roi = SC.shape[0]
        roi_names = roi_names[1:]
        return SC, num_roi, roi_names

    def read_gain(self, gain_path, gain_mat_res, gain_rgn_map_path=None):
        if gain_mat_res == 'high':
            gain_hr = np.load(gain_path)['gain_inv_square']
            gain_rgn_map = np.loadtxt(gain_rgn_map_path)
            assert self._num_snsrs_all == gain_hr.shape[0]
            gain_lr_avg_hr = np.zeros((self._num_snsrs_all, self._num_roi))
            for roi in range(1, self._num_roi + 1):
                #NOTE: looping from 1 to num_roi + 1 to ignore the unkown roi
                # which is included in high resolution gain matrix as roi 0
                idcs = np.nonzero(gain_rgn_map == roi)[0]
                gain_lr_avg_hr[:, roi - 1] = np.sum(gain_hr[:, idcs], axis=1)
                gain = tf.constant(gain_lr_avg_hr.T, dtype=tf.float32)
        elif gain_mat_res == 'low':
            gain = np.loadtxt(gain_path)
            gain = tf.constant(gain.T, dtype=tf.float32)
        else:
            raise ValueError("gain_mat_res should be either 'low' or 'high'")
        return gain

    def update_gain(self, snsr_picks):
        gain_idxs = [self._snsr_lbls_all.index(lbl) for lbl in snsr_picks]
        self._gain = tf.gather(self._gain, indices=gain_idxs, axis=1)
        self._snsr_lbls_picks = snsr_picks
        self._num_snsr_picks = len(snsr_picks)

    @tf.function(jit_compile=True)
    def split_params(self, theta):
        x0_hat = theta[0:self._num_roi]
        x_init_hat = theta[self._num_roi:2 * self._num_roi]
        z_init_hat = theta[2 * self._num_roi:3 * self._num_roi]
        tau_hat = theta[3 * self._num_roi]
        K_hat = theta[3 * self._num_roi + 1]
        amp_hat = theta[3 * self._num_roi + 2]
        offset_hat = theta[3 * self._num_roi + 3]
        eps_hat = theta[3 * self._num_roi + 4]
        return (x0_hat, x_init_hat, z_init_hat, tau_hat, K_hat, amp_hat,
                offset_hat, eps_hat)

    @tf.function(jit_compile=True)
    def join_params(self, x0_hat, x_init_hat, z_init_hat, tau_hat, K_hat,
                    amp_hat, offset_hat, eps_hat):
        theta = tf.concat((x0_hat, x_init_hat, z_init_hat, tau_hat[tf.newaxis],
                           K_hat[tf.newaxis], amp_hat[tf.newaxis],
                           offset_hat[tf.newaxis], eps_hat[tf.newaxis]),
                          axis=0)
        return theta

    @tf.function(jit_compile=True)
    def transformed_parameters(self, x0_hat, x_init_hat, z_init_hat, tau_hat,
                               K_hat, amp_hat, offset_hat, eps_hat):
        x0 = self.x0_bounded(x0_hat)
        x_init = self.x_init_bounded(x_init_hat)
        z_init = self.z_init_bounded(z_init_hat)
        tau = self.tau_bounded(tau_hat)
        K = self.K_bounded(K_hat)
        amp = self.amp_bounded(amp_hat)
        offset = self.offset_bounded(offset_hat)
        eps = self.eps_bounded(eps_hat)
        return (x0, x_init, z_init, tau, K, amp, offset, eps)

    @tf.function(jit_compile=True)
    def inv_transformed_parameters(self, x0, x_init, z_init, tau, K, amp,
                                   offset, eps):
        x0_hat = self.x0_unbounded(x0)
        x_init_hat = self.x_init_unbounded(x_init)
        z_init_hat = self.z_init_unbounded(z_init)
        tau_hat = self.tau_unbounded(tau)
        K_hat = self.K_unbounded(K)
        amp_hat = self.amp_unbounded(amp)
        offset_hat = self.offset_unbounded(offset)
        eps_hat = self.eps_unbounded(eps)
        return (x0_hat, x_init_hat, z_init_hat, tau_hat, K_hat, amp_hat,
                offset_hat, eps_hat)

    @tf.function
    def _ode_fn(self, y, x0, tau, K):
        I1 = tf.constant(4.1, dtype=tf.float32)
        x = y[0:self._num_roi]
        z = y[self._num_roi:2 * self._num_roi]
        dxdt = 1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - z + I1
        global_cplng = tf.reduce_sum(K * self._SC *
                                     (x[tf.newaxis, :] - x[:, tf.newaxis]),
                                     axis=1)
        dzdt = (1.0 / tau) * (4 * (x - x0) - z - global_cplng)
        return tf.concat((dxdt, dzdt), axis=0)

    @tf.function
    def simulate(self, nsteps, nsubsteps, time_step, y_init, x0, tau, K):
        print("euler_integrator()...")
        y = tf.TensorArray(dtype=tf.float32, size=nsteps)
        y_next = y_init

        def cond1(i, y, y_next):
            return tf.less(i, nsteps)

        def body1(i, y, y_next):
            j = tf.constant(0)

            def cond2(j, y_next):
                return tf.less(j, nsubsteps)

            def body2(j, y_next):
                y_next = y_next + time_step * self._ode_fn(y_next, x0, tau, K)
                return j + 1, y_next

            j, y_next = tf.while_loop(cond2,
                                      body2, (j, y_next),
                                      parallel_iterations=1,
                                      maximum_iterations=nsubsteps)

            y = y.write(i, y_next)
            return i + 1, y, y_next

        i = tf.constant(0)
        i, y, y_next = tf.while_loop(cond1,
                                     body1, (i, y, y_next),
                                     parallel_iterations=1,
                                     maximum_iterations=nsteps)
        return y.stack()

    @tf.function(jit_compile=True)
    def project_sensor_space(self, x, amp, offset):
        slp = amp * tf.math.log(tf.matmul(tf.math.exp(x), self._gain)) + offset
        return slp

    def setup_inference(self, nsteps, nsubsteps, time_step, mean, std,
                        obs_data, obs_space):
        self._nsteps = nsteps
        self._nsubsteps = nsubsteps
        self._time_step = time_step
        self._obs_data = obs_data
        self._obs_space = obs_space
        self._build_priors(mean, std)

    def _build_priors(self, mean, std):
        self._x0_prior = tfd.Normal(loc=mean['x0'], scale=std['x0'])
        self._x_init_prior = tfd.Normal(loc=mean['x_init'],
                                        scale=std['x_init'])
        self._z_init_prior = tfd.Normal(loc=mean['z_init'],
                                        scale=std['z_init'])

        # tau - Time scale
        self._tau_prior = tfd.Uniform(low=self._tau_lb, high=self._tau_ub)

        # K - Global coupling
        K = tfd.TruncatedNormal(loc=mean['K'],
                                scale=std['K'],
                                low=self._K_lb,
                                high=self._K_ub).sample(5000)
        K_hat = self.K_unbounded(K)
        K_hat_mean = tf.math.reduce_mean(K_hat)
        K_hat_std = tf.math.reduce_std(K_hat)
        self._K_hat_prior = tfd.Normal(loc=K_hat_mean, scale=K_hat_std)

        # slp amplitude scaling
        self._amp_hat_prior = tfd.Normal(loc=0.0, scale=1.0)

        # slp offset
        self._offset_hat_prior = tfd.Normal(loc=0.0, scale=1.0)

        # eps - Observation Noise
        eps = tfd.TruncatedNormal(loc=mean['eps'],
                                  scale=std['eps'],
                                  low=self._eps_lb,
                                  high=self._eps_ub).sample(5000)
        eps_hat = self.eps_unbounded(eps)
        eps_hat_mean = tf.math.reduce_mean(eps_hat)
        eps_hat_std = tf.math.reduce_std(eps_hat)
        self._eps_hat_prior = tfd.Normal(loc=eps_hat_mean, scale=eps_hat_std)

    @tf.function(jit_compile=True)
    def _bounded_trans_jacob_adj(self, y, lb, ub):
        t1 = ub - lb
        t2 = y - lb / t1
        log_det_jac = tf.math.log(tf.abs(t1 * t2 * (1 - t2)) + 1e-10)
        return log_det_jac

    @tf.function
    def _prior_log_prob(self, x0, x_init, z_init, tau, K_hat, amp_hat,
                        offset_hat, eps_hat):
        x0_jacob_adj = self._bounded_trans_jacob_adj(x0, self._x0_lb,
                                                     self._x0_ub)
        x0_prior_lp = tf.reduce_sum(self._x0_prior.log_prob(x0) + x0_jacob_adj,
                                    axis=0)

        x_init_jacob_adj = self._bounded_trans_jacob_adj(
            x_init, self._x_init_lb, self._x_init_ub)
        x_init_prior_lp = tf.reduce_sum(self._x_init_prior.log_prob(x_init) +
                                        x_init_jacob_adj,
                                        axis=0)

        z_init_jacob_adj = self._bounded_trans_jacob_adj(
            z_init, self._z_init_lb, self._z_init_ub)
        z_init_prior_lp = tf.reduce_sum(self._z_init_prior.log_prob(z_init) +
                                        z_init_jacob_adj,
                                        axis=0)
        tau_jacob_adj = self._bounded_trans_jacob_adj(tau, self._tau_lb,
                                                      self._tau_ub)
        tau_prior_lp = self._tau_prior.log_prob(tau) + tau_jacob_adj

        K_prior_lp = self._K_hat_prior.log_prob(K_hat)

        amp_prior_lp = self._amp_hat_prior.log_prob(amp_hat)

        offset_prior_lp = self._offset_hat_prior.log_prob(offset_hat)

        eps_prior_lp = self._eps_hat_prior.log_prob(eps_hat)

        return x0_prior_lp + x_init_prior_lp + z_init_prior_lp + \
            tau_prior_lp + K_prior_lp + amp_prior_lp + \
            offset_prior_lp + eps_prior_lp

    @tf.function
    def _likelihood_log_prob(self, x0, x_init, z_init, tau, K, amp, offset,
                             eps, obs_data, obs_space):
        y_init = tf.concat((x_init, z_init), axis=0)
        y_pred = self.simulate(self._nsteps, self._nsubsteps, self._time_step,
                               y_init, x0, tau, K)
        x_pred = y_pred[:, 0:self._num_roi]
        if (obs_space == 'sensor'):
            slp_mu = self.project_sensor_space(x_pred, amp, offset)
            # # tf.print("Nan in x_mu", tf.reduce_any(tf.math.is_nan(x_mu)))
            llp = tf.reduce_mean(
                tf.reduce_sum(tfd.Normal(loc=slp_mu,
                                         scale=eps).log_prob(obs_data),
                              axis=[1, 2]))
        if (obs_space == 'source'):
            x_pred = amp * x_pred + offset
            llp = tf.reduce_mean(
                tf.reduce_sum(tfd.Normal(loc=x_pred,
                                         scale=eps).log_prob(obs_data),
                              axis=[1, 2]))
        return llp

    @tf.function
    def log_prob(self, theta, nsamples):
        if theta.shape.ndims == 1:
            theta = theta[tf.newaxis, :]
        lp = tf.TensorArray(dtype=tf.float32, size=nsamples)
        i = tf.constant(0)

        def cond(i, lp):
            return tf.less(i, nsamples)

        def body(i, lp):
            (x0_hat, x_init_hat, z_init_hat, tau_hat, K_hat, amp_hat,
             offset_hat, eps_hat) = self.split_params(theta[i])
            (x0, x_init, z_init, tau, K, amp, offset,
             eps) = self.transformed_parameters(x0_hat, x_init_hat, z_init_hat,
                                                tau_hat, K_hat, amp_hat,
                                                offset_hat, eps_hat)
            # Compute Likelihood
            likelihood_lp = self._likelihood_log_prob(x0, x_init, z_init, tau,
                                                      K, amp, offset, eps,
                                                      self._obs_data,
                                                      self._obs_space)

            # Compute Prior probability
            prior_lp = self._prior_log_prob(x0, x_init, z_init, tau, K_hat,
                                            amp_hat, offset_hat, eps_hat)
            # Posterior log. probability
            lp_i = likelihood_lp + prior_lp
            # tf.print("likelihood = ", likelihood_lp, "prior = ", prior_lp)
            lp = lp.write(i, lp_i)
            return i + 1, lp

        i, lp = tf.while_loop(cond=cond,
                              body=body,
                              loop_vars=(i, lp),
                              maximum_iterations=nsamples)
        return lp.stack()
