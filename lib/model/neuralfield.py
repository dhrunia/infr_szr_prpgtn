import tensorflow as tf
import lib.utils.sht as tfsht
import lib.utils.projector
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions


class Epileptor2D:
    def __init__(self,
                 L_MAX,
                 N_LAT,
                 N_LON,
                 verts_irreg_fname,
                 rgn_map_irreg_fname,
                 SC_path,
                 gain_irreg_path,
                 L_MAX_PARAMS,
                 x0_lb=tf.constant(-5.0, dtype=tf.float32),
                 x0_ub=tf.constant(-1.0, dtype=tf.float32)):
        self._L_MAX = L_MAX
        self._N_LAT, self._N_LON, self._cos_theta, self._glq_wts, self._P_l_m_costheta = tfsht.prep(
            L_MAX, N_LAT, N_LON)
        self._D = tf.constant(0.01, dtype=tf.float32)
        l = tf.range(0, self._L_MAX + 1, dtype=tf.float32)
        Dll = self._D * l * (l + 1)
        Dll = tf.reshape(tf.repeat(Dll, self._L_MAX + 1),
                         (self._L_MAX + 1, self._L_MAX + 1))
        self._Dll = tf.cast(Dll, dtype=tf.complex64)

        self._nv = tf.constant(2 * self._N_LAT * self._N_LON,
                              dtype=tf.int32)  # Total no. of vertices
        self._nvph = tf.math.floordiv(self._nv,
                                     2)  # No.of vertices per hemisphere

        self._rgn_map_reg = lib.utils.projector.find_rgn_map_reg(
            N_LAT=self._N_LAT.numpy(),
            N_LON=self._N_LON.numpy(),
            cos_theta=self._cos_theta,
            verts_irreg_fname=verts_irreg_fname,
            rgn_map_irreg_fname=rgn_map_irreg_fname)
        self._unkown_roi_idcs = np.nonzero(self._rgn_map_reg == 0)[0]
        unkown_roi_mask = np.ones(self._nv)
        unkown_roi_mask[self._unkown_roi_idcs] = 0
        self._unkown_roi_mask = tf.constant(unkown_roi_mask, dtype=tf.float32)

        self._idcs_nbrs_irreg = lib.utils.projector.find_nbrs_irreg_sphere(
            N_LAT=self._N_LAT.numpy(),
            N_LON=self._N_LON.numpy(),
            cos_theta=self._cos_theta,
            verts_irreg_fname=verts_irreg_fname)

        # Constants cached for computing gradients of local coupling
        delta_phi = tf.constant(2.0 * np.pi / self._N_LON.numpy(),
                                dtype=tf.float32)
        phi = tf.range(0, 2.0 * np.pi, delta_phi, dtype=tf.float32)
        phi_db = phi[:, tf.newaxis] - phi[tf.newaxis, :]
        m = tf.range(0, self._L_MAX + 1, dtype=tf.float32)
        self._cos_m_phidb = 2.0 * tf.math.cos(tf.einsum(
            "m,db->mdb", m, phi_db))
        self._P_l_m_Dll = delta_phi * tf.math.real(
            self._Dll)[:, :, tf.newaxis] * tf.math.real(self._P_l_m_costheta)

        self._rgn_map_reg_sorted = tf.gather(self._rgn_map_reg,
                                             tf.argsort(self._rgn_map_reg))
        self._low_idcs = []
        self._high_idcs = []
        for roi in tf.unique(self._rgn_map_reg_sorted)[0]:
            roi_idcs = tf.squeeze(tf.where(self._rgn_map_reg_sorted == roi))
            self._low_idcs.append(
                roi_idcs[0] if roi_idcs.ndim > 0 else roi_idcs)
            self._high_idcs.append(roi_idcs[-1] +
                                   1 if roi_idcs.ndim > 0 else roi_idcs + 1)

        # Compute a region mapping such that all cortical rois are contiguous
        # NOTE: This shouldn't be necessary once subcortical regions are also
        # included in the simulation
        tmp = self._rgn_map_reg.numpy()
        tmp[tmp > 81] = tmp[tmp > 81] - 9
        self._vrtx_roi_map = tf.constant(tmp, dtype=tf.int32)

        SC = np.loadtxt(SC_path)

        # remove subcortical regions
        # NOTE: not required once subcortical regions are included
        # in simulation
        idcs1, idcs2 = np.meshgrid(np.unique(self._rgn_map_reg),
                                   np.unique(self._rgn_map_reg),
                                   indexing='ij')
        SC = SC[idcs1, idcs2]

        SC = SC / np.max(SC)
        SC[np.diag_indices_from(SC)] = 0.0
        self._SC = tf.constant(SC, dtype=tf.float32)
        gain_irreg = np.load(gain_irreg_path)['gain_inv_square']
        # Remove subcortical vertices
        # NOTE: This won't be necessary once subcortical regions are
        # included in simulation
        num_verts_irreg = np.loadtxt(verts_irreg_fname).shape[0]
        gain_irreg = gain_irreg[:, 0:num_verts_irreg]
        self.gain_reg = tf.constant(gain_irreg[:, self._idcs_nbrs_irreg].T,
                                    dtype=tf.float32)
        self._L_MAX_PARAMS = L_MAX_PARAMS
        _, _, self._cos_theta_params, self._glq_wts_params, self._P_l_m_costheta_params = tfsht.prep(
            self._L_MAX_PARAMS, self._N_LAT, self._N_LON)
        self._nmodes_params = tf.pow(self._L_MAX_PARAMS + 1, 2)
        self._x0_lb = x0_lb
        self._x0_ub = x0_ub

    @property
    def L_MAX(self):
        return self._L_MAX

    @property
    def N_LAT(self):
        return self._N_LAT

    @property
    def N_LON(self):
        return self._N_LON

    @property
    def nv(self):
        return self._nv

    @property
    def nvph(self):
        return self._nvph

    @property
    def rgn_map_reg(self):
        return self._rgn_map_reg

    @property
    def unkown_roi_mask(self):
        return self._unkown_roi_mask

    @property
    def nmodes_params(self):
        return self._nmodes_params

    @property
    def L_MAX_PARAMS(self):
        return self._L_MAX_PARAMS

    @property
    def x0_lb(self):
        return self._x0_lb

    @property
    def x0_ub(self):
        return self._x0_ub

    @property
    def glq_wts_params(self):
        return self._glq_wts_params

    @property
    def P_l_m_costheta_params(self):
        return self._P_l_m_costheta_params

    @tf.function
    def x0_trans_to_vrtx_space(self, theta):
        x0_lh = tfsht.synth(
            self._L_MAX_PARAMS, self._N_LON,
            tf.complex(theta[0:self._nmodes_params],
                       theta[self._nmodes_params:2 * self._nmodes_params]),
            self._P_l_m_costheta_params)
        x0_rh = tfsht.synth(
            self._L_MAX_PARAMS, self._N_LON,
            tf.complex(theta[2 * self._nmodes_params:3 * self._nmodes_params],
                       theta[3 * self._nmodes_params:4 * self._nmodes_params]),
            self._P_l_m_costheta_params)
        x0 = tf.concat([x0_lh, x0_rh], axis=0)
        return x0

    @tf.function
    def x0_bounded_trnsform(self, x0):
        return lib.utils.tnsrflw.sigmoid_transform(x0, self._x0_lb,
                                                   self._x0_ub)

    @tf.custom_gradient
    def _local_coupling(
        self,
        x,
        glq_wts,
        P_l_m_costheta,
        Dll,
        L_MAX,
        N_LAT,
        N_LON,
        P_l_m_Dll,
        cos_m_phidb,
    ):
        print("local_coupling()...")
        x_hat_lh = tf.stop_gradient(x[0:N_LAT * N_LON])
        x_hat_rh = tf.stop_gradient(x[N_LAT * N_LON:])
        x_lm_lh = tf.stop_gradient(
            tfsht.analys(L_MAX, N_LAT, N_LON, x_hat_lh, glq_wts,
                         P_l_m_costheta))
        x_lm_hat_lh = tf.stop_gradient(-1.0 * tf.reshape(Dll, [-1]) * x_lm_lh)
        x_lm_rh = tf.stop_gradient(
            tfsht.analys(L_MAX, N_LAT, N_LON, x_hat_rh, glq_wts,
                         P_l_m_costheta))
        x_lm_hat_rh = tf.stop_gradient(-1.0 * tf.reshape(Dll, [-1]) * x_lm_rh)
        local_cplng_lh = tf.stop_gradient(
            tfsht.synth(L_MAX, N_LON, x_lm_hat_lh, P_l_m_costheta))
        local_cplng_rh = tf.stop_gradient(
            tfsht.synth(L_MAX, N_LON, x_lm_hat_rh, P_l_m_costheta))
        local_cplng = tf.stop_gradient(
            tf.concat((local_cplng_lh, local_cplng_rh), axis=0))

        def grad(upstream):
            upstream_lh = tf.reshape(upstream[0:N_LAT * N_LON], (N_LAT, N_LON))
            upstream_rh = tf.reshape(upstream[N_LAT * N_LON:], (N_LAT, N_LON))
            glq_wts_grad = None
            P_l_m_costheta_grad = None
            Dll_grad = None
            L_MAX_grad = None
            N_LAT_grad = None
            N_LON_grad = None
            P_l_m_Dll_grad = None
            cos_m_phidb_grad = None

            g_lh = -1.0 * tf.einsum("cd,a,lmc,lma,mdb->ab",
                                    upstream_lh,
                                    tf.math.real(glq_wts),
                                    P_l_m_Dll[:, 1:, :],
                                    tf.math.real(P_l_m_costheta)[:, 1:, :],
                                    cos_m_phidb[1:, :, :],
                                    optimize="optimal") - tf.einsum(
                                        "cd,a,lc,la,db->ab",
                                        upstream_lh,
                                        tf.math.real(glq_wts),
                                        P_l_m_Dll[:, 0, :],
                                        tf.math.real(P_l_m_costheta)[:, 0, :],
                                        cos_m_phidb[0, :, :],
                                        optimize="optimal")
            g_rh = -1.0 * tf.einsum("cd,a,lmc,lma,mdb->ab",
                                    upstream_rh,
                                    tf.math.real(glq_wts),
                                    P_l_m_Dll[:, 1:, :],
                                    tf.math.real(P_l_m_costheta)[:, 1:, :],
                                    cos_m_phidb[1:, :, :],
                                    optimize="optimal") - tf.einsum(
                                        "cd,a,lc,la,db->ab",
                                        upstream_rh,
                                        tf.math.real(glq_wts),
                                        P_l_m_Dll[:, 0, :],
                                        tf.math.real(P_l_m_costheta)[:, 0, :],
                                        cos_m_phidb[0, :, :],
                                        optimize="optimal")
            g = tf.concat((tf.reshape(g_lh, [-1]), tf.reshape(g_rh, [-1])),
                          axis=0)
            return [
                g,
                glq_wts_grad,
                P_l_m_costheta_grad,
                Dll_grad,
                L_MAX_grad,
                N_LAT_grad,
                N_LON_grad,
                P_l_m_Dll_grad,
                cos_m_phidb_grad,
            ]

        return local_cplng, grad

    @tf.function
    def _ode_fn(self, y, x0, tau, K):
        print("epileptor2d_nf_ode_fn()...")
        # nv = 2 * self._N_LAT * self._N_LON
        x = y[0:self._nv]
        z = y[self._nv:2 * self._nv]
        I1 = tf.constant(4.1, dtype=tf.float32)
        # NOTE: alpha > 7.0 is causing DormandPrince integrator to diverge
        alpha = tf.constant(1.0, dtype=tf.float32)
        theta = tf.constant(-1.0, dtype=tf.float32)
        gamma_lc = tf.constant(5.0, dtype=tf.float32)
        x_hat = tf.math.sigmoid(alpha * (x - theta)) * self._unkown_roi_mask
        local_cplng = self._local_coupling(
            x_hat,
            self._glq_wts,
            self._P_l_m_costheta,
            self._Dll,
            self._L_MAX,
            self._N_LAT,
            self._N_LON,
            self._P_l_m_Dll,
            self._cos_m_phidb,
        )
        x_sorted = tf.gather(x, self._rgn_map_reg_sorted)
        x_roi = tfp.stats.windowed_mean(x_sorted, self._low_idcs,
                                        self._high_idcs)
        # tf.print(x_hat_roi.shape)
        # tf.print("tau = ", tau)
        global_cplng_roi = tf.reduce_sum(
            K * self._SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]),
            axis=1)
        global_cplng_vrtcs = tf.gather(global_cplng_roi, self._vrtx_roi_map)
        dx = (1.0 - tf.math.pow(x, 3) - 2 * tf.math.pow(x, 2) - z +
              I1) * self._unkown_roi_mask
        dz = ((1.0 / tau) * (4 * (x - x0) - z - global_cplng_vrtcs -
                             gamma_lc * local_cplng)) * self._unkown_roi_mask
        return tf.concat((dx, dz), axis=0)

    @tf.function
    def simulate(self, nsteps, nsubsteps, time_step, y_init, x0, tau, K):
        print("euler_integrator()...")
        y = tf.TensorArray(dtype=tf.float32, size=nsteps)
        y_next = y_init
        cond1 = lambda i, y, y_next: tf.less(i, nsteps)

        def body1(i, y, y_next):
            j = tf.constant(0)
            cond2 = lambda j, y_next: tf.less(j, nsubsteps)

            def body2(j, y_next):
                y_next = y_next + time_step * self._ode_fn(y_next, x0, tau, K)
                return j + 1, y_next

            j, y_next = tf.while_loop(cond2,
                                      body2, (j, y_next),
                                      maximum_iterations=nsubsteps)

            y = y.write(i, y_next)
            return i + 1, y, y_next

        i = tf.constant(0)
        i, y, y_next = tf.while_loop(cond1,
                                     body1, (i, y, y_next),
                                     maximum_iterations=nsteps)
        return y.stack()

    def project_sensor_space(self, x):
        slp = tf.math.log(tf.matmul(tf.math.exp(x), self.gain_reg))
        return slp

    @tf.function
    def log_prob(self, theta, slp_obs, nsteps, nsubsteps, time_step, y_init,
                 tau, K, x0_prior_mu):
        eps = tf.constant(0.1, dtype=tf.float32)
        tf.print("nan in theta", tf.reduce_any(tf.math.is_nan(theta)))
        x0 = self.x0_trans_to_vrtx_space(theta[0:4 * self._nmodes_params])
        tf.print("nan in x0", tf.reduce_any(tf.math.is_nan(x0)))
        x0_trans = self.x0_bounded_trnsform(x0) * self._unkown_roi_mask
        # x0_trans_log_det_jcbn = tf.reduce_sum(
        #     tf.math.log(
        #         tf.math.abs(
        #             (x0_trans - x0_lb) * (1 - (x0_trans - x0_lb) /
        #                                   (x0_ub - x0_lb))) * unkown_roi_mask))
        tf.print("nan in x0_trans", tf.reduce_any(tf.math.is_nan(x0_trans)))
        # tau = theta[4 * nmodes]
        # tau_trans = lib.utils.tnsrflw.sigmoid_transform(
        #     tau, tf.constant(15, dtype=tf.float32),
        #     tf.constant(100, dtype=tf.float32))
        y_pred = self.simulate(nsteps, nsubsteps, time_step, y_init, x0_trans,
                               tau, K)
        x_pred = y_pred[:, 0:self._nv] * self._unkown_roi_mask
        tf.print("nan in x_pred", tf.reduce_any(tf.math.is_nan(x_pred)))
        # slp_mu = tf.math.log(tf.matmul(tf.math.exp(x_pred), self.gain_reg))
        slp_mu = self.project_sensor_space(x_pred)
        # tf.print("Nan in x_mu", tf.reduce_any(tf.math.is_nan(x_mu)))
        likelihood = tf.reduce_sum(
            tfd.Normal(loc=slp_mu, scale=eps).log_prob(slp_obs))
        # x0_prior = tf.reduce_sum(tfd.Normal(loc=0.0, scale=5.0).log_prob(theta))
        x0_prior = tf.reduce_sum(
            tfd.Normal(loc=x0_prior_mu, scale=0.5).log_prob(x0_trans))
        lp = likelihood + x0_prior
        tf.print("likelihood = ", lp, "prior = ", x0_prior)
        return lp
