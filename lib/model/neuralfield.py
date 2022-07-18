import tensorflow as tf
import lib.utils.sht as tfsht
import lib.utils.projector
import numpy as np
import tensorflow_probability as tfp
import lib.io.tvb

tfd = tfp.distributions


class Epileptor2D:

    def __init__(self,
                 L_MAX,
                 N_LAT,
                 N_LON,
                 verts_irreg_fname,
                 rgn_map_irreg_fname,
                 conn_zip_path,
                 gain_irreg_path,
                 gain_irreg_rgn_map_path,
                 L_MAX_PARAMS,
                 param_bounds=None):
        self._L_MAX = L_MAX
        (self._N_LAT, self._N_LON, self._cos_theta, self._glq_wts,
         self._P_l_m_costheta) = tfsht.prep(L_MAX, N_LAT, N_LON)
        self._D = tf.constant(0.01, dtype=tf.float32)
        l = tf.range(0, self._L_MAX + 1, dtype=tf.float32)
        Dll = self._D * l * (l + 1)
        Dll = tf.reshape(tf.repeat(Dll, self._L_MAX + 1),
                         (self._L_MAX + 1, self._L_MAX + 1))
        self._Dll = tf.cast(Dll, dtype=tf.complex64)
        # Total number of vertices
        self._nv = tf.constant(2 * self._N_LAT * self._N_LON, dtype=tf.int32)
        # Number of vertices per hemisphere
        self._nvph = tf.math.floordiv(self._nv, 2)
        # Total number of subcortical regions
        self._ns = tf.constant(18, dtype=tf.int32)
        # Number of subcortical regions per hemisphere
        self._nsph = tf.math.floordiv(self._ns, 2)

        # Read the TVB sturcutural connectivity and the ROI names
        # NOTE: These are later re-orderd to follow the ROI ordering used
        # by this class
        SC, _, tvb_roi_names = lib.io.tvb.read_conn(
            conn_zip_path=conn_zip_path)

        # Total number of ROI
        self._nroi = SC.shape[0]

        # Indices of subcortical ROI in the provided SC
        print("Assuming indices (zero based) of subcortical roi in the " +
              "provided SC to be [73:81] for left hemisphere " +
              "and [154:162] for right hemisphere")
        idcs_subcrtx_roi = np.concatenate(
            (np.arange(73, 73 + 9, dtype=np.int32),
             np.arange(154, 154 + 9, dtype=np.int32)),
            dtype=np.int32)

        rgn_map_reg_tvb = lib.utils.projector.find_rgn_map_reg(
            N_LAT=self._N_LAT.numpy(),
            N_LON=self._N_LON.numpy(),
            cos_theta=self._cos_theta,
            verts_irreg_fname=verts_irreg_fname,
            rgn_map_irreg_fname=rgn_map_irreg_fname)

        # Methods of this class are implemented assuming the ROI ordering is:
        # Cortical Left -> Cortical Right -> Subcortical Left -> Subcortical Right
        # Compute mappings between roi indices in TVB and this class
        roi_map_tfnf_to_tvb = np.zeros(self._nroi, dtype=np.int32)
        roi_map_tfnf_to_tvb = np.concatenate(
            (np.unique(rgn_map_reg_tvb), idcs_subcrtx_roi), dtype=np.int32)
        self._roi_map_tfnf_to_tvb = tf.constant(roi_map_tfnf_to_tvb,
                                                dtype=tf.int32)
        roi_map_tvb_to_tfnf = np.zeros(self._nroi, dtype=np.int32)
        roi_map_tvb_to_tfnf[self._roi_map_tfnf_to_tvb] = np.arange(
            0, self._nroi, dtype=np.int32)
        self._roi_map_tvb_to_tfnf = tf.constant(roi_map_tvb_to_tfnf,
                                                dtype=tf.int32)

        # Region mapping
        self._rgn_map = self._build_rgn_map(rgn_map_reg_tvb)
        # Append the indices of subcortical regions
        self._rgn_map = tf.concat(
            (self._rgn_map, tf.range(145, 145 + 18, dtype=tf.int32)), axis=0)

        # Compute a mask for unkown ROI
        print("Assuming ROI 0 in provided SC is the UNKOWN region")
        self._unkown_roi_idcs = np.nonzero(self._rgn_map == 0)[0]
        unkown_roi_mask = np.ones(self._nv + self._ns)
        unkown_roi_mask[self._unkown_roi_idcs] = 0
        # # Append 1s to account for subcoritial rois
        # unkown_roi_mask = np.concatenate((unkown_roi_mask, np.ones(self._ns)),
        #                                  axis=0)
        self._unkown_roi_mask = tf.constant(unkown_roi_mask, dtype=tf.float32)

        # Find the idcs of vertices in irregular sphere that are closest to
        # the vertices in the regular sphere used by this class
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

        # Re order the SC according to follow the ROI ordering used in this class
        idcs1, idcs2 = np.meshgrid(self._roi_map_tfnf_to_tvb,
                                   self._roi_map_tfnf_to_tvb,
                                   indexing='ij')
        SC = SC[idcs1, idcs2]
        self._SC = tf.constant(SC, dtype=tf.float32)

        # ROI labels indexed according to this class
        self._roi_names = [
            tvb_roi_names[self._roi_map_tfnf_to_tvb[i]]
            for i in range(self._nroi)
        ]

        self._rgn_map_argsort = tf.argsort(self._rgn_map)
        self._rgn_map_sorted = tf.gather(self._rgn_map, self._rgn_map_argsort)
        self._low_idcs = []
        self._high_idcs = []
        for roi in tf.unique(self._rgn_map_sorted)[0]:
            roi_idcs = tf.squeeze(tf.where(self._rgn_map_sorted == roi))
            self._low_idcs.append(
                roi_idcs[0] if roi_idcs.ndim > 0 else roi_idcs)
            self._high_idcs.append(roi_idcs[-1] +
                                   1 if roi_idcs.ndim > 0 else roi_idcs + 1)

        # Prepare the gain matrix
        gain_irreg = np.load(gain_irreg_path)['gain_inv_square']
        num_verts_irreg = np.loadtxt(verts_irreg_fname).shape[0]
        gain_irreg_crtx = gain_irreg[:, 0:num_verts_irreg]
        # subcortical regions are treated as point masses
        gain_irreg_subcrtx = np.zeros(shape=(gain_irreg_crtx.shape[0],
                                             idcs_subcrtx_roi.shape[0]))
        gain_irreg_rgn_map = np.loadtxt(gain_irreg_rgn_map_path)
        for i, roi in enumerate(idcs_subcrtx_roi):
            idcs = np.nonzero(gain_irreg_rgn_map == roi)[0]
            gain_irreg_subcrtx[:, i] = np.sum(gain_irreg[:, idcs], axis=1)
        gain_reg_crtx = gain_irreg_crtx[:, self._idcs_nbrs_irreg].T
        gain_reg_subcrtx = gain_irreg_subcrtx.T
        gain_reg = np.concatenate([gain_reg_crtx, gain_reg_subcrtx], axis=0)
        self._gain_reg = tf.constant(gain_reg, dtype=tf.float32)
        # self._gain_reg_crtx = tf.constant(gain_reg_crtx, dtype=tf.float32)
        # self._gain_reg_subcrtx = tf.constant(gain_reg_subcrtx,
        #                                      dtype=tf.float32)

        # # SHT constants for inference in modes space
        # self._L_MAX_PARAMS = L_MAX_PARAMS
        # (_, _, self._cos_theta_params, self._glq_wts_params,
        #  self._P_l_m_costheta_params) = tfsht.prep(self._L_MAX_PARAMS,
        #                                            self._N_LAT, self._N_LON)
        # self._nmodes_params = tf.pow(self._L_MAX_PARAMS + 1, 2)
        self.setup_param_mode_space_constants(L_MAX_PARAMS)

        # Compute the no.of vertices in each ROI
        self._nv_per_roi = tf.constant([
            tf.where(self._rgn_map == roi).shape[0]
            for roi in range(self._nroi)
        ])
        self._vrtx_wts = tf.cast(tf.gather(1 / self._nv_per_roi,
                                           self._rgn_map),
                                 dtype=tf.float32)
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

        self._x0_lb = param_bounds['x0']['lb']
        self._x0_ub = param_bounds['x0']['ub']
        self._eps_lb = param_bounds['eps']['lb']
        self._eps_ub = param_bounds['eps']['ub']

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
    def ns(self):
        return self._ns

    @property
    def nsph(self):
        return self._nsph

    @property
    def rgn_map(self):
        return self._rgn_map

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
    def eps_lb(self):
        return self._eps_lb

    @property
    def eps_ub(self):
        return self._eps_ub

    @property
    def glq_wts_params(self):
        return self._glq_wts_params

    @property
    def P_l_m_costheta_params(self):
        return self._P_l_m_costheta_params

    @property
    def SC(self):
        return self._SC

    @SC.setter
    def SC(self, conn):
        self._SC = conn

    @property
    def gain(self):
        return self._gain_reg

    @property
    def nroi(self):
        return self._nroi

    @property
    def roi_map_tvb_to_tfnf(self):
        return self._roi_map_tvb_to_tfnf

    @property
    def roi_map_tfnf_to_tvb(self):
        return self._roi_map_tfnf_to_tvb

    @property
    def roi_names(self):
        return self._roi_names

    @property
    def nsteps(self):
        return self._nsteps

    @property
    def nsubsteps(self):
        return self._nsubsteps

    @property
    def time_step(self):
        return self._time_step

    @tf.function
    def _build_rgn_map(self, rgn_map_reg_tvb):
        rgn_map = tf.map_fn(lambda x: self._roi_map_tvb_to_tfnf[x],
                            rgn_map_reg_tvb)
        return rgn_map

    def setup_param_mode_space_constants(self, L_MAX_PARAMS):
        # SHT constants for inference in mode space
        self._L_MAX_PARAMS = L_MAX_PARAMS
        (_, _, self._cos_theta_params, self._glq_wts_params,
         self._P_l_m_costheta_params) = tfsht.prep(self._L_MAX_PARAMS,
                                                   self._N_LAT, self._N_LON)
        self._nmodes_params = tf.pow(self._L_MAX_PARAMS + 1, 2)

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
    def x0_bounded(self, x0_hat):
        return lib.utils.tnsrflw.sigmoid_transform(x0_hat, self._x0_lb,
                                                   self._x0_ub)

    @tf.function
    def eps_bounded(self, eps_hat):
        return lib.utils.tnsrflw.sigmoid_transform(eps_hat, self._eps_lb,
                                                   self._eps_ub)

    @tf.function
    def x0_unbounded(self, x0):
        return lib.utils.tnsrflw.inv_sigmoid_transform(x0, self._x0_lb,
                                                       self._x0_ub)

    @tf.function
    def eps_unbounded(self, eps):
        return lib.utils.tnsrflw.inv_sigmoid_transform(eps, self._eps_lb,
                                                       self._eps_ub)

    @tf.function
    def transformed_parameters(self, theta, param_space):
        if param_space == 'mode':
            x0_hat_crtx = self.x0_trans_to_vrtx_space(
                theta[0:4 * self._nmodes_params])
            x0_hat_subcrtx = theta[4 * self._nmodes_params:4 *
                                   self._nmodes_params + self._ns]
            x0_hat = tf.concat([x0_hat_crtx, x0_hat_subcrtx], axis=0)
            # tf.print(x0_hat.shape)
            x0 = self.x0_bounded(x0_hat)
            eps_hat = theta[4 * self._nmodes_params + self._ns]
            eps = self.eps_bounded(eps_hat)
        if param_space == 'vertex':
            x0 = self.x0_bounded(theta[0:self._nv + self._ns])
            eps = self.eps_bounded(theta[self._nv + self._ns])
        return x0, eps

    @tf.function
    def inv_transformed_parameters(self, x0, eps, param_space):
        if param_space == 'mode':
            x0_hat = self.x0_unbounded(x0)
            x0_crtx_lh = x0_hat[0:self._nvph]
            x0_crtx_rh = x0_hat[self._nvph:self._nv]
            x0_subcrtx = x0_hat[self._nv:self._nv + self._ns]
            x0_crtx_lm_lh = tfsht.analys(self._L_MAX_PARAMS, self._N_LAT,
                                         self._N_LON, x0_crtx_lh,
                                         self._glq_wts_params,
                                         self._P_l_m_costheta_params)
            x0_crtx_lm_rh = tfsht.analys(self._L_MAX_PARAMS, self._N_LAT,
                                         self._N_LON, x0_crtx_rh,
                                         self._glq_wts_params,
                                         self._P_l_m_costheta_params)
            x0_lm = tf.concat(values=(tf.math.real(x0_crtx_lm_lh),
                                      tf.math.imag(x0_crtx_lm_lh),
                                      tf.math.real(x0_crtx_lm_rh),
                                      tf.math.imag(x0_crtx_lm_rh), x0_subcrtx),
                              axis=0)
            eps_hat = self.eps_unbounded(eps)
            theta = tf.concat((x0_lm, eps_hat), axis=0)
        if param_space == 'vertex':
            x0_hat = self.x0_unbounded(x0)
            eps_hat = self.eps_unbounded(eps)
            theta = tf.concat((x0_hat, eps_hat), axis=0)
        return theta

    @tf.function
    def _roi_mean(self, x):
        x_roi = tf.TensorArray(size=len(self._low_idcs), dtype=tf.float32)
        i = 0
        for l_idx, h_idx in zip(self._low_idcs, self._high_idcs):
            x_roi = x_roi.write(i, tf.reduce_mean(x[l_idx:h_idx]))
            i += 1
        return x_roi.stack()

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
        x = y[0:self._nv + self._ns]
        # x_subcort = y[self._nv:self._nv + self._ns]
        z = y[self._nv + self._ns:2 * (self._nv + self._ns)]

        I1 = tf.constant(4.1, dtype=tf.float32)
        # NOTE: alpha > 7.0 is causing DormandPrince integrator to diverge
        alpha = tf.constant(1.0, dtype=tf.float32)
        theta = tf.constant(-1.0, dtype=tf.float32)
        gamma_lc = tf.constant(2.0, dtype=tf.float32)
        x_crtx_hat = tf.math.sigmoid(
            alpha *
            (x[0:self._nv] - theta)) * self._unkown_roi_mask[0:self._nv]
        local_cplng = self._local_coupling(
            x_crtx_hat,
            self._glq_wts,
            self._P_l_m_costheta,
            self._Dll,
            self._L_MAX,
            self._N_LAT,
            self._N_LON,
            self._P_l_m_Dll,
            self._cos_m_phidb,
        )
        # Append zeros for subcortical regions
        local_cplng = tf.concat(
            [local_cplng, tf.zeros(self._ns, dtype=tf.float32)], axis=0)
        x_sorted = tf.gather(x, self._rgn_map_argsort)
        # x_crtx_roi = self._roi_mean(x_crtx_sorted)
        x_roi = tfp.stats.windowed_mean(x_sorted, self._low_idcs,
                                        self._high_idcs)
        global_cplng_roi = tf.reduce_sum(
            K * self._SC * (x_roi[tf.newaxis, :] - x_roi[:, tf.newaxis]),
            axis=1)
        global_cplng_vrtcs = tf.gather(global_cplng_roi, self._rgn_map)
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
                                      maximum_iterations=nsubsteps)

            y = y.write(i, y_next)
            return i + 1, y, y_next

        i = tf.constant(0)
        i, y, y_next = tf.while_loop(cond1,
                                     body1, (i, y, y_next),
                                     maximum_iterations=nsteps)
        return y.stack()

    @tf.function
    def project_sensor_space(self, x):
        slp = tf.math.log(tf.matmul(tf.math.exp(x), self._gain_reg))
        return slp

    def setup_inference(self, nsteps, nsubsteps, time_step, y_init, tau, K,
                        x0_prior_mu):
        # self._obs = obs_data
        self._nsteps = nsteps
        self._nsubsteps = nsubsteps
        self._time_step = time_step
        self._y_init = y_init
        self._tau = tau
        self._K = K
        self._x0_prior_mu = x0_prior_mu

    # def _build_priors(self, x0_prior_loc, x0_prior_scale):
    #     # self._x0_prior = tfd.MixtureSameFamily(
    #     #     mixture_distribution=tfd.Categorical(probs=[0.9, 0.1]),
    #     #     components_distribution=tfd.MultivariateNormalDiag(
    #     #         loc=x0_prior_loc, scale_diag=x0_prior_scale))
    #     self._x0_prior = tfd.Normal(loc=x0_prior_loc, scale=x0_prior_scale)

    @tf.function
    def _prior_log_prob(self, x0, roi_weighted):
        if roi_weighted:
            x0_crtx_prior_lp = tf.reduce_sum(
                tfd.Normal(loc=self._x0_prior_mu[0:self._nv],
                           scale=0.5).log_prob(x0[0:self._nv]) *
                self._vrtx_wts[0:self._nv])
        else:
            x0_crtx_prior_lp = tf.reduce_sum(
                tfd.Normal(loc=self._x0_prior_mu[0:self._nv],
                           scale=0.5).log_prob(x0[0:self._nv]))
        x0_subcrtx_prior_lp = tf.reduce_sum(
            tfd.Normal(loc=self._x0_prior_mu[self._nv:self._nv + self._ns],
                       scale=0.5).log_prob(x0[self._nv:self._nv + self._ns]))
        x0_prior_lp = x0_crtx_prior_lp + x0_subcrtx_prior_lp
        return x0_prior_lp

    @tf.function
    def _likelihood_log_prob(self, x0, eps, obs_data, obs_space):
        y_pred = self.simulate(self._nsteps, self._nsubsteps, self._time_step,
                               self._y_init, x0, self._tau, self._K)
        x_pred = y_pred[:, 0:self._nv + self._ns] * self._unkown_roi_mask
        # tf.print("nan in x_pred", tf.reduce_any(tf.math.is_nan(x_pred)))
        if (obs_space == 'sensor'):
            slp_mu = self.project_sensor_space(x_pred)
            # # tf.print("Nan in x_mu", tf.reduce_any(tf.math.is_nan(x_mu)))
            llp = tf.reduce_mean(
                tf.reduce_sum(tfd.Normal(loc=slp_mu,
                                         scale=eps).log_prob(obs_data),
                              axis=[1, 2]))
        if (obs_space == 'source'):
            llp = tf.reduce_sum(
                tfd.Normal(loc=x_pred, scale=eps).log_prob(obs_data) *
                self._vrtx_wts)
        return llp

    @tf.function
    def log_prob(self, theta, obs_data, param_space, obs_space,
                 prior_roi_weighted):
        nsamples = theta.shape[0]
        lp = tf.TensorArray(dtype=tf.float32, size=nsamples)
        i = tf.constant(0)

        def cond(i, lp):
            return tf.less(i, nsamples)

        def body(i, lp):
            # eps = tf.constant(0.1, dtype=tf.float32)
            # tf.print("nan in theta", tf.reduce_any(tf.math.is_nan(theta)))
            x0, eps = self.transformed_parameters(theta[i], param_space)
            x0_unkown_masked = x0 * self._unkown_roi_mask

            prior_lp = self._prior_log_prob(x0_unkown_masked,
                                            prior_roi_weighted)
            likelihood_lp = self._likelihood_log_prob(x0_unkown_masked, eps,
                                                      obs_data, obs_space)
            lp_i = prior_lp + likelihood_lp
            tf.print("likelihood = ", likelihood_lp, "prior = ", prior_lp)
            lp = lp.write(i, lp_i)
            return i + 1, lp

        i, lp = tf.while_loop(cond=cond,
                              body=body,
                              loop_vars=(i, lp),
                              parallel_iterations=1)
        return lp.stack()
