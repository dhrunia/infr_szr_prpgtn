# %%
import tensorflow as tf
import tensorflow_probability as tfp
import lib.model.neuralfield
import numpy as np
import os

tfd = tfp.distributions
# %%
L_MAX_lc = 32

for N_LAT in tf.range(64, 129):
    dyn_mdl = lib.model.neuralfield.Epileptor2D(
        L_MAX=L_MAX_lc,
        N_LAT=N_LAT,
        N_LON=2 * N_LAT,
        verts_irreg_fname="datasets/data_jd/id004_bj/tvb/ico7/vertices.txt",
        rgn_map_irreg_fname=
        "datasets/data_jd/id004_bj/tvb/Cortex_region_map_ico7.txt",
        conn_zip_path="datasets/data_jd/id004_bj/tvb/connectivity.vep.zip",
        gain_irreg_path=
        "datasets/data_jd/id004_bj/tvb/gain_inv_square_ico7.npz",
        gain_irreg_rgn_map_path=
        "datasets/data_jd/id004_bj/tvb/gain_region_map_ico7.txt",
        L_MAX_PARAMS=16,
        diff_coeff=0.00047108,
        alpha=2.0,
        theta=-1.0)
    x_init_true = tf.constant(-2.0, dtype=tf.float32) * tf.ones(
    dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32) * \
        dyn_mdl.unkown_roi_mask
    z_init_true = tf.constant(5.0, dtype=tf.float32) * tf.ones(
        dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32) * \
            dyn_mdl.unkown_roi_mask
    y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
    tau_true = tf.constant(25, dtype=tf.float32, shape=())
    # K_true = tf.constant(1.0, dtype=tf.float32, shape=())
    # x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
    x0_true = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
    ez_hyp_roi_tvb = [116, 127, 157]
    ez_hyp_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_hyp_roi_tvb]
    ez_hyp_vrtcs = np.concatenate(
        [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
    x0_true[ez_hyp_vrtcs] = -1.8
    pz_hyp_roi_tvb = [114, 148]
    pz_hyp_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in pz_hyp_roi_tvb]
    pz_hyp_vrtcs = np.concatenate(
        [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in pz_hyp_roi])
    # x0_true[pz_hyp_vrtcs] = -2.1
    x0_true = tf.constant(x0_true, dtype=tf.float32) * dyn_mdl.unkown_roi_mask
    t = dyn_mdl.SC.numpy()
    t[dyn_mdl.roi_map_tvb_to_tfnf[114], dyn_mdl.roi_map_tvb_to_tfnf[116]] = 2.0
    t[dyn_mdl.roi_map_tvb_to_tfnf[148], dyn_mdl.roi_map_tvb_to_tfnf[127]] = 2.0
    K_true = t.max()
    t = t / t.max()
    dyn_mdl.SC = tf.constant(t, dtype=tf.float32)
    amp_true = dyn_mdl.amp_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
    offset_true = dyn_mdl.offset_bounded(
        tfd.Normal(loc=0.0, scale=1.0).sample())
    eps_true = 0.1

    nsteps = tf.constant(300, dtype=tf.int32)
    sampling_period = tf.constant(0.1, dtype=tf.float32)
    time_step = tf.constant(0.05, dtype=tf.float32)
    nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                        dtype=tf.int32)
    gamma_lc = 1.0

    y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true,
                             x0_true, tau_true, K_true, gamma_lc)
    x_obs = y_obs[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
    slp_true = amp_true * dyn_mdl.project_sensor_space(x_obs) + offset_true

    n_sample_aug = 50
    obs_data_aug = tf.TensorArray(dtype=tf.float32, size=n_sample_aug)
    for j in range(n_sample_aug):
        data_noised = slp_true + \
            tf.random.normal(shape=slp_true.shape, mean=0, stddev=eps_true)
        obs_data_aug = obs_data_aug.write(j, data_noised)
    obs_data_aug = obs_data_aug.stack()
    save_dir = f'datasets/syn_data/id004_bj/LMAX_lc_{L_MAX_lc}'
    os.makedirs(save_dir, exist_ok=True)
    np.savez(f'{save_dir}/sim_N_LAT{N_LAT}.npz',
             x0=x0_true.numpy(),
             x_init=x_init_true.numpy(),
             z_init=z_init_true.numpy(),
             SC=dyn_mdl.SC.numpy(),
             ez_roi_tvb=ez_hyp_roi_tvb,
             pz_roi_tvb=pz_hyp_roi_tvb,
             tau=tau_true.numpy(),
             amp=amp_true.numpy(),
             offset=offset_true.numpy(),
             x=x_obs.numpy(),
             slp=slp_true.numpy(),
             slp_aug=obs_data_aug.numpy(),
             eps=eps_true)
