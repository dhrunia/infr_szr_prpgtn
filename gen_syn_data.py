# %%
import tensorflow as tf
import tensorflow_probability as tfp
import lib.model.neuralfield
import numpy as np
import os

tfd = tfp.distributions
# %% Various Spatial resolution i.e. N_LAT = [64:128]
L_MAX_lc = 32
data_dir = 'datasets/data_jd/id022_te'
save_dir = f'datasets/syn_data/id022_te/LMAX_lc_{L_MAX_lc}'
os.makedirs(save_dir, exist_ok=True)
for N_LAT in tf.range(64, 129):
    dyn_mdl = lib.model.neuralfield.Epileptor2D(
        L_MAX=L_MAX_lc,
        N_LAT=N_LAT,
        N_LON=2 * N_LAT,
        verts_irreg_fname=f"{data_dir}/tvb/ico7/vertices.txt",
        rgn_map_irreg_fname=f"{data_dir}/tvb/Cortex_region_map_ico7.txt",
        conn_zip_path=f"{data_dir}/tvb/connectivity.vep.zip",
        gain_irreg_path=f"{data_dir}/tvb/gain_inv_square_ico7.npz",
        gain_irreg_rgn_map_path=f"{data_dir}/tvb/gain_region_map_ico7.txt",
        L_MAX_PARAMS=16)
    nv_total = dyn_mdl.nv + dyn_mdl.ns
    x_init_true = tf.constant(-2.0, dtype=tf.float32) * tf.ones(
    nv_total, dtype=tf.float32) * \
        dyn_mdl.unkown_roi_mask
    z_init_true = tf.constant(5.0, dtype=tf.float32) * tf.ones(
        nv_total, dtype=tf.float32) * \
            dyn_mdl.unkown_roi_mask
    y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
    tau_true = tf.constant(25, dtype=tf.float32, shape=())
    K_true = tf.constant(1.0, dtype=tf.float32, shape=())
    x0_true = -3.0 * np.ones(nv_total)
    ez_true_roi_tvb = [57, 61, 56, 55]
    ez_true_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_true_roi_tvb]
    ez_true_vrtcs = np.concatenate(
        [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_true_roi])
    x0_true[ez_true_vrtcs] = -1.8
    x0_true = tf.constant(x0_true, dtype=tf.float32) * dyn_mdl.unkown_roi_mask
    amp_true = dyn_mdl.amp_bounded(tfd.Normal(loc=0.0, scale=1.0).sample())
    offset_true = dyn_mdl.offset_bounded(
        tfd.Normal(loc=0.0, scale=1.0).sample())
    eps_true = 0.1

    nsteps = tf.constant(300, dtype=tf.int32)
    sampling_period = tf.constant(0.1, dtype=tf.float32)
    time_step = tf.constant(0.05, dtype=tf.float32)
    nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                        dtype=tf.int32)

    y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true,
                             x0_true, tau_true, K_true)
    x_obs = y_obs[:, 0:nv_total] * dyn_mdl.unkown_roi_mask
    z_obs = y_obs[:, nv_total:2 * (nv_total)] * dyn_mdl.unkown_roi_mask
    slp_true = amp_true * dyn_mdl.project_sensor_space(x_obs) + offset_true
    print(f"Saving simulated for N_LAT={N_LAT}")
    np.savez(f'{save_dir}/sim_N_LAT{N_LAT}.npz',
             x0=x0_true.numpy(),
             x_init=x_init_true.numpy(),
             z_init=z_init_true.numpy(),
             SC=dyn_mdl.SC.numpy(),
             rgn_map=dyn_mdl.rgn_map.numpy(),
             K=K_true.numpy(),
             ez_roi_tvb=ez_true_roi_tvb,
             ez_roi=ez_true_roi,
             ez_vrtcs=ez_true_vrtcs,
             tau=tau_true.numpy(),
             amp=amp_true.numpy(),
             offset=offset_true.numpy(),
             x=x_obs.numpy(),
             z=z_obs.numpy(),
             unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
             slp=slp_true.numpy(),
             eps=eps_true,
             nsteps=nsteps,
             sampling_period=sampling_period,
             time_step=time_step)