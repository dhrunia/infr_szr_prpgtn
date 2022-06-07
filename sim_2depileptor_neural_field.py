# %%
from time import time
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import lib.model.neuralfield
import matplotlib.pyplot as plt
import os
import lib.plots.neuralfield
import lib.plots.seeg
# %%
results_dir = 'tmp'
os.makedirs(results_dir, exist_ok=True)
figs_dir = f'{results_dir}/figures'
os.makedirs(figs_dir, exist_ok=True)
dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=128,
    N_LAT=129,
    N_LON=257,
    verts_irreg_fname='datasets/id004_bj_jd/tvb/ico7/vertices.txt',
    rgn_map_irreg_fname='datasets/id004_bj_jd/tvb/Cortex_region_map_ico7.txt',
    conn_zip_path='datasets/id004_bj_jd/tvb/connectivity.vep.zip',
    gain_irreg_path='datasets/id004_bj_jd/tvb/gain_inv_square_ico7.npz',
    gain_irreg_rgn_map_path='datasets/id004_bj_jd/tvb/gain_region_map_ico7.txt',
    L_MAX_PARAMS=10)
# %%
x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
x0_true = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
ez_hyp_roi_tvb = [116, 127, 151]
ez_hyp_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_hyp_roi_tvb]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
# pz_hyp_roi = [140]
# pz_hyp_vrtcs = np.concatenate(
#     [np.nonzero(roi == dyn_mdl.rgn_map_reg)[0] for roi in pz_hyp_roi])
# x0_true[pz_hyp_vrtcs] = -2.1
x0_true = tf.constant(x0_true, dtype=tf.float32)
t = dyn_mdl.SC.numpy()
t[dyn_mdl.roi_map_tvb_to_tfnf[140], dyn_mdl.roi_map_tvb_to_tfnf[116]] = 5.0
dyn_mdl.SC = tf.constant(t, dtype=tf.float32)
# %%
lib.plots.neuralfield.spatial_map(
    x0_true.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    clim={
        'min': -3.5,
        'max': -1.0
    },
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)
start_time = time()
y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                         tau_true, K_true)
print(f"Elapsed {time() - start_time} seconds")
x_obs = y_obs[:, 0:dyn_mdl.nv + dyn_mdl._ns] * dyn_mdl.unkown_roi_mask
slp_obs = dyn_mdl.project_sensor_space(x_obs)

# %%
lib.plots.neuralfield.create_video(
    x_obs.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f'{figs_dir}/source_activity',
    movie_name='movie.mp4',
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())
# %%
lib.plots.seeg.plot_slp(slp_obs.numpy(),
                        save_dir=figs_dir,
                        fig_name='obs_slp.png')
plt.savefig(f'{figs_dir}/obs_slp.png')
