# %%
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# tf.keras.backend.set_floatx('float64')
# tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import time
import lib.utils.tnsrflw
import lib.plots.neuralfield
import lib.plots.seeg
import os
import lib.model.neuralfield
from lib.sampler import SVGD

tfd = tfp.distributions
tfb = tfp.bijectors

# %%
results_dir = 'results/exp54'
os.makedirs(results_dir, exist_ok=True)
figs_dir = f'{results_dir}/figures'
os.makedirs(figs_dir, exist_ok=True)

dyn_mdl = lib.model.neuralfield.Epileptor2D(
    L_MAX=32,
    N_LAT=65,  #129,
    N_LON=129,  #257,
    verts_irreg_fname='datasets/data_jd/id004_bj/tvb/ico7/vertices.txt',
    rgn_map_irreg_fname=
    'datasets/data_jd/id004_bj/tvb/Cortex_region_map_ico7.txt',
    conn_zip_path='datasets/data_jd/id004_bj/tvb/connectivity.vep.zip',
    gain_irreg_path='datasets/data_jd/id004_bj/tvb/gain_inv_square_ico7.npz',
    gain_irreg_rgn_map_path=
    'datasets/data_jd/id004_bj/tvb/gain_region_map_ico7.txt',
    L_MAX_PARAMS=16)

# %%
x_init_true = tf.constant(-2.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
z_init_true = tf.constant(5.0, dtype=tf.float32) * \
    tf.ones(dyn_mdl.nv + dyn_mdl.ns, dtype=tf.float32)
y_init_true = tf.concat((x_init_true, z_init_true), axis=0)
tau_true = tf.constant(25, dtype=tf.float32, shape=())
K_true = tf.constant(1.0, dtype=tf.float32, shape=())
# x0_true = tf.constant(tvb_syn_data['x0'], dtype=tf.float32)
x0_true = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
ez_hyp_roi_tvb = [157]  #[116, 127, 157]
ez_hyp_roi = [dyn_mdl.roi_map_tvb_to_tfnf[roi] for roi in ez_hyp_roi_tvb]
ez_hyp_vrtcs = np.concatenate(
    [np.nonzero(roi == dyn_mdl.rgn_map)[0] for roi in ez_hyp_roi])
x0_true[ez_hyp_vrtcs] = -1.8
x0_true = tf.constant(x0_true, dtype=tf.float32)
# t = dyn_mdl.SC.numpy()
# t[dyn_mdl.roi_map_tvb_to_tfnf[140], dyn_mdl.roi_map_tvb_to_tfnf[116]] = 5.0
# dyn_mdl.SC = tf.constant(t, dtype=tf.float32)
# %%
lib.plots.neuralfield.spatial_map(
    x0_true.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    clim={
        'min': -5.0,
        'max': 0.0
    },
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    fig_dir=f"{figs_dir}/ground_truth",
    fig_name="x0_gt.png")
# %%
nsteps = tf.constant(300, dtype=tf.int32)
sampling_period = tf.constant(0.1, dtype=tf.float32)
time_step = tf.constant(0.05, dtype=tf.float32)
nsubsteps = tf.cast(tf.math.floordiv(sampling_period, time_step),
                    dtype=tf.int32)

y_obs = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_true,
                         tau_true, K_true)
x_obs = y_obs[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_obs = dyn_mdl.project_sensor_space(x_obs)
# %%
lib.plots.seeg.plot_slp(slp_obs.numpy(),
                        save_dir=f'{figs_dir}/ground_truth',
                        fig_name='slp_obs.png')
# %%
n_sample_aug = 50
obs_data_aug = tf.TensorArray(dtype=tf.float32, size=n_sample_aug)
for j in range(n_sample_aug):
    data_noised = slp_obs + \
        tf.random.normal(shape=slp_obs.shape, mean=0, stddev=0.1)
    obs_data_aug = obs_data_aug.write(j, data_noised)
obs_data_aug = obs_data_aug.stack()
# %%
x0_prior_mu = -3.0 * np.ones(dyn_mdl.nv + dyn_mdl.ns)
# x0_prior_mu[-6] = -1.5
x0_prior_mu = tf.constant(x0_prior_mu,
                          dtype=tf.float32) * dyn_mdl.unkown_roi_mask
dyn_mdl.setup_inference(nsteps=nsteps,
                        nsubsteps=nsubsteps,
                        time_step=time_step,
                        y_init=y_init_true,
                        tau=tau_true,
                        K=K_true,
                        x0_prior_mu=x0_prior_mu,
                        obs_data=obs_data_aug,
                        param_space='mode',
                        obs_space='sensor',
                        prior_roi_weighted=False)
# %%
nparams = 4 * dyn_mdl.nmodes_params + dyn_mdl.ns + 1
init_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(nparams),
                                     scale_diag=tf.ones(nparams))
samples = init_dist.sample(10)
svgd_sampler = SVGD(dyn_mdl=dyn_mdl)
# %%
start_time = time.time()
n_iters = 100
samples = svgd_sampler.update(samples, n_iters, stepsize=1e-3)
print(f"Elapse {time.time() - start_time} seconds")
# %%
x0_samples = tf.TensorArray(dtype=tf.float32,
                            size=samples.shape[0],
                            clear_after_read=False)
eps_samples = tf.TensorArray(dtype=tf.float32,
                             size=samples.shape[0],
                             clear_after_read=False)
for i, theta in enumerate(samples.numpy()):
    x0_i, eps_i = dyn_mdl.transformed_parameters(theta, param_space='mode')
    x0_samples = x0_samples.write(i, x0_i)
    eps_samples = eps_samples.write(i, eps_i)
x0_samples = x0_samples.stack()
eps_samples = eps_samples.stack()
x0_mean = tf.reduce_mean(x0_samples, axis=0).numpy()
x0_std = tf.math.reduce_std(x0_samples, axis=0).numpy()
eps_mean = tf.reduce_mean(eps_samples, axis=0).numpy()
eps_std = tf.math.reduce_std(eps_samples, axis=0).numpy()
# %%
# fig_name = 'x0_gt_vs_inferred_mean_and_std.png'
# lib.plots.neuralfield.x0_gt_vs_infer(x0_true, x0_mean, x0_std,
#                                      dyn_mdl.N_LAT.numpy(),
#                                      dyn_mdl.N_LON.numpy(),
#                                      dyn_mdl.unkown_roi_mask, figs_dir,
#                                      fig_name)
lib.plots.neuralfield.spatial_map(
    x0_mean,
    N_LAT=dyn_mdl.N_LAT,
    N_LON=dyn_mdl.N_LON,
    clim={
        'min': -3.0,
        'max': -1.0
    },
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    fig_dir=f"{figs_dir}",
    fig_name="x0_posterior_mean.png")
lib.plots.neuralfield.spatial_map(
    x0_std,
    N_LAT=dyn_mdl.N_LAT,
    N_LON=dyn_mdl.N_LON,
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    fig_dir=f"{figs_dir}",
    fig_name="x0_posterior_std.png")
# %%
for i, smpl in enumerate(x0_samples.numpy()):
    lib.plots.neuralfield.spatial_map(
    smpl,
    N_LAT=dyn_mdl.N_LAT,
    N_LON=dyn_mdl.N_LON,
    clim={
        'min': -3.0,
        'max': -1.0
    },
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy(),
    fig_dir=f"{figs_dir}",
    fig_name=f"x0_posterior_sample{i}.png")
# %%
y_ppc = dyn_mdl.simulate(nsteps, nsubsteps, time_step, y_init_true, x0_mean,
                         tau_true, K_true)
x_ppc = y_ppc[:, 0:dyn_mdl.nv + dyn_mdl.ns] * dyn_mdl.unkown_roi_mask
slp_ppc = dyn_mdl.project_sensor_space(x_ppc)
# %%
lib.plots.neuralfield.create_video(
    x_ppc.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f'{figs_dir}/infer',
    movie_name='movie.mp4',
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())

# %%
out_dir = f'{figs_dir}/ground_truth'
lib.plots.neuralfield.create_video(
    x_obs.numpy(),
    N_LAT=dyn_mdl.N_LAT.numpy(),
    N_LON=dyn_mdl.N_LON.numpy(),
    out_dir=f'{figs_dir}/ground_truth',
    movie_name='movie.mp4',
    unkown_roi_mask=dyn_mdl.unkown_roi_mask.numpy())


# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=200)
lib.plots.seeg.plot_slp(slp_obs.numpy(), ax=axs[0], title='Observed')
lib.plots.seeg.plot_slp(slp_ppc.numpy(), ax=axs[1], title='Predicted')
fig.savefig(f'{figs_dir}/slp_obs_vs_pred.png', facecolor='white')
