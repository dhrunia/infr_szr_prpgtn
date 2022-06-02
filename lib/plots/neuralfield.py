import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import lib.utils.consts as consts

def create_video(x, N_LAT, N_LON, out_dir, movie_name=None, clim=None):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(f'{out_dir}/*')
    for f in files:
        os.remove(f)

    if clim is None:
        clim = {'min': np.min(x), 'max': np.max(x)}

    for i in range(x.shape[0]):
        _, axs = plt.subplots(1, 2, dpi=200, figsize=(7, 4), tight_layout=True)
        spatial_map(x[i],
                    N_LAT=N_LAT,
                    N_LON=N_LON,
                    clim=clim,
                    axs=axs,
                    fig_dir=out_dir,
                    fig_name=f'x_{i+1:06d}.png')
        plt.close()

    (ffmpeg.input(f"{out_dir}/x_%06d.png", framerate=16).filter_(
        'scale', size='hd1080', force_original_aspect_ratio='increase').output(
            f'{out_dir}/{movie_name}').run())


def x0_gt_vs_infer(x0_gt,
                   x0_infer_mean,
                   x0_infer_std,
                   N_LAT,
                   N_LON,
                   unkown_roi_mask,
                   fig_dir,
                   fig_name=None):
    nvph = N_LAT * N_LON
    x0_infer_mean[np.where(unkown_roi_mask == 0)[0]] = -3.0
    # x0_infer_std[np.where(_unkown_roi_mask == 0)[0]] = -3.0
    # x0_hat[np.where(_unkown_roi_mask == 0)[0]] = -3.0
    plt.figure(dpi=200, figsize=(7, 4))
    x0_gt_lh = np.reshape(x0_gt[0:nvph], (N_LAT, N_LON))
    x0_gt_rh = np.reshape(x0_gt[nvph:], (N_LAT, N_LON))
    x0_infer_mean_lh = np.reshape(x0_infer_mean[0:nvph], (N_LAT, N_LON))
    x0_infer_mean_rh = np.reshape(x0_infer_mean[nvph:], (N_LAT, N_LON))
    x0_infer_std_lh = np.reshape(x0_infer_std[0:nvph], (N_LAT, N_LON))
    x0_infer_std_rh = np.reshape(x0_infer_std[nvph:], (N_LAT, N_LON))
    clim_min = np.min([np.min(x0_gt), np.min(x0_infer_mean)])
    clim_max = np.max([np.max(x0_gt), np.max(x0_infer_mean)])
    plt.subplot(321)
    plt.imshow(x0_gt_lh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Ground Truth - Left hemisphere", fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(322)
    plt.imshow(x0_gt_rh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Ground Truh - Right hemisphere", fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)

    plt.subplot(323)
    plt.imshow(x0_infer_mean_lh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Inferred Mean - Left hemisphere", fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(324)
    plt.imshow(x0_infer_mean_rh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Inferred Mean - Right hemisphere", fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.tight_layout()

    plt.subplot(325)
    plt.imshow(x0_infer_std_lh, interpolation=None, cmap='hot')
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Left hemisphere", fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(326)
    plt.imshow(x0_infer_std_rh, interpolation=None, cmap='hot')
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Right hemisphere", fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.tight_layout()
    if (fig_name is not None):
        plt.savefig(f'{fig_dir}/{fig_name}',
                    facecolor='white',
                    bbox_inches='tight')


def spatial_map(x,
                N_LAT,
                N_LON,
                title={
                    'lh': 'Left Hemisphsere',
                    'rh': 'Right Hemisphere'
                },
                clim=None,
                fig_dir=None,
                fig_name=None,
                axs=None):
    nvph = N_LAT * N_LON
    x_lh = np.reshape(x[0:nvph], (N_LAT, N_LON))
    x_rh = np.reshape(x[nvph:], (N_LAT, N_LON))
    if axs is None:
        fig, axs = plt.subplots(1,
                                2,
                                dpi=200,
                                figsize=(7, 4),
                                tight_layout=True)
    if clim is None:
        clim = {'min': np.min(x), 'max': np.max(x)}
    fig = axs[0].get_figure()
    im = axs[0].imshow(x_lh, interpolation=None, cmap='hot')
    im.set_clim(clim['min'], clim['max'])
    axs[0].set_title(title['lh'], fontsize=consts.FS_LARGE)
    axs[0].set_xlabel("Longitude", fontsize=consts.FS_MED)
    axs[0].set_ylabel("Latitude", fontsize=consts.FS_MED)
    axs[0].tick_params(labelsize=consts.FS_SMALLl)
    cbar = fig.colorbar(im, ax=axs[0], shrink=0.2, fraction=0.3)
    cbar.ax.tick_params(labelsize=consts.FS_SMALLl)
    im = axs[1].imshow(x_rh, interpolation=None, cmap='hot')
    im.set_clim(clim['min'], clim['max'])
    axs[1].set_title(title['rh'], fontsize=consts.FS_LARGE)
    axs[1].set_xlabel("Longitude", fontsize=consts.FS_MED)
    axs[1].set_ylabel("Latitude", fontsize=consts.FS_MED)
    axs[1].tick_params(labelsize=consts.FS_SMALLl)
    cbar = fig.colorbar(im, ax=axs[1], shrink=0.2, fraction=0.3)
    cbar.ax.tick_params(labelsize=consts.FS_SMALLl)
    if (fig_name is not None):
        fig.savefig(f"{fig_dir}/{fig_name}",
                    bbox_inches='tight',
                    facecolor='white')
