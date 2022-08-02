import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import lib.utils.consts as consts


def create_video(x,
                 N_LAT,
                 N_LON,
                 out_dir,
                 movie_name=None,
                 clim=None,
                 unkown_roi_mask=None):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(f'{out_dir}/*')
    for f in files:
        os.remove(f)

    if clim is None:
        clim = {'min': np.min(x), 'max': np.max(x)}

    for i in range(x.shape[0]):
        spatial_map(x[i],
                    N_LAT=N_LAT,
                    N_LON=N_LON,
                    clim=clim,
                    fig_dir=out_dir,
                    fig_name=f'x_{i+1:06d}.png',
                    unkown_roi_mask=unkown_roi_mask)
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
    plt.title("Standard Deviation - Left hemisphere",
              fontsize=consts.FS_SMALLl)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(326)
    plt.imshow(x0_infer_std_rh, interpolation=None, cmap='hot')
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Right hemisphere",
              fontsize=consts.FS_SMALLl)
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
                title=None,
                clim=None,
                fig_dir=None,
                fig_name=None,
                ax=None,
                unkown_roi_mask=None):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    nvph = N_LAT * N_LON
    nv = 2 * nvph
    if clim is None:
        clim = {'min': np.min(x), 'max': np.max(x)}

    if unkown_roi_mask is not None:
        x[np.nonzero(unkown_roi_mask == 0)] = clim['min']

    x_lh = np.reshape(x[0:nvph], (N_LAT, N_LON))
    x_rh = np.reshape(x[nvph:nv], (N_LAT, N_LON))
    x_subcrtx_lh = x[nv:nv + 9][np.newaxis, :]
    x_subcrtx_rh = x[nv + 9:][np.newaxis, :]

    if ax is None:
        fig = plt.figure(figsize=(7, 4), dpi=200, constrained_layout=True)
        gs = fig.add_gridspec(2,
                              3,
                              height_ratios=[0.9, 0.1],
                              width_ratios=[0.49, 0.49, 0.02])
        ax = {}
        ax['crtx_lh'] = fig.add_subplot(gs[0, 0])
        ax['crtx_rh'] = fig.add_subplot(gs[0, 1])
        ax['subcrtx_lh'] = fig.add_subplot(gs[1, 0])
        ax['subcrtx_rh'] = fig.add_subplot(gs[1, 1])
        ax['clr_bar'] = fig.add_subplot(gs[:, 2])

    if title is None:
        title = {
            'crtx_lh': 'Left Hemisphsere',
            'crtx_rh': 'Right Hemisphere',
            'subcrtx_lh': 'Left Subcortical',
            'subcrtx_rh': 'Right Subcortical'
        }

    im = ax['crtx_lh'].imshow(x_lh,
                              interpolation=None,
                              cmap='hot',
                              vmin=clim['min'],
                              vmax=clim['max'])
    ax['crtx_lh'].set_title(title['crtx_lh'], fontsize=consts.FS_LARGE)
    ax['crtx_lh'].set_xlabel("Longitude", fontsize=consts.FS_MED)
    ax['crtx_lh'].set_ylabel("Latitude", fontsize=consts.FS_MED)
    ax['crtx_lh'].tick_params(labelsize=consts.FS_SMALL)
    im = ax['crtx_rh'].imshow(x_rh,
                              interpolation=None,
                              cmap='hot',
                              vmin=clim['min'],
                              vmax=clim['max'])
    ax['crtx_rh'].set_title(title['crtx_rh'], fontsize=consts.FS_LARGE)
    ax['crtx_rh'].set_xlabel("Longitude", fontsize=consts.FS_MED)
    ax['crtx_rh'].set_ylabel("Latitude", fontsize=consts.FS_MED)
    ax['crtx_rh'].tick_params(labelsize=consts.FS_SMALL)
    im = ax['subcrtx_lh'].imshow(x_subcrtx_lh,
                                 interpolation=None,
                                 cmap='hot',
                                 vmin=clim['min'],
                                 vmax=clim['max'])
    ax['subcrtx_lh'].axes.yaxis.set_visible(False)
    ax['subcrtx_lh'].set_title(title['subcrtx_lh'], fontsize=consts.FS_LARGE)
    ax['subcrtx_lh'].tick_params(labelsize=consts.FS_SMALL)
    im = ax['subcrtx_rh'].imshow(x_subcrtx_rh,
                                 interpolation=None,
                                 cmap='hot',
                                 vmin=clim['min'],
                                 vmax=clim['max'])
    ax['subcrtx_rh'].axes.yaxis.set_visible(False)
    ax['subcrtx_rh'].set_title(title['subcrtx_rh'], fontsize=consts.FS_LARGE)
    ax['subcrtx_rh'].tick_params(labelsize=consts.FS_SMALL)
    sm = ScalarMappable(norm=Normalize(vmin=clim['min'], vmax=clim['max']),
                        cmap='hot')
    cbar = plt.colorbar(sm, cax=ax['clr_bar'])
    ax['clr_bar'].tick_params(labelsize=consts.FS_SMALL)

    if fig_dir is None:
        fig_dir = os.getcwd()

    os.makedirs(fig_dir, exist_ok=True)

    if fig_name is not None:
        plt.savefig(f"{fig_dir}/{fig_name}",
                    bbox_inches='tight',
                    facecolor='white')
