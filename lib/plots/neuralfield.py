# import ffmpeg
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import lib.utils.consts as consts
import pyshtools as pysh
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def create_video(x,
                 N_LAT,
                 N_LON,
                 out_dir,
                 movie_name=None,
                 clim=None,
                 unkown_roi_mask=None,
                 vis_type='rect',
                 dpi=200,
                 ds_freq=1):
    os.makedirs(out_dir, exist_ok=True)
    # files = glob.glob(f'{out_dir}/*')
    # for f in files:
    #     os.remove(f)

    if clim is None:
        clim = {'min': np.min(x), 'max': np.max(x)}

    if vis_type == 'rect':
        fig, ax = setup_rect_spat_map_axs(dpi=dpi)
        moviewriter = anim.FFMpegWriter(fps=16)
        with moviewriter.saving(fig=fig,
                                outfile=f"{out_dir}/{movie_name}",
                                dpi=dpi):
            for i in range(x.shape[0]):
                for key in ax.keys():
                    ax[key].clear()
                rect_spat_map(x[i],
                              N_LAT=N_LAT,
                              N_LON=N_LON,
                              clim=clim,
                              unkown_roi_mask=unkown_roi_mask,
                              ax=ax)
                moviewriter.grab_frame()
    elif vis_type == 'spherical':
        fig, ax = setup_spherical_spat_map_axs(dpi=dpi)
        coords = setup_spherical_coords(N_LAT=N_LAT, N_LON=N_LON)
        moviewriter = anim.FFMpegWriter(fps=16)
        with moviewriter.saving(fig=fig,
                                outfile=f"{out_dir}/{movie_name}",
                                dpi=dpi):
            for i in range(x.shape[0]):
                for key in ax.keys():
                    ax[key].clear()
                spherical_spat_map(x[i],
                                   N_LAT=N_LAT,
                                   N_LON=N_LON,
                                   coords=coords,
                                   clim=clim,
                                   unkown_roi_mask=unkown_roi_mask,
                                   ax=ax,
                                   dpi=dpi,
                                   ds_freq=ds_freq)
                moviewriter.grab_frame()

    plt.close()

    # (ffmpeg.input(f"{out_dir}/x_%06d.png", framerate=16).filter_(
    #     'scale', size='hd1080', force_original_aspect_ratio='increase').output(
    #         f'{out_dir}/{movie_name}').run())


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
    plt.title("Ground Truth - Left hemisphere", fontsize=consts.FS_SMALL)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(322)
    plt.imshow(x0_gt_rh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Ground Truh - Right hemisphere", fontsize=consts.FS_SMALL)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)

    plt.subplot(323)
    plt.imshow(x0_infer_mean_lh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Inferred Mean - Left hemisphere", fontsize=consts.FS_SMALL)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(324)
    plt.imshow(x0_infer_mean_rh, interpolation=None, cmap='hot')
    plt.clim(clim_min, clim_max)
    plt.title("Inferred Mean - Right hemisphere", fontsize=consts.FS_SMALL)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.tight_layout()

    plt.subplot(325)
    plt.imshow(x0_infer_std_lh, interpolation=None, cmap='hot')
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Left hemisphere", fontsize=consts.FS_SMALL)
    plt.xlabel("Longitude", fontsize=consts.FS_MED)
    plt.ylabel("Latitude", fontsize=consts.FS_MED)
    plt.xticks(fontsize=consts.FS_MED)
    plt.yticks(fontsize=consts.FS_MED)
    plt.colorbar(fraction=0.02)
    plt.subplot(326)
    plt.imshow(x0_infer_std_rh, interpolation=None, cmap='hot')
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Right hemisphere",
              fontsize=consts.FS_SMALL)
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


def rect_spat_map(x,
                  N_LAT,
                  N_LON,
                  title=None,
                  clim=None,
                  fig_dir=None,
                  fig_name=None,
                  ax=None,
                  unkown_roi_mask=None,
                  dpi=200):

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
        fig, ax = setup_rect_spat_map_axs(dpi=dpi)

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

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)

    if fig_name is not None:
        plt.savefig(f"{fig_dir}/{fig_name}",
                    bbox_inches='tight',
                    facecolor='white')


def setup_rect_spat_map_axs(dpi):
    fig = plt.figure(figsize=(7, 4), dpi=dpi, constrained_layout=True)
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
    return fig, ax


def spherical_spat_map(x,
                       N_LAT,
                       N_LON,
                       unkown_roi_mask=None,
                       coords=None,
                       ax=None,
                       clim=None,
                       fig_dir=None,
                       fig_name=None,
                       dpi=200,
                       ds_freq=1):

    nvph = N_LAT * N_LON
    nv = 2 * nvph

    if clim is None:
        clim = {'min': np.min(x), 'max': np.max(x)}

    if unkown_roi_mask is not None:
        x[np.nonzero(unkown_roi_mask == 0)] = clim['min']

    if coords is None:
        coords = setup_spherical_coords(N_LAT=N_LAT, N_LON=N_LON)

    # close_plot = False
    if ax is None:
        fig, ax = setup_spherical_spat_map_axs(dpi=dpi)
        # close_plot = True

    x_crtx_lh = np.reshape(x[0:nvph], (N_LAT, N_LON))
    x_crtx_rh = np.reshape(x[nvph:2 * nvph], (N_LAT, N_LON))
    x_subcrtx_lh = x[nv:nv + 9][:, np.newaxis]
    x_subcrtx_rh = x[nv + 9:][:, np.newaxis]
    ax['crtx_lh_front'].scatter(coords['x_front'][0:-1:ds_freq],
                                coords['y_front'][0:-1:ds_freq],
                                coords['z_front'][0:-1:ds_freq],
                                s=8.0,
                                c=x_crtx_lh[0:-1:ds_freq, 0:N_LON // 2],
                                vmin=clim['min'],
                                vmax=clim['max'],
                                cmap='hot')
    ax['crtx_lh_front'].set_xlim(-0.65, 0.65)
    ax['crtx_lh_front'].set_zlim(-0.65, 0.65)
    ax['crtx_lh_front'].view_init(azim=270, elev=0)
    ax['crtx_lh_front'].set_axis_off()

    ax['crtx_lh_back'].scatter(coords['x_back'][0:-1:ds_freq],
                               coords['y_back'][0:-1:ds_freq],
                               coords['z_back'][0:-1:ds_freq],
                               s=8.0,
                               c=x_crtx_lh[0:-1:ds_freq, N_LON // 2:N_LON],
                               vmin=clim['min'],
                               vmax=clim['max'],
                               cmap='hot')
    ax['crtx_lh_back'].set_xlim(-0.65, 0.65)
    ax['crtx_lh_back'].set_zlim(-0.65, 0.65)
    ax['crtx_lh_back'].view_init(azim=90, elev=0)
    ax['crtx_lh_back'].set_axis_off()

    ax['crtx_rh_front'].scatter(coords['x_front'][0:-1:ds_freq],
                                coords['y_front'][0:-1:ds_freq],
                                coords['z_front'][0:-1:ds_freq],
                                s=8.0,
                                c=x_crtx_rh[0:-1:ds_freq, 0:N_LON // 2],
                                vmin=clim['min'],
                                vmax=clim['max'],
                                cmap='hot')
    ax['crtx_rh_front'].set_xlim(-0.65, 0.65)
    ax['crtx_rh_front'].set_zlim(-0.65, 0.65)
    ax['crtx_rh_front'].view_init(azim=270, elev=0)
    ax['crtx_rh_front'].set_axis_off()

    ax['crtx_rh_back'].scatter(coords['x_back'][0:-1:ds_freq],
                               coords['y_back'][0:-1:ds_freq],
                               coords['z_back'][0:-1:ds_freq],
                               s=8.0,
                               c=x_crtx_rh[0:-1:ds_freq, N_LON // 2:N_LON],
                               vmin=clim['min'],
                               vmax=clim['max'],
                               cmap='hot')
    ax['crtx_rh_back'].set_xlim(-0.65, 0.65)
    ax['crtx_rh_back'].set_zlim(-0.65, 0.65)
    ax['crtx_rh_back'].view_init(azim=90, elev=0)
    ax['crtx_rh_back'].set_axis_off()

    im = ax['subcrtx_lh'].imshow(x_subcrtx_lh,
                                 interpolation=None,
                                 cmap='hot',
                                 vmin=clim['min'],
                                 vmax=clim['max'])
    ax['subcrtx_lh'].axes.xaxis.set_visible(False)
    ax['subcrtx_lh'].tick_params(labelsize=consts.FS_SMALL)
    im = ax['subcrtx_rh'].imshow(x_subcrtx_rh,
                                 interpolation=None,
                                 cmap='hot',
                                 vmin=clim['min'],
                                 vmax=clim['max'])
    ax['subcrtx_rh'].axes.xaxis.set_visible(False)
    ax['subcrtx_rh'].tick_params(labelsize=consts.FS_SMALL)

    sm = ScalarMappable(norm=Normalize(vmin=clim['min'], vmax=clim['max']),
                        cmap='hot')
    cbar = plt.colorbar(sm, cax=ax['clr_bar'])
    ax['clr_bar'].tick_params(labelsize=consts.FS_SMALL)

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
    if fig_name is not None:
        plt.savefig(f'{fig_dir}/{fig_name}', facecolor='white')
    # if close_plot:
    #     plt.close()


def setup_spherical_coords(N_LAT, N_LON):
    cos_theta, _ = pysh.expand.SHGLQ(N_LAT - 1)
    theta = np.arccos(cos_theta)
    phi_back = np.arange(0, np.pi, 2 * np.pi / N_LON)
    phi_front = np.arange(np.pi, 2 * np.pi, 2 * np.pi / N_LON)
    theta_grid, phi_grid = np.meshgrid(theta, phi_front, indexing='ij')
    coords = {}
    coords['x_front'] = (np.cos(phi_grid) * np.sin(theta_grid))
    coords['y_front'] = (np.sin(phi_grid) * np.sin(theta_grid))
    coords['z_front'] = np.cos(theta_grid)
    theta_grid, phi_grid = np.meshgrid(theta, phi_back, indexing='ij')
    coords['x_back'] = (np.cos(phi_grid) * np.sin(theta_grid))
    coords['y_back'] = (np.sin(phi_grid) * np.sin(theta_grid))
    coords['z_back'] = np.cos(theta_grid)
    return coords


def setup_spherical_spat_map_axs(fig=None, dpi=200):
    if fig is None:
        fig = plt.figure(figsize=(3, 7), dpi=dpi, constrained_layout=False)
    fig.set_facecolor('0.75')
    gs = fig.add_gridspec(8,
                          3,
                          width_ratios=[0.1, 0.8, 0.1],
                          hspace=0,
                          wspace=0,
                          left=0.05,
                          right=0.88,
                          top=1,
                          bottom=0)
    ax = {}
    ax['subcrtx_lh'] = fig.add_subplot(gs[1:3, 0])
    ax['crtx_lh_front'] = fig.add_subplot(gs[0:2, 1], projection='3d', facecolor='0.75')
    ax['crtx_lh_back'] = fig.add_subplot(gs[2:4, 1], projection='3d', facecolor='0.75')
    ax['crtx_rh_front'] = fig.add_subplot(gs[4:6, 1], projection='3d', facecolor='0.75')
    ax['crtx_rh_back'] = fig.add_subplot(gs[6:8, 1], projection='3d', facecolor='0.75')
    ax['subcrtx_rh'] = fig.add_subplot(gs[5:7, 0])
    ax['clr_bar'] = fig.add_subplot(gs[2:6, 2])
    return fig, ax


def spat_map_infr_vs_pred(y_gt,
                          y_infr,
                          N_LAT,
                          N_LON,
                          unkown_roi_mask=None,
                          clim=None,
                          fig_dir=None,
                          fig_name=None,
                          dpi=200):
    fig = plt.figure(figsize=(10, 10))
    fig.set_facecolor('0.75')
    subfigs = fig.subfigures(1, 2)
    subfigs[0].suptitle('Ground Truth', fontsize='large')
    subfigs[1].suptitle('Inferred', fontsize='large')
    _, ax_gt = setup_spherical_spat_map_axs(fig=subfigs[0])
    _, ax_infr = setup_spherical_spat_map_axs(fig=subfigs[1])
    spherical_spat_map(
        y_gt,
        N_LAT=N_LAT,
        N_LON=N_LON,
        clim=clim,
        unkown_roi_mask=unkown_roi_mask,
        ax=ax_gt,
        dpi=dpi)
    spherical_spat_map(
        y_infr,
        N_LAT=N_LAT,
        N_LON=N_LON,
        clim=clim,
        unkown_roi_mask=unkown_roi_mask,
        ax=ax_infr,
        dpi=dpi)
    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
    if fig_name is not None:
        plt.savefig(f'{fig_dir}/{fig_name}', facecolor='white')
