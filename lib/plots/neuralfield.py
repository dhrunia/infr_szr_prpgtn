import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def create_video(x, N_LAT, N_LON, out_dir, fig_name):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(f'{out_dir}/*')
    for f in files:
        os.remove(f)
    nvph = N_LAT * N_LON
    clim_min = np.min(x)
    clim_max = np.max(x)

    for i in range(x.shape[0]):
        plt.figure(dpi=200, figsize=(7, 3))
        x_lh = np.reshape(x[i, 0:nvph], (N_LAT, N_LON))
        x_rh = np.reshape(x[i, nvph:], (N_LAT, N_LON))
        fontsize = 7
        plt.subplot(121)
        plt.imshow(x_lh, interpolation=None)
        plt.clim(clim_min, clim_max)
        plt.title("Left hemisphere")
        plt.xlabel("Longitude", fontsize=fontsize)
        plt.ylabel("Latitude", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.colorbar(fraction=0.02)
        plt.subplot(122)
        plt.imshow(x_rh, interpolation=None)
        plt.clim(clim_min, clim_max)
        plt.title("Right hemisphere")
        plt.xlabel("Longitude", fontsize=fontsize)
        plt.ylabel("Latitude", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.colorbar(fraction=0.02)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/x_{i+1:06d}.jpg")
        plt.close()
    (ffmpeg.input(f"{out_dir}/x_%06d.jpg", framerate=25).filter_(
        'scale', size='hd1080', force_original_aspect_ratio='increase').output(
            f'{out_dir}/{fig_name}').run())


def x0_gt_vs_infer(x0_gt,
                   x0_infer_mean,
                   x0_infer_std,
                   N_LAT,
                   N_LON,
                   unkown_roi_mask,
                   fig_dir,
                   fig_name=None):
    fs_small = 5
    fs_med = 7
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
    plt.imshow(x0_gt_lh, interpolation=None)
    plt.clim(clim_min, clim_max)
    plt.title("Ground Truth - Left hemisphere", fontsize=fs_small)
    plt.xlabel("Longitude", fontsize=fs_med)
    plt.ylabel("Latitude", fontsize=fs_med)
    plt.xticks(fontsize=fs_med)
    plt.yticks(fontsize=fs_med)
    plt.colorbar(fraction=0.02)
    plt.subplot(322)
    plt.imshow(x0_gt_rh, interpolation=None)
    plt.clim(clim_min, clim_max)
    plt.title("Ground Truh - Right hemisphere", fontsize=fs_small)
    plt.xlabel("Longitude", fontsize=fs_med)
    plt.ylabel("Latitude", fontsize=fs_med)
    plt.xticks(fontsize=fs_med)
    plt.yticks(fontsize=fs_med)
    plt.colorbar(fraction=0.02)

    plt.subplot(323)
    plt.imshow(x0_infer_mean_lh, interpolation=None)
    plt.clim(clim_min, clim_max)
    plt.title("Inferred Mean - Left hemisphere", fontsize=fs_small)
    plt.xlabel("Longitude", fontsize=fs_med)
    plt.ylabel("Latitude", fontsize=fs_med)
    plt.xticks(fontsize=fs_med)
    plt.yticks(fontsize=fs_med)
    plt.colorbar(fraction=0.02)
    plt.subplot(324)
    plt.imshow(x0_infer_mean_rh, interpolation=None)
    plt.clim(clim_min, clim_max)
    plt.title("Inferred Mean - Right hemisphere", fontsize=fs_small)
    plt.xlabel("Longitude", fontsize=fs_med)
    plt.ylabel("Latitude", fontsize=fs_med)
    plt.xticks(fontsize=fs_med)
    plt.yticks(fontsize=fs_med)
    plt.colorbar(fraction=0.02)
    plt.tight_layout()

    plt.subplot(325)
    plt.imshow(x0_infer_std_lh, interpolation=None)
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Left hemisphere", fontsize=fs_small)
    plt.xlabel("Longitude", fontsize=fs_med)
    plt.ylabel("Latitude", fontsize=fs_med)
    plt.xticks(fontsize=fs_med)
    plt.yticks(fontsize=fs_med)
    plt.colorbar(fraction=0.02)
    plt.subplot(326)
    plt.imshow(x0_infer_std_rh, interpolation=None)
    # plt.clim(clim_min, clim_max)
    plt.title("Standard Deviation - Right hemisphere", fontsize=fs_small)
    plt.xlabel("Longitude", fontsize=fs_med)
    plt.ylabel("Latitude", fontsize=fs_med)
    plt.xticks(fontsize=fs_med)
    plt.yticks(fontsize=fs_med)
    plt.colorbar(fraction=0.02)
    plt.tight_layout()
    if (fig_name is not None):
        plt.savefig(f'{fig_dir}/{fig_name}', facecolor='white')
