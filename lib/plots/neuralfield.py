import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def create_video(x, N_LAT, N_LON, out_dir):
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
        # plt.colorbar(fraction=0.02, fontsize=5)
        plt.subplot(122)
        plt.imshow(x_rh, interpolation=None)
        plt.clim(clim_min, clim_max)
        plt.title("Right hemisphere")
        plt.xlabel("Longitude", fontsize=fontsize)
        plt.ylabel("Latitude", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # plt.colorbar(fraction=0.02)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/x_{i+1:06d}.jpg")
        plt.close()
    (
        ffmpeg
        .input("tmp/x_%06d.jpg", framerate=25)
        .filter_('scale', size='hd1080', force_original_aspect_ratio='increase')
        .output('tmp/movie.mp4')
        .run()
    )
