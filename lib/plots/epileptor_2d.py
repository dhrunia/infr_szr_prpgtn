import matplotlib.pyplot as plt
import lib.utils.consts as consts
import os
import numpy as np


def plot_ts_imshow(y,
                   save_dir=None,
                   fig_name=None,
                   ax=None,
                   figsize=(7, 6),
                   clim=None,
                   **kwargs):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=120, layout='tight')
    if clim is None:
        clim = {'min': np.min(y), 'max': np.max(y)}
    fig = ax.get_figure()
    im = ax.imshow(y.T, interpolation=None, aspect='auto', cmap='inferno')
    im.set_clim(clim['min'], clim['max'])
    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize=consts.FS_LARGE)
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'], fontsize=consts.FS_LARGE)
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'], fontsize=consts.FS_LARGE)
    ax.tick_params(labelsize=consts.FS_MED)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, fraction=0.1)
    cbar.ax.tick_params(labelsize=consts.FS_MED)
    if fig_name is not None:
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{fig_name}', facecolor='white')


def plot_slp(slp,
             save_dir=None,
             fig_name=None,
             ax=None,
             figsize=(7, 6),
             clim=None,
             **kwargs):
    plot_ts_imshow(slp,
                   save_dir=save_dir,
                   fig_name=fig_name,
                   ax=ax,
                   figsize=figsize,
                   clim=clim,
                   xlabel='Time',
                   ylabel='Sensor',
                   **kwargs)


def plot_src(x,
             save_dir=None,
             fig_name=None,
             ax=None,
             figsize=(7, 6),
             clim=None,
             **kwargs):
    plot_ts_imshow(x,
                   save_dir=save_dir,
                   fig_name=fig_name,
                   ax=ax,
                   figsize=figsize,
                   clim=clim,
                   xlabel='Time',
                   ylabel='Region',
                   **kwargs)
