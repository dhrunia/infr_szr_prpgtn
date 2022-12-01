import matplotlib.pyplot as plt
import numpy as np
import lib.utils.consts as consts
import os


def plot_x0(x0,
            roi_names=None,
            ax=None,
            save_dir=None,
            fig_name=None,
            dpi=500):
    num_roi = x0.shape[0]
    if ax is None:
        fig = plt.figure(figsize=(3, 7), dpi=dpi, layout='tight')
        ax = plt.subplot(111)
    ax.barh(np.r_[1:num_roi + 1], x0)
    if roi_names is not None:
        ax.set_yticks(np.r_[1:num_roi + 1], roi_names, fontsize=2)
    if fig_name is not None:
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{fig_name}')


def plot_ez_hyp(ez_hyp,
                roi_names=None,
                ax=None,
                save_dir=None,
                fig_name=None,
                dpi=500):
    plot_x0(x0=ez_hyp,
            roi_names=roi_names,
            ax=ax,
            save_dir=save_dir,
            fig_name=fig_name,
            dpi=dpi)


def plot_ez_hyp_vs_rsctn_vs_pred(ez_hyp,
                                 ez_rsctn,
                                 x0_pred,
                                 save_dir,
                                 fig_name,
                                 dpi=500):
    fig, axs = plt.subplots(nrows=1,
                            ncols=3,
                            dpi=500,
                            layout='tight',
                            figsize=(8, 8))
    plot_x0(x0=ez_hyp, ax=axs[0], dpi=dpi)
    plot_x0(x0=ez_rsctn, ax=axs[0], dpi=dpi)
    plot_x0(x0=x0_pred, ax=axs[0], dpi=dpi)
    if fig_name is not None:
        if (not os.path.exists(save_dir)):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f'{save_dir}/{fig_name}')


def plot_src(x,
             roi_names=None,
             ax=None,
             save_dir=None,
             fig_name=None,
             dpi=200,
             title=None):
    plot_slp(x,
             snsr_lbls=roi_names,
             ax=ax,
             save_dir=save_dir,
             fig_name=fig_name,
             dpi=dpi,
             title=title)


def plot_slp(slp,
             snsr_lbls=None,
             save_dir=None,
             fig_name=None,
             ax=None,
             title='SEEG log power',
             clim=None,
             dpi=200):

    nt, ns = slp.shape
    if ax is None:
        fig = plt.figure(figsize=(7, 6), dpi=dpi, layout='tight')
        ax = plt.subplot()
    if clim is None:
        clim = {'min': np.min(slp), 'max': np.max(slp)}
    fig = ax.get_figure()
    im = ax.imshow(slp.T, interpolation=None, aspect='auto', cmap='hot')
    im.set_clim(clim['min'], clim['max'])
    ax.set_title(title, fontsize=consts.FS_LARGE)
    ax.set_xlabel('Time', fontsize=consts.FS_LARGE)
    ax.set_ylabel('Sensor', fontsize=consts.FS_LARGE)
    ax.tick_params(labelsize=consts.FS_MED)
    if snsr_lbls is not None:
        ax.set_yticks(np.r_[0:ns], snsr_lbls, fontsize=2)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, fraction=0.1)
    cbar.ax.tick_params(labelsize=consts.FS_MED)
    if fig_name is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/{fig_name}', facecolor='white')