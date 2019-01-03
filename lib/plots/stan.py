"""
Plots to look at Stan behavior or common requested, e.g. pair plots.

"""


def trace_nuts(csv, extras='', skip=0):
    from pylab import subplot, plot, gca, title,  grid, xticks
    if isinstance(extras, str):
        extras = extras.split()
    for csvi in csv:
        i = 1
        for key in csvi.keys():
            if key[-2:] == '__' or key in extras:
                subplot(4, 4, i)
                plot(csvi[key][skip:], alpha=0.5)
                if key in ('stepsize__', ):
                    gca().set_yscale('log')
                title(key)
                grid(1)
                if ((i - 1)/4) < 4:
                    xticks(xticks()[0], [])
                i += 1


# def pair_plots(csv, keys, skip=0):
#     import pylab as pl
#     n = len(keys)
#     if isinstance(csv, dict):
#         csv = [csv]  # following assumes list of chains' results
#     for i, key_i in enumerate(keys):
#         for j, key_j in enumerate(keys):
#             pl.subplot(n, n, i*n+j+1)
#             for csvi in csv:
#                 if i==j:
#                     pl.hist(csvi[key_i][skip:], 20, log=True)
#                 else:
#                     pl.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
#             if i==0:
#                 pl.title(key_j)
#             if j==0:
#                 pl.ylabel(key_i)
#     pl.tight_layout()

    
def nuts_diagnostics(data, figsize, figname=''):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.subplot(3, 3, 1)
    plt.plot(data['lp__'])
    plt.xlabel('Iteration')
    plt.ylabel('log prob.')
    plt.subplot(3, 3, 2)
    plt.plot(data['energy__'])
    plt.xlabel('Iteration')
    plt.ylabel('energy')
    plt.subplot(3, 3, 3)
    plt.plot(data['accept_stat__'])
    plt.xlabel('Iteration')
    plt.ylabel('accept stat.')
    plt.subplot(3, 3, 4)
    plt.plot(data['stepsize__'])
    plt.xlabel('Iteration')
    plt.ylabel('step size')
    plt.subplot(3, 3, 5)
    plt.plot(data['treedepth__'])
    plt.xlabel('Iteration')
    plt.ylabel('tree depth')
    plt.subplot(3, 3, 6)
    plt.plot(data['n_leapfrog__'])
    plt.xlabel('Iteration')
    plt.ylabel('n_leapfrog')
    plt.subplot(3, 3, 7)
    plt.plot(data['divergent__'])
    plt.xlabel('Iteration')
    plt.ylabel('divergent')
    plt.tight_layout()
    if (figname):
        plt.savefig(figname)


def x0_violin_syn(x0_infer, x0_true, ez, pz, figsize, figname='', legend_loc='upper right'):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    
    nn = x0_true.shape[0]
    ez_pz = np.concatenate((ez, pz))
    non_ez_pz = np.setdiff1d(np.arange(0, nn), ez_pz)

    plt.figure(figsize=figsize)
    plt.violinplot(x0_infer, showmeans=True, points=1000);
    plt.axhline(-2.0, color='green', alpha=0.3)
    xtick_labels = []

    for i in range(nn):
        if(i%2 == 0):
            xtick_labels.append(str(i+1))
        else:
            xtick_labels.append('')
    plt.xticks(np.r_[1:nn+1],xtick_labels);
    plt.xlabel('Region#',fontsize=15);
    plt.ylabel('$x_0$',fontsize=15);

    plt.plot(ez+1, x0_true[ez], color='red', marker='*', markersize=5, linestyle='None')
    plt.plot(pz+1, x0_true[pz], color='orange', marker='*', markersize=5, linestyle='None')
    plt.plot(non_ez_pz + 1, x0_true[non_ez_pz], color='black', marker='*', markersize=5, \
             linestyle='None')
    legend_elements = [Line2D([0], [0], linewidth=3, color='red', marker='*',
                              linestyle='None', label='EZ'),
                       Line2D([0], [0], linewidth=3, color='orange', marker='*',
                              linestyle='None', label='PZ'),
                       Line2D([0], [0], linewidth=3, color='black', marker='*',
                              linestyle='None', label='Non seizing'),
                       Line2D([0], [0], linewidth=3, color='C0', alpha=0.8, label='Inferred')]
    plt.legend(handles=legend_elements, loc=legend_loc)
    if(figname):
        plt.savefig(figname)


def x0_violin_patient(x0_infer,
                      ez_hyp,
                      figsize,
                      figname='',
                      legend_loc='upper right',
                      plt_close=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    nn = x0_infer.shape[1]
    plt.figure(figsize=figsize)
    violins = plt.violinplot(
        x0_infer, showmeans=True, points=1000)
    for i, violin_i in enumerate(violins['bodies']):
        if (i + 1 in ez_hyp):
            violin_i.set_facecolor('red')
            violin_i.set_edgecolor('red')
            violin_i.set_alpha(0.8)
    violins['cmins'].set_color('black')
    violins['cmins'].set_alpha(0.3)
    violins['cmaxes'].set_color('black')
    violins['cmaxes'].set_alpha(0.3)
    violins['cbars'].set_color('black')
    violins['cbars'].set_alpha(0.3)
    violins['cmeans'].set_color('black')
    violins['cmeans'].set_alpha(0.3)
    # plt.axhline(-2.0, color='green', alpha=0.3)
    # xtick_labels = []
    # for i in range(nn):
    #     if(i%2 == 0):
    #         xtick_labels.append(str(i+1))
    #     else:
    #         xtick_labels.append('')
    plt.xticks(np.r_[1:nn + 1:2], np.r_[1:nn + 1:2])
    plt.xlabel(
        'Region#', fontsize=15)
    plt.ylabel(
        '$x_0$', fontsize=15)
    # plt.plot(ez_hyp, -2.0+np.zeros_like(ez_hyp), color='red', marker='*',
    #          markersize=5, linestyle='None')
    legend_elements = [
        Line2D(
            [0], [0],
            linewidth=4,
            color='red',
            alpha=0.8,
            label='EZ clinical hypothesis')
    ]
    # Line2D([0], [0], linewidth=3, color='C0', alpha=0.8, label='Inferred')]
    plt.legend(handles=legend_elements, loc=legend_loc, frameon=True)
    if (figname):
        plt.savefig(figname)
    if (plt_close):
        plt.close()


def pair_plots(samples, params, figname=''):
    import numpy as np
    import matplotlib.pyplot as plt
    div_iters = np.where(samples['divergent__'] == 1)[0]
    plt.figure(figsize=(23, 13))
    nParams = len(params)
    for i in range(nParams):
        for j in range(nParams):
            plt.subplot(nParams, nParams, i * nParams + (j + 1))
            if (i == j):
                plt.hist(samples[params[i]].flatten(), bins=50, color='black')
            else:
                xvals = np.mean(
                    samples[params[j]], axis=1) if len(
                        samples[params[j]].shape) > 1 else samples[params[j]]
                yvals = np.mean(
                    samples[params[i]], axis=1) if len(
                        samples[params[i]].shape) > 1 else samples[params[i]]
                for k in range(xvals.shape[0]):
                    if (k in div_iters):
                        plt.plot(xvals[k], yvals[k], 'ro', alpha=0.8)
                    else:
                        plt.plot(xvals[k], yvals[k], 'ko', alpha=0.1)
            if (i == 0):
                plt.title(params[j], fontsize=13)
            if (j == 0):
                plt.ylabel(params[i], fontsize=13, rotation=90)
    plt.tight_layout()
    if(figname):
        plt.savefig(figname)


def plot_source(x, z, ez=[], pz=[], figname=''):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=4, label='EZ'),
        Line2D([0], [0], color='orange', linewidth=4, label='PZ')
    ]
    nn = x.shape[1]
    for i in range(nn):
        if (i in ez):
            plt.plot(x[:, i], color='red')
        elif (i in pz):
            plt.plot(x[:, i], color='orange')
        else:
            plt.plot(x[:, i], color='black')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('x', fontsize=15)
    plt.title(
        'Sample mean of posterior predicted source activity', fontsize=15)
    plt.legend(handles=legend_elements)
    plt.subplot(212)
    for i in range(nn):
        if (i in ez):
            plt.plot(z[:, i], color='red')
        elif (i in pz):
            plt.plot(z[:, i], color='orange')
        else:
            plt.plot(z[:, i], color='black')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('z', fontsize=15)
    plt.title(
        'Sample mean of posterior predicted source activity', fontsize=15)
    plt.legend(handles=legend_elements)
    if (figname):
        plt.savefig(figname)


def plot_fit_target(data_pred, data_true, figname=''):
    import matplotlib.pyplot as plt
    import numpy as np
    ns = data_pred['snsr_pwr'].shape[0]
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(data_pred['slp'], color='red', alpha=0.3)
    plt.plot(data_true['slp'], color='black', alpha=0.3)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('SLP', fontsize=15)
    plt.title('Sample mean of posterior predicted SLP', fontsize=15)
    plt.subplot(212)
    plt.bar(
        np.arange(1, ns + 1), data_pred['snsr_pwr'], color='red', alpha=0.3)
    plt.bar(
        np.arange(1, ns + 1), data_true['snsr_pwr'], color='black', alpha=0.3)
    plt.xlabel('Electrode', fontsize=15)
    plt.ylabel('Power', fontsize=15)
    plt.xticks(np.r_[1:ns + 1:3], np.r_[1:ns + 1:3])
    if (figname):
        plt.savefig(figname)
