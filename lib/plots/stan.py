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


def pair_plots(csv, keys, skip=0):
    import pylab as pl
    n = len(keys)
    if isinstance(csv, dict):
        csv = [csv]  # following assumes list of chains' results
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            pl.subplot(n, n, i*n+j+1)
            for csvi in csv:
                if i==j:
                    pl.hist(csvi[key_i][skip:], 20, log=True)
                else:
                    pl.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
            if i==0:
                pl.title(key_j)
            if j==0:
                pl.ylabel(key_i)
    pl.tight_layout()

def nuts_diagnostics(data, figsize, figname):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    plt.subplot(3,3,1)
    plt.plot(data['lp__'])
    plt.xlabel('Iteration')
    plt.ylabel('log prob.')

    plt.subplot(3,3,2)
    plt.plot(data['energy__'])
    plt.xlabel('Iteration')
    plt.ylabel('energy')

    plt.subplot(3,3,3)
    plt.plot(data['accept_stat__'])
    plt.xlabel('Iteration')
    plt.ylabel('accept stat.')

    plt.subplot(3,3,4)
    plt.plot(data['stepsize__'])
    plt.xlabel('Iteration')
    plt.ylabel('step size')

    plt.subplot(3,3,5)
    plt.plot(data['treedepth__'])
    plt.xlabel('Iteration')
    plt.ylabel('tree depth')

    plt.subplot(3,3,6)
    plt.plot(data['n_leapfrog__'])
    plt.xlabel('Iteration')
    plt.ylabel('n_leapfrog')

    plt.subplot(3,3,7)
    plt.plot(data['divergent__'])
    plt.xlabel('Iteration')
    plt.ylabel('divergent')

    plt.tight_layout() 

    if(figname):
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

def x0_violin_patient(x0_infer, ez_hyp, figsize, figname='', legend_loc='upper right'):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    
    nn = x0_infer.shape[1]

    plt.figure(figsize=figsize)
    violins = plt.violinplot(x0_infer, showmeans=True, points=1000);
    for i,violin_i in enumerate(violins['bodies']):
        if(i+1 in ez_hyp):
            violin_i.set_facecolor('red')
            violin_i.set_edgecolor('black')
            violin_i.set_alpha(0.3)
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
    plt.xticks(np.r_[1:nn+1:2],np.r_[1:nn+1:2]);
    plt.xlabel('Region#',fontsize=15);
    plt.ylabel('$x_0$',fontsize=15);

    # plt.plot(ez_hyp, -2.0+np.zeros_like(ez_hyp), color='red', marker='*',
    #          markersize=5, linestyle='None')
    legend_elements = [Line2D([0], [0], linewidth=4, color='red', alpha=0.3,
                              label='EZ clinical hypothesis')]
                       # Line2D([0], [0], linewidth=3, color='C0', alpha=0.8, label='Inferred')]
    plt.legend(handles=legend_elements, loc=legend_loc, frameon=True)
    if(figname):
        plt.savefig(figname)
    plt.close()
