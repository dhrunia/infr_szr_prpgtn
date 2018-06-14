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

def nuts_diagnostics(data):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20,10))
    plt.subplot(4,2,1)
    plt.plot(data['lp__'])
    plt.xlabel('Iteration')
    plt.ylabel('log prob.')

    plt.subplot(4,2,2)
    plt.plot(data['energy__'])
    plt.xlabel('Iteration')
    plt.ylabel('energy')

    plt.subplot(4,2,3)
    plt.plot(data['accept_stat__'])
    plt.xlabel('Iteration')
    plt.ylabel('accept stat.')

    plt.subplot(4,2,4)
    plt.plot(data['stepsize__'])
    plt.xlabel('Iteration')
    plt.ylabel('step size')

    plt.subplot(4,2,5)
    plt.plot(data['treedepth__'])
    plt.xlabel('Iteration')
    plt.ylabel('tree depth')

    plt.subplot(4,2,6)
    plt.plot(data['n_leapfrog__'])
    plt.xlabel('Iteration')
    plt.ylabel('n_leapfrog')

    plt.subplot(4,2,7)
    plt.plot(data['divergent__'])
    plt.xlabel('Iteration')
    plt.ylabel('divergent')

    plt.tight_layout();  
