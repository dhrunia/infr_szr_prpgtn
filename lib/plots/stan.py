"""
Plots to look at Stan behavior or common requested, e.g. pair plots.

"""


def trace_nuts(csv, extras='', skip=0):
    from pylab import subplot, plot, gca, title,  grid
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