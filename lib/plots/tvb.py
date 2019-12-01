def plot_connectome(sc_mat):
    from matplotlib import colors, cm

    norm = colors.LogNorm(1e-7, sc_mat.weights.max())
    im = plt.imshow(sc_mat.weights, norm=norm, cmap=cm.jet)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_title('Strcutural Connectivity', fontsize=13.0)
