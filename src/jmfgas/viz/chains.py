"""Walker trace plots for MCMC chains."""

import matplotlib.pyplot as plt


def trace_plot(chain, labels, burn_in=0):
    """Walker traces per parameter with the burn-in cutoff marked.

    chain has shape (nsteps, nwalkers, ndim).
    """
    ndim = chain.shape[2]
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True, squeeze=False)
    for i in range(ndim):
        ax = axes[i, 0]
        ax.plot(chain[:, :, i], alpha=0.3, color="k")
        ax.axvline(burn_in, color="b", ls=":", label="burn-in")
        ax.set_ylabel(labels[i])
    axes[-1, 0].set_xlabel("step")
    axes[0, 0].legend(loc="best", fontsize=9)
    return fig
