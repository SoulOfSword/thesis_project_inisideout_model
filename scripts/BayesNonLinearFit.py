import sys
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy import stats
from multiprocessing import Pool, cpu_count
import time

###############################################################################
#
# BayesNonLinearFit
#
# Inspired by BayesLineFit (Desmond & Lelli, 2018-2020)
# Extended for non-linear models by Andrea Costa, 2026
#
# This software performs MCMC-based non-linear fitting to data with errors
# in both coordinates and non-negligible intrinsic scatter.
#
###############################################################################
#
# PURPOSE:
#       MCMC-based non-linear fit to data with errors in both coordinates
#       and non-negligible intrinsic scatter.
#
# EXPLANATION:
#       Fit a non-linear model y = f(x; params) to a set of points (xi, yi)
#       with errors (err_xi, err_yi) and non-negligible intrinsic scatter (s).
#       The errors are assumed to be Gaussian-distributed and independent.
#       The intrinsic scatter is assumed to be Gaussian in the vertical direction.
#
#       For non-linear models, orthogonal scatter is more complex to compute
#       (requires finding the closest point on the curve numerically), so
#       vertical scatter is the default approach.
#
# DEPENDENCIES:
#       Python packages numpy, emcee, corner, matplotlib, scipy, multiprocessing.
#
###############################################################################


def BayesNonLinearFit(x, y, model_func, param_names, param_init,
                      err_x=None, err_y=None,
                      param_bounds=None,
                      nwalkers=50, max_iters=10000,
                      outfile_chain=None, outfile_bestfit=None,
                      outplot_convergence=None, outplot_corner=None,
                      outplot_bestfit=None,
                      plotpdf=True, quiet=False,
                      plot_xlabel="x", plot_ylabel="y", plot_title=None,
                      scatter_type="vertical"):
    """
    Performs a Bayesian fit of a non-linear model to data including intrinsic scatter.

    The model accounts for:
    - Measurement errors in both x and y coordinates
    - Intrinsic scatter (vertical or approximate orthogonal)

    Args:
        x (1D float array): Data on x-axis
        y (1D float array): Data on y-axis
        model_func (callable): Model function with signature f(x, *params) -> y
                               Must be vectorized to accept array x
        param_names (list of str): Names of the model parameters (for plotting)
        param_init (1D float array): Initial guesses for model parameters
        err_x (float or 1D float array, optional): Uncertainty on x-coordinate; default=0
        err_y (float or 1D float array, optional): Uncertainty on y-coordinate; default=0
        param_bounds (list of tuples, optional): [(min, max), ...] bounds for each parameter
                                                  Use (None, None) for unbounded
        nwalkers (int, optional): Number of emcee walkers (default: 50)
        max_iters (int, optional): Maximum number of iterations in MCMC (default: 10000)
        outfile_chain (str, optional): File to store output chain; None to suppress
        outfile_bestfit (str, optional): File to store best-fit parameters; None to suppress
        outplot_convergence (str, optional): Name of convergence plot; None to suppress
        outplot_corner (str, optional): Name of corner plot; None to suppress
        outplot_bestfit (str, optional): Name of best-fit plot; None to suppress
        plotpdf (bool, optional): Create plots in pdf (True) or png (False) format
        quiet (bool, optional): Suppress output to screen
        plot_xlabel (str, optional): Label for x-axis in plots
        plot_ylabel (str, optional): Label for y-axis in plots
        plot_title (str, optional): Title for best-fit plot
        scatter_type (str, optional): "vertical" or "approximate_orthogonal"

    Returns:
        params_result (dict): Dictionary with parameter results, each containing:
                              [ML_value, median, upper_error, lower_error]
        scatter_result (1D float array): [ML, median, upper_error, lower_error] for intrinsic scatter
        sobs (float): Maximum-likelihood rms observed scatter
        samples (2D array): MCMC samples for further analysis
    """

    # Validate inputs
    x = np.asarray(x)
    y = np.asarray(y)
    n_data = len(x)
    n_params = len(param_init)

    if len(y) != n_data:
        raise ValueError("x and y must have the same length")
    if len(param_names) != n_params:
        raise ValueError("param_names must match the number of parameters in param_init")

    # Handle errors
    if err_x is None:
        err_x = np.abs(x) / 1.e10  # Small default errors
    if err_y is None:
        err_y = np.abs(y) / 1.e10

    err_x = np.atleast_1d(err_x)
    err_y = np.atleast_1d(err_y)

    if len(err_x) == 1:
        err_x = np.full(n_data, err_x[0])
    if len(err_y) == 1:
        err_y = np.full(n_data, err_y[0])

    # Validate errors
    if np.any(err_x < 0):
        raise ValueError("x errors cannot be negative")
    if np.any(err_y < 0):
        raise ValueError("y errors cannot be negative")

    # Set up parameter bounds
    if param_bounds is None:
        param_bounds = [(None, None)] * n_params

    # Convert bounds to arrays for easier handling
    param_mins = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
    param_maxs = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])

    if not quiet:
        print("Number of data points:", n_data)
        print("Number of model parameters:", n_params)
        print("Parameter names:", param_names)
        print("-------------------------")

    ##### CORRELATION TESTS ######
    Pearson = stats.pearsonr(x, y)
    Spearman = stats.spearmanr(x, y)
    Kendall = stats.kendalltau(x, y)

    if not quiet:
        print("PEARSON'S TEST")
        print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Pearson[0]), float('%.4g'%Pearson[1])))
        print("-------------------------")
        print("SPEARMAN'S TEST")
        print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Spearman[0]), float('%.4g'%Spearman[1])))
        print("-------------------------")
        print("KENDALL'S TEST")
        print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Kendall[0]), float('%.4g'%Kendall[1])))
        print("-------------------------")

    ###### INITIAL CURVE FIT #######
    try:
        # Use scipy curve_fit for initial estimate
        popt, pcov = curve_fit(model_func, x, y, p0=param_init,
                               sigma=err_y, absolute_sigma=True,
                               maxfev=10000)
        perr = np.sqrt(np.diag(pcov))

        # Calculate initial residuals and scatter estimate
        y_pred = model_func(x, *popt)
        res_init = y - y_pred
        rms_init = np.sqrt(np.mean(res_init**2))
        s_init = np.sqrt(max(0, rms_init**2 - np.mean(err_y)**2))

        if not quiet:
            print("Initial curve_fit successful")
            print("Initial parameters:", dict(zip(param_names, popt)))
            print("Initial rms:", float('%.4g'%rms_init))
            print("-------------------------")

    except Exception as e:
        if not quiet:
            print(f"Warning: Initial curve_fit failed ({e}), using provided initial values")
        popt = np.array(param_init)
        perr = np.abs(popt) * 0.1 + 0.1  # 10% of value + small constant
        y_pred = model_func(x, *popt)
        res_init = y - y_pred
        rms_init = np.sqrt(np.mean(res_init**2))
        s_init = rms_init / 2

    # Update bounds if not specified (use wide range around initial fit)
    for i in range(n_params):
        if param_bounds[i][0] is None:
            param_mins[i] = popt[i] - 50 * perr[i]
        if param_bounds[i][1] is None:
            param_maxs[i] = popt[i] + 50 * perr[i]

    ### BAYESIAN FIT ####

    # Total number of dimensions: model params + intrinsic scatter
    ndim = n_params + 1

    # Initialize walkers
    p0 = []
    for i in range(nwalkers):
        pi = []
        for j in range(n_params):
            # Sample from normal distribution, clipped to bounds
            val = np.random.normal(popt[j], perr[j])
            val = np.clip(val, param_mins[j], param_maxs[j])
            pi.append(val)
        # Add intrinsic scatter initialization
        pi.append(np.random.uniform(s_init/10, s_init * 2))
        p0.append(pi)

    p0 = np.array(p0)

    # MCMC sampling
    index = 0
    autocorr = np.empty(max_iters)
    old_tau = np.inf

    if not quiet:
        print("Running MCMC with", cpu_count(), "cores. Please wait...")

    start = time.time()

    # Select likelihood function based on scatter type
    if scatter_type == "vertical":
        lnprob_func = lnprob_vertical_nonlinear
    elif scatter_type == "approximate_orthogonal":
        lnprob_func = lnprob_approx_orthogonal_nonlinear
    else:
        raise ValueError("scatter_type must be 'vertical' or 'approximate_orthogonal'")

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob_func,
            args=[x, err_x, y, err_y, model_func, param_mins, param_maxs, n_params],
            pool=pool
        )

        for sample in sampler.sample(p0, iterations=max_iters, progress=False):
            if sampler.iteration == max_iters - 1:
                print("Warning: The sampler did not converge. Consider increasing max_iters or adjusting bounds.")
                break

            if sampler.iteration % 100:
                continue

            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
            except:
                continue

    end = time.time()

    if not quiet:
        print("Finished after", sampler.iteration, "iterations in", round(end-start), "seconds")
        print("Mean acceptance fraction:", round(np.mean(sampler.acceptance_fraction), 3))
        print("-------------------------")

    # Process chains
    try:
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
    except:
        # If autocorrelation fails, use conservative values
        burnin = sampler.iteration // 3
        thin = 1

    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=False, thin=thin)
    log_prob_samples_flat = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

    # Filter out bad samples
    mask = log_prob_samples_flat > -1.e300
    samples = samples[mask]
    log_prob_samples_flat = log_prob_samples_flat[mask]

    if len(samples) < 100:
        print("Warning: Very few valid samples. Results may be unreliable.")

    # Convergence plot
    if outplot_convergence is not None:
        fig = plt.figure(figsize=(10, 4), dpi=300)
        for i in range(nwalkers):
            y_arr = log_prob_samples[:, i]
            x_arr = np.arange(0, len(y_arr), 1)
            plt.plot(x_arr, y_arr, '.', alpha=0.3)
        plt.xlabel("Walker step")
        plt.ylabel("ln(Likelihood)")
        plt.title("MCMC Convergence")
        if plotpdf:
            plt.savefig(outplot_convergence + ".pdf", format='pdf', dpi=300)
        else:
            plt.savefig(outplot_convergence + ".png")
        plt.close()

    # Write chain file
    if outfile_chain is not None:
        all_samples = np.concatenate((samples, log_prob_samples_flat[mask[mask]][:len(samples), None]), axis=1)
        header = " ".join(param_names) + " sigma lnLike"
        np.savetxt(outfile_chain, all_samples, header=header)

    # Extract results
    ML = np.max(log_prob_samples_flat)
    index_ML = np.where(log_prob_samples_flat == ML)[0][0]

    params_result = {}
    all_labels = param_names + ["Intrinsic Scatter"]

    for i, name in enumerate(all_labels):
        val_ML = samples[index_ML, i]
        val_med = corner.quantile(samples[:, i], 0.5)
        val_dw = corner.quantile(samples[:, i], 0.16) - val_med
        val_up = corner.quantile(samples[:, i], 0.84) - val_med
        params_result[name] = [val_ML, val_med, val_up, val_dw]

    # Calculate observed scatter
    params_ML = samples[index_ML, :n_params]
    y_pred_ML = model_func(x, *params_ML)
    res_ML = y - y_pred_ML
    rms_ML = np.sqrt(np.mean(res_ML**2))

    # Print results
    if not quiet:
        print("Maximum likelihood (ML) value:", float('%.4g'%ML))
        for name in all_labels:
            vals = params_result[name]
            print(f"{name} (ML, median, +err, -err): {float('%.4g'%vals[0])}; {float('%.4g'%vals[1])}; +{float('%.4g'%vals[2])}, {float('%.4g'%vals[3])}")
        print("ML observed scatter (vertical):", float('%.4g'%rms_ML))
        print("*** NB medians and errors only meaningful for unimodal posteriors. Check corner plot! ***")
        print("-------------------------")

    # Write best-fit file
    if outfile_bestfit is not None:
        with open(outfile_bestfit, "w") as f:
            f.write(f"Pearson r: {Pearson[0]:.6f}; p-value: {Pearson[1]:.6f}\n")
            f.write(f"Spearman rho: {Spearman[0]:.6f}; p-value: {Spearman[1]:.6f}\n")
            f.write(f"Kendall tau: {Kendall[0]:.6f}; p-value: {Kendall[1]:.6f}\n")
            f.write(f"Maximum likelihood (ML) value: {float('%.4g'%ML)}\n")
            for name in all_labels:
                vals = params_result[name]
                f.write(f"{name} (ML, median, +err, -err): {float('%.4g'%vals[0])}; {float('%.4g'%vals[1])}; +{float('%.4g'%vals[2])}, {float('%.4g'%vals[3])}\n")
            f.write(f"ML rms observed scatter (vertical): {float('%.4g'%rms_ML)}\n")

    # Corner plot
    if outplot_corner is not None:
        figure = corner.corner(
            samples,
            levels=(1. - np.exp(-0.5), 1. - np.exp(-2.0)),
            labels=all_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            label_kwargs={"fontsize": 12},
            title_kwargs={"fontsize": 10}
        )

        # Mark median values
        med_values = [params_result[name][1] for name in all_labels]
        axes = np.array(figure.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(med_values[i], color="r")
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(med_values[xi], color="r")
                ax.axhline(med_values[yi], color="r")
                ax.plot(med_values[xi], med_values[yi], "sr")

        if plotpdf:
            figure.savefig(outplot_corner + ".pdf", format='pdf', dpi=300)
        else:
            figure.savefig(outplot_corner + ".png")
        plt.close()

    # Best-fit plot
    if outplot_bestfit is not None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        # Plot data with error bars
        ax.errorbar(x, y, xerr=err_x, yerr=err_y, fmt=' ',
                    ecolor='grey', capsize=3, alpha=0.3, zorder=0)
        ax.scatter(x, y, color="magenta", edgecolor='k', linewidth=1, s=30, zorder=1)

        # Plot best-fit curve
        x_plot = np.linspace(np.min(x), np.max(x), 200)
        params_med = [params_result[name][1] for name in param_names]
        y_plot = model_func(x_plot, *params_med)
        ax.plot(x_plot, y_plot, '-r', lw=2, zorder=3, label="Best fit (median)")

        # Plot scatter band
        sigma_med = params_result["Intrinsic Scatter"][1]
        ax.fill_between(x_plot, y_plot - sigma_med, y_plot + sigma_med,
                        color='k', alpha=0.2, zorder=2, label=f"σ = {sigma_med:.3f}")

        ax.set_xlabel(plot_xlabel)
        ax.set_ylabel(plot_ylabel)
        if plot_title:
            ax.set_title(plot_title)
        ax.legend()
        ax.grid(alpha=0.3, zorder=0)

        plt.tight_layout()
        if plotpdf:
            fig.savefig(outplot_bestfit + ".pdf", format='pdf', dpi=300)
        else:
            fig.savefig(outplot_bestfit + ".png")
        plt.close()

    # Prepare return values
    scatter_result = params_result["Intrinsic Scatter"]

    return params_result, scatter_result, rms_ML, samples


def lnprob_vertical_nonlinear(theta, x_arr, err_x_arr, y_arr, err_y_arr,
                               model_func, param_mins, param_maxs, n_params):
    """
    Log-likelihood function for non-linear model with vertical scatter.

    Accounts for errors in both x and y using error propagation.
    The x-error contribution is approximated via the local derivative df/dx.
    """
    params = theta[:n_params]
    sigma = theta[n_params]  # Intrinsic scatter

    # Check bounds
    if sigma < 0:
        return -1.e300
    for i in range(n_params):
        if params[i] < param_mins[i] or params[i] > param_maxs[i]:
            return -1.e300

    try:
        # Model prediction
        y_model = model_func(x_arr, *params)

        # Numerical derivative for error propagation
        dx = np.maximum(np.abs(x_arr) * 1e-6, 1e-10)
        y_plus = model_func(x_arr + dx, *params)
        y_minus = model_func(x_arr - dx, *params)
        dydx = (y_plus - y_minus) / (2 * dx)

        # Residuals (vertical)
        residuals = y_arr - y_model

        # Total variance: y-error + propagated x-error + intrinsic scatter
        variance = err_y_arr**2 + (dydx * err_x_arr)**2 + sigma**2

        # Log-likelihood (Gaussian)
        chi_sq = residuals**2 / variance
        log_L = -0.5 * np.sum(chi_sq + np.log(2 * np.pi * variance))

        if not np.isfinite(log_L):
            return -1.e300

        return log_L

    except Exception:
        return -1.e300


def lnprob_approx_orthogonal_nonlinear(theta, x_arr, err_x_arr, y_arr, err_y_arr,
                                        model_func, param_mins, param_maxs, n_params):
    """
    Log-likelihood function for non-linear model with approximate orthogonal scatter.

    Uses a first-order approximation for the orthogonal distance to the curve.
    This is an approximation that works well when the curve is not too curved locally.
    """
    params = theta[:n_params]
    sigma = theta[n_params]  # Intrinsic scatter

    # Check bounds
    if sigma < 0:
        return -1.e300
    for i in range(n_params):
        if params[i] < param_mins[i] or params[i] > param_maxs[i]:
            return -1.e300

    try:
        # Model prediction
        y_model = model_func(x_arr, *params)

        # Numerical derivative for orthogonal projection
        dx = np.maximum(np.abs(x_arr) * 1e-6, 1e-10)
        y_plus = model_func(x_arr + dx, *params)
        y_minus = model_func(x_arr - dx, *params)
        dydx = (y_plus - y_minus) / (2 * dx)

        # Vertical distance
        delta_y = y_arr - y_model

        # Approximate orthogonal distance: d_orth ≈ |delta_y| / sqrt(1 + (dy/dx)^2)
        # For a tangent line y = y0 + m*(x - x0), orthogonal distance is |y - y0 - m*(x-x0)| / sqrt(1 + m^2)
        orthogonal_dist_sq = delta_y**2 / (1 + dydx**2)

        # Error propagation for orthogonal direction
        # Approximate: project errors onto the normal direction
        cos_alpha_sq = 1 / (1 + dydx**2)  # cos^2 of angle with vertical
        sin_alpha_sq = dydx**2 / (1 + dydx**2)

        err_orthogonal_sq = cos_alpha_sq * err_y_arr**2 + sin_alpha_sq * err_x_arr**2

        # Total variance in orthogonal direction
        variance = err_orthogonal_sq + sigma**2

        # Log-likelihood
        chi_sq = orthogonal_dist_sq / variance
        log_L = -0.5 * np.sum(chi_sq + np.log(2 * np.pi * variance))

        if not np.isfinite(log_L):
            return -1.e300

        return log_L

    except Exception:
        return -1.e300
