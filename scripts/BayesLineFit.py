import sys
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data, RealData
from scipy import stats
from multiprocessing import Pool, cpu_count
import time

# ###############################################################################
#
# Copyright (C) 2018-2020, Harry Desmond & Federico Lelli
# E-mail: harry.desmond@physics.ox.ac.uk, LelliF@cardiff.ac.uk
#
# Up-to-date versions of the software are available from:
# https://www2.physics.ox.ac.uk/contacts/people/desmond
# http://astroweb.cwru.edu/SPARC/
# https://www.lellifederico.com/
#
# If you have used this software in your research, please acknowledge
# Lelli et al. (2019, MNRAS, 484, 2367) where we describe in detail 
# the maximum-likelihood (ML) method to fit a linear model y=a*x+b
# considering either vertical or the orthogonal intrinsic scatter.
#
# This software is provided as it is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
#################################################################################
#+
# NAME:
#       BayesLineFit
#
# PURPOSE:
#       MCMC-based linear fit to data with errors in both coordinates
#       and non-negligible intrinsic scatter. See Appendix A here:
#       https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.3267L/abstract
#
# EXPLANATION:
#       Fit a linear model (y= a*x + b) to a set of points (xi, yi) with
#       errors (err_xi, err_yi) and non-negligible intrinsic scatter (s).
#       The errors are assumed to be Gaussian-distributed and independent.
#       The intrinsic scatter is assumed to be Gaussian either in the
#       vertical direction y or the direction orthogonal to the best-fit line.
#       The program uses an affine-invariant MCMC sampler to map out the full
#       posteriors of slope, intercept and intrinsic scatter, plots the results
#       and prints summary statistics.
#
# DEPENDENCIES:
#       Python packages numpy, emcee, corner, matplotlib, scipy, multiprocessing. 
#       Tested on Python 3.7.5.
#
# OUTPUTS:
#       Several diagnostics of the data and linear fit are printed to screen.
#       Asymmetric errors are estimated at the 68% confidence level of the 
#       marginalized 1-D posterior distributions of the fitting parameters. 
#       The following files are produced:
#       outfile1: data from MCMC chains
#       outfile2: best-fit values and summary statistics
#       outplot1: walker convergence plot
#       outplot2: corner plot of parameter constraints
#       outplot3: plot of best-fit line and uncertainty overlaid on the data
#
# MODIFICATION HISTORY:
#       Harry Desmond, Oxford, June 2018: main functions and MCMC implementation
#       Federico Lelli, Cardiff, February 2020: ODR and priors implementation
#       Harry Desmond, East Knoyle, April 2020: I/O implementations & py-module
#       Federico Lelli, Cardiff, June 2020: change severe errors with warnings
#       Harry Desmond, East Knoyle, July 2020: fixed likelihood plot bug
#-
#-------------------------------------------------------------------------------

###########################
### MODIFY IF NECESSARY ###
###########################

# Output file names
outfile1 = 'outchain.dat'
outfile2 = 'bestfitvalues.dat'
outplot1 = 'convergence.pdf'
outplot2 = 'cornerplot.pdf'
outplot3 = 'bestfitplot.pdf'

while True:
    infile = input("Name of data file (format: x, y, err_x, err_y; default=data.dat): ")
    if infile=="q" or infile=="quit" or infile=="exit":
        quit()
    try:
        if infile=="":
            x, y, err_x, err_y = np.loadtxt("data.dat", unpack=True)
        else:
            x, y, err_x, err_y = np.loadtxt(infile, unpack=True)
            # Modify read patten here if format is not x, y, err_x, err_y
            # e.g. x, err_x, y, err_y = np.loadtxt(infile, unpack=True, skiprows=1, usecols=[0,2,1,3])
        break
    except IOError:
        print("That file does not exist. Please try again.")

if any(t < 0. for t in err_x):
    print("At least one of your x errors is negative.")
    quit()
if any(t < 0. for t in err_y):
    print("At least one of your y errors is negative.")
    quit()
    
if np.sum(err_x)==0.:
    print("x errors are 0. Softening slightly...")
    err_x = np.abs(x)/1.e10
if np.sum(err_y)==0.:
    print("y errors are 0. Softening slightly...")
    err_y = np.abs(y)/1.e10

while True:
    fit_type = input("Orthogonal or vertical fit (orth/vert; default=orth): ")
    if fit_type=="q" or fit_type=="quit" or fit_type=="exit":
        quit()
    if fit_type=="orth" or fit_type=="":
        orthfit = True
        break
    elif fit_type=="vert":
        orthfit = False
        break
    else:
        print("Unrecognised choice. Please try again.")
        
advanced = input("Type y for advanced options ")

if advanced=="y":
    while True:
        nwalkers = input("Number of walkers (default=50): ")
        if nwalkers=="q" or nwalkers=="quit" or nwalkers=="exit":
            quit()
        if nwalkers=="":
            nwalkers=50
            break
        try:
            nwalkers = int(nwalkers)
            break
        except ValueError:
            print("Please enter an integer.")
            
    while True:
        max_iters = input("Maximum number of iterations (default=10000): ")
        if max_iters=="q" or max_iters=="quit" or max_iters=="exit":
            quit()
        if max_iters=="":
            max_iters=10000
            break
        try:
            max_iters = int(max_iters)
            break
        except ValueError:
            print("Please enter an integer.")
            
    while True:
        arr = input("Lower and upper bounds for slope (space-separated; default=broad range): ")
        if arr=="q" or arr=="quit" or arr=="exit":
            quit()
        if arr=="":
            flag_ODR_slope = True
            break
        try:
            arr = [float(i) for i in arr.split()]
        except ValueError:
            print("Need two numbers.")
            continue
        if len(arr) == 2:
            slope_min, slope_max = arr[0], arr[1]
            flag_ODR_slope = False
            break
        else:
            print("Need two numbers.")
        
    while True:
        arr = input("Lower and upper bounds for intercept (space-separated; default=broad range): ")
        if arr=="q" or arr=="quit" or arr=="exit":
            quit()
        if arr=="":
            flag_ODR_int = True
            break
        try:
            arr = [float(i) for i in arr.split()]
        except ValueError:
            print("Need two numbers.")
            continue
        if len(arr) == 2:
            int_min, int_max = arr[0], arr[1]
            flag_ODR_int = False
            break
        else:
            print("Need two numbers.")
    
    print("*** For further options edit source code BayesLineFit.py ***") 
    
else:
    nwalkers, max_iters = 50, 10000
    flag_ODR_slope, flag_ODR_int = True, True

print("")

#############################
### MODIFY AT YOUR PERIL! ###
#############################

##### DEFINE FUNCTIONS ######

#Define model to fit
def func(beta, x):
	y = beta[0]*x + beta[1]
	return y

# Likelihood for vertical fit
def lnprob_vertical(x, x_arr, err_x_arr, y_arr, err_y_arr):
  # Variables (sigma=intrinsic scatter)
  slope, intercept, sigma = x[0], x[1], x[2]
  if sigma < 0. or slope < slope_min or slope > slope_max or intercept < int_min or intercept > int_max:
      return -1.e300
  # Expected M for given V
  mean = intercept + slope * np.array(x_arr)  
  # Difference between point and line in vertical direction
  dist = np.array(y_arr) - mean
  # Sum up in quadrature contributions to scatter in vertical direction
  scatter = np.sqrt(np.array(err_y_arr)*np.array(err_y_arr) + slope*slope*np.array(err_x_arr)*np.array(err_y_arr) + sigma*sigma)
  chi_sq = dist*dist / (scatter*scatter)  
  # Define Gaussian likelihood
  L = np.exp(-chi_sq/2.) / (np.sqrt(2.*np.pi) * scatter)
  if np.min(L) < 1.e-300:
    return -1.e300
  return np.sum(np.log(L))

# Likelihood for orthogonal fit
def lnprob_orthogonal(x, x_arr, err_x_arr, y_arr, err_y_arr):  
  # Variables (sigma=intrinsic scatter)
  slope, intercept, sigma = x[0], x[1], x[2]
  if sigma < 0. or slope < slope_min or slope > slope_max or intercept < int_min or intercept > int_max:
    return -1.e300
  # Expected y for given x
  mean = intercept + slope * np.array(x_arr)
  # Difference between point and line in vertical direction
  delta_y = y_arr - mean 
  # Shortest squared distance between point and line (in orthogonal direction)
  dist2 = (y_arr - slope*x_arr - intercept)**2./(slope**2 + 1.)
  # Sum up in quadrature contributions to scatter in orthogonal direction
  scatter2 = sigma*sigma + dist2/(delta_y*delta_y)*np.array(err_y_arr)*np.array(err_y_arr) + (1. - dist2/(delta_y*delta_y))*np.array(err_x_arr)*np.array(err_x_arr)  
  chi_sq = dist2 / scatter2
  # Define Gaussian likelihood
  L = np.exp(-chi_sq/2.) / np.sqrt(2.*np.pi*scatter2)
  if np.min(L) < 1.e-300:
    return -1.e300
  return np.sum(np.log(L))

#----------------------------#

print("Number of data points:", len(x))

##### CORRELATION TESTS ######

Pearson = stats.pearsonr(x, y)
Spearman = stats.spearmanr(x, y)
Kendall = stats.kendalltau(x, y)
print("-------------------------")
print("PEARSON'S TEST")
print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Pearson[0]), float('%.4g'%Pearson[1])))
print("-------------------------")
print("SPEARMAN'S TEST")
print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Spearman[0]), float('%.4g'%Spearman[1])))
print("-------------------------")
print("KENDALL'S TEST")
print("Correlation coefficient: %s; p-value %s" % (float('%.4g'%Kendall[0]), float('%.4g'%Kendall[1])))
print("-------------------------")

###### INITIAL ODR FIT #######

model = Model(func)
data = RealData(x, y, err_x, err_y)
odr = ODR(data, model, [1,1])
odr.set_job(fit_type=0)
output = odr.run()
a_ODR = output.beta[0] #slope
b_ODR = output.beta[1] #normalization
err_a_ODR = output.sd_beta[0] #error on slope
err_b_ODR = output.sd_beta[1] #error on norm
res_ODR = (y - output.beta[0]*x - output.beta[1]) #residuals
rms_ODR = np.sqrt(np.mean(res_ODR**2., dtype=np.float64)) #observed scatter
s_ODR = np.sqrt(rms_ODR**2. - np.mean(err_y)**2.) #estimate of intrinsic scatter
if (np.isnan(a_ODR) or np.isnan(b_ODR)):
	sys.exit("-----------------------------------------------------------\n\
SEVERE ERROR: The INITIAL ODR FIT FAILED! CHECK YOUR INPUTS!\n\
--------------------------------------------------------------")
if (np.isnan(s_ODR)):
	print("-----------------------------------------------------------\n\
WARNING: OBSERVED VERTICAL SCATTER IS SMALLER THAN EXPECTED FROM Y-ERRORS. YOUR ERRORS MAY BE OVER-ESTIMATED!\n\
--------------------------------------------------------------")

if flag_ODR_slope:
    slope_min = a_ODR - 20.*err_a_ODR
    slope_max = a_ODR + 20.*err_a_ODR
if flag_ODR_int:
    int_min = b_ODR - 20.*err_b_ODR
    int_max = b_ODR + 20.*err_b_ODR

### BAYESIAN FIT ####

# Initialize walkers
ndim = 3
p0 = []
for i in range(nwalkers):
    pi = [np.random.normal(a_ODR, err_a_ODR), np.random.normal(b_ODR, err_b_ODR), np.random.uniform(rms_ODR/10., rms_ODR)]
    p0.append(pi)

index = 0
autocorr = np.empty(max_iters)
old_tau = np.inf

print("Running MCMC with", cpu_count(), "cores. Please wait...")

start = time.time()
with Pool() as pool:
    if orthfit:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_orthogonal, args=[x, err_x, y, err_y], pool=pool)        # Default sampler move
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_vertical, args=[x, err_x, y, err_y], pool=pool)

    for sample in sampler.sample(p0, iterations=max_iters, progress=False):
        if sampler.iteration==max_iters-1:
            print("The sampler did not converge. Probably either your data is pathological or a linear fit is bad or unconstrained. If you think this is not the case, try increasing the maximum number of iterations, imposing bounds on the parameters, or using a different move method in the sampling.")
            quit()
        
        if sampler.iteration % 100:
            continue

        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau    

end = time.time()
print("Converged after", sampler.iteration, "iterations in", round(end-start), "seconds")

print("Mean acceptance fraction:", round(np.mean(sampler.acceptance_fraction),3))
print("-------------------------")

# Calculate autocorrelation and make convergence plots
tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=False, thin=thin)
log_prob_samples_flat = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

mask = log_prob_samples_flat > -1.e300
samples = samples[mask]

fig = plt.figure(figsize=(10, 4), dpi=300)
for i in range(nwalkers):
	y_arr = log_prob_samples[:,i]
	x_arr = np.arange(0, len(y_arr), 1)
	plt.plot(x_arr, y_arr, '.')
plt.xlabel("Walker step")
plt.ylabel("ln(Likelihood)")
plt.savefig(outplot1, format='pdf', dpi=300)

# Write chain file
all_samples = np.concatenate((samples, log_prob_samples_flat[mask, None]), axis=1)
np.savetxt(outfile1, all_samples, header="slope intercept sigma lnLike")

# Maximum-likelihood values
ML = np.max(log_prob_samples_flat)
index = np.where(log_prob_samples_flat==ML)[0][0]
a_ML = samples[index,0]
b_ML = samples[index,1]
s_ML = samples[index,2]
# Posterior medians
a_med = corner.quantile(samples[:,0], 0.5)
b_med = corner.quantile(samples[:,1], 0.5)
s_med = corner.quantile(samples[:,2], 0.5)
# Errors (down/up)
a_dw = corner.quantile(samples[:,0], 0.16)-a_med
a_up = corner.quantile(samples[:,0], 0.84)-a_med
b_dw = corner.quantile(samples[:,1], 0.16)-b_med
b_up = corner.quantile(samples[:,1], 0.84)-b_med
s_dw = corner.quantile(samples[:,2], 0.16)-s_med
s_up = corner.quantile(samples[:,2], 0.84)-s_med
# Residuals and observed scatter
res_ML = (y - a_ML*x - b_ML)
rms_ML = np.sqrt(np.mean(res_ML**2., dtype=np.float64))
res_ML_orth = (y - a_ML*x - b_ML)/np.sqrt(a_ML*a_ML + 1.)
rms_ML_orth = np.sqrt(np.mean(res_ML_orth**2., dtype=np.float64))

# Write outputs
print("Maximum likelihood (ML) value: %s" % (float('%.4g'%ML)))
print("Slope (ML, median, upper error, lower error): %s; %s; +%s, %s" % (float('%.4g'%a_ML), float('%.4g'%a_med), float('%.4g'%a_up), float('%.4g'%a_dw)))
print("Intercept (ML, median, upper error, lower error): %s; %s; +%s; %s" % (float('%.4g'%b_ML), float('%.4g'%b_med), float('%.4g'%b_up), float('%.4g'%b_dw)))
print("Intrinsic scatter (ML, median, upper error, lower error): %s; %s; +%s; %s" % (float('%.4g'%s_ML), float('%.4g'%s_med), float('%.4g'%s_up), float('%.4g'%s_dw)))
print("ML rms observed scatter (vertical): %s" % (float('%.4g'%rms_ML)))
print("ML rms observed scatter (orthogonal): %s" % (float('%.4g'%rms_ML_orth)))
print("*** NB medians and errors only meaningful for unimodal posteriors. Check the corner plot! ***")
print("-------------------------")

f = open(outfile2, "w")
f.write("Pearson r: %f; p-value: %f \n" % (Pearson[0], Pearson[1]))
f.write("Spearman rho: %f; p-value: %f \n" % (Spearman[0], Spearman[1]))
f.write("Kendall tau: %f; p-value: %f \n" % (Kendall[0],Kendall[1]))
f.write("Maximum likelihood (ML) value: %s\n" % (float('%.4g'%ML)))
f.write("Slope (ML, median, upper error, lower error): %s; %s; +%s, %s\n" % (a_ML, a_med, a_up, a_dw))
f.write("Intercept (ML, median, upper error, lower error): %s; %s; +%s; %s\n" % (b_ML, b_med, b_up, b_dw))
f.write("Intrinsic scatter (ML, median, upper error, lower error): %s; %s; +%s; %s\n" % (s_ML, s_med, s_up, s_dw))
f.write("ML observed scatter (vertical): %s\n" % (rms_ML))
f.write("ML observed scatter (orthogonal): %s\n" % (rms_ML_orth))
f.close()

# Make corner plot
# Sigma levels in 2D = 1-exp( (x/r)**2/2)
figure = corner.corner(samples, levels=(1.-np.exp(-0.5), 1.-np.exp(-2.0)), labels=[r"Slope", r"Intercept", r"Intrinsic Scatter", r"Intrinsic Scatter"], quantiles=[0.16,0.84], show_titles=True, label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 10})
# Maximum-likelihood values
Med_value = [a_med, b_med, s_med]
axes = np.array(figure.axes).reshape((ndim, ndim))
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(Med_value[i], color="r")
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(Med_value[xi], color="r")
        ax.axhline(Med_value[yi], color="r")
        ax.plot(Med_value[xi], Med_value[yi], "sr")
figure.savefig(outplot2, format='pdf', dpi=300)

# Make best-fit plot
fig = plt.figure(figsize=(4, 4), dpi=300) 
x_vec = np.array([np.min(x), np.max(x)])
y_vec = a_med*x_vec + b_med
plt.errorbar(x, y, xerr=err_x, yerr=err_y, marker='o', markersize=1, linestyle=' ', color='k', zorder=2)
plt.plot(x_vec, y_vec, '-r', zorder=1)
if (orthfit==False):
	ymin = y_vec - s_med
	ymax = y_vec + s_med
	plt.fill_between(x_vec, ymin, ymax, color='k', alpha=0.2, zorder=0)
elif (orthfit==True):
	dist = np.sqrt( s_med**2 + (s_med*a_med)**2. )
	ymin = y_vec - dist
	ymax = y_vec + dist
	plt.fill_between(x_vec, ymin, ymax, color='k', alpha=0.2, zorder=0)
fig.savefig(outplot3, format='pdf', dpi=300)
