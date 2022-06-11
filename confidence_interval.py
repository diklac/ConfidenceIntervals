# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 23:05:37 2022

@author: dikla
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 22})


# -----------------------
# Interface
# -----------------------
mu = 42.0
sigma = 5.0
n_samples = 20
n_partition = int(np.floor(0.5*n_samples))
ci_percentile = 0.95
line_width = 3
ci_calc_repeat = 5e2
num_bootstrap = 50000

# -----------------------
# Funcs
# -----------------------
def calc_statistics(data, ci_percentile=0.95):
    '''
    Calculate mean and CI for data and confidence level ci_percentile
    Parameters
    ----------
    data : samples drawn from a normal distribution with unknown
           mean and variance.
    ci_percentile : confidence level

    Returns
    -------
    (sample mean, CI)

    '''
    n_df = len(data) - 1
    
    # sample mean and variance
    m = np.mean(data)
    s = np.sqrt(np.var(data, ddof = 1))
    
    # from Student's t table
    c = stats.t(df = n_df).ppf(1 - 0.5*(1 - ci_percentile))
    
    # the CI
    v = c*s/np.sqrt(len(data))
    CI = [m - v, m + v]
    return (m, CI)


# -----------------------
# Confidence Interval Realizations
# -----------------------
# Calculate ci_calc_repeat realizations of
# CI and check how many contained the real
# parameter value (mu)

param_in_ci = 0
for ii in range(int(ci_calc_repeat)):
    # generate the data
    d = mu + sigma*np.random.randn(n_samples)
    
    # using Student's t distribution, calc CI
    (d_mean, d_CI) = calc_statistics(d, ci_percentile)
    
    # is the parameter inside the CI realization?
    p_in_ci = (mu >= d_CI[0]) & (mu <= d_CI[1])
    if p_in_ci:
        param_in_ci += 1
    
# plot the
param_not_in_ci = ci_calc_repeat - param_in_ci
plt.figure()
plt.bar(['not inside', 'inside'], [param_not_in_ci/ci_calc_repeat, param_in_ci/ci_calc_repeat], alpha = 0.5)

    
    
# -----------------------
# Confidence Interval Not a Measure of Accuracy
# -----------------------

# generate the full data set
data_full = mu + sigma*np.random.randn(n_samples)

# and the partitioned data
inds = np.random.choice(np.arange(n_samples), n_partition, replace = False)
inds_comp = np.random.choice(np.arange(n_samples), n_samples - n_partition, replace = False)
data1 = data_full[inds]
data2 = data_full[inds_comp]

# calculate a ci_percentile CI for the full data set and the partitioned sets
ds = [data_full, data1, data2]
names= ['full', '1', '2']
plt.figure()
line_idx = 0.1*np.arange(len(ds), 0, -1)
for (d, jj) in zip(ds, line_idx):
    (m, ci) = calc_statistics(d, ci_percentile)
    plt.plot(ci, (jj, jj), 'o-', linewidth = line_width)
    
# plot the CIs and the real value of the mean
plt.plot((mu, mu), (np.min(line_idx), np.max(line_idx)), '--', color = 'orange', linewidth = line_width)
plt.yticks([], [])
plt.xlabel('weight [g]')
plt.legend(names)

# -----------------------
# Parametric Bootstrapping
# -----------------------
d0 = mu + sigma*np.random.randn(n_samples)
# Assuming the data is sampled from a Normal distribution,
# get the MLE mean and sigma
mu_MLE = np.mean(d0)
sigma_MLE = np.sqrt((1.0/n_samples)*np.sum(np.square(d0 - mu_MLE)))
print('MLE mean %.2f, sigma %.2f' % (mu_MLE, sigma_MLE))

# the bootstrap samples
bootstrap_estimates = np.zeros((num_bootstrap))
for ind in range(num_bootstrap):
    d_bootstrap = mu_MLE + sigma_MLE*np.random.randn(n_samples)
    mu_bootstrap = np.mean(d_bootstrap)
    bootstrap_estimates[ind] = mu_bootstrap

# compute and plot the empirical CDF
x = np.sort(bootstrap_estimates)
y = np.arange(1, num_bootstrap + 1)/num_bootstrap
plt.figure()
plt.scatter(x, y)
plt.title('Bootstrap Empirical CDF, %d samples' % num_bootstrap)
plt.xlabel('Bootstrap estimated mean')

# calculate the CI based on the CDF
ll = 0.5*(1 - ci_percentile)
ul = 1 - ll
lind = np.where(y >= ll)[0][0]
uind = np.where(y >= ul)[0][0]
ci_bootstrap = [x[lind], x[uind]]
print('Bootstrap %.2f CI (%f, %f)' % (ci_percentile, ci_bootstrap[0], ci_bootstrap[1]))

plt.show()
        