"""
Sam Voisin
February 2021
MCMC diagnostic utility functions
"""

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# trace plot
def plot_trace(*chains):
    """
    make trace plots
    """
    n = chains[0].size
    for chain in chains:
        plt.plot(range(n), chain, alpha=0.2)
    plt.show()

# distribution plot
def plot_posterior_dists(*chains):
    """
    plot posterior distributions for a single parameter
    """
    for chain in chains:
        sns.kdeplot(chain, shade=True)
    plt.show()

# highest density interval plot
def plot_hdi(*chains, hdi_prob=0.99, expect=True):
    """
    plot highest density interval on distribution plot
    """
    cat_chain = np.concatenate([np.array(chain) for chain in chains])
    all_mean = cat_chain.mean()
    for chain in chains:
        chain = np.array(chain)
        sns.kdeplot(chain, shade=True)
    lb, ub = az.hdi(chain, hdi_prob=hdi_prob)
    plt.vlines(x=lb, ymin=0, ymax=1, colors="red", label="HDI bounds")
    plt.vlines(x=ub, ymin=0, ymax=1, colors="red")
    if expect:
        plt.vlines(x=all_mean, ymin=0, ymax=1,
                   color="black", label="expectation")
    plt.legend()
    plt.title(f"Expectation: {all_mean:.4f}; HDI bounds ({lb:.4f}, {ub:.4f})")
    plt.show()

# autocorrelation plot
def plot_acf(*chains):
    """
    plot autocorrelation
    """
    n = chains[0].size
    plt.hlines(y=0, xmin=0, xmax=n, color="black")
    for chain in chains:
        acf = az.autocorr(np.array(chain))  # ensure a numpy array
        plt.scatter(range(n), acf, s=1)
    plt.show()

# gelman-rubin diagnostic/potential scale reduction factor
def calc_B(*chains):
    """
    calculate between-sequence variance B/n
    """
    m = len(chains)
    n = len(chains[0])
    chain_means = np.array([chain.mean() for chain in chains])
    all_mean = chain_means.mean()
    res = (chain_means - all_mean)**2
    res = res.sum()
    res *= n/(m-1)
    return res

def calc_W(*chains):
    """
    calculate avg within-sequence variance W
    """
    m = len(chains)
    n = len(chains[0])
    chain_means = [chain.mean() for chain in chains]
    chain_sses = []
    for i, chain in enumerate(chains):
        chain_sse = ((chain-chain_means[i])**2).sum()
        chain_sses.append(chain_sse)
    res = sum(chain_sses)
    res /= (m*(n-1))
    return res

def calc_Vhat(*chains):
    """
    calculate pooled posterior variance estimate (i.e. Vhat)
    """
    m = len(chains)
    n = len(chains[0])
    B = calc_B(*chains)
    W = calc_W(*chains)
    sig2plus = W*(n-1)/n+B/n  # marginal posterior variance of estimand
    res = sig2plus + B/(m*n)  # pooled posterior variance estimate
    return res

def calc_Rhat(*chains):
    """
    calculate potential scale reduction factor (i.e. gelman-rubin diagnostic)
    """
    Vhat = calc_Vhat(*chains)
    W = calc_W(*chains)
    return np.sqrt(Vhat/W)

# effective sample size
def calc_ess(*chains):
    """
    calculate effective sample size (neff)
    """
    m = len(chains)
    n = len(chains[0])
    cat_chain = np.concatenate([np.array(chain) for chain in chains])
    rhot = az.autocorr(cat_chain)
    res = m*n/(1+2*rhot.sum())
    return res


