"""
Sam Voisin
February 2021
House Prices Data Pipeline
"""

import os
import json
import pickle as pkl
import numpy as np
import pandas as pd
import arviz as az
import pymc3 as pm
import theano
import theano.tensor as T
from sklearn.model_selection import train_test_split


# read data
fp = os.path.abspath('model_set.csv')
raw_df = pd.read_csv(fp)

# data transforms
raw_df["LogSalePrice"] = np.log(raw_df.SalePrice)
raw_df.CentralAir = [1 if i == "Y" else 0 for i in raw_df.CentralAir]
raw_df.YrSold = raw_df.YrSold - raw_df.YrSold.min()  # years from 2006
raw_df.YearBuilt = raw_df.YearBuilt - raw_df.YearBuilt.min()  # years from 1872
Neighborhoods = raw_df.Neighborhood.unique()
NbdLookup = dict(zip(Neighborhoods, range(Neighborhoods.size)))
raw_df["NeighborhoodCode"] = raw_df.Neighborhood.replace(NbdLookup)

# drop unecessary cols
d_cols = ["Utilities"]
raw_df.drop(columns=d_cols, inplace=True)

### data preparation and formatting ###

# design matix
covariates = ("1stFlrSF", "LotArea")
y = raw_df.LogSalePrice
X = raw_df.loc[:, covariates]
X_nbd = raw_df.loc[:, "NeighborhoodCode"]
n_nbd = Neighborhoods.size
n, p = X.shape

# train-test split
train_idx, test_idx = train_test_split(range(n),
                                       test_size=0.2,
                                       random_state=1)
X_train = X.iloc[train_idx, :].reset_index(drop=True)
X_nbd_train = X_nbd.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx, :].reset_index(drop=True)
X_nbd_test = X_nbd.iloc[test_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

### modeling ###
hp_model = pm.Model()
#prev_trace = pm.load_trace('.pymc_1.trace', model=hp_model)

Ip_mat = np.eye(p)
zp_vec = np.zeros(p)

# specify model and perform sampling
with hp_model:
    X_train_data = pm.Data("X_train_data", X_train)
    nbd_idx = pm.Data("nbd_idx", X_nbd_train)
    y_train_data = pm.Data("y_train_data", y_train)
    # hyper priors
    chol, corr, stds = pm.LKJCholeskyCov("Omega", n=p, eta=1.,
                                         sd_dist=pm.HalfStudentT.dist(sigma=0.5,
                                                                      nu=1.),
                                         compute_corr=True)
    cov = pm.Deterministic("cov", chol.dot(chol.T))
    tau_alpha = pm.HalfStudentT("tau_alpha", sigma=0.5, nu=1.)
    alpha = pm.Normal("alpha", mu=12., sigma=0.5)
    # priors
    alpha_nbd = pm.Normal("alpha_nbd",
                          mu=alpha,
                          sigma=tau_alpha,
                          shape=(n_nbd,))
    beta_uc = pm.MvNormal("beta_uc", mu=zp_vec, cov=Ip_mat, shape=(p,))
    beta = pm.Deterministic("beta", cov.dot(beta_uc))
    sigma = pm.HalfStudentT("sigma", sigma=0.5, nu=1.)
    # likelihood
    Ey_x = T.add(alpha_nbd[nbd_idx], X_train_data.dot(beta))  # E[Y|X]
    y_obs = pm.Normal("y_obs", mu=Ey_x, sigma=sigma, observed=y_train_data)
    # sampling
    posterior = pm.sample(draws=5000, tune=20000, cores=4,
                          init="advi+adapt_diag",
                          target_accept=0.95,
                          return_inferencedata=False)
    posterior_predictive = pm.fast_sample_posterior_predictive(posterior)


# save model
pm.save_trace(trace=posterior)
post_pred_samps = pd.DataFrame(posterior_predictive["y_obs"].T)
post_pred_samps.columns = [
    f"sample_{i}" for i in range(post_pred_samps.shape[1])
    ]
post_pred_samps.to_csv("./posterior_pred_samps.csv", index=False)