"""
Sam Voisin
February 2021
House Prices Prior Predicitive Distribution
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
import pymc3 as pm
from sklearn.model_selection import train_test_split

# read data
fp = os.path.abspath('model_set.csv')
raw_df = pd.read_csv(fp)

# data transforms
raw_df["LogSalePrice"] = np.log(raw_df.SalePrice)
raw_df["CntrLogSalePrice"] = raw_df.LogSalePrice - raw_df.LogSalePrice.mean()
raw_df.CentralAir = [1 if i == "Y" else 0 for i in raw_df.CentralAir]
raw_df.YrSold = raw_df.YrSold - raw_df.YrSold.min()  # years from 2006
raw_df.YearBuilt = raw_df.YearBuilt - raw_df.YearBuilt.min()  # years from 1872

# drop unecessary cols
d_cols = ["Utilities"]
raw_df.drop(columns=d_cols, inplace=True)


### data preparation and formatting ###

# design matix
covariates = ["YrSold", "YearBuilt", "1stFlrSF", "CentralAir", "LotArea"]
y = raw_df.LogSalePrice.values
X = raw_df.loc[:, covariates].values
n, p = X.shape

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1)


### modeling ###

# specify model and sample
with pm.Model() as hp_model:
    # hyper priors
    chol, corr, stds = pm.LKJCholeskyCov("Omega", n=p, eta=1.,
                                         sd_dist=pm.Exponential.dist(2.0),
                                         compute_corr=True)
    cov = pm.Deterministic("cov", chol.dot(chol.T))
    mu_alpha = pm.Normal("mu_alpha", mu=12., sigma=1.)
    tau_alpha = pm.HalfStudentT("tau_alpha", sigma=0.5, nu=1.)
    # priors
    alpha = pm.Normal("alpha", mu=mu_alpha, sigma=tau_alpha)
    beta = pm.MvNormal("beta", mu=np.zeros(p), cov=cov, shape=(p,))
    sigma = pm.HalfStudentT("sigma", sigma=0.5, nu=1.)
    # likelihood
    Ey = alpha + X_train @ beta  # expectation of Y|X
    y_obs = pm.Normal("y_obs", mu=Ey, sigma=sigma, observed=y_train)
    # sampling
    prior_pred = pm.sample_prior_predictive(samples=100, random_seed=1)

# save model
with open("prior_pred_samps.pkl", "wb") as fh:
    pkl.dump(prior_pred, fh)