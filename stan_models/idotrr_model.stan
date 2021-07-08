data {
    int N;
    vector[N] Footprint;
    vector[N] LotArea_acres;
    vector[N] LogSalePrice;
}

parameters {
    real<lower=0> tau_alpha;
    real alpha;
    real beta_Footprint;
    real beta_LotArea;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    
    mu = alpha + Footprint * beta_Footprint + LotArea_acres * beta_LotArea;
}

model {
    // priors
    tau_alpha ~ inv_gamma(3, 0.5);
    alpha ~ normal(12, tau_alpha);
    beta_Footprint ~ normal(0, 1);
    beta_LotArea ~ normal(0, 1);
    sigma ~ inv_gamma(3, 0.5);

    // likelihood
    LogSalePrice ~ normal(mu, sigma);
}
