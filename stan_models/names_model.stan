data {
    int N;  // number of houses in nbd
    int K;  // number of covariates
    matrix[N,K] X;  // design matrix
    vector[N] LogSalePrice;  // response variable
}

parameters {
    real alpha;
    real<lower=0> tau_alpha;
    corr_matrix[K] Omega;
    vector<lower=0>[K] tau_beta;
    vector[K] beta;
    real<lower=0> sigma;
}

transformed parameters {
    vector[N] mu;
    mu = alpha + X * beta;
    matrix[K,K] Sigma_beta;
    Sigma_beta = quad_form_diag(Omega, tau_beta);
}

model {
    // hyper priors
    Omega ~ lkj_corr(1);
    tau_alpha ~ inv_gamma(3, 0.5);
    tau_beta ~ inv_gamma(3, 0.5);
    // priors
    alpha ~ normal(12, tau_alpha);
    beta ~ multi_normal(rep_vector(0, K), Sigma_beta);
    sigma ~ inv_gamma(3, 0.5);

    // likelihood
    LogSalePrice ~ normal(mu, sigma);
}
