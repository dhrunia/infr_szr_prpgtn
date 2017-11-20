data {
    int nt;
    real y[nt];
    int use_data;
    real dt;
}

parameters {
    real x_eta[nt-1];
    real z_eta[nt-1];
    real x0;
    real z0;
    real tau_;
    real dt_;
    real<lower=0, upper=1> amp;
    real off;
    real eps_;
    real sig_;
    real a;
}

transformed parameters {
    real x[nt];
    real z[nt];
    real tau = tau_;
    //real dt = 0.1 * exp(0.1 * dt_);
    real eps = 0.1 * exp(0.1 * eps_);
    real sig = 0.1 * exp(0.1 * sig_);
    x[1] = x0;
    z[1] = z0;
    for (i in 1:(nt-1)) {
        // print("i = ", i, ", x = ", x[i], ", z = ", z[i]);
        x[i + 1] = x[i] + tau*(x[i]-x[i]*x[i]*x[i]/3+z[i])*dt + sig*x_eta[i];
        z[i + 1] = z[i] + (1.0/tau)*(a - x[i])*dt + sig*z_eta[i];
    }
}

model {
    to_vector(x_eta) ~ normal(0, 1);
    to_vector(z_eta) ~ normal(0, 1);
    x0 ~ normal(1, 1);
    z0 ~ normal(0, 1);
    tau_ ~ normal(0, 1);
    dt_ ~ normal(0, 1);
    eps_ ~ normal(0, 1);
    sig_ ~ normal(0, 1);
    a ~ normal(1, 1);
    if (use_data == 1)
        to_vector(y) ~ normal(to_vector(x)*amp + off, eps);
}

generated quantities {
    real gy[nt];
    for (i in 1:nt)
        gy[i] = normal_rng(x[i]*amp + off, eps);
}