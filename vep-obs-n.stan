data {
    int nt;
    int ns;
    real seeg_log_power[nt, ns];
}

transformed data {
    int nn = ns;
    int use_data = 1;
    real y[nt, nn] = seeg_log_power;
}

parameters {
    real x_eta[nt-1, nn];
    real z_eta[nt-1, nn];
    real tau_[nn];
    real<lower=0, upper=1> amp[nn];
    real off[nn];
    real a[nn];
    // pool i.c. & noise scales
    real x0;
    real z0;
    real eps_;
    real sig_;
    real dt_;
}

transformed parameters {
    real x[nt, nn];
    real z[nt, nn];
    real tau[nn];
    real eps = 0.01 * exp(0.02 * eps_);
    real sig;//[nn];
    real dt = 0.0084 * exp(0.005 * dt_);
    sig = 0.15 * exp(0.1 * sig_);
    for (i in 1:nn) {
        x[1, i] = x0;
        z[1, i] = z0;
        tau[i] = tau_[i] + 5.0;
    }
    for (i in 1:(nt-1)) {
        for (j in 1:nn) {
            x[i + 1, j] = x[i, j] + tau[j]*(x[i, j]-x[i, j]*x[i, j]*x[i, j]/3+z[i, j])*dt + sig*x_eta[i, j];
            z[i + 1, j] = z[i, j] + (1.0/tau[j])*(a[j] - x[i, j])*dt + sig*z_eta[i, j];
        }
    }
}

model {
    to_vector(to_array_1d(x_eta)) ~ normal(0, 1);
    to_vector(to_array_1d(z_eta)) ~ normal(0, 1);
    x0 ~ normal(1, 1);
    z0 ~ normal(0, 1);
    dt ~ normal(0, 1);
    to_vector(tau_) ~ normal(0, 1);
    eps_ ~ normal(0, 1);
    sig_ ~ normal(0, 1);
    to_vector(a) ~ normal(1, 1);
    if (use_data == 1)
        for (t in 1:nt)
            for (i in 1:nn)
                y[t, i] ~ normal(x[t, i]*amp[i] + off[i], eps);
}

generated quantities {
    real gy[nt, nn];
    for (t in 1:(nt - 1))
        for (i in 1:nn)
            gy[t, i] = normal_rng(x[t, i]*amp[i] + off[i], eps);
}
