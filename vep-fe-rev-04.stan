functions {

    real gamma_a_from_u_v(real u, real v) {
        return (u*sqrt(u * u + 4 * v) + u * u + 2 * v) / (2 * v);
    }

    real gamma_b_from_a_u(real a, real u) {
        return (a - 1.0) / u;
    }

}

data {
    int nn;
    int nt;
    int ns;
    real I1;
    real tau0;
    real dt;
    matrix[ns, nn] gain;
    matrix[nt, ns] seeg_log_power;
    vector[nn] Ic;
    matrix<lower=0.0>[nn, nn] SC;
    real SC_var; // prior on variance of connectivity strengths
    real K_lo;
    real K_u;
    real K_v;
    real x0_lo;
    real x0_hi;
    real eps_hi;
    real sig_hi;
    real zlim[2];
    real tt;
    int use_data;
}

transformed data {
    matrix[ns, nn] log_gain = log(gain);
    real sigma_hi = sig_hi;
    real x0_range = x0_hi - x0_lo;
    real x0_prior_sd = x0_range/4;
    real x0_prior_mean = x0_lo + 0.5 * x0_range;
    real K_a = gamma_a_from_u_v(K_u, K_v);
    real K_b = gamma_b_from_a_u(K_a, K_u);
}

parameters {

    // functional connectivity 
    matrix<lower=0.0>[nn, nn] FC;

    // integrate and predict
    real<lower=0> K;
    vector[nn] x0;    
    real epsilon_star;
    real amplitude;
    real offset;
    real<lower=0.0, upper=1.0> time_scale;

    // time-series state non-centering:
    vector[nn] x_init;
    vector[nn] z_init;
    matrix[nn, nt - 1] x_eta;
    matrix[nn, nt - 1] z_eta;
    real<lower=0.0> sigma_star;
}

model {
    matrix[nn, nt] x;
    matrix[nn, nt] z;

    real epsilon = 0.1 * exp(0.2 * epsilon_star);
    real sigma = 0.01 * exp(0.4 * sigma_star);

    real fc_a;
    real fc_b;

    x0 ~ normal(x0_prior_mean, x0_prior_sd);
    epsilon_star ~ normal(0, 1);
    sigma_star ~ normal(0, 1);
    amplitude ~ normal(1.0, 0.5);
    amplitude ~ normal(-1.0, 0.5);
    offset ~ normal(1.8125, 0.5);

    x_init ~ normal(0, 1);
    z_init ~ normal(0, 1);
    to_vector(x_eta) ~ normal(0, 1);
    to_vector(z_eta) ~ normal(0, 1);

    K ~ gamma(4, 1.5); // thin lower + upper tail, cover about 0-10

    /* functional connectivity */
    for (i in 1:nn) {
        for (j in 1:nn) {
            if (i>=j) {
                real fc_u;
                real fc_v;
                if ((i==j) || (SC[i, j]==0.0)) {
                    fc_u = 1e-6;
                    fc_v = 1e-3;
                } else {
                    fc_u = SC[i, j];
                    fc_v = fc_u * SC_var;
                }
                fc_a = gamma_a_from_u_v(fc_u, fc_v);
                fc_b = gamma_b_from_a_u(fc_a, fc_u);
                FC[i, j] ~ gamma(fc_a, fc_b);
                FC[j, i] ~ gamma(fc_a, fc_b);
            }
        }
    }

    /* integrate & predict */
    for (i in 1:nn) {
      x[i, 1] = x_init[i] - 1.8;
      z[i, 1] = z_init[i] + 3.17;
    } 
    for (t in 1:(nt-1)) {
        for (i in 1:nn) {
            real dx = (I1 + 1.0) - pow(x[i, t],3.0) - 2.0 * pow(x[i, t], 2.0) - z[i, t];
            real gx = FC[i,] * (x[,t] - x[i,t]) - Ic[i] * (1.8 + x[i,t]);
            real dz = inv(tau0) * (4 * (x[i, t] - x0[i]) - z[i, t] - K * gx);

            x[i, t+1] = x[i, t] + dt*tt*time_scale*dx + x_eta[i, t] * sigma;
            z[i, t+1] = z[i, t] + dt*tt*time_scale*dz + z_eta[i, t] * sigma;
        }
    }
    if (use_data==1)
        to_vector(seeg_log_power) ~ normal(amplitude * (to_vector(log(gain * exp(x))) + offset), epsilon);
}

generated quantities {
    matrix[nn, nt] x;
    matrix[nn, nt] z;
    real epsilon = 0.1 * exp(0.2 * epsilon_star);
    real sigma = 0.13 * exp(0.2 * sigma_star);
    for (i in 1:nn) {
      x[i, 1] = x_init[i];
      z[i, 1] = z_init[i];
    } 
    for (t in 1:(nt-1)) {
        for (i in 1:nn) {
            real dx = (I1 + 1.0) - pow(x[i, t],3.0) - 2.0 * pow(x[i, t], 2.0) - z[i, t];
            real gx = FC[i,] * (x[,t] - x[i,t]) - Ic[i] * (1.8 + x[i,t]);
            real dz = inv(tau0) * (4 * (x[i, t] - x0[i]) - z[i, t] - K * gx);

            x[i, t+1] = x[i, t] + dt*tt*time_scale*dx + x_eta[i, t] * sigma;
            z[i, t+1] = z[i, t] + dt*tt*time_scale*dz + z_eta[i, t] * sigma;
        }
    }
}