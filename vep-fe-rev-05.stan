functions {

    real gamma_a_from_u_v(real u, real v) {
        return (u*sqrt(u * u + 4 * v) + u * u + 2 * v) / (2 * v);
    }

    real gamma_b_from_a_u(real a, real u) {
        return (a - 1.0) / u;
    }

    real FC_lpdf(matrix FC, matrix SC, real SC_var, real u_self, real v_self) {
        int nn = rows(SC);
        real ld = 0.0;
        real fc_a;
        real fc_b;
        for (i in 1:nn) {
            for (j in 1:nn) {
                if (i>=j) {
                    real fc_u;
                    real fc_v;
                    if ((i==j) || (SC[i, j]==0.0)) {
                        fc_u = u_self;
                        fc_v = v_self;
                    } else {
                        fc_u = SC[i, j];
                        fc_v = fc_u * SC_var;
                    }
                    fc_a = gamma_a_from_u_v(fc_u, fc_v);
                    fc_b = gamma_b_from_a_u(fc_a, fc_u);
                    ld = ld + 2.0 * gamma_lpdf(FC[i, j] | fc_a, fc_b);
                }
            }
        }
        return ld;
    }

    matrix vector_differencing(row_vector x) {
      matrix[num_elements(x), num_elements(x)] D;
      for (i in 1:num_elements(x)) {
        D[i] = x - x[i];
      }
      return D';
    }

    
    row_vector x_step(row_vector x, row_vector z, real I1, real time_scale, 
        row_vector x_eta, real sigma
    ) {
        int nn = num_elements(x);
        row_vector[nn] x_next;
        row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
          row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
          x_next = x + (time_scale * dx) + x_eta * sigma;
        return x_next;
    }

    row_vector z_step(row_vector x, row_vector z, row_vector x0, real K, matrix FC, vector Ic, 
        real time_scale, row_vector z_eta, real sigma, real tau0
    ) {
        int nn = num_elements(z);
        row_vector[nn] z_next;
        matrix[nn, nn] D = vector_differencing(x);
        row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D) - Ic .* to_vector(1.8 + x));
        row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - tanh(K * gx));
        z_next = z + (time_scale * dz) + z_eta * sigma;
        return z_next;
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
  matrix[nn, nn] SC_ = SC / max(SC);
  for (i in 1:nn) SC_[i, i] = 0.0;
}

parameters {

    // functional connectivity 
    matrix<lower=0.0>[nn, nn] FC;

    // integrate and predict
    real<lower=0> K;
    row_vector[nn] x0;    
    real epsilon_star;
    real<lower=0.0> amplitude;
    real offset;
    real<lower=0.02, upper=0.1> time_scale;

    // time-series state non-centering:
    vector[nn] x_init;
    vector[nn] z_init;
    matrix[nt - 1, nn] x_eta;
    matrix[nt - 1, nn] z_eta;
    real<lower=0.0> sigma_star;
}

transformed parameters {
    real epsilon = 0.1 * exp(0.2 * epsilon_star);
    real sigma = 0.13 * exp(0.2 * sigma_star);
}


model {
    matrix[nt, nn] x;
    matrix[nt, nn] z;

    real fc_a;
    real fc_b;

    x0 ~ normal(x0_prior_mean, x0_prior_sd);
    epsilon_star ~ normal(0, 1);
    sigma_star ~ normal(0, 1);
    amplitude ~ normal(6,2);
    offset ~ normal(1.8125, 0.1875);

    x_init ~ normal(0, 1);
    z_init ~ normal(0, 1);
    to_vector(x_eta) ~ normal(0, 1);
    to_vector(z_eta) ~ normal(0, 1);

    K ~ gamma(4, 1.5); // thin lower + upper tail, cover about 0-10

    /* functional connectivity prior */
    target += FC_lpdf(FC | SC, SC_var, 1e-6, 1e-3);

    /* integrate & predict */
    for (i in 1:nn) {
      x[1, i] = x_init[i];
      z[1, i] = z_init[i];
    } 
    for (t in 1:(nt-1)) {
        x[t+1] = x_step(x[t], z[t], I1, time_scale, x_eta[t], sigma);
        z[t+1] = z_step(x[t], z[t], x0, K, FC, Ic, time_scale, z_eta[t], sigma, tau0); 
    }
    if (use_data==1)
        to_vector(seeg_log_power) ~ normal(amplitude * (to_vector(log(gain * exp(x'))') + offset), epsilon);
}

generated quantities {
    matrix[nt, nn] x;
    matrix[nt, nn] z;
    for (i in 1:nn) {
      x[1, i] = x_init[i];
      z[1, i] = z_init[i];
    } 
    for (t in 1:(nt-1)) {
        x[t+1] = x_step(x[t], z[t], I1, time_scale, x_eta[t], sigma);
        z[t+1] = z_step(x[t], z[t], x0, K, FC, Ic, time_scale, z_eta[t], sigma, tau0); 
    }
}
