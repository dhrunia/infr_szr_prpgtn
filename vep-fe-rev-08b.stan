// vs. rev05, rm FC, rm gain

functions {
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
  real I1;
  real tau0;
  real dt;
  matrix[nt, nn] seeg_log_power;
  vector[nn] Ic;
  matrix<lower=0.0>[nn, nn] SC;
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
  real sigma_hi = sig_hi;
  real x0_range = x0_hi - x0_lo;
  real x0_prior_sd = x0_range/4;
  real x0_prior_mean = x0_lo + 0.5 * x0_range;
  real K_a = gamma_a_from_u_v(K_u, K_v);
  real K_b = gamma_b_from_a_u(K_a, K_u);
  real max_slp = max(seeg_log_power);
}

parameters {
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

  /* integrate & predict */
  for (i in 1:nn) {
    x[1, i] = x_init[i];
    z[1, i] = z_init[i];
  } 
  for (t in 1:(nt-1)) {
    x[t+1] = x_step(x[t], z[t], I1, time_scale, x_eta[t], sigma);
    z[t+1] = z_step(x[t], z[t], x0, K, Ic, time_scale, z_eta[t], sigma, tau0); 
  }
  if (use_data==1)
    to_vector(seeg_log_power) ~ normal(amplitude * (to_vector(x) + offset),
				       epsilon);
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
    z[t+1] = z_step(x[t], z[t], x0, K, Ic, time_scale, z_eta[t], sigma, tau0); 
  }
}
