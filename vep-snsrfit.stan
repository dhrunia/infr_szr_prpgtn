functions {

  matrix vector_differencing(row_vector x) {
    matrix[num_elements(x), num_elements(x)] D;
    for (i in 1:num_elements(x)) {
      D[i] = x - x[i];
    }
    return D;
  }

  row_vector x_step(row_vector x, row_vector z, real I1, real time_step, real time_scale, real sigma) {
    int nn = num_elements(x);
    row_vector[nn] x_next;
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    x_next = x + (time_scale * time_step * dx);
    return x_next;
  }

  row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC, vector Ic, 
		    real time_step, real time_scale, real sigma, real tau0) {
    int nn = num_elements(z);
    row_vector[nn] z_next;
    matrix[nn, nn] D = vector_differencing(x);
    row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D) - Ic .* to_vector(1.8 + x));
    row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
    z_next = z + (time_scale * time_step * dz);
    return z_next;
  }
}

data {
  int nn;
  int ns;
  int nt;
  real I1;
  real tau0;
  vector[nn] Ic;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;
  matrix[ns,nn] gain;

  // Some parameters are fixed for now with known values from simulation
  real time_scale;
  real time_step;
  real sigma;
  real k;
  real epsilon;
  real amplitude;
  real offset;
  row_vector[nn] x_init;
  row_vector[nn] z_init;

  // Modelled data
  row_vector[ns] seeg_log_power[nt];
}

parameters {
  row_vector[nn] x0_star;
  row_vector<lower=-3.0, upper=1.0>[nn] x[nt];
  row_vector<lower=1.0, upper=5.0>[nn] z[nt];
}

transformed parameters{
  row_vector[nn] x0 = -2.5 + x0_star;
}

model {
  row_vector[ns] mu_seeg_log_power[nt];

  x0_star ~ normal(0,1.0);
  x[1] ~ normal(x_init - 1.5, sigma);
  z[1] ~ normal(z_init + 2.0, sigma);
  mu_seeg_log_power[1] = amplitude * (log(gain * exp(x[1]')) + offset)';
  for (t in 1:(nt-1)) {
    x[t+1] ~ normal(x_step(x[t], z[t], I1, time_step, time_scale, sigma), sigma);
    z[t+1] ~ normal(z_step(x[t], z[t], x0, k*SC, Ic, time_step, time_scale, sigma, tau0), sigma);
    mu_seeg_log_power[t+1] = amplitude * (log(gain * exp(x[t]')) + offset)';
  }

  for (t in 1:nt){
      seeg_log_power[t] ~ normal(mu_seeg_log_power[t], epsilon);
  }
}

generated quantities {
}
