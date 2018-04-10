functions {

  matrix vector_differencing(row_vector x) {
    matrix[num_elements(x), num_elements(x)] D;
    for (i in 1:num_elements(x)) {
      D[i] = x - x[i];
    }
    return D;
  }

  row_vector x_step(row_vector x, row_vector z, real I1, real time_step) {
    int nn = num_elements(x);
    row_vector[nn] x_next;
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    x_next = x + (time_step * dx);
    return x_next;
  }

  row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC, real time_step,
		    real tau0) {
    int nn = num_elements(z);
    row_vector[nn] z_next;
    matrix[nn, nn] D = vector_differencing(x);
    row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D));
    row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
    z_next = z + (time_step * dz);
    return z_next;
  }
}

data {
  int nn;
  int ns;
  int nt;
  real I1;
  real tau0;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;
  matrix[ns,nn] gain;

  // Some parameters are fixed for now with known values from simulation
  real sigma;
  real k;
  real epsilon;
  real amplitude;
  real offset;
  real time_step;

  // Modelled data
  row_vector[ns] seeg[nt];
}

parameters {
  row_vector[nn] x0_star;
  row_vector<lower=-3.0, upper=3.0>[nn] x[nt];
  row_vector<lower=1.0, upper=5.0>[nn] z[nt];
}

transformed parameters{
  row_vector[nn] x0 = -2.5 + x0_star;
}

model {
  row_vector[ns] mu_seeg[nt];

  x0_star ~ normal(0,1.0);
  /* x[1] ~ normal(x_init - 1.5, sigma); */
  /* z[1] ~ normal(z_init + 2.0, sigma); */
  mu_seeg[1] = amplitude * (gain * x[1]' + offset)';
  for (t in 1:(nt-1)) {
    x[t+1] ~ normal(x_step(x[t], z[t], I1, time_step), sigma);
    z[t+1] ~ normal(z_step(x[t], z[t], x0, k*SC, time_step, tau0), sigma);
    mu_seeg[t+1] = amplitude * (gain * x[t]' + offset)';
  }

  for (t in 1:nt){
      seeg[t] ~ normal(mu_seeg[t], epsilon);
  }
}

generated quantities {
}
