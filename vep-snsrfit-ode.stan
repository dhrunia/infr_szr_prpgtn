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

  row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC,
		    real time_step, real tau0) {
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
  real time_step;
  int nsteps;
  real tau0;
  
  matrix[ns,nn] gain;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;
  real<lower=0.0> K;

  real epsilon;

  // Modelled data
  row_vector[ns] slp[nt];
}

parameters {
  row_vector[nn] x0_star;
  row_vector[nn] x_init_star;
  row_vector[nn] z_init_star;
  real amplitude_star;
  real offset;
  //  matrix<lower=0.0, upper=10.0>[nn, nn] FC;
}

transformed parameters{
  row_vector[nn] x_t;
  row_vector[nn] z_t;
  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
  row_vector[ns] mu_slp[nt];
  row_vector[nn] x0 = -2.5 + x0_star;
  row_vector[nn] x_init = -2.0 + x_init_star;
  row_vector[nn] z_init = 3.0 + z_init_star;
  real amplitude = exp(pow(0.5, 2) + log(1.0) + 0.5*amplitude_star);

  x_t = x_init;
  z_t = z_init;
  for (t in 1:nt) {
    for(i in 1:nsteps){
      x_t = x_step(x_t, z_t, I1, time_step);
      z_t = z_step(x_t, z_t, x0, K*SC, time_step, tau0);
    }
    x[t] = x_t;
    z[t] = z_t;
    mu_slp[t] = amplitude * (log(gain * exp(x[t])')' + offset);
  }
}

model {
  x0_star ~ normal(0, 1.0);
  amplitude_star ~ normal(0, 1.0);
  offset ~ normal(0, 1.0);
  /* for (i in 1:nn){ */
  /*   for (j in 1:nn){ */
  /*     FC[i,j] ~ normal(K*SC[i,j], 0.01); */
  /*   } */
  /* } */
  x_init_star ~ normal(0, 1.0);
  z_init_star ~ normal(0, 1.0);
  for (t in 1:nt){
    slp[t] ~ normal(mu_slp[t], epsilon);
  }
}

generated quantities {
}
