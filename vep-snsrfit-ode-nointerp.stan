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
  matrix[ns,nn] gain;
  matrix<lower=0.0>[nn, nn] SC;

  // Modelled data
  row_vector[ns] slp[nt]; //seeg log power
  row_vector[ns] snsr_pwr; //seeg sensor power

  // Data on priors
  row_vector[nn] x0_mu;
}

transformed data{
  real I1 = 3.1;
  real time_step = 0.1;
}

parameters {
  row_vector[nn] x0;
  real amplitude;
  real offset;
  real K;
  real tau0;
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  real eps_slp;
  real eps_snsr_pwr;
}

transformed parameters{   
  // Euler integration of the 2D Epileptor
  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
  row_vector[ns] mu_slp[nt];
  row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns);
  for (t in 1:nt) {
    if(t == 1){
      x[t] = x_step(x_init, z_init, I1, time_step);
      z[t] = z_step(x_init, z_init, x0, K*SC, time_step, tau0);
    }
    else{
      x[t] = x_step(x[t-1], z[t-1], I1, time_step);
      z[t] = z_step(x[t-1], z[t-1], x0, K*SC, time_step, tau0);
    }
    mu_slp[t] = amplitude * (log(gain * exp(x[t])')' + offset);
    mu_snsr_pwr += mu_slp[t] .* mu_slp[t];
  }
  mu_snsr_pwr = mu_snsr_pwr / nt;
}

model {
  x0 ~ normal(x0_mu, 1.0);
  amplitude ~ normal(1.0, 10.0)T[0,];
  offset ~ normal(0, 10.0);
  tau0 ~ normal(20, 10.0)T[5,];
  K ~ normal(1.0, 10.0)T[0,];
  for (i in 1:nn){
    x_init[i] ~ normal(-2.0, 10.0);
    z_init[i] ~ normal(3.5, 10.0);
  }
  eps_slp ~ normal(1, 10)T[0,];
  eps_snsr_pwr ~ normal(1, 10)T[0,];
  for (t in 1:nt) {
    target += normal_lpdf(slp[t] | mu_slp[t], eps_slp);
  }
  target += normal_lpdf(snsr_pwr | mu_snsr_pwr, eps_snsr_pwr);
}

generated quantities {
}
