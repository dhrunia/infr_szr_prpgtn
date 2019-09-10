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

  matrix[ns,nn] gain;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;

  row_vector[nn] x_init;
  row_vector[nn] z_init;

  // Modelled data
  row_vector[ns] slp[nt]; //seeg log power
  row_vector[ns] snsr_pwr; //seeg sensor power
}

parameters {
  row_vector[nn] x0_star;
  real amplitude_star;
  real offset_star;
  real K_star;
  real tau0_star;
  real epsilon_slp_star;
  real epsilon_snsr_pwr_star;
}

transformed parameters{
  row_vector[nn] x0 = -2.5 + x0_star;
  real amplitude = exp(pow(1.0, 2) + log(1.0) + 1.0*amplitude_star);
  real offset = offset_star;
  real tau0 = exp(pow(1.0, 2) + log(30.0) + 1.0*tau0_star);
  real K = exp(pow(1.0, 2) + log(1.0) + 1.0*K_star);
  real epsilon_slp = exp(pow(1.0, 2) + log(1.0) + 1.0*epsilon_slp_star);
  real epsilon_snsr_pwr = exp(pow(1.0, 2) + log(100.0) + 1.0*epsilon_snsr_pwr_star);

  // Euler integration of the epileptor without noise 
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
    for (i in 1:ns){
      mu_snsr_pwr[i] += pow(mu_slp[t][i], 2);
    }
  }
}

model {
  target += normal_lpdf(x0_star | 0, 1.0);
  target += normal_lpdf(amplitude_star | 0, 1.0);
  target += normal_lpdf(offset_star | 0, 1.0);
  target += normal_lpdf(tau0_star | 0, 1.0);
  target += normal_lpdf(K_star | 0, 1.0);
  target += normal_lpdf(epsilon_slp_star | 0, 1.0);
  target += normal_lpdf(epsilon_snsr_pwr_star | 0, 1.0);
  for (t in 1:nt) {
    target += normal_lpdf(slp[t] | mu_slp[t], epsilon_slp);
  }
  target += normal_lpdf(snsr_pwr | mu_snsr_pwr, epsilon_snsr_pwr);
}

generated quantities {
}
