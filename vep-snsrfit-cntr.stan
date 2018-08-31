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
  real tau0;
  
  matrix[ns,nn] gain;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;
  real sigma;
  real epsilon_slp;
  real epsilon_snsr_pwr;

  // Modelled data
  row_vector[ns] slp[nt]; //seeg log power
  row_vector[ns] snsr_pwr; //seeg sensor power
}

parameters {
  row_vector[nn] x0_star;
  row_vector[nn] x_init_star;
  row_vector[nn] z_init_star;
  real amplitude_x_star;
  real amplitude_z_star;
  real amplitude_slp_star;
  real offset_x;
  real offset_z;
  real offset_slp;
  /* real epsilon_star; */
  //  matrix<lower=0.0, upper=10.0>[nn, nn] FC;
  real time_step_star;
  /* real sigma_star; */
  real K_star;
  /* real tau0_star; */

  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
}

transformed parameters{
  row_vector[nn] x0 = -2.5 + x0_star;
  row_vector[nn] x_init = -2.0 + x_init_star;
  row_vector[nn] z_init = 3.0 + z_init_star;
  real amplitude_x = exp(pow(0.5, 2) + log(1.0) + 0.5*amplitude_x_star);
  real amplitude_z = exp(pow(0.5, 2) + log(1.0) + 0.5*amplitude_z_star);
  real amplitude_slp = exp(pow(0.5, 2) + log(1.0) + 0.5*amplitude_slp_star);
  /* real offset_slp = exp(pow(0.5, 2) + log(0.001) + 0.5*offset_slp_star); */
  /* real epsilon = exp(pow(1.0, 2) + log(0.01) + 1.0*epsilon_star); */
  real time_step = exp(pow(0.6, 2) + log(0.5) + 0.6*time_step_star);
  /* real sigma = exp(pow(1.0, 2) + log(0.1) + 1.0*sigma_star) */
  /* real tau0 = exp(pow(1.0, 2) + log(30.0) + 1.0*tau0_star); */
  real K = exp(pow(1.0, 2) + log(1.0) + 1.0*K_star);
}

model {
  row_vector[ns] mu_slp[nt];
  row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns);

  x0_star ~ normal(0, 1.0);
  amplitude_x_star ~ normal(0, 1.0);
  amplitude_z_star ~ normal(0, 1.0);
  amplitude_slp_star ~ normal(0, 1.0);
  offset_x ~ normal(0, 1.0);
  offset_z ~ normal(0, 1.0);
  offset_slp ~ normal(0, 1.0);
  /* epsilon_star ~ normal(0, 1.0); */
  /* for (i in 1:nn){ */
  /*   for (j in 1:nn){ */
  /*     FC[i,j] ~ normal(K*SC[i,j], 0.01); */
  /*   } */
  /* } */
  x_init_star ~ normal(0, 1.0);
  z_init_star ~ normal(0, 1.0);

  time_step_star ~ normal(0, 1.0);
  /* tau0_star ~ normal(0, 1.0); */
  K_star ~ normal(0, 1.0);
  
  for (t in 1:nt) {
    if(t == 1){
      x[t] ~ normal(x_step(amplitude_x*(x_init + offset_x), amplitude_z * (z_init + offset_z), I1, time_step), sigma);
      z[t] ~ normal(z_step(amplitude_x *(x_init + offset_x), amplitude_z * (z_init + offset_z), x0, K*SC, time_step, tau0), sigma);
    }
    else{
      x[t] ~ normal(x_step(amplitude_x*(x[t-1] + offset_x), amplitude_z * (z[t-1] + offset_z), I1, time_step), sigma);
      z[t] ~ normal(z_step(amplitude_x *(x[t-1] + offset_x), amplitude_z * (z[t-1] + offset_z), x0, K*SC, time_step, tau0), sigma);
    }
    mu_slp[t] = amplitude_slp * (log(gain * exp(x[t])')' + offset_slp);
    slp[t] ~ normal(mu_slp[t], epsilon_slp);
    for (i in 1:ns){
      mu_snsr_pwr[i] += pow(mu_slp[t][i], 2);
    }
  }
  snsr_pwr ~ normal(mu_snsr_pwr, epsilon_snsr_pwr);
}

generated quantities {
}
