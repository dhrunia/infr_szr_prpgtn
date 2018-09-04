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
  
  row_vector[nn] x0;
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  real amplitude;
  real offset;
  real time_step;
  real K;
  real tau0;

  real epsilon_slp;
  real epsilon_snsr_pwr;

  // Modelled data
  row_vector[ns] slp_true[nt]; //seeg log power
  row_vector[ns] snsr_pwr_true; //seeg sensor power

  // Point at which to evaluate posterior
  row_vector[ns] slp_mean[nt]; //seeg log power
  row_vector[ns] snsr_pwr_mean; //seeg sensor power
}

transformed data{
  row_vector[nn] x0_star = x0 + 2.5;
  row_vector[nn] x_init_star = x_init + 2.0;
  row_vector[nn] z_init_star = z_init - 3.0;
  real amplitude_star = (log(amplitude) - pow(0.5,2) - log(1.0))/0.5;
  real time_step_star = (log(time_step) - pow(0.6,2) - log(0.5))/0.6;
  real K_star = (log(K) - pow(1.0,2) - log(1.0))/1.0;
  real tau0_star = (log(tau0) - pow(1.0,2) - log(30.0))/1.0;
}

parameters {
}

transformed parameters{
}

model {
}

generated quantities {
  real posterior_prob = 0.0;

  for(i in 1:nn){
    posterior_prob += normal_lpdf(x0_star[i] | 0, 1.0);
    posterior_prob += normal_lpdf(x_init_star[i] | 0, 1.0);
    posterior_prob += normal_lpdf(z_init_star[i] | 0, 1.0);
  }
  posterior_prob += normal_lpdf(amplitude_star | 0, 1.0);
  posterior_prob += normal_lpdf(offset | 0, 1.0);

  posterior_prob += normal_lpdf(time_step_star | 0, 1.0);
  posterior_prob += normal_lpdf(tau0_star | 0, 1.0);
  posterior_prob += normal_lpdf(K_star | 0, 1.0);

  for (t in 1:nt) {
    for (i in 1:ns){
      posterior_prob += normal_lpdf(slp_true[t,i] | slp_mean[t,i], epsilon_slp);
    }
  }
  for (i in 1:ns)
    posterior_prob += normal_lpdf(snsr_pwr_true[i] | snsr_pwr_mean[i], epsilon_snsr_pwr);
}
