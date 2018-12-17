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
  
  row_vector x_z_interp(row_vector x, row_vector z, real I1, real time_step, real dt,
			row_vector x0, matrix SC, real tau0) {
    int nn = num_elements(x);
    row_vector[2*nn] x_z_t;
    row_vector[nn] x_t;
    row_vector[nn] z_t;
    real nsteps = floor(time_step/dt);
    real step_count = 0;
    x_t = x;
    z_t = z;
    while(step_count != nsteps){
      x_t = x_step(x_t, z_t, I1, dt);
      z_t = z_step(x_t, z_t, x0, SC, dt, tau0);
      step_count += 1;
    }
    x_z_t[1:nn] = x_t;
    x_z_t[nn+1:2*nn] = z_t;
    return x_z_t;
  }
}

data {
  int nn;
  int ns;
  int nt;
  real I1;
  /* real tau0; */
  /* real time_step; */
  real dt;

  matrix[ns,nn] gain;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;

  row_vector[ns] epsilon_slp;
  row_vector[ns] epsilon_snsr_pwr;

  // Modelled data
  row_vector[ns] slp[nt]; //seeg log power
  row_vector[ns] snsr_pwr; //seeg sensor power
}

parameters {
  row_vector[nn] x0_star;
  row_vector[nn] x_init_star;
  row_vector[nn] z_init_star;
  real amplitude_star;
  real offset_star;
  real time_step_star;
  real K_star;
  real tau0_star;
  //  matrix<lower=0.0, upper=10.0>[nn, nn] FC;
  real<lower=1> alpha;
}

transformed parameters{
  row_vector[nn] x0 = -2.5 + (1/alpha)*x0_star;
  row_vector[nn] x_init = -2.0 + (1/alpha)*x_init_star;
  row_vector[nn] z_init = 3.0 + (1/alpha)*z_init_star;
  real amplitude = exp(pow(1.0, 2) + log(1.0) + 1.0*(1/alpha)*amplitude_star);
  real offset = (1/alpha)*offset_star;
  real time_step = exp(pow(1.0, 2) + log(0.5) + 1.0*(1/alpha)*time_step_star);
  real tau0 = exp(pow(1.0, 2) + log(30.0) + 1.0*(1/alpha)*tau0_star);
  real K = exp(pow(1.0, 2) + log(1.0) + 1.0*(1/alpha)*K_star);

  // Euler integration of the epileptor without noise 
  row_vector[2*nn] x_z_t;
  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
  row_vector[ns] mu_slp[nt];
  row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns);
  for (t in 1:nt) {
    if(t ==1)
      x_z_t = x_z_interp(x_init, z_init, I1, time_step, dt, x0, K*SC, tau0);
    else
      x_z_t = x_z_interp(x[t-1], z[t-1], I1, time_step, dt, x0, K*SC, tau0);
    x[t] = x_z_t[1:nn];
    z[t] = x_z_t[nn+1:2*nn];
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
  /* for (i in 1:nn){ */
  /*   for (j in 1:nn){ */
  /*     FC[i,j] ~ normal(K*SC[i,j], 0.01); */
  /*   } */
  /* } */
  target += normal_lpdf(x_init_star | 0, 1.0);
  target += normal_lpdf(z_init_star | 0, 1.0);
  target += normal_lpdf(time_step_star | 0, 1.0);
  target += normal_lpdf(tau0_star | 0, 1.0);
  target += normal_lpdf(K_star | 0, 1.0);
  for (t in 1:nt) {
    target += normal_lpdf(slp[t] | mu_slp[t], epsilon_slp);
  }
  target += normal_lpdf(snsr_pwr | mu_snsr_pwr, epsilon_snsr_pwr);
  target += -(3*nn + 5) * log(alpha);
}

generated quantities {
}
