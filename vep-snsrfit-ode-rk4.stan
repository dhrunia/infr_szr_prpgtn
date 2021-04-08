functions {

  matrix vector_differencing(row_vector x) {
    matrix[num_elements(x), num_elements(x)] D;
    for (i in 1:num_elements(x)) {
      D[i] = x - x[i];
    }
    return D;
  }

  row_vector dydt(row_vector y, real I1, row_vector x0, matrix FC, real tau0) {
    int nn = num_elements(y)/2;
    row_vector[2*nn] y_dot;
    row_vector[nn] x = y[1:nn];
    row_vector[nn] z = y[nn+1:2*nn];
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    matrix[nn, nn] D = vector_differencing(x);
    row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D));
    y_dot[1:nn] = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    y_dot[nn+1:2*nn] = inv(tau0) * (4 * (x - x0) - z - gx);
    return y_dot;
  }

  row_vector y_step(row_vector y, real I1, row_vector x0, matrix FC, real tau0, real h) {
    //   h: Integration time step
    int nn = num_elements(y)/2;
    row_vector[2*nn] y_next;
    row_vector[2*nn] k1 = dydt(y, I1, x0, FC, tau0);
    row_vector[2*nn] k2 = dydt(y + h*(k1/2), I1, x0, FC, tau0);
    row_vector[2*nn] k3 = dydt(y + h*(k2/2), I1, x0, FC, tau0);
    row_vector[2*nn] k4 = dydt(y + h*k3, I1, x0, FC, tau0);
    y_next = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
    return y_next;
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
  // row_vector[nn] x_init = rep_row_vector(-2.0, nn);
  // row_vector[nn] z_init = rep_row_vector(3.5, nn);
}

parameters {
  row_vector[nn] x0;
  real<lower=0> alpha;
  real beta;
  real<lower=0> K;
  real<lower=5> tau0;
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  real<lower=0> eps_slp;
  real<lower=0> eps_snsr_pwr;
}

transformed parameters{   
  // RK4 integration of the 2D Epileptor
  row_vector[2*nn] y[nt];
  row_vector[ns] mu_slp[nt];
  row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns);
  for (t in 1:nt) {
    if(t == 1){
        y[t] = y_step(append_col(x_init, z_init), I1, x0, K*SC, tau0, time_step);
    }
    else{
        y[t] = y_step(y[t-1], I1, x0, K*SC, tau0, time_step);
    }
    mu_slp[t] = alpha * log(gain * exp(y[t,1:nn]'))' + beta;
    mu_snsr_pwr += mu_slp[t] .* mu_slp[t];
  }
  mu_snsr_pwr = mu_snsr_pwr / nt;
}

model {
  x0 ~ normal(x0_mu, 100.0);
  alpha ~ normal(1.0, 10.0);
  beta ~ normal(0, 10.0);
  tau0 ~ normal(20, 10.0);
  K ~ normal(1.0, 10.0);
  for (i in 1:nn){
    x_init[i] ~ normal(-2.0, 10.0);
    z_init[i] ~ normal(3.5, 10.0);
  }
  eps_slp ~ normal(1, 10);
  eps_snsr_pwr ~ normal(1, 10);
  for (t in 1:nt) {
    target += normal_lpdf(slp[t] | mu_slp[t], eps_slp);
  }
  target += normal_lpdf(snsr_pwr | mu_snsr_pwr, eps_snsr_pwr);
}

generated quantities {
}