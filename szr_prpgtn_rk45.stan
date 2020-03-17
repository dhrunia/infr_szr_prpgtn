functions {
  real[] epileptor_2D(real t, real[] y_t, real[] theta, real[] data_r, int[] data_i){
    int nn = data_i[1];
    matrix[nn,nn] SC = to_matrix(data_r[1:nn*nn], nn, nn);
    real I1 = data_r[nn*nn + 1];
    real tau0 = theta[1];
    real K = theta[2];
    row_vector[nn] x0 = to_row_vector(theta[3:3+nn-1]);
    row_vector[nn] x = to_row_vector(y_t[1:nn]);
    row_vector[nn] z = to_row_vector(y_t[nn+1:2*nn]);
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    matrix[nn, nn] D;
    row_vector[nn] gx;
    row_vector[nn] dx;
    row_vector[nn] dz;
    real dydt[2*nn];

    for (i in 1:nn) {
      D[i] = x - x[i];
    }
    gx = to_row_vector(rows_dot_product(K*SC, D));
    dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    dz = inv(tau0) * (4 * (x - x0) - z - gx);
    dydt[1:nn] = to_array_1d(dx);
    dydt[nn+1:2*nn] = to_array_1d(dz);
    /* print("tau0=",tau0, "K=", K, "I1=", I1, "x0=", x0); */
    /* print("x=", x); */
    /* print("z=", z); */
    return dydt;
  }
}

data {
  int nn;
  int ns;
  int nt;
  real ts[nt];
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
  real t0 = 0;
  int data_i[1];
  real data_r[nn*nn + 1];
  data_i[1] = nn;
  data_r[1:nn*nn] = to_array_1d(SC);
  data_r[nn*nn + 1] = I1;
}

parameters {
  row_vector<upper=-1.5>[nn] x_init;
  row_vector<lower=2.75>[nn] z_init;
  row_vector[nn] x0;
  real<lower=0> alpha;
  real beta;
  real<lower=0> K;
  real<lower=20> tau0;
  real<lower=0> eps_slp;
  real<lower=0> eps_snsr_pwr;
}

transformed parameters{   
  // RK45 integration of the 2D Epileptor
  real theta[nn+2];
  real y_init[2*nn];
  real y[nt,2*nn];
  
  theta[1] = tau0;
  theta[2] = K;
  theta[3:3+nn-1] = to_array_1d(x0);
  y_init[1:nn] = to_array_1d(x_init);
  y_init[nn+1:2*nn] = to_array_1d(z_init);
  y = integrate_ode_rk45(epileptor_2D, y_init, t0, ts, theta, data_r, data_i);
}

model {

  row_vector[ns] mu_slp[nt];
  row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns);
  row_vector[nn] x_t;
  row_vector[nn] z_t;

  x_init ~ normal(-2.0, 10);
  z_init ~ normal(3.5, 10);
  x0 ~ normal(x0_mu, 1.0);
  alpha ~ normal(1.0, 10.0);
  beta ~ normal(0, 10.0);
  tau0 ~ normal(20, 10.0);
  K ~ normal(1.0, 10.0);
  eps_slp ~ normal(1, 10);
  eps_snsr_pwr ~ normal(1, 10);

  for (t in 1:nt) {
    x_t = to_row_vector(y[t,1:nn]);
    z_t = to_row_vector(y[t,nn+1:2*nn]);
    mu_slp[t] = alpha * (log(gain * exp(x_t)')') + beta;
    mu_snsr_pwr += mu_slp[t] .* mu_slp[t];
    /* print("tau0=",tau0,"K=",K,"x0=",x0,"x_t=",x_t,"z_t=",z_t,"x_init=",x_init,"z_init=",z_init); */
    target += normal_lpdf(slp[t] | mu_slp[t], eps_slp);
  }
  mu_snsr_pwr = mu_snsr_pwr / nt;
  target += normal_lpdf(snsr_pwr | mu_snsr_pwr, eps_snsr_pwr);
}

generated quantities {
  /* row_vector[nn] x[nt]; */
  /* row_vector[nn] z[nt]; */
  /* row_vector[ns] mu_slp[nt]; */
  /* row_vector[ns] mu_snsr_pwr = rep_row_vector(0, ns); */

  /* /\* print("y_init=", y_init); *\/ */
  /* /\* print("tau0=",tau0,"K=",K,"x0=",x0,"x_init=",x_init,"z_init=",z_init,"y_init=", y_init, "t0=",t0,"ts=",ts); *\/ */
  
  /* for (t in 1:nt) { */
  /*   /\* if(t == 1) *\/ */
  /*   /\*   y[t] = to_array_1d(to_vector(y_init) + 0.1*to_vector(epileptor_2D(1, y_init, theta, data_r, data_i))); *\/ */
  /*   /\* else *\/ */
  /*   /\*   y[t] = to_array_1d(to_vector(y[t-1]) + 0.1*to_vector(epileptor_2D(1, y[t-1], theta, data_r, data_i))); *\/ */
  /*   x[t] = to_row_vector(y[t,1:nn]); */
  /*   z[t] = to_row_vector(y[t,nn+1:2*nn]); */
  /*   /\* print("t=",t); *\/ */
  /*   /\* print("x=",x[t]); *\/ */
  /*   /\* print("z=",z[t]); *\/ */
  /*   mu_slp[t] = alpha * (log(gain * exp(x[t])')') + beta; */
  /*   mu_snsr_pwr += mu_slp[t] .* mu_slp[t]; */
  /* } */
  /* mu_snsr_pwr = mu_snsr_pwr / nt; */
}
