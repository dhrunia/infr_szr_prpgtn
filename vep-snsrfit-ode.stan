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
  matrix[ns,nn] gain;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;

  // Modelled data
  row_vector[ns] seeg[nt];
}

parameters {
  row_vector[nn] x0_star;
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  row_vector<lower=0.0>[ns] amplitude;
  row_vector<lower=0.0>[ns] offset;
  real<lower=0.0> time_step;
  real<lower=0.0> tau0;
  real<lower=0.0> K;
  real<lower=0.0> epsilon;
  matrix<lower=0.0, upper=1.0>[nn, nn] FC;
}

transformed parameters{
  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
  row_vector[ns] mu_seeg[nt];
  row_vector[nn] x0 = -2.5 + x0_star;

  for (t in 1:nt) {
    if(t == 1){
      x[t] = x_step(x_init, z_init, I1, time_step);
      z[t] = z_step(x_init, z_init, x0, K*FC, time_step, tau0);
    }
    else{
      x[t] = x_step(x[t-1], z[t-1], I1, time_step);
      z[t] = z_step(x[t-1], z[t-1], x0, K*FC, time_step, tau0);
    }
    mu_seeg[t] = (gain * x[t]')';
  }
}

model {
  x0_star ~ normal(0, 1.0);
  amplitude ~ normal(1.0, 1.0);
  epsilon ~ normal(0.1, 1.0);
  time_step ~ normal(0.1, 1.0);
  tau0 ~ normal(30, 10.0);
  for (i in 1:nn){
    for (j in 1:nn){
      FC[i,j] ~ normal(SC[i,j], 0.01);
    }
  }
  x_init ~ normal(-2.0, 0.5);
  z_init ~ normal(3.0, 0.5);
  for (t in 1:nt){
    seeg[t] ~ normal(amplitude .* (mu_seeg[t] + offset), epsilon);
  }
}

generated quantities {
}
