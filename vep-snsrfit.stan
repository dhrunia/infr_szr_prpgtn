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
  real sigma;

  // Modelled data
  row_vector[ns] seeg[nt];
}

parameters {
  row_vector[nn] x0_star;
  row_vector<lower=-3.0, upper=3.0>[nn] x[nt];
  row_vector<lower=1.0, upper=5.0>[nn] z[nt];
  row_vector<lower=0.0>[ns] amplitude;
  row_vector<lower=0.0>[ns] offset;
  real<lower=0.0> time_step;
  real<lower=0.0> tau0;
  real<lower=0.0> K;
  real<lower=0.0> epsilon;
  matrix<lower=0.0, upper=1.0>[nn, nn] FC;
}

transformed parameters{
  row_vector[nn] x0 = -2.5 + x0_star;
}

model {
  row_vector[ns] mu_seeg[nt];

  x0_star ~ normal(0, 1.0);
  amplitude ~ normal(1.0, 1.0);
  epsilon ~ normal(0.1, 1.0);
  time_step ~ normal(0.1, 1.0);
  tau0 ~ normal(30, 1.0);
  for (i in 1:nn){
    for (j in 1:nn){
      FC[i,j] ~ normal(SC[i,j], 1.0);
    }
  }
  x[1] ~ normal(-2.0, sigma);
  z[1] ~ normal(3.0, sigma);
  mu_seeg[1] = (gain * x[1]')';
  for (t in 1:(nt-1)) {
    x[t+1] ~ normal(x_step(x[t], z[t], I1, time_step), sigma);
    z[t+1] ~ normal(z_step(x[t], z[t], x0, K*FC, time_step, tau0), sigma);
    mu_seeg[t+1] = (gain * x[t]')';
  }
  for (t in 1:nt){
    seeg[t] ~ normal(amplitude .* (mu_seeg[t] + offset), epsilon);
  }
}

generated quantities {
}
