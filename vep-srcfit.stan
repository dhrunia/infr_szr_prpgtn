functions {

  matrix vector_differencing(row_vector x) {
    matrix[num_elements(x), num_elements(x)] D;
    for (i in 1:num_elements(x)) {
      D[i] = x - x[i];
    }
    return D';
  }

  row_vector x_step(row_vector x, row_vector z, real I1, real time_step, real time_scale, real sigma) {
    int nn = num_elements(x);
    row_vector[nn] x_next;
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    x_next = x + (time_scale * time_step * dx);
    return x_next;
  }

  row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC, vector Ic, 
		    real time_step, real time_scale, row_vector z_eta, real sigma, real tau0) {
    int nn = num_elements(z);
    row_vector[nn] z_next;
    matrix[nn, nn] D = vector_differencing(x);
    row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D) - Ic .* to_vector(1.8 + x));
    row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
    z_next = z + (time_scale * time_step * dz) + sqrt(time_step) * sigma * z_eta;
    return z_next;
  }
}

data {
  int nn;
  int nt;
  real I1;
  real tau0;
  vector[nn] Ic;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;
  

  // All parameters except x0 are taken as input
  real time_scale;
  real time_step;
  real sigma;
  real k;
  real sigma_xobs;
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  row_vector[nn] z_eta[nt - 1];

  // modelled data
  row_vector[nn] xobs[nt];
}

/* transformed data { */
/*     matrix[ns, nn] log_gain = log(gain); */
/*     matrix [nn, nn] SC_ = SC; */
/*     for (i in 1:nn) SC_[i, i] = 0; */
/*     SC_ /= max(SC_) * rows(SC_); */
/* } */

parameters {
  row_vector[nn] x0;    
}

model {
  row_vector[nn] mu_xobs[nt];
  row_vector[nn] mu_zobs[nt];

  x0 ~ normal(-2.5,1);
  mu_xobs[1] = x_init - 1.5;
  mu_zobs[1] = z_init + 2.0;
  for (t in 1:(nt-1)) {
    mu_xobs[t+1] = x_step(mu_xobs[t], mu_zobs[t], I1, time_step, time_scale, sigma);
    mu_zobs[t+1] = z_step(mu_xobs[t], mu_zobs[t], x0, k*SC, Ic, time_step, time_scale, z_eta[t], sigma, tau0);
  }

  for (t in 1:nt){
      xobs[t] ~ normal(mu_xobs[t], sigma_xobs);
      //      zobs[t] ~ normal(mu_zobs[t], sigma_zobs);
  }
}

