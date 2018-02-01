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
  int ns;
  real I1;
  real tau0;
  vector[nn] Ic;
  matrix<lower=0.0, upper=1.0>[nn, nn] SC;
  real sigma;
  real k;

  // Parameters taken as input for simulation
  row_vector[nn] x0;    
  real time_scale;
  real time_step;
  // time-series state non-centering:
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  row_vector[nn] z_eta[nt - 1];
}

/* transformed data { */
/*     matrix[ns, nn] log_gain = log(gain); */
/*     matrix [nn, nn] SC_ = SC; */
/*     for (i in 1:nn) SC_[i, i] = 0; */
/*     SC_ /= max(SC_) * rows(SC_); */
/* } */

parameters {
}

model {
}

generated quantities {
  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
  row_vector[nn] xobs[nt];
  row_vector[nn] zobs[nt];
  real sigma_xobs = 0.5;
  real sigma_zobs = 0.5;
  
  x[1] = x_init - 1.5;
  z[1] = z_init + 2.0;
  for (t in 1:(nt-1)) {
    x[t+1] = x_step(x[t], z[t], I1, time_step, time_scale, sigma);
    z[t+1] = z_step(x[t], z[t], x0, k*SC, Ic, time_step, time_scale, z_eta[t], sigma, tau0);
  }

  for (t in 1:nt){
    for (i in 1:nn){
      xobs[t,i] = normal_rng(x[t,i],sigma_xobs);
      zobs[t,i] = normal_rng(z[t,i],sigma_zobs);
    }
  }
}
