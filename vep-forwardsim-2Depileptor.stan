functions {

  matrix vector_differencing(row_vector x) {
    matrix[num_elements(x), num_elements(x)] D;
    for (i in 1:num_elements(x)) {
      D[i] = x - x[i];
    }
    return D;
  }

  row_vector x_step(row_vector x, row_vector z, real I1, real time_step, real time_scale, real sigma) {
    int nn = num_elements(x);
    row_vector[nn] x_next;
    row_vector[nn] I1_vec = rep_row_vector(I1 + 1.0, nn);
    row_vector[nn] dx = I1_vec - (x .* x .* x) - 2.0 * (x .* x) - z;
    x_next = x + (time_scale * time_step * dx);
    return x_next;
  }

  row_vector z_step(row_vector x, row_vector z, row_vector x0, matrix FC, 
		    real time_step, real time_scale, row_vector z_eta, real sigma, real tau0) {
    int nn = num_elements(z);
    row_vector[nn] z_next;
    matrix[nn, nn] D = vector_differencing(x);
    row_vector[nn] gx = to_row_vector(rows_dot_product(FC, D));
    row_vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - gx);
    z_next = z + (time_scale * time_step * dz) + sqrt(time_step) * sigma * z_eta;
    return z_next;
  }
}

data {
  int nn;
  int ns;
  int nt;
  real I1;
  real tau0;
  matrix[nn, nn] SC;
  matrix[ns,nn] gain;

  // Parameters taken as input for simulation
  row_vector[nn] x0;    
  real time_scale;
  real time_step;
  int nsteps;
  real sigma;
  real k;
  real epsilon;
  
  // time-series state non-centering:
  row_vector[nn] x_init;
  row_vector[nn] z_init;
  row_vector[nn] z_eta[nt];
}

parameters {
}

model {
}

generated quantities {
  row_vector[nn] x_t;
  row_vector[nn] z_t;
  row_vector[nn] x[nt];
  row_vector[nn] z[nt];
  row_vector[ns] seeg[nt];

  x_t = x_init;
  z_t = z_init;
  for (t in 1:nt) {
    for(i in 1:nsteps){
      x_t = x_step(x_t, z_t, I1, time_step, time_scale, sigma);
      z_t = z_step(x_t, z_t, x0, k*SC, time_step, time_scale, z_eta[t], sigma, tau0);
    }
    x[t] = x_t;
    z[t] = z_t;
    seeg[t] = (gain * x[t]')';
  }
}
