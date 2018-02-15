functions {

}

data {
  int nn;
  int ns;
  int nt;
  matrix[ns,nn] gain;

  // Some Parameters are fixed for now
  real epsilon;
  real amplitude;
  real offset;
  
  // Modelled data
  row_vector seeg_log_power[ns,nt];
}

parameters {
  matrix<lower=-3.0,upper=1.0>[nn, nt] x;
}

model {
  matrix[ns,nt] mu_seeg_log_power;
  mu_seeg_log_power = amplitude * (log(gain * exp(x)) + offset);
  seeg_log_power[t] ~ normal(mu_seeg_log_power, epsilon);
}

generated quantities {
}
