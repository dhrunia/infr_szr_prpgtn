functions {

}

data {
  int nn;
  int ns;
  int nt;
  matrix[ns,nn] gain;
  real epsilon;
  real sigma;
  
  // Modelled data
  row_vector[ns] seeg[nt];
}

parameters {
  row_vector[nn] x[nt];
  real offset;
  real alpha;
}

model {
  row_vector[ns] mu_seeg[nt];
  alpha ~ normal(0,1);
  offset ~ normal(0,1);
  for (t in 2:nt){
    x[t] ~ normal(alpha * x[t-1], sigma);
    mu_seeg[t] = log(gain*exp(x[t])')' + offset;
    seeg[t] ~ normal(mu_seeg[t], epsilon);
  }
}
