functions {

}

data {
  int nn;
  int ns;
  int nt;
  matrix[ns,nn] gain;
  real epsilon;
  
  // Modelled data
  row_vector[ns] seeg[nt];
}

parameters {
  row_vector[nn] x[nt];
}

model {
  for (t in 1:nt)
    seeg[t] ~ normal((gain*x[t]')', epsilon);
}
