data {
  int nn;
  /* int ns; */
  int nt;
  /* matrix[ns,nn] gain; */

  // Modelled data
  /* row_vector[ns] slp[nt]; //seeg log power */
  /* row_vector[ns] snsr_pwr; //seeg sensor power */
  matrix[nt, nn] x;
}

transformed data{
  row_vector[nn] eps_slp = rep_row_vector(1.0, nn);
}

parameters{
  row_vector[nn] lambda_on;
  row_vector[nn] lambda_off;
  row_vector[nn] t_on;
  row_vector[nn] t_off;
  row_vector[nn] alpha;
  row_vector[nn] beta;
  row_vector[nn] dt;
  /* row_vector[nn] eps_slp; */
}

transformed parameters{
  matrix[nt,nn] ts = rep_matrix(0, nt, nn);
  matrix[nt,nn] x_mu;
  
  for (t_i in 2:nt)
    ts[t_i] = ts[t_i-1] + dt;
  
  x_mu = rep_matrix(alpha, nt).*(inv(1 + exp(-rep_matrix(lambda_on, nt).*ts + rep_matrix(t_on, nt))).*inv(1+exp(rep_matrix(lambda_off, nt).*ts - rep_matrix(t_off, nt)))) + rep_matrix(beta, nt);
}

model{
  lambda_on ~ lognormal(1,1);
  lambda_off ~ lognormal(1,1);
  t_on ~ lognormal(5, 1);
  t_off ~ lognormal(5,1);
  alpha ~ lognormal(4, 1);
  beta ~ normal(-50, 5);
  dt ~ lognormal(0, 1);
  /* eps_slp ~ lognormal(0, 0.1); */
  for (t_i in 1:nt)
    x[t_i] ~ normal(x_mu[t_i], eps_slp);
}
