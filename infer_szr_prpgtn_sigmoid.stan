data {
  int nn;
  /* int ns; */
  int nt;
  /* matrix[ns,nn] gain; */

  // Modelled data
  /* row_vector[ns] slp[nt]; //seeg log power */
  /* row_vector[ns] snsr_pwr; //seeg sensor power */
  vector[nt] x;
}

transformed data{
}

parameters{
  real lambda_on;
  real lambda_off;
  real t_on;
  real t_off;
  real alpha;
  real beta;
  real dt;
  real eps_slp;
}

transformed parameters{
  vector[nt] ts = rep_vector(0, nt);
  vector[nt] x_mu;
  for (t_i in 2:nt)
    ts[t_i] = ts[t_i-1] + dt;
  
  x_mu = alpha*(inv(1 + exp(-lambda_on*ts + t_on)).*inv(1+exp(lambda_off*ts - t_off))) + beta;
}

model{
  lambda_on ~ lognormal(1,1);
  lambda_off ~ lognormal(1,1);
  t_on ~ lognormal(0, 1);
  t_off ~ lognormal(0,1);
  alpha ~ lognormal(4, 1);
  beta ~ normal(-50, 5);
  dt ~ lognormal(0, 1);
  eps_slp ~ lognormal(1, 1);
  x ~ normal(x_mu, eps_slp);
}
