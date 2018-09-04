data{
  int ns;
  int nt;
  row_vector[ns] slp_sim_ground_truth[nt];
  row_vector[ns] slp_true[nt];
}
parameters{
  real amplitude_star;
  real offset;
}
transformed parameters{
  real amplitude = exp(pow(0.5, 2) + log(1.0) + 0.5*amplitude_star);
}
model{
  amplitude_star ~ normal(0, 1.0);
  offset ~ normal(0, 1.0);
  for (t in 1:nt)
    slp_true[t] ~ normal(amplitude * (slp_sim_ground_truth[t] + offset), 0.1);
}
