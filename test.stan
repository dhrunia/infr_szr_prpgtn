data{
  int nn;
  int nt;
  row_vector[nn] zeta[nt];
}

parameters{

}

model{

}

generated quantities{
  for (t in 1:nt)
    print(sum(zeta[t]))
}
