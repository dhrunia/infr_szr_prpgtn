# VEP BENCHMARKING

This repo contains files for benchmarking parameter inference using stan to identify epileptic networks
given SEEG data

## Contents

_Stan files_

- SDE model: vep-snsrfit.stan

_Data_

Two synthetic datasets are generated for benchmarking:
1. Dataset 1(id001_ac) consists of one EZ (epileptogenic zone) and two PZ (propagation zone)
2. Dataset 2(id001_cj) consists of two EZ and three PZ of which propagation is solely due to connectivity to one region

_Jupyter notebooks_

- [TVB\_forward\_sim.ipynb](TVB_forward_sim.ipynb) - Simulate SEEG data using 6D epileptor

## Benchmarking

Start with bechmarking hyperparameter _sigma_ and add more dimensions to benchmarking incrementally
