# VEP Debugging

This repo contains files to identify spatio-temporal seizure propagation patterns in epilepsy
using SEEG data

## Contents

_Model files_

- [vep.stan](vep.stan): A probabilistic model of SEEG log. power using 2D epileptor as the prior
  on source power profile

_Fitting_

- [vep-ode-hmc.ipynb](vep-ode-hmc.ipynb): Inference with HMC
- [vep-ode-optim.ipynb](vep-ode-optim.ipynb): Inference using MAP
