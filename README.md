# VEP BENCHMARKING

This repo contains files to identify spatio-temporal seizure propagation patterns epileptic
using SEEG data

## Contents

_Model files_

- [szr_prpgtn.stan](szr_prpgtn.stan): A probabilistic model of SEEG log. power using 2D epileptor as the prior
  on source power profile

_Fitting_
- [vep-fit-syndata.ipynb](vep-fit-syndata.ipynb): Fitting synthetic data
- [vep-fit-retrodata.ipynb](vep-fit-retrodata.ipynb): Fitting retrospective data
