# VEP

This repo contains files for inferring epileptic networks in intracranial
data.

## Contents

_Stan files_

- ODE formulation: [`vep-ode-rev-04.stan`](vep-ode-rev-04.stan)
- Linear stability analysis model: [`vep-lsa.stan`](vep-lsa.stan)
- ~~SDE formulation: [`sde.stan`](sde.stan)~~
- ~~Single channel for observation model: [`one.stan`](one.stan)~~

_Data_

- Data with new preprocessing and extra entities for interictal time series: [new-data.R.zip](https://github.com/maedoc/vep.stan/files/1521920/new-data.R.zip)

_Jupyter notebooks_

- [`old-workflow`](old-workflow.ipynb) - ...
- [`fwd-sim.ipynb`](fwd-sim.ipynb) - forward simulation code

## Priors

### SDE

e.g. `vep-fe-rev-05.stan`.

- `FC` - positive definite, w/ mode == SC
- `K` - <10, probably much less (or rescale SC/FC if problematic)
- `x0` - for healthy nodes, <-1.8, unhealthy nodes ~ -1.8
- `amplitude` - ~1, 
- `offset` - ~0
- `time_scale` - 0.025; no more than a 2/3 cycles for oscillator for given time window
- `x_init` - ~ -1.5
- `z_init` - ~ 2.0
- `epsilon` - ~0.5
- `sigma` - 0.1


## Origin

_Comments on VEP model origin_

Stan models based on

    The Virtual Epileptic Patient: individualized whole-brain models of epilepsy spread 
    Jirsa VK, Proix T, Perdikis D, Woodman MM, Wang H, Chauvel P, Gonzalez-Martinez J, Bernard C, Bénar C, Guye M, Bartolomei F 
    NeuroImage 2016

    http://www.sciencedirect.com/science/article/pii/S1053811916300891

from page 380, section "Data fitting"


    Obtaining estimates of the parameters of the network model, given the
    available functional data is performed within a Bayesian framework,
    using a reduced model and reduced set of functional data. First, we
    follow the 2D reduction of the Epileptor under the assumption that the
    slow variable dynamics sufficiently capture seizure propagation. The

in the code below, this slow variable 2D system is the x & z in the 
parameters section.

    sEEG data are windowed and Fourier transformed to obtain estimates of
    their spectral density over time, and the power above 10 Hz is summed
    to capture the temporal variation of the fast activity. These time

this enters as the seeg_log_power in the data section

    series are corrected to a preictal baseline, log-transformed and
    linearly detrended over the time window encompassing the seizure.

    Contacts are selected which present greater high-frequency activity
    than their neighbors on the same electrode. Given that, contrary to
    M/EEG, the sEEG lead field is highly sparse, three nodes per contact
    are used in the network model; we assume other nodes are not recruited
    and thus rest on a fixed point: the effect of these nodes with
    constant states on other nodes enters in the model through a constant
    sum over the corresponding elements of the structural connectivity

these are the Ic & SC data

    matrix. Next, we use an observation model that incorporates the sEEG
    forward solution described above, under the assumption that the x1
    variable describes fluctuations in the log power of high frequency
    activity, predicting sensor log power, with normally distributed
    observation error.  Uninformative priors are placed on the hidden
    states’ initial conditions, while their evolution follows an Euler-
    Maruyama discretization of the corresponding stochastic differential
    equations with linear additive noise. Uninformative priors are also
    placed on the excitability parameter per node x0, observation baseline

the x0 is the parameter which we are truly interested in

    power, scale and noise, and finally the length of the seizure is also
    allowed to freely vary to match that of a given recorded seizure.
    Structural connectivity specifies a gamma prior on the connectivity
    used in the generative method (discuss: serves as functional
    connectivity).

The choice of gamma is not informed by anatomy, at this point,
 but looked OK to my eye.

## Model structure

A few comments on the model structure:

### gamma parameters

I use scipy to visualize chosen gamma prior, which has different parametrization
than Stan, hence this transformation

```
K_a = gamma_a_from_u_v(K_u, K_v);
K_b = gamma_b_from_a_u(K_a, K_u);
K ~ gamma(K_a, K_b) T[0, 10];
```

### Effective or functional connectivity

FC/EC, effective connectivity is a neuroscience term for a
statistical estimator of causal influence, estimated from data.

The simplest FC, and most often used, is correlation, but here we are allowing
for an FC defined by this neural network model, using SC as a prior, with the
imperfect analogy that SC is the road, and FC is the traffic. 

```
    for (i in 1:nn) {
        for (j in 1:nn) {
            if (i>=j) {
                if ((i==j) || (SC[i, j]==0.0)) {
                    fc_u = 1e-6;
                    fc_v = 1e-3;
                } else {
                    fc_u = SC[i, j];
                    fc_v = fc_u * SC_var;
                }
                fc_a = gamma_a_from_u_v(fc_u, fc_v);
                fc_b = gamma_b_from_a_u(fc_a, fc_u);
                FC[i, j] ~ gamma(fc_a, fc_b);
                FC[j, i] ~ gamma(fc_a, fc_b);
            }
        }
    }
```

compute cooupling to all nodes j to this node i 
```
for (j in 1:nn) if (i!=j) gx = gx + FC[i, j]*(x[j, t] - x[i, t]);
```
compute derivatives
```
dx = 1.0 - x[i, t]*x[i, t]*x[i, t] - 2.0*x[i, t]*x[i, t] - z[i, t] + I1;
```
Euler Maruyama equivalent scheme
```
x[i, t+1] ~ normal(x[i, t] + dt*tt*dx, sig); // T[xlim[1], xlim[2]];
```
predict log power using `gain` matrix as obvservation model.
```
seeg_log_power[t] ~ normal(amp*(log(gain*exp(col(x, t))) + offset), eps);
```