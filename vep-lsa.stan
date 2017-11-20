/* VEP model using stability analysis to predict recruitment */

functions {
    /* F(z) is the nonlinearity used in the 1D reduction of the Epileptor */
    real F(real z) {
        return 1.0/4.0*(-16.0/3.0-sqrt(8*z-629.6/27.0));
    }

    /* vectorized form of F(z) */
    vector Fv(vector z) {
        vector[num_elements(z)] out;
        for (i in 1:num_elements(z))
            out[i] <- F(z[i]);
        return out;
    }

    /* maps a disc sys eig val to prob of destabilizing */
    vector destable(vector decay) {
        vector[num_elements(decay)] dest;

        for (i in 1:num_elements(decay))
            dest[i] <- 1.0/(1.0 + exp(-decay[i]^2*2)) * 2.0 - 1.0;

        return dest;
    }

    /* calculate per-node afferent coupling term */
    vector coupling(vector z, matrix W) {
        vector[num_elements(z)] aff;
        for (i in 1:num_elements(z)) {
            aff[i] <- 0.0;
            for (j in 1:num_elements(z))
                aff[i] <- aff[i] + W[i, j] * (F(z[j]) - F(z[i]));
        }
        return aff;
    }

    /* calculate x0 from z fixed point

        dz/dt = 1/tau0 (4(F(z) - x0) - z - I) == 0

        F(z) - x0  = (z + I)/4
      
        x0  = F(z) - (z + I)/4

    where I is known because we have the fixed points, i.e.

        I = sum W_ij (F(z_j) - F(z_i))
    */
    vector x0_of_z0(vector z0, matrix W) {
        return Fv(z0) - (z0 + coupling(z0, W)) * 0.25;
    }

    /* rescale connectivity */
    matrix rescale_connectivity(matrix W, int[] in_mask, real in_coef, real out_coef) {
        matrix[rows(W), cols(W)] W_;
        real temp;
        for (i in 1:rows(W)) {
            for (j in 1:cols(W)) {
                if (in_mask[i] || in_mask[j]) {
                    temp <- in_coef;
                } else {
                    temp <- out_coef;
                }
                W_[i, j] <- temp * W[i, j];
            }
        }
        return W_;
    }
}

data  {
    /* dimensions: number of nodes & number of seeg channels */
    int nn;
    // int n_seeg;

    /* arrays of 0 & 1, where 1 indicates node has special quality, hyp or ez */
    int node_is_hyp[n_node];

    real tau0;  /* time scale separation */
    real dt;    /* time step size */
    real eps;   /* perturbation size */
    real z0_hi; /* upper limit on z0 */

    /* patient's lead field & possibly non-normalized connectivity */
    // matrix[n_seeg, n_node] seeg_gain;
    matrix[n_node, n_node] connectivity;

    /* observation */
    int recruited[n_node];

    int diagnose;
}

transformed data {
    real z_crit;
    matrix[n_node, n_node] norm_conn; /* normalized connectivity */

    norm_conn <- connectivity;
    for (i in 1:n_node) 
        norm_conn[i, i] <- 0.0;
    norm_conn <- norm_conn / max(norm_conn);

    /* critical z value, below this is unstable */
    z_crit <- (629.6 / 27.0) / 8.0;
}

parameters {
    /* coupling scaling for normal & hypothalamic connections */
    real<lower=0, upper=50> K;
    real<lower=0, upper=20> Khyp;

    /* z fixed point */
    vector<lower=z_crit, upper=z0_hi>[n_node] z0;  
}

transformed parameters {
    /* excitability from fixed point */
    vector[n_node] x0;
    x0 <- x0_of_z0(z0, rescale_connectivity(norm_conn, node_is_hyp, Khyp, K));
}

model {
    vector[n_node] z1;          /* perturbation from fixed point */
    vector[n_node] z2;          /* x1 after one step in time */
    vector[n_node] decay;       /* decay of perturbation */

    /* perturb state*/
    z1 <- z0 + eps * (z_crit - z0);

    /* step once in time */
    z2 <- z1 + dt * ((1.0/tau0) * (4.0*(Fv(z1) - x0) - z1 
        - coupling(z1, rescale_connectivity(norm_conn, node_is_hyp, Khyp, K))));

    /* compute perturbation decay */
    decay <- (z2 - z0) ./ (z1 - z0);

    /* diagnostic output */
    if (diagnose) {
        print("model diagnostics");
        print("Khyp = ", Khyp, ";");
        print("K = ", K, ";");
        print("x0 = ", x0, ";");
        print("z0 = ", z0, ";");
        print("z1 = ", z1, ";");
        print("z2 = ", z2, ";");
        print("decay = ", decay, ";");
        print("destable = ", destable(decay), ";");
        print("lp = ", bernoulli_log(recruited, destable(decay)));
    }

    /* predict recruitment based on node decay */
    recruited ~ bernoulli(destable(decay));
}
