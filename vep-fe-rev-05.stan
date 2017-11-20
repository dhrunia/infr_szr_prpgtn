/* ODE form, based on rev04 w/ time & noise scale fixes */

functions {

    real gamma_a_from_u_v(real u, real v) {
        return (u*sqrt(u * u + 4 * v) + u * u + 2 * v) / (2 * v);
    }

    real gamma_b_from_a_u(real a, real u) {
        return (a - 1.0) / u;
    }

    vector vec_gx(matrix FC, vector Ic, vector x) {
        int m = rows(FC);
        vector[m] gx;
        for (i in 1:m)
            gx[i] = FC[i,] * (x - x[i]);
        return gx - Ic .* (1.8 + x);
    }

    real[] vep_net_ode(real time, real[] state, real[] theta, real[] x_r, int[] x_i) {
        int nn = x_i[1];
        vector[nn] x = to_vector(state[1:nn]);
        vector[nn] x2 = square(x);
        vector[nn] z = to_vector(state[nn+1:2*nn]);
        real I1 = x_r[1];
        real tt = x_r[2];
        real tau0 = x_r[3];
        vector[nn] Ic = to_vector(x_r[4:3 + nn]);
        // inverse of vep_net_ode_theta
        real K = theta[1];
        real time_scale = theta[2];
        vector[nn] x0 = to_vector(theta[3:2+nn]);
        matrix[nn, nn] FC = to_matrix(theta[3+nn:2+nn*(nn + 1)], nn, nn);
        // derivatives
        vector[nn] dx = (I1 + 1.0) - x2 .* x - 2.0 * x2 - z;
        vector[nn] dz = inv(tau0) * (4 * (x - x0) - z - K * vec_gx(FC, Ic, x));
        // pack
        return to_array_1d(tt * time_scale * append_row(dx, dz));
    }

    real[] vep_net_ode_theta(int dbg, real K, real time_scale, vector x0, matrix FC)  {
        int nn = rows(FC);
        real theta[2+nn*(nn + 1)];
        theta[1] = K;
        theta[2] = time_scale;
        theta[3:2+nn] = to_array_1d(x0);
        theta[3+nn:2+nn*(nn + 1)] = to_array_1d(FC);
        if (dbg) for (i in 1:size(theta)) print("vep_net_ode_theta[",i,"] = ", theta[i]);
        return theta;
    }

    real[] vep_net_ode_x_r(real I1, real tt, real tau0, vector Ic) {
        int nn = rows(Ic);
        real x_r[3 + nn];
        x_r[1] = I1;
        x_r[2] = tt;
        x_r[3] = tau0;
        x_r[4:3 + nn] = to_array_1d(Ic);
        return x_r;
    }

    int[] vep_net_ode_x_i(int nn) {
        int x_i[1];
        x_i[1] = nn;
        return x_i;
    }

    matrix[] integrate_vep_net_ode(
            int dbg,
            real[] ode_times,
            matrix FC, real K, vector x0, real I1, real tt, real time_scale,
            real tau0, vector Ic,
            vector x_init, vector z_init)
    {
        int nn = rows(FC);
        int nt = size(ode_times) + 1;
        matrix[nn, nt] sol[2];
        real sol_[nt, 2 * nn];
        sol_[1,] = to_array_1d(append_row(x_init, z_init));
        sol_[2:nt,] = integrate_ode_rk45(
            vep_net_ode,
            sol_[1,],
            0.0,
            ode_times,
            vep_net_ode_theta(dbg, K, time_scale, x0, FC),
            vep_net_ode_x_r(I1, tt, tau0, Ic),
            vep_net_ode_x_i(nn));
        sol[1] = to_matrix(sol_[:, 1:nn])';
        sol[2] = to_matrix(sol_[:, nn+1:2*nn])';
        return sol;
    }
}

data {
    int nn;
    int nt;
    int ns;
    real I1;
    real tau0;
    real dt;
    matrix[ns, nn] gain;
    matrix[nt, ns] seeg_log_power;
    vector[nn] Ic;
    matrix<lower=0.0>[nn, nn] SC;
    real SC_var; // prior on variance of connectivity strengths
    real K_lo;
    real K_u;
    real K_v;
    real x0_lo;
    real x0_hi;
    real eps_hi;
    real sig_hi;
    real zlim[2];
    real tt;
    int use_data;
    int dbg;
}

transformed data {
    matrix[ns, nn] log_gain = log(gain);
    real sigma_hi = sig_hi;
    real x0_range = x0_hi - x0_lo;
    real x0_prior_sd = x0_range/4;
    real x0_prior_mean = x0_lo + 0.5 * x0_range;
    real K_a = gamma_a_from_u_v(K_u, K_v);
    real K_b = gamma_b_from_a_u(K_a, K_u);
    real ode_times[nt-1];

    for (t in 1:(nt-1))
        ode_times[t] = t*dt;
}

parameters {

    // functional connectivity
    matrix<lower=0.0>[nn, nn] FC;

    // integrate and predict
    real<lower=0> K;
    vector[nn] x0;
    real epsilon_star;
    real amplitude;
    real offset;
    real<lower=0.0, upper=1.0> time_scale;

    // time-series state non-centering:
    vector[nn] x_init;
    vector[nn] z_init;
}

transformed parameters {
    matrix[nn, nt] xz[2];
    real epsilon = 0.1 * exp(0.2 * epsilon_star);

    xz = integrate_vep_net_ode(
        dbg, ode_times,
        FC, K, x0, I1, tt, time_scale, tau0, Ic,
        x_init - 1.8, z_init + 3.17
    ); // nb transpose here
}

model {

    real fc_a;
    real fc_b;

    x0 ~ normal(-1.7, 1.0);
    epsilon_star ~ normal(0, 1);
    amplitude ~ normal(1.0, 0.5);
    amplitude ~ normal(-1.0, 0.5);
    offset ~ normal(1.8125, 0.5);

    x_init ~ normal(0, 1);
    z_init ~ normal(0, 1);

    K ~ gamma(4, 1.5); // thin lower + upper tail, cover about 0-10

    /* functional connectivity */
    for (i in 1:nn) {
        for (j in 1:nn) {
            if (i>=j) {
                real fc_u;
                real fc_v;
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

    if (use_data==1)
        to_vector(seeg_log_power) ~ normal(amplitude * (to_vector(log(gain * exp(xz[1]))) + offset), epsilon);
}
