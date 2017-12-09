functions {
  vector unpack_xfp(int n, real[] p) {
    vector[n] xfp;
    if (size(p) == (2 + 2*n))
      xfp = to_vector(p[2+n:2+2*n]);
    else
      xfp = rep_vector(-1.8, n);
    return xfp;
  }
  
  real[] ode(real t, real[] y, real[] p, real[] r, int[] i) {
    int n = size(y) / 3;

    vector[n] x = to_vector(y[1:n]);
    vector[n] z = to_vector(y[n+1:2*n]);

    real T = r[1];
    real tau = r[2];
    matrix[n,n] w = to_matrix(r[3:n*n+2], n, n);
    
    real k = p[1];
    vector[n] x0 = to_vector(p[2:1+n]);
    vector[n] xfp = unpack_xfp(n, p);
    
    vector[n] gx = rows_dot_product(w, rep_matrix(x, n) - rep_matrix(x, n));
    vector[n] dx = 3.1 - x .* x .* x - 2 * x .* x - z;
    vector[n] dz = (4 * (x - x0) - k * gx - z) / tau;
    vector[n] de = (T - t) * (x - xfp);

    return to_array_1d(append_row(append_row(dx, dz), de));
  }

  real[] ode_r(real T, real tau, matrix w) {
    real r[2+num_elements(w)];
    r[1] = T;
    r[2] = tau;
    r[3:num_elements(w)+2] = to_array_1d(w);
    return r;
  }

  int[] ode_i() {
    int i[0];
    return i;
  }

  real[] ode_t(real T) {
    return {T};
  }

  real[] ode_p(real k, vector x0, vector xz0) {
    real p[rows(x0) + 1 + rows(xz0)];
    p[1] = k;
    p[2:1+rows(x0)] = to_array_1d(x0);
    if (rows(xz0) > 0)
      p[2+rows(x0):1+rows(x0)+rows(xz0)] = to_array_1d(xz0);
    return p;
  }
  
  // wraps ode for use w/ algebra_system
  vector ode_0(vector y, vector p, real[] r, int[] i) {
    int n = rows(y) / 2;
    real y_[3*n];
    real p_[rows(p)] = to_array_1d(p);
    y_[1:2*n] = to_array_1d(y);
    y_[2*n+1:3*n] = rep_array(0, n);
    return to_vector(ode(0.0, y_, p_, r, i)[1:n*2]);
  }

  vector fixed_points_guess(int n) {
    vector[2*n] y0;
    real xi = -1.9;
    real zi = 3.1 - xi .* xi .* xi - 2 * xi .* xi;
    y0[1:n] = rep_vector(xi, n);
    y0[n+1:2*n] = rep_vector(zi, n);
    return y0;
  }

  vector fixed_points(matrix w, real tau, real[] p) {
    return algebra_solver(ode_0, fixed_points_guess(rows(w)),
			  to_vector(p), ode_r(0.0, tau, w), ode_i());
  }
  
  vector predict_ei(matrix w, vector ic_xz, vector x0, real k, real T, real tau, vector xz0) {
    int n = rows(w);
    real sol[1, 3*n];
    real ic[3 * n];
    real ei[n];

    ic[1:2*n] = to_array_1d(ic_xz);
    ic[2*n+1:3*n] = rep_array(0, n);

    sol = integrate_ode_rk45(ode, ic, 0.0, ode_t(T),
			     ode_p(k, x0, xz0), ode_r(T, tau, w), ode_i());

    ei = sol[1, 2*n+1:3*n];
    for (i in 1:n)
      if (ei[i] < 0)
	ei[i] = 0.0;
    // TODO should be able to normalize 
    return to_vector(ei) / max(ei);
  }

  real[] ode_t_full(real T, int ns) {
    real t[ns];
    for (i in 1:ns)
      t[i] = T/ns * i;
    return t;
  }

  real[,] full_sol(matrix w, vector ic_xz, vector x0, real k, real T, real tau, vector xz0, int ns) {
    int n = rows(w);
    real sol[ns, 3*n];
    real ic[3 * n];
    real ei[n];

    ic[1:2*n] = to_array_1d(ic_xz);
    ic[2*n+1:3*n] = rep_array(0, n);

    sol = integrate_ode_rk45(ode, ic, 0.0, ode_t_full(T, ns),
			     ode_p(k, x0, xz0), ode_r(T, tau, w), ode_i());
    return sol;
  }
}

data {
  int n;
  int ns;
  real T;
  real tau;
  matrix[n, n] w;
  vector[n] ei;
}

parameters {
  vector[n * 2] ic_xz;
  vector[n] x0_;
  real k_;
}

transformed parameters {
  real k = 0.01 * exp(0.1*k_) / n;
  // x0_ inits between -2 2 but that range is problematic here
  vector[n] x0 = x0_/4 - 4;
  vector[n * 2] xz0 = fixed_points(w, tau, ode_p(k, x0, rep_vector(0,0)));
  vector[n] eih = predict_ei(w, ic_xz, x0, k, T, tau, xz0);
}

model {
  ic_xz[1:n] ~ normal(-1.5, 0.1);
  ic_xz[n:2*n] ~ normal(2.0, 0.1);
  k_ ~ normal(0, 1);
  x0_ ~ normal(0, 1);
  ei ~ normal(eih, 0.1);
  // but, need fixed points to be stable
  for (i in 1:n)
    if (xz0[i] > -1.8)
      reject("x fp ", i, " > -1.8 = ", xz0[i], "(x0=", x0[i], ", k=", k, ")");
  target += normal_lpdf(log(-(xz0[1:n] + 1.8)) | 0, 0.5);
}

generated quantities {
  real xze[ns, 3*n];
  if (ns > 0)
    xze = full_sol(w, ic_xz, x0, k, T, tau, xz0, ns);
}
