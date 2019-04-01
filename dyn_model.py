import numpy as np
# import theano


class Epileptor_2D:
    def dx(self, x, z, I1):
        dx_eval = 1 - x**3 - 2 * x**2 - z + I1
        return dx_eval

    def dz(self, x, z, K, SC, tau0, x0):
        nn = x.size
        x_diff = np.repeat(x[:, np.newaxis], nn, axis=1) - x
        gx = K * SC * x_diff.T
        dz_eval = (1 / tau0) * (4 * (x - x0) - z - gx.sum(axis=1))
        return dz_eval

    def sim(self, params):
        x = np.zeros([params['nt'], params['nn']])
        z = np.zeros([params['nt'], params['nn']])
        for t in range(params['nt']):
            if (t == 0):
                x[t] = params['x_init'] + params['time_step'] * self.dx(
                    params['x_init'], params['z_init'], params['I1'])
                z[t] = params['z_init'] + params['time_step'] * self.dz(
                    params['x_init'], params['z_init'], params['K'],
                    params['SC'], params['tau0'], params['x0'])
            else:
                x[t] = x[t - 1] + params['time_step'] * self.dx(
                    x[t - 1], z[t - 1], params['I1'])
                z[t] = z[t - 1] + params['time_step'] * self.dz(
                    x[t - 1], z[t - 1], params['K'], params['SC'],
                    params['tau0'], params['x0'])
        return {'x': x, 'z': z}


# class Epileptor_2D:
#     def __init__(self, nt):
#         self.dt = theano.tensor.dscalar('dt')
#         self.x_init = theano.tensor.dvector('x_init')
#         self.z_init = theano.tensor.dvector('z_init')
#         self.SC = theano.tensor.dmatrix('SC')
#         self.I1 = theano.tensor.dscalar('I1')
#         self.K = theano.tensor.dscalar('K')
#         self.x0 = theano.tensor.dvector('x0')
#         self.tau0 = theano.tensor.dscalar('tau0')
#         self.nn = theano.tensor.iscalar('nn')
#         self.nt = nt
#         self.output, self.updates = theano.scan(
#             fn=self.step,
#             outputs_info=[self.x_init, self.z_init],
#             non_sequences=[
#                 self.dt, self.SC, self.K, self.x0, self.I1, self.tau0, self.nn
#             ],
#             n_steps=self.nt)
#         self.x = self.output[0]
#         self.z = self.output[1]
#         self.f = theano.function(
#             inputs=[
#                 self.x_init, self.z_init, self.dt, self.SC, self.K, self.x0,
#                 self.I1, self.tau0, self.nn
#             ],
#             outputs=[self.x, self.z],
#             updates=self.updates)

#     def dx(self, x, z, I1):
#         dx_eval = 1 - x**3 - 2 * x**2 - z + 3.1
#         return dx_eval

#     def dz(self, x, z, SC, K, x0, tau0, nn):
#         x_diff = np.repeat(x[:, np.newaxis], nn, axis=1) - x
#         # x_diff = np.repeat(x, nn, axis=0) - np.repeat(x, nn, axis=0).T
#         gx = K * SC * x_diff
#         dz_eval = (1 / tau0) * (4 * (x - x0) - z - gx.sum(axis=1))
#         return dz_eval

#     def step(self, x_prev, z_prev, dt, SC, K, x0, I1, tau0, nn):
#         x_next = x_prev + dt * self.dx(x_prev, z_prev, I1)
#         z_next = z_prev + dt * self.dz(x_prev, z_prev, SC, K, x0, tau0, nn)
#         return x_next, z_next

#     def sim(self, x_init_val, z_init_val, dt_val, SC_val, K_val, x0_val,
#             I1_val, tau0_val, nn_val):
#         return self.f(x_init_val, z_init_val, dt_val, SC_val, K_val, x0_val,
#                       I1_val, tau0_val, nn_val)
