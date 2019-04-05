import pymc3 as pm
import numpy as np
import lib.pymc3.transforms as tr
import theano
import theano.tensor as tt


def logNormal(x, mode, sd):
    return np.exp(sd**2 + np.log(mode) + sd * x)


def dx(x, z, I1):
    dx_eval = 1 - x**3 - 2 * x**2 - z + 3.1
    return dx_eval


def dz(x, z, SC, K, x0, tau0):
    nn = x.size
    x_diff = np.repeat(x[:, np.newaxis], nn, axis=1) - x
    gx = K * SC * x_diff.T
    dz_eval = (1 / tau0) * (4 * (x - x0) - z - gx.sum(axis=1))
    return dz_eval


def step(x_prev, z_prev, dt, SC, K, x0, I1, tau0):
    x_next = x_prev + dt*dx(x_prev, z_prev, I1)
    z_next = z_prev + dt*dz(x_prev, z_prev, SC, K, x0, tau0)
    return x_next, z_next


def step_sde(x_eta, x_prev, z_prev, dt, SC, K, x0, I1, tau0):
    x_next = x_prev + dt*dx(x_prev, z_prev, I1) + tt.sqrt(dt)*x_eta
    z_next = z_prev + dt*dz(x_prev, z_prev, SC, K, x0, tau0)
    return x_next, z_next


class vep_ode:
    def __init__(self, consts, obs):
        self.consts = consts
        self.obs = obs
        self.model = pm.Model()
        with self.model:
            x0_star = pm.Normal(
                'x0_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            x0 = pm.Deterministic('x0', -2.5 + x0_star)
            amplitude_star = pm.Normal('amplitude_star', mu=0.0, sd=1.0)
            amplitude = pm.Deterministic(
                'amplitude', logNormal(amplitude_star, mode=1.0, sd=1.0))
            offset_star = pm.Normal('offset_star', mu=0.0, sd=1.0)
            offset = pm.Deterministic('offset', offset_star)
            K_star = pm.Normal('K_star', mu=0.0, sd=1.0)
            K = pm.Deterministic('K', logNormal(K_star, mode=1.0, sd=1.0))
            tau0_star = pm.Normal('tau0_star', mu=0.0, sd=1.0)
            tau0 = pm.Deterministic('tau0',
                                    logNormal(tau0_star, mode=30.0, sd=1.0))
            x_init_star = pm.Normal(
                'x_init_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            x_init = pm.Deterministic('x_init', -2.0 + x_init_star)
            z_init_star = pm.Normal(
                'z_init_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            z_init = pm.Deterministic('z_init', 3.5 + z_init_star)
            # Cast constants in the model as tensors using theano shared variables
            time_step = theano.shared(self.consts['time_step'], 'time_step')
            # x_init = theano.shared(self.consts['x_init'], 'x_init')
            # z_init = theano.shared(self.consts['z_init'], 'z_init')
            SC = theano.shared(self.consts['SC'], 'SC')
            I1 = theano.shared(self.consts['I1'], 'I1')
            output, updates = theano.scan(
                fn=step,
                outputs_info=[x_init, z_init],
                non_sequences=[time_step, SC, K, x0, I1, tau0],
                n_steps=self.consts['nt'])
            x_sym = output[0]
            z_sym = output[1]
            x = pm.Deterministic('x', x_sym)
            z = pm.Deterministic('z', z_sym)
            _mu_slp = amplitude * tt.transpose(
                tt.log(
                    tt.dot(self.consts['gain'], tt.exp(tt.transpose(x_sym))) +
                    offset))
            mu_slp = pm.Deterministic('mu_slp', _mu_slp)
            _mu_snsr_pwr = (mu_slp * mu_slp).sum(axis=0)
            _mu_snsr_pwr = _mu_snsr_pwr / _mu_snsr_pwr.max()
            mu_snsr_pwr = pm.Deterministic('mu_snsr_pwr', _mu_snsr_pwr)
            slp = pm.Normal(
                'slp',
                mu=mu_slp,
                sd=self.consts['eps_slp'],
                shape=(self.consts['nt'], self.consts['ns']),
                observed=self.obs['slp'])
            snsr_pwr = pm.Normal(
                'snsr_pwr',
                mu=mu_snsr_pwr,
                sd=self.consts['eps_snsr_pwr'],
                shape=self.consts['ns'],
                observed=self.obs['snsr_pwr'])


class vep_ode_scaled:
    def __init__(self, consts, obs):
        self.consts = consts
        self.obs = obs
        self.alpha = self.consts['alpha']
        self.model = pm.Model()
        with self.model:
            # alpha = pm.Uniform('alpha', lower=0)
            x0_star = pm.Normal(
                'x0_star',
                mu=0.0,
                sd=1.0,
                shape=self.consts['nn'],
                transform=tr.Linear(self.alpha, 0))
            x0 = pm.Deterministic('x0', -2.5 + x0_star)
            amplitude_star = pm.Normal(
                'amplitude_star',
                mu=0.0,
                sd=1.0,
                transform=tr.Linear(self.alpha, 0))
            amplitude = pm.Deterministic(
                'amplitude', logNormal(amplitude_star, mode=1.0, sd=1.0))
            offset_star = pm.Normal(
                'offset_star', mu=0, sd=1.0, transform=tr.Linear(self.alpha, 0))
            offset = pm.Deterministic('offset', offset_star)
            K_star = pm.Normal(
                'K_star', mu=0.0, sd=1.0, transform=tr.Linear(self.alpha, 0))
            K = pm.Deterministic('K', logNormal(K_star, mode=1.0, sd=1.0))
            tau0_star = pm.Normal(
                'tau0_star', mu=0.0, sd=1.0, transform=tr.Linear(self.alpha, 0))
            tau0 = pm.Deterministic('tau0',
                                    logNormal(tau0_star, mode=30.0, sd=1.0))
            x_init_star = pm.Normal(
                'x_init_star', mu=0.0, sd=1.0, shape=self.consts['nn'], transform=tr.Linear(self.alpha, 0))
            x_init = pm.Deterministic('x_init', -2.0 + x_init_star)
            z_init_star = pm.Normal(
                'z_init_star', mu=0.0, sd=1.0, shape=self.consts['nn'], transform=tr.Linear(self.alpha, 0))
            z_init = pm.Deterministic('z_init', 3.5 + z_init_star)
            # Cast constants in the model as tensors using theano shared variables
            time_step = theano.shared(self.consts['time_step'], 'time_step')
            # x_init = theano.shared(self.consts['x_init'], 'x_init')
            # z_init = theano.shared(self.consts['z_init'], 'z_init')
            SC = theano.shared(self.consts['SC'], 'SC')
            I1 = theano.shared(self.consts['I1'], 'I1')
            output, updates = theano.scan(
                fn=step,
                outputs_info=[x_init, z_init],
                non_sequences=[time_step, SC, K, x0, I1, tau0],
                n_steps=self.consts['nt'])
            x_sym = output[0]
            z_sym = output[1]
            x = pm.Deterministic('x', x_sym)
            z = pm.Deterministic('z', z_sym)
            mu_slp = pm.Deterministic('mu_slp', (amplitude * tt.transpose(
                tt.log(
                    tt.dot(self.consts['gain'], tt.exp(tt.transpose(x_sym))) +
                    offset))))
            # _mu_snsr_pwr = (mu_slp * mu_slp).sum(axis=0)
            # _mu_snsr_pwr = _mu_snsr_pwr / _mu_snsr_pwr.max()
            # mu_snsr_pwr = pm.Deterministic('mu_snsr_pwr', _mu_snsr_pwr)
            slp = pm.Normal(
                'slp',
                mu=mu_slp,
                sd=self.consts['eps_slp'],
                shape=(self.consts['nt'], self.consts['ns']),
                observed=self.obs['slp'])
            # snsr_pwr = pm.Normal(
            #     'snsr_pwr',
            #     mu=mu_snsr_pwr,
            #     sd=self.consts['epsilon_snsr_pwr'],
            #     shape=self.consts['ns'],
            #     observed=self.obs['snsr_pwr'])


class vep_ode_hyperinfer:
    def __init__(self, consts, obs):
        self.consts = consts
        self.obs = obs
        self.model = pm.Model()
        with self.model:
            x0_star = pm.Normal(
                'x0_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            x0 = pm.Deterministic('x0', -2.5 + x0_star)
            amplitude_star = pm.Normal('amplitude_star', mu=0.0, sd=1.0)
            amplitude = pm.Deterministic(
                'amplitude', logNormal(amplitude_star, mode=1.0, sd=1.0))
            offset_star = pm.Normal('offset_star', mu=0.0, sd=1.0)
            offset = pm.Deterministic('offset', offset_star)
            K_star = pm.Normal('K_star', mu=0.0, sd=1.0)
            K = pm.Deterministic('K', logNormal(K_star, mode=1.0, sd=1.0))
            tau0_star = pm.Normal('tau0_star', mu=0.0, sd=1.0)
            tau0 = pm.Deterministic('tau0',
                                    logNormal(tau0_star, mode=30.0, sd=1.0))
            x_init_star = pm.Normal(
                'x_init_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            x_init = pm.Deterministic('x_init', -2.0 + x_init_star)
            z_init_star = pm.Normal(
                'z_init_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            z_init = pm.Deterministic('z_init', 3.5 + z_init_star)
            eps_slp_star = pm.Normal('eps_slp_star', mu=0.0, sd=1.0)
            eps_slp = pm.Deterministic(
                'eps_slp', logNormal(eps_slp_star, mode=0.1, sd=1.0))
            # eps_snsr_pwr_star = pm.Normal('eps_snsr_pwr_star', mu=0.0, sd=1.0)
            # eps_snsr_pwr = pm.Deterministic(
            #     'eps_snsr_pwr', logNormal(eps_snsr_pwr_star, mode=0.1, sd=1.0))
            # Cast constants in the model as tensors using theano shared variables
            time_step = theano.shared(self.consts['time_step'], 'time_step')
            # x_init = theano.shared(self.consts['x_init'], 'x_init')
            # z_init = theano.shared(self.consts['z_init'], 'z_init')
            SC = theano.shared(self.consts['SC'], 'SC')
            I1 = theano.shared(self.consts['I1'], 'I1')
            output, updates = theano.scan(
                fn=step,
                outputs_info=[x_init, z_init],
                non_sequences=[time_step, SC, K, x0, I1, tau0],
                n_steps=self.consts['nt'])
            x_sym = output[0]
            z_sym = output[1]
            x = pm.Deterministic('x', x_sym)
            z = pm.Deterministic('z', z_sym)
            _mu_slp = amplitude * tt.transpose(
                tt.log(
                    tt.dot(self.consts['gain'], tt.exp(tt.transpose(x_sym))) +
                    offset))
            mu_slp = pm.Deterministic('mu_slp', _mu_slp - _mu_slp.mean(axis=0))
            # _mu_snsr_pwr = (mu_slp * mu_slp).sum(axis=0)
            # _mu_snsr_pwr = _mu_snsr_pwr / _mu_snsr_pwr.max()
            # mu_snsr_pwr = pm.Deterministic('mu_snsr_pwr', _mu_snsr_pwr)
            slp = pm.Normal(
                'slp',
                mu=mu_slp,
                sd=eps_slp,
                shape=(self.consts['nt'], self.consts['ns']),
                observed=self.obs['slp'])
            # snsr_pwr = pm.Normal(
            #     'snsr_pwr',
            #     mu=mu_snsr_pwr,
            #     sd=eps_snsr_pwr,
            #     shape=self.consts['ns'],
            #     observed=self.obs['snsr_pwr'])


class vep_ode_normpriors:
    def __init__(self, consts, params_init, obs):
        self.consts = consts
        self.params_init = params_init
        self.obs = obs
        self.model = pm.Model()
        with self.model:
            x0 = pm.Normal('x0', mu=-2.5, sd=1.0, shape=self.consts['nn'])
            amplitude = pm.Bound(
                pm.Normal, lower=0)(
                    'amplitude', mu=1.0, sd=1.0)
            offset = pm.Normal('offset', mu=0.0, sd=1.0)
            K = pm.Bound(pm.Normal, lower=0)('K', mu=1.0, sd=1.0)
            tau0 = pm.Bound(
                pm.Normal, lower=10, upper=2000)(
                    'tau0', mu=10, sd=1.0)
            # Cast constants in the model as tensors
            # using theano shared variables
            time_step = theano.shared(self.consts['time_step'], 'time_step')
            x_init = theano.shared(self.consts['x_init'], 'x_init')
            z_init = theano.shared(self.consts['z_init'], 'z_init')
            SC = theano.shared(self.consts['SC'], 'SC')
            I1 = theano.shared(self.consts['I1'], 'I1')
            output, updates = theano.scan(
                fn=step,
                outputs_info=[x_init, z_init],
                non_sequences=[time_step, SC, K, x0, I1, tau0],
                n_steps=self.consts['nt'])
            x_sym = output[0]
            z_sym = output[1]
            x = pm.Deterministic('x', x_sym)
            z = pm.Deterministic('z', z_sym)
            mu_slp = pm.Deterministic('mu_slp', (amplitude * tt.transpose(
                tt.log(
                    tt.dot(self.consts['gain'], tt.exp(tt.transpose(x_sym))) +
                    offset))))
            _mu_snsr_pwr = (mu_slp * mu_slp).sum(axis=0)
            _mu_snsr_pwr = _mu_snsr_pwr / _mu_snsr_pwr.max()
            mu_snsr_pwr = pm.Deterministic('mu_snsr_pwr', _mu_snsr_pwr)
            slp = pm.Normal(
                'slp',
                mu=mu_slp,
                sd=self.consts['epsilon_slp'],
                shape=(self.consts['nt'], self.consts['ns']),
                observed=self.obs['slp'])
            snsr_pwr = pm.Normal(
                'snsr_pwr',
                mu=mu_snsr_pwr,
                sd=self.consts['epsilon_snsr_pwr'],
                shape=self.consts['ns'],
                observed=self.obs['snsr_pwr'])


class vep_ode_normpriors_hyperinfer:
    def __init__(self, consts, params_init, obs):
        self.consts = consts
        self.params_init = params_init
        self.obs = obs
        self.model = pm.Model()
        with self.model:
            x0 = pm.Normal('x0', mu=-2.5, sd=1.0, shape=self.consts['nn'])
            x0.tag.test_value = self.params_init['x0']
            amplitude = pm.Bound(
                pm.Normal, lower=0)(
                    'amplitude', mu=1.0, sd=1.0)
            amplitude.tag.test_value = self.params_init['amplitude']
            offset = pm.Normal('offset', mu=0.0, sd=1.0)
            offset.tag.test_value = self.params_init['offset']
            K = pm.Bound(pm.Normal, lower=0)('K', mu=1.0, sd=1.0)
            K.tag.test_value = self.params_init['K']
            tau0 = pm.Bound(
                pm.Normal, lower=10, upper=2000)(
                    'tau0', mu=10, sd=1.0)
            tau0.tag.test_value = self.params_init['tau0']
            # Cast constants in the model as tensors
            # using theano shared variables
            time_step = theano.shared(self.consts['time_step'], 'time_step')
            x_init = theano.shared(self.consts['x_init'], 'x_init')
            z_init = theano.shared(self.consts['z_init'], 'z_init')
            SC = theano.shared(self.consts['SC'], 'SC')
            I1 = theano.shared(self.consts['I1'], 'I1')
            output, updates = theano.scan(
                fn=step,
                outputs_info=[x_init, z_init],
                non_sequences=[time_step, SC, K, x0, I1, tau0],
                n_steps=self.consts['nt'])
            x_sym = output[0]
            z_sym = output[1]
            x = pm.Deterministic('x', x_sym)
            z = pm.Deterministic('z', z_sym)
            mu_slp = pm.Deterministic('mu_slp', (amplitude * tt.transpose(
                tt.log(
                    tt.dot(self.consts['gain'], tt.exp(tt.transpose(x_sym))) +
                    offset))))
            _mu_snsr_pwr = (mu_slp * mu_slp).sum(axis=0)
            _mu_snsr_pwr = _mu_snsr_pwr / _mu_snsr_pwr.max()
            mu_snsr_pwr = pm.Deterministic('mu_snsr_pwr', _mu_snsr_pwr)
            epsilon_slp = pm.Bound(pm.Normal, lower=0)('epsilon_slp', mu=0.1, sd=1.0)
            epsilon_snsr_pwr = pm.Bound(pm.Normal, lower=0)('epsilon_snsr_pwr', mu=0.1, sd=1.0)
            slp = pm.Normal(
                'slp',
                mu=mu_slp,
                sd=epsilon_slp,
                shape=(self.consts['nt'], self.consts['ns']),
                observed=self.obs['slp'])
            snsr_pwr = pm.Normal(
                'snsr_pwr',
                mu=mu_snsr_pwr,
                sd=epsilon_snsr_pwr,
                shape=self.consts['ns'],
                observed=self.obs['snsr_pwr'])


class vep_sde_noncntrd:
    def __init__(self, consts, obs):
        self.consts = consts
        self.obs = obs
        self.model = pm.Model()
        with self.model:
            x0_star = pm.Normal('x0_star', mu=0.0, sd=1.0, shape=self.consts['nn'])
            x0 = pm.Deterministic('x0', -2.5 + x0_star)
            amplitude_star = pm.Normal('amplitude_star', mu=0.0, sd=1.0)
            amplitude = pm.Deterministic(
                'amplitude', logNormal(amplitude_star, mode=1.0, sd=1.0))
            offset_star = pm.Normal('offset_star', mu=0.0, sd=1.0)
            offset = pm.Deterministic('offset', offset_star)
            K_star = pm.Normal('K_star', mu=0.0, sd=1.0)
            K = pm.Deterministic('K', logNormal(K_star, mode=1.0, sd=1.0))
            tau0_star = pm.Normal('tau0_star', mu=0.0, sd=1.0)
            tau0 = pm.Deterministic('tau0',
                                    logNormal(tau0_star, mode=30.0, sd=1.0))
            x_eta = pm.Normal('x_eta', mu=0, sd=1.0, shape=(self.consts['nt'], self.consts['nn']))
            # Cast constants in the model as tensors using theano shared variables
            time_step = theano.shared(self.consts['time_step'], 'time_step')
            x_init = theano.shared(self.consts['x_init'], 'x_init')
            z_init = theano.shared(self.consts['z_init'], 'z_init')
            SC = theano.shared(self.consts['SC'], 'SC')
            I1 = theano.shared(self.consts['I1'], 'I1')

            output, updates = theano.scan(fn=step_sde,
                                          sequences=x_eta,
                                        outputs_info=[x_init, z_init],
                                        non_sequences=[time_step, SC, K, x0, I1, tau0])
            
            x_sym = output[0]
            z_sym = output[1]
            x = pm.Deterministic('x', x_sym)
            z = pm.Deterministic('z', z_sym)
            mu_slp = pm.Deterministic(
                'mu_slp',
                (amplitude *
                tt.transpose(tt.log(tt.dot(self.consts['gain'], tt.exp(tt.transpose(x_sym))) + offset))))
            # _mu_snsr_pwr = (mu_slp * mu_slp).sum(axis=0)
            # _mu_snsr_pwr = _mu_snsr_pwr / _mu_snsr_pwr.max()
            # mu_snsr_pwr = pm.Deterministic('mu_snsr_pwr', _mu_snsr_pwr)
            mu_slp = mu_slp - tt.sum(mu_slp, axis=0)
            slp = pm.Normal(
                'slp',
                mu=mu_slp,
                sd=self.consts['epsilon_slp'],
                shape=(self.consts['nt'], self.consts['ns']),
                observed=self.obs['slp'])
            # snsr_pwr = pm.Normal(
            #     'snsr_pwr',
            #     mu=mu_snsr_pwr,
            #     sd=self.consts['epsilon_snsr_pwr'],
            #     shape=self.consts['ns'],
            #     observed=self.obs['snsr_pwr'])
