import scipy.stats as stats
import numpy as np

def gen_ic(loc, scale):
        x0_clip_a, x0_clip_b = -5.0, 0.0
        x0_loc = loc['x0']
        x0 = stats.truncnorm.rvs(a=(x0_clip_a - x0_loc)/scale, b=(
            x0_clip_b - x0_loc)/scale, loc=x0_loc, scale=scale)

        alpha_clip_a, alpha_clip_b = 0.0, 50.0
        alpha_loc = loc['alpha']
        alpha = stats.truncnorm.rvs(a=(alpha_clip_a - alpha_loc)/scale, b=(
            alpha_clip_b - alpha_loc)/scale, loc=alpha_loc, scale=scale)[0]

        beta_clip_a, beta_clip_b = -10.0, 10.0
        beta_loc = loc['beta']
        beta = stats.truncnorm.rvs(a=(beta_clip_a - beta_loc)/scale, b=(
            beta_clip_b - beta_loc)/scale, loc=beta_loc, scale=scale)[0]

        K_clip_a, K_clip_b = 0.0, 5.0
        K_loc = loc['K']
        K = stats.truncnorm.rvs(a=(K_clip_a - K_loc)/scale, b=(
            K_clip_b - K_loc)/scale, loc=K_loc, scale=scale)[0]

        tau0_clip_a, tau0_clip_b = 15.0, 100.0
        tau0_loc = loc['tau0']
        tau0 = stats.truncnorm.rvs(a=(tau0_clip_a - tau0_loc)/scale, b=(
            tau0_clip_b - tau0_loc)/scale, loc=tau0_loc, scale=scale)[0]

        eps_slp_clip_a, eps_slp_clip_b = 0.5, 10.0
        eps_slp_loc = loc['eps_slp']
        eps_slp = stats.truncnorm.rvs(a=(eps_slp_clip_a - eps_slp_loc)/scale, b=(
            eps_slp_clip_b - eps_slp_loc)/scale, loc=eps_slp_loc, scale=scale)[0]

        eps_snsr_pwr_clip_a, eps_snsr_pwr_clip_b = 0.5, 10.0
        eps_snsr_pwr_loc = loc['eps_snsr_pwr']
        eps_snsr_pwr = stats.truncnorm.rvs(a=(eps_snsr_pwr_clip_a - eps_snsr_pwr_loc)/scale, b=(
            eps_snsr_pwr_clip_b - eps_snsr_pwr_loc)/scale, loc=eps_snsr_pwr_loc, scale=scale)[0]

        x_init_clip_a, x_init_clip_b = -2.5, -1.5
        x_init_loc = loc['x_init']
        x_init = stats.truncnorm.rvs(a=(x_init_clip_a - x_init_loc)/scale, b=(
            x_init_clip_b - x_init_loc)/scale, loc=x_init_loc, scale=scale)

        z_init_clip_a, z_init_clip_b = 3.0, 5.0
        z_init_loc = loc['z_init']
        z_init = stats.truncnorm.rvs(a=(z_init_clip_a - z_init_loc)/scale, b=(
            z_init_clip_b - z_init_loc)/scale, loc=z_init_loc, scale=scale)

        param_init = {'x0': x0, 'alpha': alpha,
                      'beta': beta, 'K': K, 'tau0': tau0, 'x_init': x_init, 'z_init': z_init,
                      'eps_slp': eps_slp, 'eps_snsr_pwr': eps_snsr_pwr}
        return param_init