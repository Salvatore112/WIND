import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")

class AbstractOptimizer(ABC):
    def __init__(self, dim, hyperparameters):
        self.dim = dim
        self.hp = hyperparameters
        self.state = {}
    
    @abstractmethod
    def step(self, x, oracle_data):
        pass

class BenchmarkEnvironment:
    def __init__(self, config):
        self.dim = config.get('dim', 2)
        self.rho = config.get('rho', 0.5)
        self.drift = config.get('drift_speed', 0.0)
        self.noise_type = config.get('noise_type', 'gaussian')
        self.noise_scale = config.get('noise_scale', 0.5)
        self.geometry = config.get('geometry', 'ideal')
        self.cond = config.get('condition_number', 1.0)
        self.theta = np.zeros(self.dim)
        self.M = self._generate_transform()
    
    def _generate_transform(self):
        if self.geometry != 'distorted' or self.cond == 1.0:
            return np.eye(self.dim)
        S = np.diag(np.linspace(1, self.cond, self.dim))
        R = ortho_group.rvs(self.dim) if self.dim > 1 else np.eye(1)
        return R @ S
    
    def update_drift(self):
        if self.drift == 0:
            return self.theta
        trend = np.ones(self.dim) / np.sqrt(self.dim) * (self.drift * 0.3)
        walk = np.random.randn(self.dim)
        norm = np.linalg.norm(walk)
        walk = (walk / (norm + 1e-9)) * (self.drift * 0.7)
        self.theta += trend + walk
        return self.theta
    
    def get_oracle(self, x):
        diff = x - self.theta
        if self.geometry == 'rosenbrock':
            z = diff + 1.0
            val = np.sum(100*(z[1:] - z[:-1]**2)**2 + (1 - z[:-1])**2)
            grad = np.zeros_like(z)
            grad[:-1] += -400*z[:-1]*(z[1:] - z[:-1]**2) - 2*(1 - z[:-1])
            grad[1:] += 200*(z[1:] - z[:-1]**2)
        else:
            u = self.M @ diff
            p = self.rho + 1
            val = np.sum(np.abs(u)**p) / p
            grad_u = (np.abs(u)**self.rho) * np.sign(u)
            grad = self.M.T @ grad_u
        noise = self._generate_noise(grad.shape)
        return val, grad + noise
    
    def _generate_noise(self, shape):
        if self.noise_scale <= 0:
            return np.zeros(shape)
        if self.noise_type == 'gaussian':
            raw = np.random.normal(0, 1, shape)
        elif self.noise_type == 'pareto':
            alpha = 2.5
            u = np.random.uniform(size=shape)
            raw = ((1.0 - u)**(-1.0/alpha) - 1.0) * np.sign(np.random.uniform(size=shape)-0.5)
        else:
            raw = np.random.normal(0, 1, shape)
        return raw * self.noise_scale

class STHRDBenchmark:
    def __init__(self):
        self.results = []
    
    def evaluate(self, optimizer_class, optim_params, phases=['stress']):
        self.results = []
        if 'heatmap' in phases:
            rho_vals = [0.1, 0.5, 0.9]
            drift_vals = [0.0, 0.05, 0.1]
            for d in drift_vals:
                for r in rho_vals:
                    capture_traj = (d == 0.05 and r == 0.5)
                    cfg = {'dim': 3 if capture_traj else 2, 'rho': r, 'drift_speed': d,
                           'noise_type': 'gaussian', 'noise_scale': 0.5}
                    stats = self._run_trials(optimizer_class, optim_params, cfg, trials=20, capture=capture_traj)
                    self.results.append({
                        'type': 'heatmap', 'rho': r, 'drift': d,
                        'success': stats['success_rate'],
                        'errors': stats['errors'],
                        'traj': stats.get('traj')
                    })
        if 'stress' in phases:
            scenarios = [
                {'name': 'Gaussian Ideal', 'noise': 'gaussian', 'cond': 1},
                {'name': 'Pareto Tail', 'noise': 'pareto', 'cond': 1},
                {'name': 'Valley (C=10)', 'noise': 'gaussian', 'cond': 10},
                {'name': 'Valley (C=100)', 'noise': 'gaussian', 'cond': 100},
            ]
            for s in scenarios:
                cfg = {'dim': 2, 'rho': 0.5, 'drift_speed': 0.02,
                       'noise_type': s['noise'], 'noise_scale': 0.5,
                       'geometry': 'distorted', 'condition_number': s['cond']}
                stats = self._run_trials(optimizer_class, optim_params, cfg, trials=50)
                self.results.append({
                    'type': 'stress', 'noise': s['name'],
                    'cond': s['cond'], 'errors': stats['errors']
                })
        return self.results
    
    def _run_trials(self, opt_cls, opt_params, env_config, trials=10, steps=600, capture=False):
        errors = []
        success = 0
        thresh = 10.0 * (1.0 + env_config.get('noise_scale',0)) * (env_config.get('condition_number', 1)**0.5)
        traj_data = None
        for i in range(trials):
            env = BenchmarkEnvironment(env_config)
            opt = opt_cls(env_config['dim'], opt_params)
            x = np.zeros(env_config['dim'])
            run_vals = []
            th_hist, x_hist = [], []
            for _ in range(steps):
                theta = env.update_drift()
                val, grad = env.get_oracle(x)
                x = opt.step(x, {'grad': grad, 'value': val})
                run_vals.append(val)
                if capture and i == 0:
                    th_hist.append(theta.copy())
                    x_hist.append(x.copy())
            final_err = np.mean(run_vals[-int(steps*0.2):])
            if np.isnan(final_err) or final_err > 1e6:
                final_err = 1e6
            errors.append(final_err)
            if final_err < thresh:
                success += 1
            if capture and i == 0:
                traj_data = (np.array(th_hist), np.array(x_hist))
        return {'errors': np.array(errors), 'success_rate': success/trials, 'traj': traj_data}
    
    def generate_dataset(self, filename, config, steps=1000):
        env = BenchmarkEnvironment(config)
        x_dummy = np.zeros(config['dim'])
        data = []
        for t in range(steps):
            theta = env.update_drift()
            val_0, grad_0 = env.get_oracle(np.zeros(config['dim']))
            row = {
                'step': t,
                'theta_x': theta[0], 'theta_y': theta[1],
                'grad_0_x': grad_0[0], 'grad_0_y': grad_0[1],
                'val_0': val_0
            }
            data.append(row)
        df = pd.DataFrame(data)
        if filename.endswith('.csv'):
            df.to_csv(filename, index=False)
        elif filename.endswith('.json'):
            df.to_json(filename, orient='records')