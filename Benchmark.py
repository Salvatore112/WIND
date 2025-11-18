# @title ST-HRD Benchmark (High-Res for Colab)
# @markdown Запустите эту ячейку. Расчет сетки с шагом 0.05 займет некоторое время (прогресс-бар внизу).

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from scipy.stats import ortho_group
from abc import ABC, abstractmethod
from tqdm.notebook import tqdm # Используем widget версию для Colab
import itertools
import warnings
from math import pi
from IPython.display import display, Markdown # Для красивого вывода в Colab

# --- Настройки для Colab ---
%matplotlib inline
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120 # Повышаем четкость графиков
plt.rcParams['figure.figsize'] = (20, 12)

# =============================================================================
# 1. АЛГОРИТМЫ И СРЕДА
# =============================================================================

class BaseOptimizer(ABC):
    def __init__(self, dim, hp):
        self.dim = dim
        self.hp = hp
        self.state = {}
    @abstractmethod
    def step(self, x, oracle_data): pass

class SGDOptimizer(BaseOptimizer):
    def step(self, x, oracle_data):
        grad = oracle_data.get('grad')
        lr = self.hp.get('lr', 0.01)
        momentum = self.hp.get('momentum', 0.0)
        if 'v' not in self.state: self.state['v'] = np.zeros_like(x)
        self.state['v'] = momentum * self.state['v'] + lr * grad
        return x - self.state['v']

class BenchmarkEnvironment:
    def __init__(self, config):
        self.dim = config['dim']
        self.rho = config['rho']
        self.drift = config['drift_speed']
        self.noise_type = config['noise_type']
        self.noise_scale = config['noise_scale']
        self.geometry = config.get('geometry', 'ideal')
        self.cond = config.get('condition_number', 1.0)
        self.theta = np.zeros(self.dim)
        self.M = self._generate_transform()

    def _generate_transform(self):
        if self.geometry != 'distorted' or self.cond == 1.0: return np.eye(self.dim)
        S = np.diag(np.linspace(1, self.cond, self.dim))
        R = ortho_group.rvs(self.dim) if self.dim > 1 else np.eye(1)
        return R @ S

    def update_drift(self):
        if self.drift == 0: return self.theta
        trend = np.ones(self.dim) / np.sqrt(self.dim) * (self.drift * 0.3)
        walk = np.random.randn(self.dim)
        walk = (walk / (np.linalg.norm(walk)+1e-9)) * (self.drift * 0.7)
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
        return val, grad + self._generate_strict_noise(grad.shape)

    def _generate_strict_noise(self, shape):
        if self.noise_scale <= 0: return np.zeros(shape)
        if self.noise_type == 'gaussian':
            raw = np.random.normal(0, 1, shape)
        elif self.noise_type == 'pareto':
            alpha = 2.5 
            u = np.random.uniform(size=shape)
            raw = ((1.0 - u)**(-1.0/alpha) - 1.0) * np.sign(np.random.uniform(size=shape)-0.5)
        else:
            raw = np.random.normal(0, 1, shape)
        return raw * self.noise_scale

# =============================================================================
# 2. ИНСТРУМЕНТЫ (Tuner & Runner)
# =============================================================================

class HyperTuner:
    def __init__(self, optim_cls, grid, steps=200, trials=2):
        self.optim_cls = optim_cls
        self.grid = grid
        self.steps = steps
        self.trials = trials

    def tune(self, cfg):
        keys, values = zip(*self.grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best_score = float('inf')
        best_params = combos[0]
        for params in combos:
            scores = []
            for _ in range(self.trials):
                env = BenchmarkEnvironment(cfg)
                opt = self.optim_cls(cfg['dim'], params)
                x = np.zeros(cfg['dim'])
                vals = []
                for _ in range(self.steps):
                    env.update_drift()
                    v, g = env.get_oracle(x)
                    x = opt.step(x, {'value':v, 'grad':g})
                    vals.append(v)
                scores.append(np.mean(vals[-int(self.steps*0.2):]))
            avg = np.mean(scores)
            if avg < best_score: best_score = avg; best_params = params
        return best_params

def run_adaptive_experiment(optim_cls, params, cfg, min_runs=50, max_runs=1000, tol=0.05, steps=600):
    final_errors = []
    success = 0
    total = 0
    traj_data = None
    thresh = 10.0 * (1.0 + cfg['noise_scale'])
    if cfg.get('geometry')=='rosenbrock': thresh *= 10

    while total < max_runs:
        batch = 10
        for _ in range(batch):
            if total >= max_runs: break
            env = BenchmarkEnvironment(cfg)
            opt = optim_cls(cfg['dim'], params)
            x = np.zeros(cfg['dim'])
            vals, th_h, x_h = [], [], []
            for _ in range(steps):
                th = env.update_drift()
                v, g = env.get_oracle(x)
                x = opt.step(x, {'value':v, 'grad':g})
                vals.append(v)
                if total==0: th_h.append(th.copy()); x_h.append(x.copy())

            l_avg = np.mean(vals[-int(steps*0.2):])
            if np.isnan(l_avg) or l_avg > 1e9: l_avg = 1e9
            else:
                if l_avg < thresh: success += 1
            final_errors.append(l_avg)
            if total==0: traj_data = (np.array(th_h), np.array(x_h))
            total += 1

        if total >= min_runs:
            arr = np.array(final_errors)
            mu = np.mean(arr)
            sem = np.std(arr)/np.sqrt(total)
            ci = 1.96 * sem
            rel_err = ci / (mu + 1e-9)
            if rel_err < tol: break
            
    return np.array(final_errors), success/total, traj_data

# =============================================================================
# 3. ВИЗУАЛИЗАЦИЯ (COLAB DASHBOARD)
# =============================================================================

class BenchmarkVisualizer:
    def __init__(self, db):
        self.db = db
        self.colors = sns.color_palette("deep")

    def show_full_report(self):
        display(Markdown("## 📊 ST-HRD Benchmark Report"))
        
        display(Markdown("### 1. Metric Tables"))
        self._display_tables()
        
        display(Markdown("### 2. Analytical Dashboard"))
        self._plot_dashboard()

    def _get_pivot(self, metric):
        data = [r for r in self.db if r['type']=='heatmap']
        if not data: return None
        df = pd.DataFrame([{
            'rho': r['rho'], 'drift': r['drift'], 
            'val': r[metric] if metric == 'success' else np.median(r['errors'])
        } for r in data])
        return df.pivot(index="drift", columns="rho", values="val")

    def _display_tables(self):
        # Формируем таблицу Stress Test для красивого вывода
        st_data = [r for r in self.db if r['type']=='stress']
        if st_data:
            rows = []
            # Находим базу (Gaussian Ideal)
            base_run = next((r for r in st_data if 'Ideal' in r['noise'] and 'Gaussian' in r['noise']), st_data[0])
            base_med = np.median(base_run['errors'])
            
            for r in st_data:
                e = np.array(r['errors'])
                rows.append({
                    'Scenario': r['noise'], 
                    'Cond. Num': r.get('cond',1),
                    'Median Error': np.median(e),
                    'Mean Error': np.mean(e),
                    'Max Error': np.max(e),
                    'Degradation': np.median(e)/base_med
                })
            
            df = pd.DataFrame(rows)
            # Стилизация таблицы Pandas
            styled = df.style.format({
                'Median Error': '{:.4f}',
                'Mean Error': '{:.4f}',
                'Max Error': '{:.4f}',
                'Degradation': '{:.1f}x'
            }).background_gradient(subset=['Median Error', 'Degradation'], cmap='Reds')
            
            display(styled)

    def _plot_dashboard(self):
        fig = plt.figure(figsize=(22, 12))
        
        # Создаем сетку подграфиков
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        ax2 = fig.add_subplot(2, 4, 2)
        ax3 = fig.add_subplot(2, 4, 3) 
        ax4 = fig.add_subplot(2, 4, 4, polar=True)
        ax5 = fig.add_subplot(2, 4, 5) 
        ax6 = fig.add_subplot(2, 4, 6) 
        ax7 = fig.add_subplot(2, 4, 7, projection='3d')
        ax8 = fig.add_subplot(2, 4, 8)

        self.plot_error_surface(ax1)
        self.plot_error_heatmap(ax2)
        self.plot_stability_heatmap(ax3)
        self.plot_radar(ax4)
        self.plot_degradation(ax5)
        self.plot_convergence(ax6)
        
        traj_run = next((r for r in self.db if r.get('traj') is not None), None)
        self.plot_traj_3d(ax7, traj_run)
        self.plot_ribbon(ax8, traj_run)

        plt.tight_layout()
        plt.show()

    # --- Plot Implementations ---
    def plot_error_surface(self, ax):
        pivot = self._get_pivot('errors')
        if pivot is None: return
        X, Y = np.meshgrid(pivot.columns, pivot.index)
        Z = pivot.values
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title("1A. Error Topology (Median)")
        ax.set_xlabel("Rho"); ax.set_ylabel("Drift"); ax.set_zlabel("Error")
        ax.view_init(30, 135)

    def plot_error_heatmap(self, ax):
        pivot = self._get_pivot('errors')
        if pivot is None: return
        # High Res: отключаем аннотации цифрами
        sns.heatmap(pivot.sort_index(ascending=False), annot=False, 
                    cmap="viridis", ax=ax, cbar_kws={'label': 'Median Error'})
        ax.set_title("1B. Error Heatmap")

    def plot_stability_heatmap(self, ax):
        pivot = self._get_pivot('success')
        if pivot is None: return
        sns.heatmap(pivot.sort_index(ascending=False), annot=False, 
                    cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Success Rate'})
        ax.set_title("2. Stability Region")

    def plot_radar(self, ax):
        cats = ['Gaussian', 'Pareto', 'Valley']
        vals = []
        for c in cats:
            runs = [r for r in self.db if r['type']=='stress' and 
                   (c in r['noise'] or (c=='Valley' and r.get('cond')==10))]
            vals.append(np.median(np.concatenate([x['errors'] for x in runs])) if runs else 1e9)
        
        scores = [1/(1+v*5) for v in vals]
        scores += scores[:1]
        angles = [n/3*2*pi for n in range(4)]
        ax.plot(angles, scores, color=self.colors[2], lw=2)
        ax.fill(angles, scores, alpha=0.2, color=self.colors[2])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats)
        ax.set_title("3. Robustness")

    def plot_degradation(self, ax):
        data = [r for r in self.db if r['type']=='stress' and 'cond' in r]
        if not data: return
        df = pd.DataFrame(data)
        conds = sorted(df['cond'].unique())
        meds, errs = [], []
        for c in conds:
            vals = np.concatenate(df[df['cond']==c]['errors'].values)
            meds.append(np.median(vals))
            errs.append(np.std(vals)/np.sqrt(len(vals)))
        ax.errorbar(conds, meds, yerr=errs, fmt='-o', capsize=4, color=self.colors[3])
        ax.set_title("4. Degradation")
        ax.set_xscale('log'); ax.set_xlabel("Cond. Num"); ax.set_ylabel("Median Error")

    def plot_convergence(self, ax):
        runs = [r for r in self.db if 'Pareto' in r.get('noise', '')]
        if not runs: return
        e = runs[0]['errors']
        ax.plot(np.cumsum(e)/np.arange(1,len(e)+1), color=self.colors[0])
        ax.set_title("5. LLN Convergence"); ax.set_xlabel("N Runs")

    def plot_traj_3d(self, ax, data):
        if not data: return
        th, x = data['traj']
        l = min(300, len(th))
        ax.plot(th[-l:,0], th[-l:,1], th[-l:,2], 'k--', alpha=0.3, label='Target')
        ax.plot(x[-l:,0], x[-l:,1], x[-l:,2], 'r-', lw=1.5, label='Algo')
        ax.set_title("6. Trajectory 3D"); ax.legend()

    def plot_ribbon(self, ax, data):
        if not data: return
        th, x = data['traj']
        rho = data['rho']
        errs = np.sum(np.abs(x-th)**(rho+1), axis=1)
        sm = np.convolve(errs, np.ones(20)/20, mode='valid')
        ax.plot(sm, color=self.colors[1])
        ax.fill_between(np.arange(len(sm)), sm*0.5, sm*1.5, alpha=0.2, color=self.colors[1])
        ax.set_title("7. Dynamics"); ax.set_ylabel("Loss")

# =============================================================================
# 4. ЗАПУСК (High Res Grid)
# =============================================================================

if __name__ == "__main__":
    display(Markdown("# 🚀 Starting ST-HRD Benchmark Calculation"))
    
    db = []
    # Сетка алгоритма (упрощена для скорости в Colab, для статьи расширьте)
    grid = {'lr': [0.1, 0.05], 'momentum': [0.0, 0.5]} 
    tuner = HyperTuner(SGDOptimizer, grid, steps=150, trials=2)

    # --- 1. HIGH-RES GRID (5% steps) ---
    rho_vals = np.round(np.arange(0.05, 1.05, 0.05), 2)
    drift_vals = np.round(np.arange(0.00, 0.11, 0.01), 2)
    
    total_points = len(rho_vals) * len(drift_vals)
    print(f"Phase 1: Mapping Landscape ({total_points} points)...")
    
    pbar = tqdm(total=total_points, desc="Heatmap Progress")
    
    for d in drift_vals:
        for r in rho_vals:
            # Capture 3D for center point
            is_capture = (d == 0.05 and r == 0.5)
            dim = 3 if is_capture else 2
            
            cfg = {'dim': dim, 'rho': r, 'drift_speed': d, 
                   'noise_type': 'gaussian', 'noise_scale': 0.5, 'geometry': 'ideal'}
            
            best_hp = tuner.tune(cfg)
            # Relaxed stats for heatmap speed (min=25, tol=15%)
            errs, succ, traj = run_adaptive_experiment(
                SGDOptimizer, best_hp, cfg, min_runs=25, max_runs=100, tol=0.15
            )
            
            db.append({'type':'heatmap', 'rho':r, 'drift':d, 
                       'success':succ, 'errors':errs, 'traj': traj if is_capture else None})
            pbar.update(1)
    pbar.close()

    # --- 2. STRESS TESTS ---
    print("Phase 2: Stress Testing...")
    scenarios = [
        {'name': 'Gaussian Ideal', 'noise': 'gaussian', 'cond': 1},
        {'name': 'Pareto Tail', 'noise': 'pareto', 'cond': 1},
        {'name': 'Valley (C=10)', 'noise': 'gaussian', 'cond': 10},
        {'name': 'Valley (C=100)', 'noise': 'gaussian', 'cond': 100},
    ]
    for s in tqdm(scenarios, desc="Stress Scenarios"):
        cfg = {'dim': 2, 'rho': 0.5, 'drift_speed': 0.02, 
               'noise_type': s['noise'], 'noise_scale': 0.5, 
               'geometry': 'distorted', 'condition_number': s['cond']}
        
        best_hp = tuner.tune(cfg)
        # Strict stats for tables
        errs, succ, _ = run_adaptive_experiment(
            SGDOptimizer, best_hp, cfg, min_runs=100, max_runs=500, tol=0.05
        )
        db.append({'type':'stress', 'noise':s['name'], 'cond':s['cond'], 'errors':errs})

    # --- 3. REPORT ---
    viz = BenchmarkVisualizer(db)
    viz.show_full_report()