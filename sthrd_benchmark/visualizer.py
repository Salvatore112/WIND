import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from math import pi
from IPython.display import display

class BenchmarkVisualizer:
    def __init__(self, results):
        self.db = results
    
    def show_dashboard(self):
        self._show_metrics()
        self.plot_heatmap()
        self.plot_radar()
        self.plot_degradation()
        self.plot_trajectory()
    
    def _show_metrics(self):
        st_data = [r for r in self.db if r['type']=='stress']
        if st_data:
            rows = []
            base_run = next((r for r in st_data if 'Ideal' in r['noise']), st_data[0])
            base_med = np.median(base_run['errors'])
            for r in st_data:
                e = np.array(r['errors'])
                rows.append({
                    'Scenario': r['noise'], 'Cond': r.get('cond',1),
                    'Median Err': np.median(e), 'Degradation': f"{np.median(e)/base_med:.1f}x"
                })
            display(pd.DataFrame(rows))
    
    def plot_heatmap(self):
        data = [r for r in self.db if r['type']=='heatmap']
        if not data:
            return
        df = pd.DataFrame([{'rho': r['rho'], 'drift': r['drift'], 'val': r['success']} for r in data])
        pivot = df.pivot(index="drift", columns="rho", values="val")
        fig = go.Figure(data=go.Heatmap(
            z=pivot.sort_index(ascending=False).values,
            x=pivot.columns,
            y=pivot.sort_index(ascending=False).index,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1
        ))
        fig.update_layout(title="Success Rate Landscape", xaxis_title="Rho", yaxis_title="Drift")
        fig.show()
    
    def plot_radar(self):
        cats = ['Gaussian', 'Pareto', 'Valley']
        vals = []
        for c in cats:
            runs = [r for r in self.db if r['type']=='stress' and
                   (c in r['noise'] or (c=='Valley' and r.get('cond')==10))]
            vals.append(np.median(np.concatenate([x['errors'] for x in runs])) if runs else 1e9)
        scores = 1.0 / (1.0 + np.log1p(vals))
        scores = np.concatenate((scores, [scores[0]]))
        angles = np.linspace(0, 2*pi, len(cats), endpoint=False).tolist()
        angles += [angles[0]]
        fig = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=[f'{a*180/pi:.0f}' for a in angles],
            fill='toself',
            line=dict(color='blue')
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(tickvals=angles[:-1], ticktext=cats)
            ),
            title="Robustness Profile"
        )
        fig.show()
    
    def plot_degradation(self):
        data = [r for r in self.db if r['type']=='stress' and 'cond' in r]
        if not data:
            return
        df = pd.DataFrame(data)
        conds = sorted(df['cond'].unique())
        means = [np.mean(np.concatenate(df[df['cond']==c]['errors'].values)) for c in conds]
        fig = go.Figure(data=go.Scatter(
            x=conds,
            y=means,
            mode='lines+markers',
            line=dict(color='red')
        ))
        fig.update_layout(
            xaxis=dict(type='log', title="Condition Number"),
            yaxis_title="Mean Error",
            title="Degradation by Geometry"
        )
        fig.show()
    
    def plot_trajectory(self):
        traj_run = next((r for r in self.db if r.get('traj') is not None), None)
        if not traj_run:
            return
        th, x = traj_run['traj']
        dists = np.linalg.norm(x - th, axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=dists,
            mode='lines',
            name='Distance to Target',
            line=dict(color='blue')
        ))
        fig.add_hline(y=np.mean(dists), line_dash="dash", line_color="red", annotation_text="Avg Lag")
        fig.update_layout(
            title=f"Tracking Dynamics (Drift={traj_run['drift']}, Rho={traj_run['rho']})",
            xaxis_title="Step",
            yaxis_title="Euclidean Dist"
        )
        fig.show()