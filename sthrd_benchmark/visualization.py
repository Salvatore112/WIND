import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


class BenchmarkVisualizer:
    def __init__(self, results):
        self.results = results

    def plot_trajectory_3d(self, max_points=500):
        if "history" not in self.results:
            return

        history = self.results["history"]
        if not history["x"]:
            return

        n_points = min(len(history["x"]), max_points)

        indices = np.linspace(0, len(history["x"]) - 1, n_points, dtype=int)

        x_traj = [history["x"][i] for i in indices]
        theta_traj = [history["theta"][i] for i in indices]

        if len(x_traj[0]) < 3:
            print(
                f"Trajectory has dimension {len(x_traj[0])}, need at least 3 for 3D plot"
            )
            return

        fig = go.Figure()

        x_vals = [x[0] for x in x_traj]
        y_vals = [x[1] for x in x_traj]
        z_vals = [x[2] for x in x_traj]

        theta_x = [t[0] for t in theta_traj]
        theta_y = [t[1] for t in theta_traj]
        theta_z = [t[2] for t in theta_traj]

        fig.add_trace(
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode="lines",
                name="Algorithm",
                line=dict(color="red", width=4),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=theta_x,
                y=theta_y,
                z=theta_z,
                mode="lines",
                name="Target",
                line=dict(color="blue", width=4, dash="dash"),
            )
        )

        fig.update_layout(
            title="3D Trajectory",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        return fig

    def plot_trajectory_2d(self, max_points=500):
        if "history" not in self.results:
            return

        history = self.results["history"]
        if not history["x"]:
            return

        n_points = min(len(history["x"]), max_points)

        indices = np.linspace(0, len(history["x"]) - 1, n_points, dtype=int)

        x_traj = [history["x"][i] for i in indices]
        theta_traj = [history["theta"][i] for i in indices]

        fig = go.Figure()

        x_vals = [x[0] for x in x_traj]
        y_vals = [x[1] for x in x_traj]

        theta_x = [t[0] for t in theta_traj]
        theta_y = [t[1] for t in theta_traj]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Algorithm",
                line=dict(color="red", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=theta_x,
                y=theta_y,
                mode="lines",
                name="Target",
                line=dict(color="blue", width=3, dash="dash"),
            )
        )

        fig.update_layout(
            title="2D Trajectory", xaxis_title="X", yaxis_title="Y", showlegend=True
        )

        return fig

    def plot_error_convergence(self):
        if "history" not in self.results:
            return

        history = self.results["history"]
        if not history["x"]:
            return

        errors = [
            np.linalg.norm(x - theta)
            for x, theta in zip(history["x"], history["theta"])
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["time"],
                y=errors,
                mode="lines",
                name="Tracking Error",
                line=dict(color="red", width=2),
            )
        )

        fig.update_layout(
            title="Tracking Error Convergence",
            xaxis_title="Time Step",
            yaxis_title="Error",
        )

        return fig

    def plot_metrics_dashboard(self):
        metrics = self.results.get("metrics", {})

        if not metrics:
            return

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        fig = go.Figure(data=[go.Bar(x=metric_names, y=metric_values)])

        fig.update_layout(
            title="Performance Metrics", xaxis_title="Metric", yaxis_title="Value"
        )

        return fig

    def plot_comparison_radar(self, comparison_results):
        if not comparison_results:
            return

        algorithm_names = list(comparison_results.keys())
        metrics = list(comparison_results[algorithm_names[0]].keys())

        fig = go.Figure()

        for algo_name in algorithm_names:
            values = [
                comparison_results[algo_name].get(metric, 0) for metric in metrics
            ]
            fig.add_trace(
                go.Scatterpolar(r=values, theta=metrics, fill="toself", name=algo_name)
            )

        max_val = max(
            [
                max([v for v in comparison_results[algo].values() if v is not None])
                for algo in algorithm_names
            ]
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, max_val * 1.1 if max_val > 0 else 1]
                )
            ),
            title="Algorithm Comparison Radar",
            showlegend=True,
        )

        return fig

    def plot_heatmap_analysis(
        self, grid_results_df, x_param, y_param, value_column="avg_error"
    ):
        if grid_results_df.empty:
            return

        pivot = grid_results_df.pivot(
            index=y_param, columns=x_param, values=value_column
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis"
            )
        )

        fig.update_layout(
            title=f"Heatmap: {value_column} vs {x_param} and {y_param}",
            xaxis_title=x_param,
            yaxis_title=y_param,
        )

        return fig

    def create_dashboard(self):
        figs = []

        traj_fig = self.plot_trajectory_3d()
        if traj_fig:
            figs.append(traj_fig)
        else:
            traj_2d_fig = self.plot_trajectory_2d()
            if traj_2d_fig:
                figs.append(traj_2d_fig)

        error_fig = self.plot_error_convergence()
        if error_fig:
            figs.append(error_fig)

        metrics_fig = self.plot_metrics_dashboard()
        if metrics_fig:
            figs.append(metrics_fig)

        return figs

    def show_dashboard(self):
        figs = self.create_dashboard()
        for fig in figs:
            fig.show()
