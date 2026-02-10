"""
Scientifically validated visualization module for ST-HRD benchmark.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class MetricVisualizer:
    """
    Visualizes the evolution of metrics over time for a single or multiple runs.
    """

    def __init__(self, results: List[Any]):
        self.results = results
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self) -> pd.DataFrame:
        data = []
        for res in self.results:
            if not hasattr(res, "metrics_history") or not res.metrics_history:
                continue

            T = len(next(iter(res.metrics_history.values())))
            steps = np.arange(T)

            seed = getattr(res.metadata, "seed", 0) if hasattr(res, "metadata") else 0
            algorithm = (
                getattr(res.optimizer_info, "name", "unknown")
                if hasattr(res, "optimizer_info")
                else "unknown"
            )

            run_df = pd.DataFrame({"step": steps, "seed": seed, "algorithm": algorithm})

            for metric_name, values in res.metrics_history.items():
                run_df[metric_name] = values

            data.append(run_df)

        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    def plot_metric_over_time(
        self,
        metric_name: str,
        title: Optional[str] = None,
        log_scale: bool = False,
        show_ci: bool = True,
    ) -> go.Figure:
        if self.df.empty or metric_name not in self.df.columns:
            return go.Figure().update_layout(title="No Data")

        stats = (
            self.df.groupby(["algorithm", "step"])[metric_name]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
        stats["upper"] = stats["mean"] + stats["ci"]
        stats["lower"] = stats["mean"] - stats["ci"]

        fig = go.Figure()

        colors = px.colors.qualitative.Plotly

        for i, algo in enumerate(stats["algorithm"].unique()):
            algo_data = stats[stats["algorithm"] == algo]
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=algo_data["step"],
                    y=algo_data["mean"],
                    mode="lines",
                    name=f"{algo}",
                    line=dict(color=color),
                )
            )

            if show_ci:
                fig.add_trace(
                    go.Scatter(
                        x=algo_data["step"],
                        y=algo_data["upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=algo_data["step"],
                        y=algo_data["lower"],
                        mode="lines",
                        fill="tonexty",
                        fillcolor=self._hex_to_rgba(color, 0.2),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        yaxis_type = "log" if log_scale else "linear"
        fig.update_layout(
            title=title or f"{metric_name} over Time",
            xaxis_title="Step (t)",
            yaxis_title=metric_name,
            yaxis_type=yaxis_type,
            template="plotly_white",
        )
        return fig

    def _hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        lv = len(hex_color)
        rgb = tuple(int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


class ComparativeVisualizer:
    """
    Visualizes aggregated performance across multiple experiments/environments.
    """

    def __init__(self, summary_df: pd.DataFrame):
        self.df = summary_df

    def plot_radar_chart(self, metrics: List[str]) -> go.Figure:
        if self.df.empty or not metrics:
            return go.Figure().update_layout(title="No Data")

        agg = self.df.groupby("algorithm")[metrics].mean()

        normalized = (agg.max() - agg) / (agg.max() - agg.min() + 1e-9)
        normalized = normalized.fillna(1.0)

        fig = go.Figure()

        for algo in normalized.index:
            values = normalized.loc[algo].values.tolist()
            values += [values[0]]
            metric_labels = metrics + [metrics[0]]

            fig.add_trace(
                go.Scatterpolar(r=values, theta=metric_labels, fill="toself", name=algo)
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Algorithm Robustness Profile (Higher is Better)",
            template="plotly_white",
        )
        return fig

    def plot_box_comparison(self, metric: str) -> go.Figure:
        if self.df.empty or metric not in self.df.columns:
            return go.Figure().update_layout(title="No Data")

        fig = px.box(
            self.df,
            x="algorithm",
            y=metric,
            color="algorithm",
            points="all",
            title=f"Distribution of {metric}",
        )
        return fig


class TrajectoryVisualizer:
    """
    Visualizes the path taken by the optimizer vs the moving optimum.
    """

    def __init__(self, history_x: List[np.ndarray], history_theta: List[np.ndarray]):
        self.X = np.array(history_x)
        self.Theta = np.array(history_theta)
        if len(self.X) == 0 or len(self.Theta) == 0:
            self.T = 0
            self.dim = 0
        else:
            self.T = len(self.X)
            self.dim = self.X.shape[1]

    def plot_2d_projection(self, dims: Tuple[int, int] = (0, 1)) -> go.Figure:
        if self.T == 0:
            return go.Figure().update_layout(title="No Trajectory Data")

        d1, d2 = dims
        if self.dim <= max(d1, d2):
            return go.Figure().update_layout(title="Dimensions out of range")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.Theta[:, d1],
                y=self.Theta[:, d2],
                mode="lines",
                name="Optimum (Theta)",
                line=dict(color="black", dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.X[:, d1],
                y=self.X[:, d2],
                mode="markers+lines",
                name="Agent (x)",
                marker=dict(
                    color=np.arange(self.T),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Time Step"),
                ),
            )
        )

        fig.update_layout(
            title=f"Trajectory Projection (Dim {d1} vs {d2})",
            xaxis_title=f"x_{d1}",
            yaxis_title=f"x_{d2}",
            template="plotly_white",
        )
        return fig

    def plot_3d_trajectory(self, dims: Tuple[int, int, int] = (0, 1, 2)) -> go.Figure:
        if self.T == 0 or self.dim < 3:
            return go.Figure().update_layout(title="Not enough dimensions for 3D")

        d1, d2, d3 = dims
        if self.dim <= max(d1, d2, d3):
            return go.Figure().update_layout(title="Dimensions out of range")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=self.Theta[:, d1],
                y=self.Theta[:, d2],
                z=self.Theta[:, d3],
                mode="lines",
                name="Optimum",
                line=dict(color="black", width=4),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=self.X[:, d1],
                y=self.X[:, d2],
                z=self.X[:, d3],
                mode="lines",
                name="Agent",
                line=dict(color="red", width=4),
            )
        )

        fig.update_layout(title="3D Trajectory", template="plotly_white")
        return fig
