from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
from core import DynamicEnvironment, Landscape
from oracle import Observation


class Metric(ABC):
    """
    Abstract base metric with standardized tail processing.
    """

    def __init__(
        self, name: str, direction: str = "minimize", rho: Optional[float] = None
    ):
        if direction not in ["minimize", "maximize"]:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if rho is not None and not (0 < rho <= 1.0):
            raise ValueError(f"Hölder exponent rho must be in (0, 1], got rho={rho}")
        self.name = name
        self.direction = direction
        self.rho = rho
        self._history: List[float] = []
        self._current_value: Optional[float] = None

    @abstractmethod
    def update(
        self,
        t: int,
        x: np.ndarray,
        theta: np.ndarray,
        observation: Observation,
        environment: DynamicEnvironment,
    ) -> None:
        pass

    def get_result(self, tail_fraction: float = 0.2) -> float:
        if not self._history:
            return float("nan")

        n = len(self._history)
        if tail_fraction <= 0.0 or tail_fraction >= 1.0:
            data = self._history
        else:
            start_idx = int(n * (1.0 - tail_fraction))
            if start_idx >= n:
                start_idx = max(0, n - 1)
            data = self._history[start_idx:]

        if not data:
            return float("nan")

        return self._aggregate(data)

    def get_history(self) -> List[float]:
        return self._history.copy()

    @abstractmethod
    def _aggregate(self, history: List[float]) -> float:
        pass

    def reset(self) -> None:
        self._history = []
        self._current_value = None


class TrackingErrorMetric(Metric):
    """
    Generalized tracking error: ||x_t - theta_t||_p with landscape-aware normalization.
    """

    def __init__(
        self,
        norm: Union[str, float] = "l2",
        matrix_A: Optional[np.ndarray] = None,
        normalize_by_dim: bool = False,
    ):
        name = f"error_{norm}" if isinstance(norm, str) else f"error_p{norm}"
        super().__init__(name=name, direction="minimize", rho=None)
        self.norm = norm
        self.A = matrix_A
        self.normalize = normalize_by_dim
        self._dim = None

    def update(self, t, x, theta, observation, environment):
        diff = x - theta

        if self.norm == "l2":
            error = np.linalg.norm(diff, ord=2)
        elif self.norm == "l1":
            error = np.linalg.norm(diff, ord=1)
        elif self.norm == "linf":
            error = np.linalg.norm(diff, ord=np.inf)
        elif self.norm == "mahalanobis":
            if self.A is None:
                if hasattr(environment.landscape, "A"):
                    self.A = environment.landscape.A
                else:
                    self.A = np.eye(len(x))
            quad_form = diff.T @ self.A @ diff
            error = np.sqrt(max(0.0, float(quad_form)))
        elif isinstance(self.norm, (int, float)):
            p = float(self.norm)
            if p < 1.0:
                raise ValueError(f"p-norm requires p >= 1, got p={p}")
            error = np.linalg.norm(diff, ord=p)
        else:
            raise ValueError(f"Unknown norm specification: {self.norm}")

        if self.normalize:
            if self._dim is None:
                self._dim = x.shape[0]
            error /= np.sqrt(self._dim)

        self._history.append(error)
        self._current_value = error

    def _aggregate(self, history: List[float]) -> float:
        return float(np.mean(history))


class MaxCoordinateErrorMetric(Metric):
    """
    Maximum absolute deviation across coordinates.
    """

    def __init__(self):
        super().__init__(name="max_coord_error", direction="minimize", rho=None)

    def update(self, t, x, theta, observation, environment):
        error = np.max(np.abs(x - theta))
        self._history.append(error)
        self._current_value = error

    def _aggregate(self, history: List[float]) -> float:
        return float(np.max(history))


class LyapunovMetric(Metric):
    """
    Lyapunov function V_n(x) = ‖x - θ_n‖_{ρ+1}^{ρ+1} for stabilization analysis.
    """

    def __init__(self, rho: float = 1.0):
        if not (0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        super().__init__(name=f"lyapunov_rho{rho:.2f}", direction="minimize", rho=rho)
        self.p_norm = rho + 1.0

    def update(self, t, x, theta, observation, environment):
        distance = np.linalg.norm(x - theta, ord=self.p_norm)
        lyapunov_value = distance**self.p_norm

        self._history.append(lyapunov_value)
        self._current_value = lyapunov_value

    def _aggregate(self, history: List[float]) -> float:
        return float(np.mean(history))


class NormalizedLyapunovMetric(Metric):
    """
    Normalized Lyapunov function V_n(x) / A^{ρ+1} for drift-invariant comparison.
    """

    def __init__(self, rho: float = 1.0, drift_magnitude_A: float = 0.1):
        if not (0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if drift_magnitude_A <= 0:
            raise ValueError(
                f"Drift magnitude A must be > 0, got A={drift_magnitude_A}"
            )
        super().__init__(
            name=f"norm_lyapunov_rho{rho:.2f}_A{drift_magnitude_A:.2f}",
            direction="minimize",
            rho=rho,
        )
        self.p_norm = rho + 1.0
        self.A = drift_magnitude_A
        self.normalization_factor = self.A**self.p_norm

    def update(self, t, x, theta, observation, environment):
        distance = np.linalg.norm(x - theta, ord=self.p_norm)
        lyapunov_value = distance**self.p_norm
        normalized = lyapunov_value / self.normalization_factor

        self._history.append(normalized)
        self._current_value = normalized

    def _aggregate(self, history: List[float]) -> float:
        return float(np.mean(history))


class AsymptoticBoundMetric(Metric):
    """
    Asymptotic upper bound estimator for Lyapunov function.
    """

    def __init__(self, rho: float = 1.0):
        if not (0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        super().__init__(
            name=f"asymptotic_bound_rho{rho:.2f}", direction="minimize", rho=rho
        )
        self.p_norm = rho + 1.0
        self._running_max: float = 0.0

    def update(self, t, x, theta, observation, environment):
        distance = np.linalg.norm(x - theta, ord=self.p_norm)
        lyapunov_value = distance**self.p_norm

        self._running_max = max(self._running_max, lyapunov_value)
        self._history.append(self._running_max)
        self._current_value = self._running_max

    def _aggregate(self, history: List[float]) -> float:
        if not history:
            return float("nan")
        tail_start = int(len(history) * 0.8)
        tail_values = history[tail_start:]
        return float(np.max(tail_values)) if tail_values else float("nan")


class InstantaneousLossMetric(Metric):
    """
    Instantaneous function value: f_t(x_t).
    """

    def __init__(self):
        super().__init__(name="instant_loss", direction="minimize", rho=None)

    def update(self, t, x, theta, observation, environment):
        loss_val = environment.landscape.loss(x, theta)
        self._history.append(loss_val)
        self._current_value = loss_val

    def _aggregate(self, history: List[float]) -> float:
        return float(np.mean(history))


class DynamicRegretMetric(Metric):
    """
    Dynamic Regret: R_T = Σ_{t=1}^T [f_t(x_t) - f_t(theta_t)].
    """

    def __init__(self, normalize_by_path: bool = False):
        super().__init__(name="dynamic_regret", direction="minimize", rho=None)
        self.normalize_by_path = normalize_by_path
        self._path_variation = 0.0
        self._prev_theta: Optional[np.ndarray] = None
        self._cumulative_regret = 0.0

    def update(self, t, x, theta, observation, environment):
        f_x = environment.landscape.loss(x, theta)
        f_theta = observation.optimum_value

        regret_t = f_x - f_theta
        self._cumulative_regret += regret_t

        if self.normalize_by_path and self._prev_theta is not None:
            self._path_variation += np.linalg.norm(theta - self._prev_theta)

        self._prev_theta = theta.copy()
        self._history.append(self._cumulative_regret)
        self._current_value = self._cumulative_regret

    def _aggregate(self, history: List[float]) -> float:
        if not history:
            return float("nan")
        total_regret = history[-1]

        if self.normalize_by_path and self._path_variation > 1e-6:
            return total_regret / self._path_variation

        return total_regret


class TimeToRecoveryMetric(Metric):
    """
    Time-to-Recovery (TTR): Steps to return to ε-neighborhood after abrupt jump.
    """

    def __init__(self, jump_threshold: float = 1.0, epsilon: float = 0.1):
        super().__init__(name="ttr", direction="minimize", rho=None)
        self.jump_threshold = jump_threshold
        self.epsilon = epsilon
        self._prev_theta: Optional[np.ndarray] = None
        self._recovery_events: List[int] = []
        self._active_recovery: Optional[Dict[str, int]] = None

    def update(self, t, x, theta, observation, environment):
        is_jump = False
        if self._prev_theta is not None:
            drift_mag = np.linalg.norm(theta - self._prev_theta)
            if drift_mag > self.jump_threshold:
                is_jump = True

        current_error = np.linalg.norm(x - theta)

        if self._active_recovery is not None:
            self._active_recovery["steps"] += 1
            if current_error <= self.epsilon:
                self._recovery_events.append(self._active_recovery["steps"])
                self._active_recovery = None

        if is_jump and current_error > self.epsilon:
            self._active_recovery = {"steps": 0}

        self._prev_theta = theta.copy()
        current_val = self._active_recovery["steps"] if self._active_recovery else 0
        self._history.append(float(current_val))
        self._current_value = float(current_val)

    def _aggregate(self, history: List[float]) -> float:
        if not self._recovery_events:
            return float("nan")
        return float(np.mean(self._recovery_events))


class DriftAdaptationMetric(Metric):
    """
    Drift Adaptation Score: Cosine similarity between optimizer motion and drift direction.
    """

    def __init__(self):
        super().__init__(name="drift_adaptation", direction="maximize", rho=None)
        self._prev_x: Optional[np.ndarray] = None
        self._prev_theta: Optional[np.ndarray] = None

    def update(self, t, x, theta, observation, environment):
        if t == 0:
            self._prev_x = x.copy()
            self._prev_theta = theta.copy()
            self._history.append(0.0)
            self._current_value = 0.0
            return

        dx = x - self._prev_x
        dtheta = theta - self._prev_theta

        norm_dx = np.linalg.norm(dx)
        norm_dtheta = np.linalg.norm(dtheta)

        if norm_dx > 1e-9 and norm_dtheta > 1e-9:
            cosine = np.dot(dx, dtheta) / (norm_dx * norm_dtheta)
            cosine = max(-1.0, min(1.0, cosine))
        else:
            cosine = 0.0

        self._history.append(cosine)
        self._current_value = cosine

        self._prev_x = x.copy()
        self._prev_theta = theta.copy()

    def _aggregate(self, history: List[float]) -> float:
        return float(np.mean(history))


class QueryEfficiencyMetric(Metric):
    """
    Query Efficiency: Mean geometric error per oracle query.
    """

    def __init__(self):
        super().__init__(name="query_efficiency", direction="minimize", rho=None)
        self._error_history: List[float] = []
        self._query_history: List[int] = []

    def update(self, t, x, theta, observation, environment):
        error = np.linalg.norm(x - theta)
        self._error_history.append(error)
        self._query_history.append(observation.query_index)
        self._history.append(error)
        self._current_value = error

    def _aggregate(self, history: List[float]) -> float:
        if not self._error_history or not self._query_history:
            return float("nan")

        n = len(self._error_history)
        tail_start = int(n * 0.8)
        tail_errors = self._error_history[tail_start:]
        mean_tail_error = (
            np.mean(tail_errors) if tail_errors else np.mean(self._error_history)
        )

        total_queries = self._query_history[-1]
        if total_queries <= 0:
            return float("inf")

        return float(mean_tail_error / total_queries)

    def reset(self) -> None:
        super().reset()
        self._error_history = []
        self._query_history = []


class MetricsCollection:
    """
    Container orchestrating multiple metrics with unified update protocol.
    """

    def __init__(self, metrics: List[Metric]):
        if not metrics:
            raise ValueError("MetricsCollection requires at least one metric")
        self.metrics = metrics

        rho_values = {m.rho for m in metrics if m.rho is not None}
        if len(rho_values) > 1:
            raise ValueError(
                f"Inconsistent Hölder exponents across metrics: {sorted(rho_values)}"
            )

    def update(
        self,
        t: int,
        x: np.ndarray,
        theta: np.ndarray,
        observation: Observation,
        environment: DynamicEnvironment,
    ) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            metric.update(t, x, theta, observation, environment)
            if metric._current_value is not None:
                results[metric.name] = metric._current_value
        return results

    def get_results(self, tail_fraction: float = 0.2) -> Dict[str, float]:
        return {
            metric.name: metric.get_result(tail_fraction=tail_fraction)
            for metric in self.metrics
        }

    def get_histories(self) -> Dict[str, List[float]]:
        return {metric.name: metric.get_history() for metric in self.metrics}

    def get_directions(self) -> Dict[str, str]:
        return {metric.name: metric.direction for metric in self.metrics}

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def get_current_values(self) -> Dict[str, float]:
        return {
            metric.name: metric._current_value
            for metric in self.metrics
            if metric._current_value is not None
        }
