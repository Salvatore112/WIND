from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Protocol, runtime_checkable
import numpy as np
from core import DynamicEnvironment, Noise


@dataclass(frozen=True)
class Observation:
    """
    Immutable observation container enforcing the Information Barrier.
    """

    x: np.ndarray
    t: int
    value: Optional[float]
    grad: Optional[np.ndarray]
    optimum_value: float
    query_index: int = 0
    mode: str = "first-order"

    def __post_init__(self):
        object.__setattr__(self, "x", self.x.copy())
        if self.grad is not None:
            object.__setattr__(self, "grad", self.grad.copy())

        forbidden = ["theta", "optimal_point", "minimum_location", "_theta"]
        for attr in forbidden:
            if hasattr(self, attr):
                raise RuntimeError(
                    f"Information Barrier Violation: '{attr}' exposed in Observation."
                )

        if self.value is not None and not np.isfinite(self.value):
            raise ValueError(f"Non-finite value observed: {self.value}")
        if self.grad is not None and not np.all(np.isfinite(self.grad)):
            raise ValueError("Non-finite gradient observed.")


@runtime_checkable
class OracleProtocol(Protocol):
    """
    Protocol defining the strict temporal interaction flow.
    """

    def start_step(self, t: int) -> None: ...
    def query(self, x: np.ndarray) -> Observation: ...
    def end_step(self) -> None: ...
    def reset(self) -> None: ...
    def get_current_theta(self) -> np.ndarray: ...


class Oracle(ABC):
    """
    Abstract base oracle handling Protocol A enforcement and noise application.
    """

    def __init__(
        self,
        environment: DynamicEnvironment,
        value_noise: Optional[Noise] = None,
        grad_noise: Optional[Noise] = None,
        seed: Optional[int] = None,
    ):
        if environment is None:
            raise ValueError("Environment cannot be None")

        self.environment = environment
        self.value_noise = value_noise
        self.grad_noise = grad_noise
        self.rng = np.random.default_rng(seed)

        self.n_queries = 0
        self.n_grad_queries = 0
        self.n_value_queries = 0

        self._current_t = -1
        self._step_active = False
        self._cached_theta: Optional[np.ndarray] = None

        if hasattr(environment, "_register_oracle"):
            environment._register_oracle(self)

    def reset(self) -> None:
        self.n_queries = 0
        self.n_grad_queries = 0
        self.n_value_queries = 0
        self._current_t = -1
        self._step_active = False
        self._cached_theta = None

        if self.value_noise and hasattr(self.value_noise, "reset"):
            self.value_noise.reset()
        if self.grad_noise and hasattr(self.grad_noise, "reset"):
            self.grad_noise.reset()

    def start_step(self, t: int) -> None:
        if self._step_active:
            raise RuntimeError("Step already active. Call end_step() first.")

        if self.environment._is_advancing:
            raise RuntimeError(
                "Environment is currently advancing. "
                "Protocol A violation: cannot lock during environment step."
            )

        self._current_t = t
        self._cached_theta = self.environment.get_current_theta(for_analysis=True)
        self._step_active = True

        if hasattr(self.environment, "_lock_for_oracle"):
            self.environment._lock_for_oracle()

    def end_step(self) -> None:
        if not self._step_active:
            raise RuntimeError("No step active. Call start_step() first.")

        self._step_active = False
        if hasattr(self.environment, "_unlock_for_oracle"):
            self.environment._unlock_for_oracle()

    def get_current_theta(self) -> np.ndarray:
        if not self._step_active or self._cached_theta is None:
            raise RuntimeError(
                "get_current_theta() called outside active step. "
                "Protocol A requires: start_step() → query() → end_step() sequence."
            )
        return self._cached_theta.copy()

    def _apply_noise(
        self, clean_value: Optional[float], clean_grad: Optional[np.ndarray], t: int
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        noisy_value = clean_value
        noisy_grad = clean_grad

        if self.value_noise is not None and clean_value is not None:
            val_arr = np.array([clean_value], dtype=np.float64)
            noisy_arr = self.value_noise.apply(val_arr, t)
            noisy_value = float(noisy_arr[0])

        if self.grad_noise is not None and clean_grad is not None:
            noisy_grad = self.grad_noise.apply(clean_grad.copy(), t)

        return noisy_value, noisy_grad

    @abstractmethod
    def query(self, x: np.ndarray) -> Observation:
        pass


class FirstOrderOracle(Oracle):
    """
    Standard Gradient Oracle with strict compliance modes.
    """

    def __init__(
        self,
        environment: DynamicEnvironment,
        value_noise: Optional[Noise] = None,
        grad_noise: Optional[Noise] = None,
        seed: Optional[int] = None,
        blind_value: bool = False,
    ):
        super().__init__(environment, value_noise, grad_noise, seed)
        self.blind_value = blind_value

    def query(self, x: np.ndarray) -> Observation:
        theta = self.get_current_theta()

        clean_val = self.environment.landscape.loss(x, theta)
        clean_grad = self.environment.landscape.grad(x, theta)
        optimum_val = self.environment.landscape.loss(theta, theta)

        noisy_val, noisy_grad = self._apply_noise(
            clean_val, clean_grad, self._current_t
        )
        final_val = None if self.blind_value else noisy_val

        self.n_queries += 1
        self.n_grad_queries += 1
        if not self.blind_value:
            self.n_value_queries += 1

        return Observation(
            x=x.copy(),
            t=self._current_t,
            value=final_val,
            grad=noisy_grad,
            optimum_value=optimum_val,
            query_index=self.n_queries,
            mode="first-order-blind" if self.blind_value else "first-order",
        )


class ZeroOrderOracle(Oracle):
    """
    Black-box Oracle returning ONLY function values.
    """

    def query(self, x: np.ndarray) -> Observation:
        theta = self.get_current_theta()

        clean_val = self.environment.landscape.loss(x, theta)
        optimum_val = self.environment.landscape.loss(theta, theta)

        noisy_val, _ = self._apply_noise(clean_val, None, self._current_t)

        self.n_queries += 1
        self.n_value_queries += 1

        return Observation(
            x=x.copy(),
            t=self._current_t,
            value=noisy_val,
            grad=None,
            optimum_value=optimum_val,
            query_index=self.n_queries,
            mode="zero-order",
        )


class HybridOracle(Oracle):
    """
    Oracle switching between First-Order and Zero-Order modes.
    """

    def __init__(
        self,
        environment: DynamicEnvironment,
        schedule: List[Tuple[str, int]],
        value_noise: Optional[Noise] = None,
        grad_noise: Optional[Noise] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(environment, value_noise, grad_noise, seed)
        if not schedule:
            raise ValueError("Schedule cannot be empty")
        self.schedule = schedule
        self._schedule_idx = 0
        self._steps_in_mode = 0
        self._current_mode = schedule[0][0]

    def start_step(self, t: int) -> None:
        while (
            self._schedule_idx < len(self.schedule) - 1
            and self._steps_in_mode >= self.schedule[self._schedule_idx][1]
        ):
            self._schedule_idx += 1
            self._steps_in_mode = 0

        self._current_mode = self.schedule[self._schedule_idx][0]
        self._steps_in_mode += 1

        super().start_step(t)

    def query(self, x: np.ndarray) -> Observation:
        theta = self.get_current_theta()
        optimum_val = self.environment.landscape.loss(theta, theta)

        if self._current_mode == "first-order":
            clean_val = self.environment.landscape.loss(x, theta)
            clean_grad = self.environment.landscape.grad(x, theta)
            noisy_val, noisy_grad = self._apply_noise(
                clean_val, clean_grad, self._current_t
            )

            self.n_queries += 1
            self.n_value_queries += 1
            self.n_grad_queries += 1

            return Observation(
                x=x.copy(),
                t=self._current_t,
                value=noisy_val,
                grad=noisy_grad,
                optimum_value=optimum_val,
                query_index=self.n_queries,
                mode=f"hybrid-fo-{self._schedule_idx}",
            )

        else:
            clean_val = self.environment.landscape.loss(x, theta)
            noisy_val, _ = self._apply_noise(clean_val, None, self._current_t)

            self.n_queries += 1
            self.n_value_queries += 1

            return Observation(
                x=x.copy(),
                t=self._current_t,
                value=noisy_val,
                grad=None,
                optimum_value=optimum_val,
                query_index=self.n_queries,
                mode=f"hybrid-zo-{self._schedule_idx}",
            )


_ORACLE_REGISTRY = {
    "first-order": FirstOrderOracle,
    "zero-order": ZeroOrderOracle,
    "hybrid": HybridOracle,
}


def make_oracle(name: str, environment: DynamicEnvironment, **kwargs) -> Oracle:
    """
    Factory method for oracle instantiation from configuration.
    """
    if name not in _ORACLE_REGISTRY:
        raise ValueError(
            f"Unknown oracle type: '{name}'. "
            f"Available: {list(_ORACLE_REGISTRY.keys())}"
        )

    return _ORACLE_REGISTRY[name](environment=environment, **kwargs)
