"""
Core physics engine for ST-HRD (Scientifically Tested High-Rate Dynamic) Benchmark.
This module defines the fundamental physical laws of the simulation:
Drifts: How the optimum theta_t evolves over time (Stationary, Linear, Adaptive, etc.).
Landscapes: The geometry of the loss function f_t(x) = L(x, theta_t).
Noises: Stochastic corruptions applied to observations (Gaussian, Heavy-tailed, Correlated).
DynamicEnvironment: The container binding these laws together.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from dataclasses import dataclass


class Drift(ABC):
    """
    Abstract base class for environment dynamics (optimum evolution).
    """

    @abstractmethod
    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the next state theta_{t+1}.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        pass


class StationaryDrift(Drift):
    """
    Stationary environment: theta_{t+1} = theta_t.
    """

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return theta.copy()

    def reset(self) -> None:
        pass


class LinearDrift(Drift):
    """
    Linear trend: theta_{t+1} = theta_t + velocity.
    """

    def __init__(self, velocity: np.ndarray):
        self.velocity = np.array(velocity)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return theta + self.velocity

    def reset(self) -> None:
        pass


class RandomWalkDrift(Drift):
    """
    Gaussian Random Walk (Brownian Motion).
    """

    def __init__(
        self, sigma: float = 0.1, sparsity: float = 0.0, seed: Optional[int] = None
    ):
        self.sigma = sigma
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        dim = theta.shape[0]
        noise = self.rng.normal(0, self.sigma, size=dim)

        if self.sparsity > 0:
            mask = self.rng.random(dim) > self.sparsity
            noise *= mask

        return theta + noise

    def reset(self) -> None:
        pass


class CyclicDrift(Drift):
    """
    Cyclic/Seasonal drift.
    """

    def __init__(self, amplitude: float, period: int, center: np.ndarray):
        self.amplitude = amplitude
        self.period = period
        self.center = np.array(center)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        phase = 2 * np.pi * (t + 1) / self.period
        offset = self.amplitude * np.sin(phase)
        return self.center + offset * np.ones_like(theta)

    def reset(self) -> None:
        pass


class JumpDrift(Drift):
    """
    Abrupt changes (Teleportation / Regime Shift).
    """

    def __init__(
        self, interval: int, jump_magnitude: float, dim: int, seed: Optional[int] = None
    ):
        self.interval = interval
        self.jump_magnitude = jump_magnitude
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if (t + 1) % self.interval == 0:
            direction = self.rng.normal(size=self.dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            return theta + direction * self.jump_magnitude
        return theta

    def reset(self) -> None:
        pass


class AdaptiveDrift(Drift):
    """
    Adaptive Drift with mode selection.
    """

    def __init__(
        self, alpha: float = 0.1, threshold: float = 1.0, mode: str = "pursuit"
    ):
        self.alpha = alpha
        self.threshold = threshold
        if mode not in ["pursuit", "evasion"]:
            raise ValueError("Mode must be 'pursuit' or 'evasion'")
        self.mode = mode

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if action is None:
            return theta

        diff = theta - action
        dist = np.linalg.norm(diff)

        if dist < self.threshold and dist > 1e-8:
            direction = diff / dist
            sign = 1.0 if self.mode == "evasion" else -1.0
            return theta + sign * self.alpha * direction

        return theta

    def reset(self) -> None:
        pass


class Landscape(ABC):
    """
    Abstract base class for the loss landscape geometry.
    """

    @abstractmethod
    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Compute function value f_t(x)."""
        pass

    @abstractmethod
    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute gradient nabla f_t(x)."""
        pass


class QuadraticLandscape(Landscape):
    """
    Quadratic Loss: f(x) = 1/2 * (x-theta)^T * A * (x-theta).
    """

    def __init__(
        self, dim: int, condition_number: float = 1.0, seed: Optional[int] = None
    ):
        self.dim = dim
        self.kappa = condition_number
        self.rng = np.random.default_rng(seed)
        self.A = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        H = self.rng.normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(H)

        if self.dim > 1:
            eigenvalues = np.linspace(1, self.kappa, self.dim)
        else:
            eigenvalues = np.array([float(self.kappa)])

        return Q @ np.diag(eigenvalues) @ Q.T

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = x - theta
        return 0.5 * float(diff.T @ self.A @ diff)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.A @ (x - theta)


class PNormLandscape(Landscape):
    """
    p-Norm Loss: f(x) = (1/p) * ||x - theta||_p^p.
    """

    def __init__(self, p: float = 2.0):
        if p < 1.0:
            raise ValueError("p must be >= 1 for convexity.")
        self.p = p

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = np.abs(x - theta)
        return float(np.sum(diff**self.p) / self.p)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        diff = x - theta
        return np.sign(diff) * (np.abs(diff) ** (self.p - 1))


class RosenbrockLandscape(Landscape):
    """
    Rosenbrock 'Banana' function.
    """

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        z = x - theta + 1.0

        val = 0.0
        for i in range(len(x) - 1):
            val += 100.0 * (z[i + 1] - z[i] ** 2) ** 2 + (1.0 - z[i]) ** 2
        return float(val)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        z = x - theta + 1.0
        grad = np.zeros_like(x)
        dim = len(x)

        for i in range(dim - 1):
            grad[i] += -400.0 * z[i] * (z[i + 1] - z[i] ** 2) - 2.0 * (1.0 - z[i])
            grad[i + 1] += 200.0 * (z[i + 1] - z[i] ** 2)

        return grad


class MultiExtremalLandscape(Landscape):
    """
    Gaussian Mixture Landscape with EXACT normalization guaranteeing L(theta, theta) = 0.
    """

    def __init__(
        self, k_centers: int = 3, width: float = 1.0, seed: Optional[int] = None
    ):
        if k_centers < 1:
            raise ValueError("k_centers must be >= 1")
        self.width_sq = width**2
        self.k = k_centers
        self._seed = seed
        self.offsets = None

    def _lazy_init(self, dim: int):
        if self.offsets is None:
            rng = np.random.default_rng(self._seed)
            self.offsets = np.zeros((self.k, dim))
            if self.k > 1:
                self.offsets[1:] = rng.normal(0, 2.0, size=(self.k - 1, dim))

    def _raw_value(self, x: np.ndarray, theta: np.ndarray) -> float:
        centers = theta + self.offsets
        diffs = x - centers
        dists_sq = np.sum(diffs**2, axis=1)

        weights = np.full(self.k, 0.3)
        weights[0] = 1.0

        terms = weights * np.exp(-dists_sq / self.width_sq)
        return -np.sum(terms)

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        self._lazy_init(x.shape[0])

        f_raw_x = self._raw_value(x, theta)
        f_raw_theta = self._raw_value(theta, theta)

        return float(f_raw_x - f_raw_theta)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self._lazy_init(x.shape[0])

        centers = theta + self.offsets
        diffs = x - centers
        dists_sq = np.sum(diffs**2, axis=1)

        weights = np.full(self.k, 0.3)
        weights[0] = 1.0

        exps = np.exp(-dists_sq / self.width_sq)
        coeffs = weights * exps * (2.0 / self.width_sq)

        grad = np.sum(coeffs[:, np.newaxis] * diffs, axis=0)
        return grad


class RobustLandscape(Landscape):
    """
    Huber Loss.
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = np.abs(x - theta)
        is_small = diff <= self.delta
        squared = 0.5 * diff**2
        linear = self.delta * (diff - 0.5 * self.delta)
        return float(np.sum(np.where(is_small, squared, linear)))

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        diff = x - theta
        abs_diff = np.abs(diff)
        is_small = abs_diff <= self.delta
        return np.where(is_small, diff, self.delta * np.sign(diff))


class Noise(ABC):
    """
    Abstract base class for measurement noise.
    """

    @abstractmethod
    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        """
        Apply noise to the signal.
        """
        pass

    def reset(self) -> None:
        """Reset internal state."""
        pass


class GaussianNoise(Noise):
    """Standard additive Gaussian noise."""

    def __init__(self, sigma: float, seed: Optional[int] = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        if self.sigma <= 0:
            return signal
        noise = self.rng.normal(0, self.sigma, size=signal.shape)
        return signal + noise


class HeavyTailedNoise(Noise):
    """
    Pareto/Heavy-tailed noise.
    """

    def __init__(self, alpha: float, scale: float = 1.0, seed: Optional[int] = None):
        self.alpha = alpha
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        u = self.rng.random(size=signal.shape)
        pareto = (1.0 - u) ** (-1.0 / self.alpha) - 1.0
        signs = self.rng.choice([-1.0, 1.0], size=signal.shape)
        noise = signs * self.scale * pareto

        noise = np.clip(noise, -1e3 * self.scale, 1e3 * self.scale)
        return signal + noise


class CorrelatedNoise(Noise):
    """
    AR(1) Correlated Noise.
    """

    def __init__(
        self, sigma: float, phi: float = 0.8, dim: int = 1, seed: Optional[int] = None
    ):
        self.sigma = sigma
        self.phi = phi
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._state = None

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        if self._state is None or self._state.shape != signal.shape:
            self._state = np.zeros_like(signal)

        epsilon = self.rng.normal(0, self.sigma, size=signal.shape)
        scaling = np.sqrt(1 - self.phi**2)
        self._state = self.phi * self._state + scaling * epsilon

        return signal + self._state


class QuantizedNoise(Noise):
    """
    Quantized Noise.
    """

    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        if self.delta <= 0:
            return signal
        return np.round(signal / self.delta) * self.delta


class MultiplicativeNoise(Noise):
    """
    Multiplicative Noise: signal * (1 + epsilon).
    """

    def __init__(self, sigma_rel: float = 0.1, seed: Optional[int] = None):
        self.sigma_rel = sigma_rel
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        epsilon = self.rng.normal(0, self.sigma_rel, size=signal.shape)
        return signal * (1.0 + epsilon)


class SparseNoise(Noise):
    """
    Sparse (Bernoulli) Noise.
    """

    def __init__(self, sigma: float, p: float = 0.1, seed: Optional[int] = None):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.sigma = sigma
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        mask = self.rng.binomial(1, self.p, size=signal.shape).astype(float)
        noise = self.rng.normal(0, self.sigma, size=signal.shape)
        return signal + (mask * noise)


@dataclass
class EnvironmentConfig:
    """Configuration for reproducible environment creation."""

    dim: int
    x_bounds: Tuple[float, float]
    drift_type: str
    drift_params: Dict[str, Any]
    landscape_type: str
    landscape_params: Dict[str, Any]
    noise_type: str
    noise_params: Dict[str, Any]


class DynamicEnvironment:
    """
    The physical world simulator.
    """

    def __init__(
        self,
        dim: int,
        drift: Drift,
        landscape: Landscape,
        initial_theta: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.dim = dim
        self.drift = drift
        self.landscape = landscape
        self.bounds = bounds

        if initial_theta is not None:
            self._theta = np.array(initial_theta, dtype=np.float64)
        else:
            self._theta = np.zeros(dim)

        self.t = 0

        self._locked_by_oracle = False
        self._is_advancing = False
        self._registered_oracle = None

    def get_current_theta(self, for_analysis: bool = False) -> np.ndarray:
        """
        Get current state.
        """
        return self._theta.copy()

    def step(self, action: Optional[np.ndarray] = None) -> None:
        """
        Advance environment time: t -> t+1.
        """
        if self._locked_by_oracle:
            raise RuntimeError(
                "Cannot advance environment while Oracle is active (in a step). "
                "Call oracle.end_step() first."
            )

        self._is_advancing = True

        self._theta = self.drift.step(self._theta, self.t, action)

        if self.bounds is not None:
            low, high = self.bounds
            self._theta = np.clip(self._theta, low, high)

        self.t += 1

        self._is_advancing = False

    def reset(self, initial_theta: Optional[np.ndarray] = None) -> None:
        """Reset environment to t=0."""
        self.t = 0
        if initial_theta is not None:
            self._theta = np.array(initial_theta)
        else:
            self._theta = np.zeros(self.dim)

        self.drift.reset()
        if self._registered_oracle:
            self._registered_oracle.reset()

    def _register_oracle(self, oracle: Any) -> None:
        self._registered_oracle = oracle

    def _lock_for_oracle(self) -> None:
        """Called by Oracle.start_step()."""
        if self._is_advancing:
            raise RuntimeError("Cannot lock environment while it is advancing.")
        self._locked_by_oracle = True

    def _unlock_for_oracle(self) -> None:
        """Called by Oracle.end_step()."""
        self._locked_by_oracle = False


_DRIFT_REGISTRY = {
    "stationary": StationaryDrift,
    "linear": LinearDrift,
    "random_walk": RandomWalkDrift,
    "cyclic": CyclicDrift,
    "jump": JumpDrift,
    "adaptive": AdaptiveDrift,
}

_LANDSCAPE_REGISTRY = {
    "quadratic": QuadraticLandscape,
    "pnorm": PNormLandscape,
    "rosenbrock": RosenbrockLandscape,
    "multiextremal": MultiExtremalLandscape,
    "robust": RobustLandscape,
}

_NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "heavy_tailed": HeavyTailedNoise,
    "correlated": CorrelatedNoise,
    "quantized": QuantizedNoise,
    "multiplicative": MultiplicativeNoise,
    "sparse": SparseNoise,
}


def make_drift(name: str, **kwargs) -> Drift:
    if name not in _DRIFT_REGISTRY:
        raise ValueError(
            f"Unknown drift type: {name}. Available: {list(_DRIFT_REGISTRY.keys())}"
        )
    return _DRIFT_REGISTRY[name](**kwargs)


def make_landscape(name: str, **kwargs) -> Landscape:
    if name not in _LANDSCAPE_REGISTRY:
        raise ValueError(
            f"Unknown landscape: {name}. Available: {list(_LANDSCAPE_REGISTRY.keys())}"
        )
    return _LANDSCAPE_REGISTRY[name](**kwargs)


def make_noise(name: str, **kwargs) -> Optional[Noise]:
    if name is None or name == "none":
        return None
    if name not in _NOISE_REGISTRY:
        raise ValueError(
            f"Unknown noise: {name}. Available: {list(_NOISE_REGISTRY.keys())}"
        )
    return _NOISE_REGISTRY[name](**kwargs)


def make_environment(config: Dict[str, Any], seed: int = 42) -> DynamicEnvironment:
    """
    Factory method to create a fully configured environment.
    """
    dim = config.get("dim", 2)

    d_conf = config.get("drift", {"type": "stationary"})
    d_type = d_conf.pop("type")
    if "seed" not in d_conf and d_type in ["random_walk", "jump"]:
        d_conf["seed"] = seed
    if "dim" not in d_conf and d_type in ["jump"]:
        d_conf["dim"] = dim
    drift = make_drift(d_type, **d_conf)

    l_conf = config.get("landscape", {"type": "quadratic"})
    l_type = l_conf.pop("type")
    if "dim" not in l_conf and l_type in ["quadratic"]:
        l_conf["dim"] = dim
    if "seed" not in l_conf and l_type in ["quadratic", "multiextremal"]:
        l_conf["seed"] = seed + 1
    landscape = make_landscape(l_type, **l_conf)

    bounds = config.get("x_bounds", None)

    return DynamicEnvironment(dim=dim, drift=drift, landscape=landscape, bounds=bounds)
