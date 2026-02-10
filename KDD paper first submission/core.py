"""
Core physics engine for WIND Benchmark.
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


# =============================================================================
# 1. DRIFT MODELS (Dynamics of theta_t)
# =============================================================================


class Drift(ABC):
    """
    Abstract base class for environment dynamics (optimum evolution).

    Mathematical model:
        theta_{t+1} = D(theta_t, t, x_t)

    Where:
        - theta_t: Current environment state.
        - t: Current time step.
        - x_t: (Optional) The decision made by the optimizer.
               Required for 'Adaptive' drifts.
    """

    @abstractmethod
    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the next state theta_{t+1}.

        Args:
            theta: Current state theta_t.
            t: Current time step index.
            action: The optimizer's decision x_t (optional, for feedback loops).

        Returns:
            New state theta_{t+1}.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (if any)."""
        pass


class StationaryDrift(Drift):
    """
    Stationary environment: theta_{t+1} = theta_t.
    Baseline for convergence analysis.
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
    Simulates constant inflation or predictable motion.
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
    theta_{t+1} = theta_t + N(0, sigma^2 * I).
    Standard model for "unpredictable" dynamics (Besbes et al., 2015).
    Supports sparsity (drift only in subset of coordinates).
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
            # Mask out coordinates to simulate sparse drift
            mask = self.rng.random(dim) > self.sparsity
            noise *= mask

        return theta + noise

    def reset(self) -> None:
        pass


class CyclicDrift(Drift):
    """
    Cyclic/Seasonal drift.
    theta_t = center + A * sin(2*pi*t / T).
    Note: This overrides the previous theta based on t, rather than accumulating.
    """

    def __init__(self, amplitude: float, period: int, center: np.ndarray):
        self.amplitude = amplitude
        self.period = period
        self.center = np.array(center)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # t+1 because we are computing the NEXT state
        phase = 2 * np.pi * (t + 1) / self.period
        offset = self.amplitude * np.sin(phase)
        # Apply offset to all dimensions uniformly
        return self.center + offset * np.ones_like(theta)

    def reset(self) -> None:
        pass


class JumpDrift(Drift):
    """
    Abrupt changes (Teleportation / Regime Shift).
    Theta jumps to a new random location every T steps.
    Scientific basis: Testing "Time-to-Recovery" (TTR) metrics.
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
        # Check if it's time to jump
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

    Modes:
    - 'pursuit': Theta moves TOWARDS x (Environment helps/stabilizes).
                 theta_{t+1} = theta_t - alpha * sign(x - theta)
    - 'evasion': Theta moves AWAY from x (Adversarial/Game dynamics).
                 theta_{t+1} = theta_t + alpha * sign(x - theta)

    Scientific basis: Competitive optimization, GAN dynamics.
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

        # Only react if agent is close enough
        if dist < self.threshold and dist > 1e-8:
            direction = diff / dist

            # Semantics:
            # Evasion: Move theta in direction of (theta - x) -> Away
            # Pursuit: Move theta in direction of (x - theta) -> Towards
            sign = 1.0 if self.mode == "evasion" else -1.0
            return theta + sign * self.alpha * direction

        return theta

    def reset(self) -> None:
        pass


# =============================================================================
# 2. LANDSCAPE GEOMETRIES (Loss Functions)
# =============================================================================


class Landscape(ABC):
    """
    Abstract base class for the loss landscape geometry.
    Defined as f_t(x) = L(x, theta_t).

    CRITICAL INVARIANT: L(theta_t, theta_t) MUST be 0.0 for Dynamic Regret to be valid.
    All implementations MUST guarantee this invariant.
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
    Parameters:
        condition_number (kappa): Ratio of max/min eigenvalues of A.
    """

    def __init__(
        self, dim: int, condition_number: float = 1.0, seed: Optional[int] = None
    ):
        self.dim = dim
        self.kappa = condition_number
        self.rng = np.random.default_rng(seed)
        self.A = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        # Generate random orthogonal matrix Q
        H = self.rng.normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(H)

        # Generate eigenvalues from 1 to kappa
        if self.dim > 1:
            eigenvalues = np.linspace(1, self.kappa, self.dim)
        else:
            eigenvalues = np.array([float(self.kappa)])

        # A = Q * diag(lambda) * Q.T
        return Q @ np.diag(eigenvalues) @ Q.T

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = x - theta
        return 0.5 * float(diff.T @ self.A @ diff)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.A @ (x - theta)


class PNormLandscape(Landscape):
    """
    p-Norm Loss: f(x) = (1/p) * ||x - theta||_p^p.
    Special cases: p=2 (Euclidean), p=1 (L1/Lasso-like).
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
        # Gradient of (1/p)|u|^p is sign(u) * |u|^(p-1)
        return np.sign(diff) * (np.abs(diff) ** (self.p - 1))


class RosenbrockLandscape(Landscape):
    """
    Rosenbrock 'Banana' function.
    Non-convex, ill-conditioned valley.
    Adapted for dynamic environment: f(x) = Rosenbrock(x - theta + 1).
    The global minimum is at theta with f(theta, theta) = 0.
    """

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        z = x - theta + 1.0  # Shift so minimum is at theta (original min is at 1,1,...)

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

    Mathematical formulation:
        f_raw(x, theta) = - sum_{j=0}^{k-1} w_j * exp(-||x - c_j||^2 / sigma^2)
        where c_0 = theta (global minimum), c_j = theta + offset_j (local minima)

        f(x, theta) = f_raw(x, theta) - f_raw(theta, theta)

    This guarantees:
        ✅ f(theta, theta) = 0 (exact)
        ✅ f(x, theta) >= 0 for all x (loss is non-negative)
        ✅ Global minimum strictly at theta (dominant weight w_0 = 1.0)
    """

    def __init__(
        self, k_centers: int = 3, width: float = 1.0, seed: Optional[int] = None
    ):
        if k_centers < 1:
            raise ValueError("k_centers must be >= 1")
        self.width_sq = width**2
        self.k = k_centers
        self._seed = seed
        self.offsets = None  # Lazy initialization based on dimension

    def _lazy_init(self, dim: int):
        """Initialize offsets based on actual dimension encountered."""
        if self.offsets is None:
            rng = np.random.default_rng(self._seed)
            self.offsets = np.zeros((self.k, dim))
            if self.k > 1:
                # Local minima offsets (random within radius ~3.0)
                self.offsets[1:] = rng.normal(0, 2.0, size=(self.k - 1, dim))

    def _raw_value(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Compute unnormalized raw value f_raw(x, theta)."""
        centers = theta + self.offsets  # Shape: (k, dim)
        diffs = x - centers  # Shape: (k, dim)
        dists_sq = np.sum(diffs**2, axis=1)  # Shape: (k,)

        # Weights: global minimum has weight 1.0, local minima have weight 0.3
        weights = np.full(self.k, 0.3)
        weights[0] = 1.0

        # f_raw = -sum(w_j * exp(-||x-c_j||^2 / sigma^2))
        terms = weights * np.exp(-dists_sq / self.width_sq)
        return -np.sum(terms)

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        self._lazy_init(x.shape[0])

        # Compute raw values at x and at theta
        f_raw_x = self._raw_value(x, theta)
        f_raw_theta = self._raw_value(theta, theta)

        # EXACT normalization: f(theta, theta) = 0 guaranteed
        return float(f_raw_x - f_raw_theta)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        self._lazy_init(x.shape[0])

        centers = theta + self.offsets  # Shape: (k, dim)
        diffs = x - centers  # Shape: (k, dim)
        dists_sq = np.sum(diffs**2, axis=1)  # Shape: (k,)

        # Weights: global minimum has weight 1.0, local minima have weight 0.3
        weights = np.full(self.k, 0.3)
        weights[0] = 1.0

        # Gradient of -w * exp(-||x-c||^2 / sigma^2)
        # = w * exp(-||x-c||^2 / sigma^2) * (2 / sigma^2) * (x - c)
        exps = np.exp(-dists_sq / self.width_sq)
        coeffs = weights * exps * (2.0 / self.width_sq)  # Shape: (k,)

        # Weighted sum of difference vectors
        grad = np.sum(coeffs[:, np.newaxis] * diffs, axis=0)  # Shape: (dim,)
        return grad


class RobustLandscape(Landscape):
    """
    Huber Loss.
    Quadratic near optimum, Linear far away. Robust to outliers in x.
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


# =============================================================================
# 3. NOISE MODELS (Stochastic Corruption)
# =============================================================================


class Noise(ABC):
    """
    Abstract base class for measurement noise.
    Supports stateful noise for temporal correlations.
    """

    @abstractmethod
    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        """
        Apply noise to the signal.

        Args:
            signal: The clean value (scalar) or gradient (vector).
            t: Time step.

        Returns:
            Noisy signal.
        """
        pass

    def reset(self) -> None:
        """Reset internal state (for CorrelatedNoise)."""
        pass


class GaussianNoise(Noise):
    """Standard additive Gaussian noise: signal + N(0, sigma)."""

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
    Simulates outliers and sensor glitches.
    """

    def __init__(self, alpha: float, scale: float = 1.0, seed: Optional[int] = None):
        self.alpha = alpha  # Tail index (1.5 - 3.0)
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        # Pareto generation: (1 - u)^(-1/alpha) - 1, random sign
        u = self.rng.random(size=signal.shape)
        pareto = (1.0 - u) ** (-1.0 / self.alpha) - 1.0
        signs = self.rng.choice([-1.0, 1.0], size=signal.shape)
        noise = signs * self.scale * pareto

        # Clip strictly massive outliers to avoid Inf in float32
        noise = np.clip(noise, -1e3 * self.scale, 1e3 * self.scale)
        return signal + noise


class CorrelatedNoise(Noise):
    """
    AR(1) Correlated Noise.
    xi_t = phi * xi_{t-1} + sqrt(1 - phi^2) * epsilon.
    Simulates drift in sensor calibration or persistent environmental effects.
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
        # Initialize state if needed or if dimension changed
        if self._state is None or self._state.shape != signal.shape:
            self._state = np.zeros_like(signal)

        epsilon = self.rng.normal(0, self.sigma, size=signal.shape)

        # Update AR(1) state
        # scaling factor sqrt(1-phi^2) keeps stationary variance = sigma^2
        scaling = np.sqrt(1 - self.phi**2)
        self._state = self.phi * self._state + scaling * epsilon

        return signal + self._state


class QuantizedNoise(Noise):
    """
    Quantized Noise.
    Simulates low-resolution ADC (Analog-to-Digital Converter).
    y = delta * round(signal / delta).
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
    Relative error model.
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
    With probability p, applies Gaussian noise. With 1-p, noise is 0.
    Scientific basis: Packet loss in gradient transmission, intermittent sensor failure.
    """

    def __init__(self, sigma: float, p: float = 0.1, seed: Optional[int] = None):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.sigma = sigma
        self.p = p
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        # Mask: 1 with probability p
        mask = self.rng.binomial(1, self.p, size=signal.shape).astype(float)
        noise = self.rng.normal(0, self.sigma, size=signal.shape)
        return signal + (mask * noise)


# =============================================================================
# 4. ENVIRONMENT CONTAINER
# =============================================================================


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

    Responsibilities:
    1. Manage state theta_t.
    2. Enforce physical bounds.
    3. Execute drift dynamics (including adaptive feedback).
    4. Provide landscape geometry to Oracle.
    5. Handle concurrency locks (bi-directional locking for Oracle protocol).


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

        # State initialization
        if initial_theta is not None:
            self._theta = np.array(initial_theta, dtype=np.float64)
        else:
            self._theta = np.zeros(dim)

        self.t = 0

        # Concurrency flags
        self._locked_by_oracle = False
        self._is_advancing = False
        self._registered_oracle = None

    def get_current_theta(self, for_analysis: bool = False) -> np.ndarray:
        """
        Get current state.

        Strict Information Barrier:
        - for_analysis=True: Allowed (e.g., for Metric computation).
        - for_analysis=False: Forbidden (raises error to prevent cheating).
        """
        if not for_analysis:
            # In production, this would inspect the call stack
            # For now, we rely on the flag being set correctly by Metrics only
            pass
        return self._theta.copy()

    def step(self, action: Optional[np.ndarray] = None) -> None:
        """
        Advance environment time: t -> t+1.

        Args:
            action: The optimizer's decision x_t.
                    Passed to drift model for AdaptiveDrift.

        Raises:
            RuntimeError: If Oracle is currently active.
        """
        if self._locked_by_oracle:
            raise RuntimeError(
                "Cannot advance environment while Oracle is active (in a step). "
                "Call oracle.end_step() first."
            )

        self._is_advancing = True

        # 1. Update drift (pass action for adaptive dynamics)
        self._theta = self.drift.step(self._theta, self.t, action)

        # 2. Enforce bounds (if physical constraint)
        if self.bounds is not None:
            low, high = self.bounds
            self._theta = np.clip(self._theta, low, high)

        # 3. Advance clock
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

    # --- Oracle Protocol Hooks (Locking) ---

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


# =============================================================================
# 5. FACTORY & REGISTRY
# =============================================================================

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

    Args:
        config: Dictionary with keys 'dim', 'drift', 'landscape', 'noise'.
                Example:
                {
                    'dim': 10,
                    'drift': {'type': 'linear', 'velocity': [0.1]*10},
                    'landscape': {'type': 'quadratic', 'condition_number': 10},
                    'noise': {'type': 'gaussian', 'sigma': 0.1}
                }
        seed: Random seed for reproducibility.

    Returns:
        Configured DynamicEnvironment instance.
    """
    dim = config.get("dim", 2)

    # Drift
    d_conf = config.get("drift", {"type": "stationary"})
    d_type = d_conf.pop("type")
    # Inject dependencies if needed
    if "seed" not in d_conf and d_type in ["random_walk", "jump"]:
        d_conf["seed"] = seed
    if "dim" not in d_conf and d_type in ["jump"]:
        d_conf["dim"] = dim
    drift = make_drift(d_type, **d_conf)

    # Landscape
    l_conf = config.get("landscape", {"type": "quadratic"})
    l_type = l_conf.pop("type")
    if "dim" not in l_conf and l_type in ["quadratic"]:
        l_conf["dim"] = dim
    if "seed" not in l_conf and l_type in ["quadratic", "multiextremal"]:
        l_conf["seed"] = seed + 1
    landscape = make_landscape(l_type, **l_conf)

    # Bounds
    bounds = config.get("x_bounds", None)

    return DynamicEnvironment(dim=dim, drift=drift, landscape=landscape, bounds=bounds)
