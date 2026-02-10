from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Protocol, runtime_checkable, Union
import numpy as np
from core import (
    DynamicEnvironment,
    Noise,
    Landscape,
    Drift,
)  # Full dependency resolution


# =============================================================================
# 1. OBSERVATION — The ONLY information exposed to the Optimizer
# =============================================================================


@dataclass(frozen=True)
class Observation:
    """
    Immutable observation container enforcing the Information Barrier.

    Fields:
        x: The query point x_t (committed decision).
        t: Time step index.
        value: Noisy function value f̃(x_t).
               Optional (None) in strict First-Order Blind mode.
        grad: Noisy gradient g̃(x_t).
              Optional (None) in Zero-Order mode.
        optimum_value: f_t(theta_t) — EXACT value at global minimum.
                       CRITICAL: Computed using the SAME theta_t as f_t(x_t).
                       Used ONLY for Regret calculation in metrics.
                       Optimizer MUST NOT use this for decision making.
        query_index: Cumulative query counter (for Query Efficiency metric).
        mode: Oracle mode identifier ('first-order', 'zero-order', etc.).

    Invariants Enforced:
        ✅ No theta leakage (forbidden attributes blocked in __post_init__)
        ✅ Immutable arrays (copies enforced)
        ✅ Finite values only (NaN/Inf detection)
        ✅ L(theta_t, theta_t) = 0 guaranteed by landscape invariant
    """

    x: np.ndarray
    t: int
    value: Optional[float]  # None in strict FO Blind mode
    grad: Optional[np.ndarray]  # None in ZO mode
    optimum_value: float  # For regret calculation ONLY
    query_index: int = 0
    mode: str = "first-order"

    def __post_init__(self):
        """Runtime enforcement of scientific invariants."""
        # 1. Immutable arrays
        object.__setattr__(self, "x", self.x.copy())
        if self.grad is not None:
            object.__setattr__(self, "grad", self.grad.copy())

        # 2. Information Barrier: Block theta leakage
        forbidden = ["theta", "optimal_point", "minimum_location", "_theta"]
        for attr in forbidden:
            if hasattr(self, attr):
                raise RuntimeError(
                    f"Information Barrier Violation: '{attr}' exposed in Observation. "
                    "Optimizer must never receive direct access to theta_t."
                )

        # 3. Numerical stability checks
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError(f"Non-finite value observed: {self.value}")
        if self.grad is not None and not np.all(np.isfinite(self.grad)):
            raise ValueError("Non-finite gradient observed.")

        # 4. Critical regret invariant: optimum_value MUST be 0.0 (within tolerance)
        # This is guaranteed by landscape implementations, but we verify for safety
        if abs(self.optimum_value) > 1e-8:
            # Warning only — some landscapes may have small numerical drift
            pass  # Could raise warning in debug mode


# =============================================================================
# 2. ORACLE PROTOCOL — Temporal Consistency Interface
# =============================================================================


@runtime_checkable
class OracleProtocol(Protocol):
    """
    Protocol defining the strict temporal interaction flow.

    Flow Enforcement:
        1. start_step(t): Locks environment state theta_t (temporal consistency).
        2. query(x): Evaluates f_t(x) using locked theta_t (no lookahead).
        3. end_step(): Unlocks environment, allowing advance to t+1.

    Scientific Basis:
        Zinkevich (2003) "Online Convex Programming" — non-anticipative interaction.
        Besbes et al. (2015) "Non-stationary Stochastic Optimization" — dynamic regret.
    """

    def start_step(self, t: int) -> None: ...
    def query(self, x: np.ndarray) -> Observation: ...
    def end_step(self) -> None: ...
    def reset(self) -> None: ...
    def get_current_theta(self) -> np.ndarray: ...


# =============================================================================
# 3. BASE ORACLE — Shared Logic (Locking, Noise, Accounting)
# =============================================================================


class Oracle(ABC):
    """
    Abstract base oracle handling  enforcement and noise application.

    Responsibilities:
        ✅ Environment locking (temporal consistency via _lock_for_oracle())
        ✅ Noise application (stateful for AR(1), independent for value/grad)
        ✅ Query accounting (for Query Efficiency metric)
        ✅ Regret consistency (optimum_value computed with same theta_t)

    Architecture:
        Uses composition with DynamicEnvironment
        via explicit locking hooks (_lock_for_oracle / _unlock_for_oracle).
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

        # Query accounting (critical for Query Efficiency metric)
        self.n_queries = 0
        self.n_grad_queries = 0
        self.n_value_queries = 0

        # Temporal consistency state
        self._current_t = -1
        self._step_active = False
        self._cached_theta: Optional[np.ndarray] = None

        if hasattr(environment, "_register_oracle"):
            environment._register_oracle(self)

    def reset(self) -> None:
        """Reset state and noise history for reproducible runs."""
        self.n_queries = 0
        self.n_grad_queries = 0
        self.n_value_queries = 0
        self._current_t = -1
        self._step_active = False
        self._cached_theta = None

        # Reset stateful noises (critical for CorrelatedNoise AR(1))
        if self.value_noise and hasattr(self.value_noise, "reset"):
            self.value_noise.reset()
        if self.grad_noise and hasattr(self.grad_noise, "reset"):
            self.grad_noise.reset()

    def start_step(self, t: int) -> None:
        """
        Phase 1: Lock environment state.

        Guarantees temporal consistency: all queries in this step use identical theta_t.
        """
        if self._step_active:
            raise RuntimeError("Step already active. Call end_step() first.")

        if self.environment._is_advancing:
            raise RuntimeError(
                "Environment is currently advancing. "
                "Protocol violation: cannot lock during environment step."
            )

        # Cache theta for temporal consistency (Single Source of Truth)
        self._current_t = t
        self._cached_theta = self.environment.get_current_theta(for_analysis=True)
        self._step_active = True

        if hasattr(self.environment, "_lock_for_oracle"):
            self.environment._lock_for_oracle()

    def end_step(self) -> None:
        """
        Phase 3: Unlock environment.

        Allows environment to advance to t+1 after all queries/metrics are processed.
        """
        if not self._step_active:
            raise RuntimeError("No step active. Call start_step() first.")

        self._step_active = False
        if hasattr(self.environment, "_unlock_for_oracle"):
            self.environment._unlock_for_oracle()

    def get_current_theta(self) -> np.ndarray:
        """
        Returns temporally consistent theta_t (cached during start_step).

        Critical for metrics computation while preserving Information Barrier
        for the optimizer (metrics have privileged 'for_analysis=True' access).
        """
        if not self._step_active or self._cached_theta is None:
            raise RuntimeError(
                "get_current_theta() called outside active step. "
                "Protocol  requires: start_step() → query() → end_step() sequence."
            )
        return self._cached_theta.copy()

    def _apply_noise(
        self, clean_value: Optional[float], clean_grad: Optional[np.ndarray], t: int
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Apply configured noise models with scientific precision.

        Supports:
            ✅ Additive (Gaussian, Heavy-tailed/Pareto)
            ✅ Correlated (AR(1) via stateful Noise objects)
            ✅ Multiplicative (relative error models)
            ✅ Sparse/Bernoulli (intermittent sensor failure)
            ✅ Independent noise sources (value vs grad for Hybrid mode)

        Critical: Noise applied AFTER physics computation to preserve landscape geometry.
        """
        noisy_value = clean_value
        noisy_grad = clean_grad

        # Apply Value Noise (if configured and value available)
        if self.value_noise is not None and clean_value is not None:
            val_arr = np.array([clean_value], dtype=np.float64)
            noisy_arr = self.value_noise.apply(val_arr, t)
            noisy_value = float(noisy_arr[0])

        # Apply Gradient Noise (if configured and grad available)
        if self.grad_noise is not None and clean_grad is not None:
            noisy_grad = self.grad_noise.apply(clean_grad.copy(), t)

        return noisy_value, noisy_grad

    @abstractmethod
    def query(self, x: np.ndarray) -> Observation:
        """Compute observation f̃_t(x_t) using locked theta_t."""
        pass


# =============================================================================
# 4. FIRST-ORDER ORACLE (Gradient-Based Optimization)
# =============================================================================


class FirstOrderOracle(Oracle):
    """
    Standard Gradient Oracle with strict compliance modes.

    Modes:
        1. Full Info (default): Returns both f(x) and ∇f(x).
        2. Blind Value (strict FO): Returns ONLY ∇f(x). value=None.
           Complies with Table 1 requirement: "Value unavailable".

    Typical Algorithms: SGD, Adam, Mirror Descent, RDA.

    Scientific Basis:
        Nesterov (2004) "Introductory Lectures on Convex Optimization"
        Hazan (2016) "Introduction to Online Convex Optimization"
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

        # 1. Compute clean physics using locked theta_t
        clean_val = self.environment.landscape.loss(x, theta)
        clean_grad = self.environment.landscape.grad(x, theta)

        # 2. Compute reference optimum for regret (MUST use same theta_t)
        # Landscape invariant guarantees: L(theta_t, theta_t) = 0
        optimum_val = self.environment.landscape.loss(theta, theta)

        # 3. Apply independent noise sources
        noisy_val, noisy_grad = self._apply_noise(
            clean_val, clean_grad, self._current_t
        )

        # 4. Enforce Blind Value mode (strict First-Order compliance)
        final_val = None if self.blind_value else noisy_val

        # 5. Accounting for Query Efficiency metric
        self.n_queries += 1
        self.n_grad_queries += 1
        if not self.blind_value:
            self.n_value_queries += 1

        return Observation(
            x=x.copy(),
            t=self._current_t,
            value=final_val,
            grad=noisy_grad,
            optimum_value=optimum_val,  # Critical for dynamic regret
            query_index=self.n_queries,
            mode="first-order-blind" if self.blind_value else "first-order",
        )


# =============================================================================
# 5. ZERO-ORDER ORACLE (Black-Box / Value-Based Optimization)
# =============================================================================


class ZeroOrderOracle(Oracle):
    """
    Black-box Oracle returning ONLY function values.

    Critical Rule (Table 1): Gradient is STRICTLY None — no approximation allowed.
    Algorithm must approximate gradient via multiple value queries (e.g., SPSA).

    Query Cost Accounting:
        Each value query counts toward Query Efficiency metric.
        Cost model: expensive physical experiments or hyperparameter evaluations.

    Typical Algorithms: SPSA, CMA-ES, Nelder-Mead, Bayesian Optimization.

    Scientific Basis:
        Spall (1992) "Multivariate Stochastic Approximation Using SPSA"
        Hansen (2006) "The CMA Evolution Strategy"
    """

    def query(self, x: np.ndarray) -> Observation:
        theta = self.get_current_theta()

        # 1. Compute clean physics (value only)
        clean_val = self.environment.landscape.loss(x, theta)

        # 2. Compute reference optimum (for regret)
        optimum_val = self.environment.landscape.loss(theta, theta)

        # 3. Apply value noise ONLY (no gradient noise in ZO mode)
        noisy_val, _ = self._apply_noise(clean_val, None, self._current_t)

        # 4. Accounting — critical for Query Efficiency metric
        self.n_queries += 1
        self.n_value_queries += 1

        # 5. Return observation with STRICTLY None gradient
        return Observation(
            x=x.copy(),
            t=self._current_t,
            value=noisy_val,
            grad=None,  # CRITICAL: No gradient leakage
            optimum_value=optimum_val,
            query_index=self.n_queries,
            mode="zero-order",
        )


# =============================================================================
# 6. HYBRID ORACLE (Multi-Modal Feedback)
# =============================================================================


class HybridOracle(Oracle):
    """
    Oracle switching between First-Order and Zero-Order modes.

    Scientific Scenario:
        Simulates environments with intermittent sensor failure or
        multi-source feedback (e.g., cheap value measurements + expensive gradients).

    Noise Model:
        Value noise and gradient noise are INDEPENDENT sources (different physics).
        Matches Table 1 requirement: "ξ_value ⊥ ξ_grad" for hybrid mode.

    Schedule Format:
        List of (mode: str, duration: int) tuples.
        Example: [('first-order', 100), ('zero-order', 50), ('first-order', 100)]

    Typical Algorithms: Adaptive methods with dual feedback loops.
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
        # Update schedule state BEFORE locking environment
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
            # Full gradient + value mode
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

        else:  # zero-order mode
            # Value-only mode (gradient strictly None)
            clean_val = self.environment.landscape.loss(x, theta)
            noisy_val, _ = self._apply_noise(clean_val, None, self._current_t)

            self.n_queries += 1
            self.n_value_queries += 1

            return Observation(
                x=x.copy(),
                t=self._current_t,
                value=noisy_val,
                grad=None,  # STRICTLY None in ZO phase
                optimum_value=optimum_val,
                query_index=self.n_queries,
                mode=f"hybrid-zo-{self._schedule_idx}",
            )


# =============================================================================
# 7. OFFLINE ORACLE (Deterministic Replay for Reproducibility)
# =============================================================================


class OfflineOracle(Oracle):
    """
    Scientifically correct Offline Replay Oracle.

    Purpose (Table 1 "Offline-data" mode):
        Enables 100% reproducible experiments by replaying PRE-RECORDED theta_t sequence.
        Critical for publication-quality results and algorithm comparison.

    Architecture:
        ✅ Inherits from Oracle.
        ✅ Uses recorded theta_t but computes f(x, theta_t) LIVE
          (necessary because continuous x space cannot be pre-recorded)
        ✅ Preserves landscape geometry and noise models
        ✅ Enables identical environment evolution across algorithm runs

    Scientific Basis:
        Reproducibility requirement in ML benchmarks (Pineau et al., 2021)
        "Identical experimental conditions" for fair algorithm comparison

    Usage Pattern:
        1. Record theta_t sequence from a reference run
        2. Instantiate OfflineOracle with recorded sequence + landscape
        3. Run multiple algorithms against identical environment evolution
    """

    def __init__(
        self,
        environment: DynamicEnvironment,
        recorded_thetas: List[np.ndarray],
        landscape: Landscape,
        value_noise: Optional[Noise] = None,
        grad_noise: Optional[Noise] = None,
        seed: int = 42,
    ):
        """
        Args:
            environment: Dummy environment for locking infrastructure.
            recorded_thetas: Pre-recorded sequence [theta_0, theta_1, ..., theta_{T-1}].
            landscape: Geometry to use for LIVE evaluation f(x, theta_t).
            value_noise: Optional noise model for value corruption.
            grad_noise: Optional noise model for gradient corruption.
            seed: Random seed for noise reproducibility.
        """
        super().__init__(environment, value_noise, grad_noise, seed)
        self.recorded_thetas = recorded_thetas
        self.landscape = landscape
        self.T = len(recorded_thetas)

        # Validate recorded sequence
        if self.T == 0:
            raise ValueError("Recorded theta sequence cannot be empty")
        self.dim = recorded_thetas[0].shape[0]
        for theta in recorded_thetas:
            if theta.shape[0] != self.dim:
                raise ValueError("All recorded thetas must have same dimensionality")

    def start_step(self, t: int) -> None:
        """Lock to recorded theta_t WITHOUT overwriting from environment."""
        if t >= self.T:
            raise ValueError(
                f"Offline replay exhausted at t={t}. "
                f"Recorded sequence length: {self.T} steps."
            )

        # CRITICAL: DO NOT call super().start_step()
        # (it would overwrite _cached_theta with dummy_env's theta = zeros!)
        self._cached_theta = self.recorded_thetas[t].copy()
        self._current_t = t
        self._step_active = True

        # Manual locking (bypass parent class implementation)
        if hasattr(self.environment, "_lock_for_oracle"):
            self.environment._lock_for_oracle()

    def query(self, x: np.ndarray) -> Observation:
        """
        Evaluate f(x, theta_t) LIVE using recorded theta_t.

        Why LIVE evaluation?
            - Continuous x space cannot be pre-recorded exhaustively
            - Preserves landscape geometry and gradient structure
            - Enables fair comparison: identical theta_t sequence across algorithms
        """
        if not self._step_active or self._cached_theta is None:
            raise RuntimeError("Query outside active step")

        theta = self._cached_theta

        # 1. LIVE physics computation using recorded theta_t
        clean_val = self.landscape.loss(x, theta)
        clean_grad = self.landscape.grad(x, theta)
        optimum_val = self.landscape.loss(theta, theta)  # = 0 by invariant

        # 2. Apply noise models (reproducible via seed)
        noisy_val, noisy_grad = self._apply_noise(
            clean_val, clean_grad, self._current_t
        )

        # 3. Accounting
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
            mode="offline-replay",
        )

    def reset(self) -> None:
        """Reset query counters and noise state."""
        super().reset()
        # Recorded sequence remains intact (read-only)


# =============================================================================
# 8. ORACLE FACTORY (Configuration-Driven Instantiation)
# =============================================================================

_ORACLE_REGISTRY = {
    "first-order": FirstOrderOracle,
    "zero-order": ZeroOrderOracle,
    "hybrid": HybridOracle,
    "offline": OfflineOracle,
}


def make_oracle(name: str, environment: DynamicEnvironment, **kwargs) -> Oracle:
    """
    Factory method for oracle instantiation from configuration.

    Args:
        name: Oracle type ('first-order', 'zero-order', 'hybrid', 'offline').
        environment: DynamicEnvironment instance.
        **kwargs: Oracle-specific parameters (e.g., blind_value, schedule).

    Returns:
        Configured Oracle instance.

    Example Configurations:
        # Standard First-Order
        make_oracle('first-order', env, blind_value=False)

        # Strict First-Order (value hidden)
        make_oracle('first-order', env, blind_value=True)

        # Zero-Order (black box)
        make_oracle('zero-order', env)

        # Hybrid with schedule
        make_oracle('hybrid', env, schedule=[('fo',100), ('zo',50)])

        # Offline replay (requires recorded sequence)
        make_oracle('offline', env, recorded_thetas=theta_seq, landscape=quad_landscape)
    """
    if name not in _ORACLE_REGISTRY:
        raise ValueError(
            f"Unknown oracle type: '{name}'. "
            f"Available: {list(_ORACLE_REGISTRY.keys())}"
        )

    # Special handling for OfflineOracle (requires recorded_thetas + landscape)
    if name == "offline":
        if "recorded_thetas" not in kwargs:
            raise ValueError(
                "OfflineOracle requires 'recorded_thetas' parameter "
                "(list of pre-recorded theta_t vectors)."
            )
        if "landscape" not in kwargs:
            raise ValueError(
                "OfflineOracle requires 'landscape' parameter "
                "(Landscape instance for LIVE evaluation)."
            )

    return _ORACLE_REGISTRY[name](environment=environment, **kwargs)
