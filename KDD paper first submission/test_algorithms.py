import pytest
import numpy as np
import json
import tempfile
import random
from pathlib import Path
from unittest.mock import Mock

# Core components
from core import (
    StationaryDrift,
    LinearDrift,
    RandomWalkDrift,
    CyclicDrift,
    JumpDrift,
    AdaptiveDrift,
    QuadraticLandscape,
    PNormLandscape,
    RosenbrockLandscape,
    MultiExtremalLandscape,
    RobustLandscape,
    GaussianNoise,
    HeavyTailedNoise,
    CorrelatedNoise,
    DynamicEnvironment,
)
from oracle import (
    FirstOrderOracle,
    ZeroOrderOracle,
    HybridOracle,
    OfflineOracle,
    Observation,
    OracleProtocol,
)
from metrics import (
    MetricsCollection,
    TrackingErrorMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    DynamicRegretMetric,
    InstantaneousLossMetric,
)
from benchmark import (
    BenchmarkRunner,
    ExperimentResult,
    OptimizerInfo,
    EnvironmentConfig,
    OptimizerProtocol,
    BatchRunner,
)


# ============================================================================
# 1. LANDSCAPE INVARIANTS (Critical for valid dynamic regret)
# ============================================================================
def test_landscape_invariant_loss_at_optimum_is_zero():
    """
    CRITICAL INVARIANT: L(theta, theta) MUST be exactly 0 for all landscapes.

    Scientific basis: Dynamic regret R_T = Σ[f_t(x_t) - f_t(theta_t)] requires
    f_t(theta_t) = 0 to be meaningful. Violation invalidates all regret analysis.
    """
    dim = 5
    theta = np.random.randn(dim)

    landscapes = [
        QuadraticLandscape(dim=dim, condition_number=10.0),
        PNormLandscape(p=2.0),
        PNormLandscape(p=1.5),
        RosenbrockLandscape(),
        MultiExtremalLandscape(k_centers=3, width=1.0),
        RobustLandscape(delta=1.0),
    ]

    for landscape in landscapes:
        loss = landscape.loss(theta, theta)
        assert (
            abs(loss) < 1e-8
        ), f"Landscape {landscape.__class__.__name__} violates L(theta,theta)=0: got {loss}"
        x_random = theta + np.random.randn(dim) * 2.0
        assert (
            landscape.loss(x_random, theta) >= -1e-8
        ), f"Landscape {landscape.__class__.__name__} produces negative loss"


def test_landscape_gradient_at_optimum_is_zero():
    """
    Gradient must vanish at global minimum (first-order optimality condition).

    Scientific basis: Necessary condition for unconstrained minimization.
    Critical for gradient-based optimizers to converge to optimum.
    """
    dim = 3
    theta = np.random.randn(dim)

    landscapes = [
        QuadraticLandscape(dim=dim, condition_number=5.0),
        PNormLandscape(p=2.0),
        RobustLandscape(delta=1.0),
    ]

    for landscape in landscapes:
        grad = landscape.grad(theta, theta)
        assert np.allclose(
            grad, np.zeros(dim), atol=1e-8
        ), f"Landscape {landscape.__class__.__name__} gradient not zero at optimum: {grad}"


# ============================================================================
# 2. Temporal consistency
# ============================================================================
def test_protocol_a_locking_mechanism():
    """
    Environment must refuse advancement while Oracle is active.

    Scientific basis requires strict temporal separation:
      1. Oracle locks environment state theta_t
      2. All queries use identical theta_t (no lookahead)
      3. Environment advances only after Oracle unlocks

    Violation would enable cheating via future information leakage.
    """
    env = DynamicEnvironment(
        dim=2,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=2, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env)

    oracle.start_step(0)
    assert env._locked_by_oracle, "Environment not locked after start_step()"

    with pytest.raises(
        RuntimeError, match="Cannot advance environment while Oracle is active"
    ):
        env.step()

    oracle.end_step()
    assert not env._locked_by_oracle, "Environment not unlocked after end_step()"
    env.step()  # Should succeed now


def test_temporal_consistency_same_theta_for_multiple_queries():
    """
    All queries within same step must observe identical theta_t.

    Scientific basis: Non-anticipative interaction (Besbes et al. 2015).
    Critical for fair comparison: optimizers must not receive inconsistent feedback.
    """
    env = DynamicEnvironment(
        dim=3,
        drift=LinearDrift(velocity=np.array([0.1, 0.1, 0.1])),
        landscape=QuadraticLandscape(dim=3, condition_number=2.0),
        initial_theta=np.zeros(3),
    )
    oracle = FirstOrderOracle(environment=env)

    # Step 0: multiple queries observe same theta
    oracle.start_step(0)
    obs1 = oracle.query(np.array([1.0, 0.0, 0.0]))
    obs2 = oracle.query(np.array([0.0, 1.0, 0.0]))
    obs3 = oracle.query(np.array([0.0, 0.0, 1.0]))
    assert obs1.t == obs2.t == obs3.t == 0
    oracle.end_step()

    # Advance environment
    env.step()

    # Step 1: new theta produces different function values at same x
    oracle.start_step(1)
    obs4 = oracle.query(np.array([1.0, 0.0, 0.0]))
    oracle.end_step()

    # Critical invariant: f_t(x) must change because theta changed
    assert not np.isclose(obs1.value, obs4.value, atol=1e-3), (
        f"Function value did not change after environment advance: "
        f"step0={obs1.value:.6f}, step1={obs4.value:.6f}"
    )


# ============================================================================
# 3. INFORMATION BARRIER (No theta leakage)
# ============================================================================
def test_observation_information_barrier():
    """
    Observation must NEVER expose theta_t directly to optimizer.

    Scientific basis: Information barrier prevents cheating.
    Optimizer must learn theta_t only through function evaluations, not direct access.
    """
    env = DynamicEnvironment(
        dim=2,
        drift=RandomWalkDrift(sigma=0.1),
        landscape=QuadraticLandscape(dim=2, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env)

    x_original = np.array([1.0, 2.0])
    oracle.start_step(0)
    obs = oracle.query(x_original)
    oracle.end_step()

    # Verify forbidden attributes are absent
    forbidden_attrs = ["theta", "optimal_point", "minimum_location", "_theta"]
    for attr in forbidden_attrs:
        assert not hasattr(
            obs, attr
        ), f"Information leak: Observation has '{attr}' attribute"

    # Verify original query point immutability
    x_original_copy = x_original.copy()
    obs.x[0] = 999.0  # Modify the COPY inside Observation

    # Original array MUST remain unchanged (prevents reference leakage)
    assert np.array_equal(
        x_original, x_original_copy
    ), "Original query point was mutated through Observation.x!"


def test_blind_value_mode_hides_function_value():
    """
    Strict first-order mode must return value=None.

    Scientific basis: Table 1 requirement "Value unavailable" for pure gradient oracles.
    Ensures fair comparison between algorithms with different oracle access.
    """
    env = DynamicEnvironment(
        dim=1,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=1, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env, blind_value=True)

    oracle.start_step(0)
    obs = oracle.query(np.array([1.0]))
    oracle.end_step()

    assert obs.value is None, "Blind value mode leaked function value"
    assert obs.grad is not None, "Blind value mode incorrectly hid gradient"


# ============================================================================
# 4. LYAPUNOV METRICS (Theoretical correctness)
# ============================================================================
def test_lyapunov_metric_rho_consistency():
    """
    Lyapunov metric must use correct p-norm = rho + 1.

    Scientific basis: Definition 1 requires V_n(x) = ||x - theta_n||_{rho+1}^{rho+1}
    for rho-Hölder smooth landscapes. Incorrect norm invalidates stabilization analysis.
    """
    rho = 0.5
    metric = LyapunovMetric(rho=rho)

    x = np.array([2.0, 0.0])
    theta = np.array([0.0, 0.0])

    p = rho + 1.0
    distance = np.linalg.norm(x - theta, ord=p)
    expected = distance**p

    mock_env = Mock()
    mock_env.landscape = Mock()
    mock_obs = Mock()
    mock_obs.optimum_value = 0.0

    metric.update(0, x, theta, mock_obs, mock_env)

    assert np.isclose(metric._current_value, expected, rtol=1e-5)


def test_normalized_lyapunov_drift_invariance():
    """
    Normalized Lyapunov must produce consistent denormalized values.

    Scientific basis: Condition D requires E[||theta_n - theta_{n-1}||_{rho+1}^{rho+1}] <= A^{rho+1}.
    Normalization by A^{rho+1} enables fair comparison across drift magnitudes.
    """
    rho = 1.0
    A1, A2 = 0.1, 0.2

    metric1 = NormalizedLyapunovMetric(rho=rho, drift_magnitude_A=A1)
    metric2 = NormalizedLyapunovMetric(rho=rho, drift_magnitude_A=A2)

    x = np.array([0.3, 0.0])
    theta = np.array([0.0, 0.0])

    mock_env = Mock()
    mock_env.landscape = Mock()
    mock_obs = Mock()
    mock_obs.optimum_value = 0.0

    metric1.update(0, x, theta, mock_obs, mock_env)
    metric2.update(0, x, theta, mock_obs, mock_env)

    # Verify denormalization recovers identical raw Lyapunov value
    raw_value_1 = metric1._current_value * (A1 ** (rho + 1.0))
    raw_value_2 = metric2._current_value * (A2 ** (rho + 1.0))
    expected_raw = 0.09  # ||[0.3,0]||_2^2

    assert np.isclose(
        raw_value_1, expected_raw, rtol=1e-5
    ), f"Denormalization failed for A1: got {raw_value_1}, expected {expected_raw}"
    assert np.isclose(
        raw_value_2, expected_raw, rtol=1e-5
    ), f"Denormalization failed for A2: got {raw_value_2}, expected {expected_raw}"


# ============================================================================
# 5. DYNAMIC REGRET INVARIANT
# ============================================================================
def test_dynamic_regret_optimum_value_consistency():
    """
    Regret calculation must use f_t(theta_t) computed with SAME theta_t as f_t(x_t).

    Scientific basis: Dynamic regret R_T = Σ[f_t(x_t) - f_t(theta_t)] requires consistent
    evaluation points. Using different theta_t would invalidate regret bounds.
    """
    env = DynamicEnvironment(
        dim=2,
        drift=LinearDrift(velocity=np.array([0.1, 0.1])),
        landscape=QuadraticLandscape(dim=2, condition_number=3.0),
        initial_theta=np.array([1.0, 1.0]),
    )
    oracle = FirstOrderOracle(environment=env)

    oracle.start_step(0)
    theta_t0 = env.get_current_theta(for_analysis=True)
    x = np.array([2.0, 2.0])
    obs = oracle.query(x)
    oracle.end_step()

    # Critical invariant: optimum_value MUST be 0 (within tolerance)
    assert abs(obs.optimum_value) < 1e-8


# ============================================================================
# 6. BENCHMARK RUNNER (End-to-end execution)
# ============================================================================
class MockOptimizer(OptimizerProtocol):
    """Simple mock optimizer for testing benchmark infrastructure."""

    def __init__(self, dim: int, lr: float = 0.1):
        self.dim = dim
        self.lr = lr
        self.name = "MockSGD"

    def step(self, observation) -> np.ndarray:
        if observation.grad is not None:
            return observation.x - self.lr * observation.grad
        else:
            # Zero-order fallback (reproducible when global seed is fixed)
            return observation.x + np.random.randn(self.dim) * 0.01

    def reset(self) -> None:
        pass


def test_benchmark_runner_full_execution():
    """
    End-to-end test of BenchmarkRunner with trajectory recording.

    Validates:
      - Successful execution without exceptions
      - Correct trajectory dimensions (T+1 points including initial state)
      - All required metrics computed and available
      - Non-negative dynamic regret (guaranteed by L(theta,theta)=0 invariant)
    """
    dim = 3
    T = 50

    env = DynamicEnvironment(
        dim=dim,
        drift=RandomWalkDrift(sigma=0.05, seed=42),
        landscape=QuadraticLandscape(dim=dim, condition_number=5.0, seed=43),
        initial_theta=np.zeros(dim),
    )
    oracle = FirstOrderOracle(
        environment=env,
        value_noise=GaussianNoise(sigma=0.01, seed=44),
        seed=45,
    )
    metrics = MetricsCollection(
        [
            TrackingErrorMetric(norm="l2"),
            DynamicRegretMetric(),
            LyapunovMetric(rho=1.0),
        ]
    )

    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True,
        tail_fraction=0.2,
    )

    optimizer = MockOptimizer(dim=dim, lr=0.2)
    result = runner.run(optimizer, T=T, x0=np.ones(dim) * 2.0, seed=42)

    # Validate result structure
    assert result.status == "SUCCESS"
    assert result.trajectory is not None
    assert len(result.trajectory["x"]) == T + 1
    assert len(result.trajectory["theta"]) == T + 1

    # Verify metrics computed
    assert "error_l2" in result.final_metrics
    assert "dynamic_regret" in result.final_metrics
    assert "lyapunov_rho1.00" in result.final_metrics

    # Verify regret non-negative (landscape invariant)
    assert result.final_metrics["dynamic_regret"] >= -1e-6


def test_batch_runner_statistical_aggregation():
    """
    BatchRunner must correctly aggregate metrics across multiple seeds.

    Validates statistical aggregation pipeline for significance testing.
    """

    def env_factory():
        return DynamicEnvironment(
            dim=2,
            drift=StationaryDrift(),
            landscape=QuadraticLandscape(dim=2, condition_number=2.0),
        )

    def oracle_factory(env):
        return FirstOrderOracle(environment=env)

    def metric_factory():
        return MetricsCollection([TrackingErrorMetric(norm="l2")])

    def optimizer_factory():
        return MockOptimizer(dim=2, lr=0.1)

    batch_runner = BatchRunner(
        environment_factory=env_factory,
        oracle_factory=oracle_factory,
        metric_factory=metric_factory,
        record_trajectory=False,
    )

    seeds = [42, 43]
    results = batch_runner.run(optimizer_factory, seeds=seeds, T=20)

    assert len(results) == len(seeds)

    errors = [
        r.final_metrics["error_l2"] for r in results if "error_l2" in r.final_metrics
    ]
    assert len(errors) == len(seeds)


# ============================================================================
# 7. OFFLINE ORACLE (Deterministic replay)
# ============================================================================
def test_offline_oracle_reproducibility():
    """
    OfflineOracle must replay identical theta_t sequence across runs.

    Scientific basis: Reproducibility requirement (Pineau et al. 2021).
    Enables fair algorithm comparison under identical environment evolution.
    """
    dim = 2
    T = 5

    # Record theta sequence from live environment
    live_env = DynamicEnvironment(
        dim=dim,
        drift=RandomWalkDrift(sigma=0.1, seed=2024),
        landscape=QuadraticLandscape(dim=dim, condition_number=1.0),
        initial_theta=np.zeros(dim),
    )

    thetas_recorded = [live_env.get_current_theta(for_analysis=True).copy()]
    for _ in range(T):
        live_env.step()
        thetas_recorded.append(live_env.get_current_theta(for_analysis=True).copy())

    # Create OfflineOracle with recorded sequence
    dummy_env = DynamicEnvironment(
        dim=dim,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=dim, condition_number=1.0),
    )

    offline_oracle = OfflineOracle(
        environment=dummy_env,
        recorded_thetas=thetas_recorded,
        landscape=QuadraticLandscape(dim=dim, condition_number=1.0),
        seed=999,
    )

    # Verify identical theta_t across two runs
    def run_with_oracle(oracle, T):
        thetas_observed = []
        for t in range(T + 1):
            oracle.start_step(t)
            thetas_observed.append(oracle.get_current_theta().copy())
            oracle.query(np.zeros(dim))
            oracle.end_step()
        return thetas_observed

    run1 = run_with_oracle(offline_oracle, T)
    offline_oracle.reset()
    run2 = run_with_oracle(offline_oracle, T)

    for t in range(T + 1):
        assert np.allclose(
            run1[t], run2[t], atol=1e-10
        ), f"Theta mismatch between runs at t={t}"
        assert np.allclose(run1[t], thetas_recorded[t], atol=1e-10), (
            f"OfflineOracle theta mismatch at t={t}: "
            f"recorded={thetas_recorded[t]}, observed={run1[t]}"
        )


# ============================================================================
# 8. EDGE CASES & ERROR HANDLING
# ============================================================================
def test_metrics_collection_rho_validation():
    """
    MetricsCollection must validate consistent rho values across metrics.

    Scientific basis: Theoretical validity of Lyapunov analysis requires
    consistent Hölder exponent rho across all rho-aware metrics.
    """
    # Valid: consistent rho values
    valid_metrics = [
        LyapunovMetric(rho=0.7),
        NormalizedLyapunovMetric(rho=0.7, drift_magnitude_A=0.1),
    ]
    MetricsCollection(valid_metrics)

    # Invalid: inconsistent rho values
    invalid_metrics = [
        LyapunovMetric(rho=0.5),
        NormalizedLyapunovMetric(rho=0.8, drift_magnitude_A=0.1),
    ]
    with pytest.raises(ValueError, match="Inconsistent Hölder exponents"):
        MetricsCollection(invalid_metrics)


def test_oracle_query_outside_active_step():
    """
    Querying Oracle outside active step must raise clear error.

    Enforces start_step() → query() → end_step() sequence.
    Prevents accidental lookahead or state corruption.
    """
    env = DynamicEnvironment(
        dim=1,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=1, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env)

    with pytest.raises(RuntimeError, match="outside active step"):
        oracle.query(np.array([1.0]))

    oracle.start_step(0)
    oracle.query(np.array([1.0]))
    oracle.end_step()

    with pytest.raises(RuntimeError, match="outside active step"):
        oracle.query(np.array([1.0]))


# ============================================================================
# 9. JSON EXPORT (Reproducibility signature)
# ============================================================================
def test_experiment_result_json_export():
    """
    Exported JSON must contain complete reproducibility signature.

    Scientific basis: Reproducibility requirement for scientific publication.
    Single-file export enables independent verification of results.
    """
    dim = 2
    T = 10

    env = DynamicEnvironment(
        dim=dim,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=dim, condition_number=2.0),
    )
    oracle = FirstOrderOracle(environment=env)
    metrics = MetricsCollection([TrackingErrorMetric(norm="l2")])

    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True,
    )

    optimizer = MockOptimizer(dim=dim, lr=0.1)
    result = runner.run(optimizer, T=T, x0=np.array([1.0, 0.0]), seed=123)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "experiment.json"
        result.save_to_json(str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            exported = json.load(f)

        # Verify critical reproducibility fields
        assert exported["_format"] == "sthrd-v1.0"
        assert exported["_has_ground_truth"] is True
        assert "optimizer_info" in exported
        assert "environment_config" in exported
        assert "trajectory" in exported
        assert len(exported["trajectory"]["x"]) == T + 1


# ============================================================================
# 10. REPRODUCIBILITY (Independent runs with identical seeds)
# ============================================================================
def test_reproducibility_independent_runs():
    """
    Verify reproducibility across independent runs with identical seed configuration.

    Scientific basis: Reproducibility is fundamental requirement for scientific benchmarks.
    Components using np.random.default_rng() require explicit seed propagation at init time.
    """
    dim = 3
    T = 30
    seed = 12345

    # Factory for identical environment configuration
    def make_env():
        return DynamicEnvironment(
            dim=dim,
            drift=RandomWalkDrift(sigma=0.05, seed=seed),
            landscape=QuadraticLandscape(dim=dim, condition_number=5.0, seed=seed + 1),
            initial_theta=np.zeros(dim),
        )

    # Run 1
    env1 = make_env()
    oracle1 = FirstOrderOracle(
        environment=env1,
        value_noise=GaussianNoise(sigma=0.01, seed=seed),
        seed=seed,
    )
    metrics1 = MetricsCollection([TrackingErrorMetric(norm="l2")])
    runner1 = BenchmarkRunner(env1, oracle1, metrics1, record_trajectory=True)
    opt1 = MockOptimizer(dim=dim, lr=0.2)
    result1 = runner1.run(opt1, T=T, x0=np.ones(dim) * 2.0, seed=seed)

    # Run 2 (independent instance with identical seed configuration)
    env2 = make_env()
    oracle2 = FirstOrderOracle(
        environment=env2,
        value_noise=GaussianNoise(sigma=0.01, seed=seed),
        seed=seed,
    )
    metrics2 = MetricsCollection([TrackingErrorMetric(norm="l2")])
    runner2 = BenchmarkRunner(env2, oracle2, metrics2, record_trajectory=True)
    opt2 = MockOptimizer(dim=dim, lr=0.2)
    result2 = runner2.run(opt2, T=T, x0=np.ones(dim) * 2.0, seed=seed)

    # Verify identical trajectories
    assert np.allclose(
        np.array(result1.trajectory["x"]),
        np.array(result2.trajectory["x"]),
        atol=1e-10,
    ), "Non-reproducible execution across independent runs with identical seeds"

    # Verify identical ground truth trajectories
    assert np.allclose(
        np.array(result1.trajectory["theta"]),
        np.array(result2.trajectory["theta"]),
        atol=1e-10,
    ), "Non-reproducible ground truth trajectory"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
