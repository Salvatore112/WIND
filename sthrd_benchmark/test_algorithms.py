import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from core import (
    StationaryDrift,
    LinearDrift,
    RandomWalkDrift,
    QuadraticLandscape,
    PNormLandscape,
    RosenbrockLandscape,
    MultiExtremalLandscape,
    RobustLandscape,
    GaussianNoise,
    DynamicEnvironment,
)
from oracle import (
    FirstOrderOracle,
    ZeroOrderOracle,
    HybridOracle,
    Observation,
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
    OptimizerProtocol,
)


def test_landscape_invariant_loss_at_optimum_is_zero():
    """
    CRITICAL INVARIANT: L(theta, theta) MUST be exactly 0 for all landscapes.
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
    Gradient must vanish at global minimum.
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


def test_protocol_a_locking_mechanism():
    """
    Environment must refuse advancement while Oracle is active.
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
    env.step()


def test_temporal_consistency_same_theta_for_multiple_queries():
    """
    All queries within same step must observe identical theta_t.
    """
    env = DynamicEnvironment(
        dim=3,
        drift=LinearDrift(velocity=np.array([0.1, 0.1, 0.1])),
        landscape=QuadraticLandscape(dim=3, condition_number=2.0),
        initial_theta=np.zeros(3),
    )
    oracle = FirstOrderOracle(environment=env)

    oracle.start_step(0)
    obs1 = oracle.query(np.array([1.0, 0.0, 0.0]))
    obs2 = oracle.query(np.array([0.0, 1.0, 0.0]))
    obs3 = oracle.query(np.array([0.0, 0.0, 1.0]))
    assert obs1.t == obs2.t == obs3.t == 0
    oracle.end_step()

    env.step()

    oracle.start_step(1)
    obs4 = oracle.query(np.array([1.0, 0.0, 0.0]))
    oracle.end_step()

    assert not np.isclose(obs1.value, obs4.value, atol=1e-3), (
        f"Function value did not change after environment advance: "
        f"step0={obs1.value:.6f}, step1={obs4.value:.6f}"
    )


def test_observation_information_barrier():
    """
    Observation must NEVER expose theta_t directly to optimizer.
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

    forbidden_attrs = ["theta", "optimal_point", "minimum_location", "_theta"]
    for attr in forbidden_attrs:
        assert not hasattr(
            obs, attr
        ), f"Information leak: Observation has '{attr}' attribute"

    x_original_copy = x_original.copy()
    obs.x[0] = 999.0

    assert np.array_equal(
        x_original, x_original_copy
    ), "Original query point was mutated through Observation.x!"


def test_blind_value_mode_hides_function_value():
    """
    Strict first-order mode must return value=None.
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


def test_lyapunov_metric_rho_consistency():
    """
    Lyapunov metric must use correct p-norm = rho + 1.
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

    raw_value_1 = metric1._current_value * (A1 ** (rho + 1.0))
    raw_value_2 = metric2._current_value * (A2 ** (rho + 1.0))
    expected_raw = 0.09

    assert np.isclose(
        raw_value_1, expected_raw, rtol=1e-5
    ), f"Denormalization failed for A1: got {raw_value_1}, expected {expected_raw}"
    assert np.isclose(
        raw_value_2, expected_raw, rtol=1e-5
    ), f"Denormalization failed for A2: got {raw_value_2}, expected {expected_raw}"


def test_dynamic_regret_optimum_value_consistency():
    """
    Regret calculation must use f_t(theta_t) computed with SAME theta_t as f_t(x_t).
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

    assert abs(obs.optimum_value) < 1e-8


class MockOptimizer(OptimizerProtocol):
    def __init__(self, dim: int, lr: float = 0.1):
        self.dim = dim
        self.lr = lr
        self.name = "MockSGD"

    def step(self, observation) -> np.ndarray:
        if observation.grad is not None:
            return observation.x - self.lr * observation.grad
        else:
            return observation.x + np.random.randn(self.dim) * 0.01

    def reset(self) -> None:
        pass


def test_benchmark_runner_full_execution():
    """
    End-to-end test of BenchmarkRunner with trajectory recording.
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

    assert result.status == "SUCCESS"
    assert result.trajectory is not None
    assert len(result.trajectory["x"]) == T + 1
    assert len(result.trajectory["theta"]) == T + 1

    assert "error_l2" in result.final_metrics
    assert "dynamic_regret" in result.final_metrics
    assert "lyapunov_rho1.00" in result.final_metrics

    assert result.final_metrics["dynamic_regret"] >= -1e-6


def test_metrics_collection_rho_validation():
    """
    MetricsCollection must validate consistent rho values across metrics.
    """
    valid_metrics = [
        LyapunovMetric(rho=0.7),
        NormalizedLyapunovMetric(rho=0.7, drift_magnitude_A=0.1),
    ]
    MetricsCollection(valid_metrics)

    invalid_metrics = [
        LyapunovMetric(rho=0.5),
        NormalizedLyapunovMetric(rho=0.8, drift_magnitude_A=0.1),
    ]
    with pytest.raises(ValueError, match="Inconsistent Hölder exponents"):
        MetricsCollection(invalid_metrics)


def test_oracle_query_outside_active_step():
    """
    Querying Oracle outside active step must raise clear error.
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


def test_experiment_result_json_export():
    """
    Exported JSON must contain complete reproducibility signature.
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

        assert exported["_format"] == "sthrd-v1.0"
        assert exported["_has_ground_truth"] is True
        assert "optimizer_info" in exported
        assert "environment_config" in exported
        assert "trajectory" in exported
        assert len(exported["trajectory"]["x"]) == T + 1


def test_reproducibility_independent_runs():
    """
    Verify reproducibility across independent runs with identical seed configuration.
    """
    dim = 3
    T = 30
    seed = 12345

    def make_env():
        return DynamicEnvironment(
            dim=dim,
            drift=RandomWalkDrift(sigma=0.05, seed=seed),
            landscape=QuadraticLandscape(dim=dim, condition_number=5.0, seed=seed + 1),
            initial_theta=np.zeros(dim),
        )

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

    assert np.allclose(
        np.array(result1.trajectory["x"]),
        np.array(result2.trajectory["x"]),
        atol=1e-10,
    ), "Non-reproducible execution across independent runs with identical seeds"

    assert np.allclose(
        np.array(result1.trajectory["theta"]),
        np.array(result2.trajectory["theta"]),
        atol=1e-10,
    ), "Non-reproducible ground truth trajectory"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
