from .environment import DynamicEnvironment
from .benchmark import BenchmarkRunner, BatchRunner
from .oracle import FirstOrderOracle, ZeroOrderOracle, HybridOracle
from .drift import (
    Drift,
    StationaryDrift,
    LinearDrift,
    RandomWalkDrift,
    CyclicDrift,
    TeleportationDrift,
)
from .landscape import (
    Landscape,
    QuadraticLandscape,
    QuadraticRavineLandscape,
    PNormLandscape,
    RosenbrockLandscape,
)
from .noise import (
    Noise,
    GaussianNoise,
    ParetoNoise,
    BernoulliSparseNoise,
    QuantizationNoise,
)
from .metrics import (
    Metric,
    TrackingErrorMetric,
    DynamicRegretMetric,
    TimeToRecoveryMetric,
)

__version__ = "0.1.0"
__author__ = "ST-HRD Development Team"

__all__ = [
    "DynamicEnvironment",
    "BenchmarkRunner",
    "BatchRunner",
    "FirstOrderOracle",
    "ZeroOrderOracle",
    "HybridOracle",
    "Drift",
    "StationaryDrift",
    "LinearDrift",
    "RandomWalkDrift",
    "CyclicDrift",
    "TeleportationDrift",
    "Landscape",
    "QuadraticLandscape",
    "QuadraticRavineLandscape",
    "PNormLandscape",
    "RosenbrockLandscape",
    "Noise",
    "GaussianNoise",
    "ParetoNoise",
    "BernoulliSparseNoise",
    "QuantizationNoise",
    "Metric",
    "TrackingErrorMetric",
    "DynamicRegretMetric",
    "TimeToRecoveryMetric",
]


def make_default_environment(dim=50, seed=None):
    from .drift import RandomWalkDrift
    from .landscape import QuadraticLandscape

    drift = RandomWalkDrift(dim=dim, step_size=0.05, seed=seed)
    landscape = QuadraticLandscape(dim=dim, condition_number=100.0)
    return DynamicEnvironment(drift=drift, landscape=landscape, seed=seed)


def make_benchmark(dim=50, n_steps=1000, noise_scale=0.1, seed=None):
    env = make_default_environment(dim=dim, seed=seed)
    oracle = FirstOrderOracle(
        environment=env,
        grad_noise=GaussianNoise(std=noise_scale, seed=seed),
        value_noise=GaussianNoise(std=noise_scale * 0.1, seed=seed),
    )
    metrics = [
        TrackingErrorMetric(normalize_by_dim=True, name="NormalizedError"),
        TrackingErrorMetric(norm=1, name="L1Error"),
    ]
    return BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        config={"n_steps": n_steps, "eval_frequency": 10},
    )
