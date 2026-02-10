"""
Benchmark orchestration module for ST-HRD.
Manages experiment execution, result collection, and unified export to single files.
"""

import json
import random
import numpy as np
import datetime
import inspect
import time
from pathlib import Path
from typing import Protocol, Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict


class OptimizerProtocol(Protocol):
    """
    Protocol for optimization algorithms compatible with ST-HRD benchmark.
    """

    def step(self, observation: Any) -> np.ndarray:
        """Compute next query point based on observation."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...


@dataclass
class OptimizerInfo:
    """
    Complete optimizer signature for scientific reproducibility.
    """

    name: str
    class_name: str
    module: str
    hyperparameters: Dict[str, Any]
    oracle_type: str
    query_cost_per_step: int
    memory_complexity: str

    @classmethod
    def from_optimizer(cls, optimizer: Any) -> "OptimizerInfo":
        """
        Extract complete signature from optimizer instance.
        """
        name = getattr(optimizer, "name", optimizer.__class__.__name__)

        hyperparams = {}
        for attr in dir(optimizer):
            if attr.startswith("_") or attr in ["step", "reset", "name", "class_name"]:
                continue

            try:
                val = getattr(optimizer, attr)
                if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                    hyperparams[attr] = val
                elif hasattr(val, "shape") and not callable(val):
                    hyperparams[f"{attr}_shape"] = list(val.shape)
            except Exception:
                continue

        oracle_type = "unknown"
        try:
            sig = inspect.signature(optimizer.step)
            params = list(sig.parameters.values())

            if len(params) > 0 and hasattr(params[0].annotation, "__name__"):
                anno = params[0].annotation
                if "grad" in str(anno).lower() or "gradient" in str(anno).lower():
                    oracle_type = "first-order"

            if oracle_type == "unknown":
                param_names = [p.name.lower() for p in params]
                if any("grad" in name for name in param_names):
                    oracle_type = "first-order"
                else:
                    oracle_type = "zero-order"
        except Exception:
            oracle_type = "unknown"

        query_cost = 1
        if oracle_type == "zero-order":
            if "SPSA" in optimizer.__class__.__name__ or "spsa" in name.lower():
                query_cost = 2

        class_name = optimizer.__class__.__name__.lower()
        if any(x in class_name for x in ["adam", "sgd", "momentum", "nesterov"]):
            mem_complexity = "O(d)"
        elif "cma" in class_name or "evolution" in class_name:
            mem_complexity = "O(d²)"
        else:
            mem_complexity = "unknown"

        return cls(
            name=name,
            class_name=optimizer.__class__.__name__,
            module=optimizer.__class__.__module__,
            hyperparameters=hyperparams,
            oracle_type=oracle_type,
            query_cost_per_step=query_cost,
            memory_complexity=mem_complexity,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentConfig:
    """
    Complete environment configuration for reproducibility.
    """

    dim: int
    drift_type: str
    drift_parameters: Dict[str, Any]
    landscape_type: str
    landscape_parameters: Dict[str, Any]
    noise_type: str
    noise_parameters: Dict[str, Any]
    bounds: Optional[Tuple[float, float]] = None

    @classmethod
    def from_environment(cls, env: Any) -> "EnvironmentConfig":
        drift_type = env.drift.__class__.__name__
        drift_params = {}
        drift_attrs = [
            "velocity",
            "sigma",
            "alpha",
            "threshold",
            "mode",
            "period",
            "amplitude",
            "center",
            "interval",
            "jump_magnitude",
        ]
        for attr in drift_attrs:
            if hasattr(env.drift, attr):
                try:
                    val = getattr(env.drift, attr)
                    if isinstance(val, (int, float, str, list, dict, bool, np.ndarray)):
                        if isinstance(val, np.ndarray):
                            drift_params[attr] = val.tolist()
                        else:
                            drift_params[attr] = val
                except Exception:
                    continue

        landscape_type = env.landscape.__class__.__name__
        landscape_params = {}
        landscape_attrs = [
            "condition_number",
            "p",
            "k_centers",
            "width",
            "delta",
            "a",
            "b",
            "scale",
        ]
        for attr in landscape_attrs:
            if hasattr(env.landscape, attr):
                try:
                    val = getattr(env.landscape, attr)
                    if isinstance(val, (int, float, str, bool)):
                        landscape_params[attr] = val
                except Exception:
                    continue

        noise_type = "none"
        noise_params = {}

        if hasattr(env, "_oracle") and env._oracle is not None:
            oracle = env._oracle
            if hasattr(oracle, "value_noise") and oracle.value_noise is not None:
                noise_obj = oracle.value_noise
                noise_type = noise_obj.__class__.__name__
                noise_attrs = ["sigma", "alpha", "scale", "phi", "p", "lam"]
                for attr in noise_attrs:
                    if hasattr(noise_obj, attr):
                        try:
                            val = getattr(noise_obj, attr)
                            if isinstance(val, (int, float, str, bool)):
                                noise_params[attr] = val
                        except Exception:
                            continue

        return cls(
            dim=env.dim,
            drift_type=drift_type,
            drift_parameters=drift_params,
            landscape_type=landscape_type,
            landscape_parameters=landscape_params,
            noise_type=noise_type,
            noise_parameters=noise_params,
            bounds=getattr(env, "bounds", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """
    Complete experiment result with unified export capabilities.
    """

    status: str
    error_message: Optional[str]
    optimizer_info: OptimizerInfo
    environment_config: EnvironmentConfig
    final_metrics: Dict[str, float]
    metrics_history: Dict[str, List[float]]
    trajectory: Optional[Dict[str, List[Union[np.ndarray, float]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0

    def __post_init__(self):
        if self.trajectory is not None:
            assert "x" in self.trajectory, "Trajectory missing 'x' (algorithm path)"
            assert (
                "theta" in self.trajectory
            ), "Trajectory missing 'theta' (ground truth optimum)"
            assert len(self.trajectory["x"]) == len(
                self.trajectory["theta"]
            ), "Trajectory length mismatch between x and theta"

    def to_dict(self) -> Dict[str, Any]:
        result_dict = asdict(self)

        if self.trajectory:
            traj_serialized = {}
            for key, values in self.trajectory.items():
                if isinstance(values, list) and len(values) > 0:
                    if isinstance(values[0], np.ndarray):
                        traj_serialized[key] = [v.tolist() for v in values]
                    else:
                        traj_serialized[key] = values
                else:
                    traj_serialized[key] = values
            result_dict["trajectory"] = traj_serialized

        result_dict["_format"] = "sthrd-v1.0"
        result_dict["_has_ground_truth"] = (
            self.trajectory is not None and "theta" in self.trajectory
        )
        result_dict["_reproducible"] = True

        return result_dict

    def save_to_json(self, filepath: str) -> Path:
        """
        Save entire experiment to a SINGLE JSON file.
        """
        data = self.to_dict()
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Saved complete experiment to single file: {output_path}")
        print(
            f"   → Optimizer: {self.optimizer_info.name} ({self.optimizer_info.class_name})"
        )
        print(f"   → Oracle type: {self.optimizer_info.oracle_type}")
        print(
            f"   → Ground truth (theta_t) included: YES ({len(self.trajectory['theta'])} steps)"
        )
        print(f"   → File size: {output_path.stat().st_size / 1024:.1f} KB")

        return output_path

    def save_to_csv(self, filepath: str) -> Path:
        """
        Save experiment to a SINGLE CSV file in long format.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas required for CSV export. Install with: pip install pandas"
            )

        if self.trajectory is None:
            raise ValueError(
                "Trajectory not recorded. Use record_trajectory=True in BenchmarkRunner."
            )

        x_array = np.array(self.trajectory["x"])
        theta_array = np.array(self.trajectory["theta"])
        steps, dim = x_array.shape

        rows = []
        exp_id = self.metadata.get(
            "experiment_id",
            f"sthrd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        for t in range(steps):
            for d in range(dim):
                row = {
                    "experiment_id": exp_id,
                    "optimizer_name": self.optimizer_info.name,
                    "optimizer_class": self.optimizer_info.class_name,
                    "oracle_type": self.optimizer_info.oracle_type,
                    "step": t,
                    "dim": d,
                    "x_value": float(x_array[t, d]),
                    "theta_value": float(theta_array[t, d]),
                    "coordinate": f"x_{d}",
                    "is_optimal": False,
                    "metric_name": "tracking_error_l2",
                    "metric_value": float(
                        self.metrics_history.get("error_l2", [np.nan] * steps)[t]
                    ),
                }
                for hp_name, hp_value in self.optimizer_info.hyperparameters.items():
                    if isinstance(hp_value, (int, float, str, bool)):
                        row[f"optimizer_{hp_name}"] = hp_value
                rows.append(row)

            inst_loss = self.metrics_history.get(
                "instantaneous_loss", [np.nan] * steps
            )[t]
            inst_regret = self.metrics_history.get(
                "instantaneous_regret", [np.nan] * steps
            )[t]

            row = {
                "experiment_id": exp_id,
                "optimizer_name": self.optimizer_info.name,
                "optimizer_class": self.optimizer_info.class_name,
                "oracle_type": self.optimizer_info.oracle_type,
                "step": t,
                "dim": -1,
                "x_value": float(inst_loss),
                "theta_value": 0.0,
                "coordinate": "f_x_t",
                "is_optimal": True,
                "metric_name": "instantaneous_regret",
                "metric_value": float(inst_regret),
            }
            for hp_name, hp_value in self.optimizer_info.hyperparameters.items():
                if isinstance(hp_value, (int, float, str, bool)):
                    row[f"optimizer_{hp_name}"] = hp_value
            rows.append(row)

        df = pd.DataFrame(rows)
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✅ Saved to single CSV: {output_path}")
        print(f"   → Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   → Includes ground truth theta_value column")
        print(f"   → File size: {output_path.stat().st_size / 1024:.1f} KB")

        return output_path

    def get_ground_truth(self) -> np.ndarray:
        if self.trajectory is None or "theta" not in self.trajectory:
            raise ValueError(
                "Ground truth not available. Use record_trajectory=True in BenchmarkRunner."
            )
        return np.array(self.trajectory["theta"])

    def get_algorithm_path(self) -> np.ndarray:
        if self.trajectory is None or "x" not in self.trajectory:
            raise ValueError("Trajectory not recorded.")
        return np.array(self.trajectory["x"])


class BenchmarkRunner:
    """
    Executes a single optimization experiment in dynamic environment.
    """

    def __init__(
        self,
        environment: Any,
        oracle: Any,
        metrics: Any,
        record_trajectory: bool = False,
        tail_fraction: float = 0.2,
    ):
        self.environment = environment
        self.oracle = oracle
        self.metrics = metrics
        self.record_trajectory = record_trajectory
        self.tail_fraction = tail_fraction

        self.trajectory = None
        self.metrics_history = {}
        self._step = 0

    def run(
        self,
        optimizer: OptimizerProtocol,
        T: int,
        x0: np.ndarray,
        seed: Optional[int] = None,
    ):
        """
        Execute optimization experiment for T timesteps.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        start_time = time.time()
        optimizer.reset()
        self.metrics.reset()
        self.environment.reset()
        self.oracle.reset()

        if self.record_trajectory:
            self.trajectory = {"x": [], "theta": [], "action": []}

        self.metrics_history = {m.name: [] for m in self.metrics.metrics}

        x = x0.copy()
        current_theta = self.environment.get_current_theta(for_analysis=True)

        if self.record_trajectory:
            self.trajectory["x"].append(x.copy())
            self.trajectory["theta"].append(current_theta.copy())
            if hasattr(self.environment.drift, "last_action"):
                self.trajectory["action"].append(
                    self.environment.drift.last_action.copy()
                )
            else:
                self.trajectory["action"].append(np.zeros_like(x))

        for t in range(T):
            self._step = t

            self.oracle.start_step(t)
            observation = self.oracle.query(x)
            self.oracle.end_step()

            theta_t = self.environment.get_current_theta(for_analysis=True)

            self.metrics.update(t, x, theta_t, observation, self.environment)

            current_metrics = self.metrics.get_current_values()
            for name, value in current_metrics.items():
                self.metrics_history[name].append(value)

            if self.record_trajectory:
                self.trajectory["x"].append(x.copy())
                self.trajectory["theta"].append(theta_t.copy())
                if hasattr(self.environment.drift, "last_action"):
                    self.trajectory["action"].append(
                        self.environment.drift.last_action.copy()
                    )
                else:
                    self.trajectory["action"].append(np.zeros_like(x))

            action = (
                x
                if hasattr(self.environment.drift, "step")
                and "action"
                in inspect.signature(self.environment.drift.step).parameters
                else None
            )
            self.environment.step(action=action)

            x = optimizer.step(observation)

        final_metrics = self.metrics.get_results(tail_fraction=self.tail_fraction)

        optimizer_info = OptimizerInfo.from_optimizer(optimizer)
        environment_config = EnvironmentConfig.from_environment(self.environment)

        metadata = {
            "experiment_id": f"sthrd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": seed,
            "total_steps": T,
            "dimension": self.environment.dim,
            "oracle_type": optimizer_info.oracle_type,
            "record_trajectory": self.record_trajectory,
            "tail_fraction": self.tail_fraction,
        }

        runtime = time.time() - start_time

        result = ExperimentResult(
            status="SUCCESS",
            error_message=None,
            optimizer_info=optimizer_info,
            environment_config=environment_config,
            final_metrics=final_metrics,
            metrics_history=self.metrics_history,
            trajectory=self.trajectory if self.record_trajectory else None,
            metadata=metadata,
            runtime=runtime,
        )

        return result


class BatchRunner:
    """
    Executes multiple experiments with different seeds for statistical significance.
    """

    def __init__(
        self,
        environment_factory: callable,
        oracle_factory: callable,
        metric_factory: callable,
        record_trajectory: bool = False,
        tail_fraction: float = 0.2,
    ):
        self.environment_factory = environment_factory
        self.oracle_factory = oracle_factory
        self.metric_factory = metric_factory
        self.record_trajectory = record_trajectory
        self.tail_fraction = tail_fraction

    def run(
        self, optimizer_factory: callable, seeds: List[int], T: int = 500, **env_kwargs
    ) -> List[ExperimentResult]:
        results = []

        for seed in seeds:
            print(f"Running seed {seed}...", end=" ")

            env = self.environment_factory(**env_kwargs)
            oracle = self.oracle_factory(env)
            metrics = self.metric_factory()
            optimizer = optimizer_factory()

            runner = BenchmarkRunner(
                env,
                oracle,
                metrics,
                record_trajectory=self.record_trajectory,
                tail_fraction=self.tail_fraction,
            )
            result = runner.run(
                optimizer, T=T, x0=np.random.randn(env.dim) * 0.1, seed=seed
            )

            results.append(result)
            print(f"DONE (error={result.final_metrics.get('error_l2', -1):.3f})")

        return results

    def save_batch_to_json(
        self,
        results: List[ExperimentResult],
        filepath: str,
        algorithm_name: str = "unknown",
    ) -> Path:
        metric_names = results[0].final_metrics.keys()
        aggregated = {}
        for name in metric_names:
            values = [r.final_metrics[name] for r in results if name in r.final_metrics]
            aggregated[f"{name}_mean"] = float(np.mean(values))
            aggregated[f"{name}_std"] = float(np.std(values))
            aggregated[f"{name}_min"] = float(np.min(values))
            aggregated[f"{name}_max"] = float(np.max(values))

        batch_data = {
            "batch_metadata": {
                "algorithm_name": algorithm_name,
                "num_seeds": len(results),
                "seeds": [r.metadata.get("seed") for r in results],
                "dimension": results[0].environment_config.dim,
                "drift_type": results[0].environment_config.drift_type,
                "drift_speed_rho": results[0].environment_config.drift_parameters.get(
                    "rho", 0.0
                ),
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "algorithm_name": algorithm_name,
            "results": [r.to_dict() for r in results],
            "aggregated_metrics": aggregated,
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ Saved batch of {len(results)} experiments to: {output_path}")
        print(f"   → Algorithm: {algorithm_name}")
        print(f"   → Seeds: {batch_data['batch_metadata']['seeds']}")
        print(
            f"   → Aggregated error_l2: {aggregated.get('error_l2_mean', -1):.3f} ± {aggregated.get('error_l2_std', -1):.3f}"
        )

        return output_path
