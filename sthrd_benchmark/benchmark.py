import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools


class BenchmarkRunner:
    def __init__(self, environment, oracle, metrics, config=None):
        self.env = environment
        self.oracle = oracle
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.config = config or {}
        self.results = {}

    def run(self, algorithm, n_steps=None, initial_x=None, verbose=True):
        n_steps = n_steps or self.config.get("n_steps", 1000)
        eval_freq = self.config.get("eval_frequency", 1)

        if initial_x is None:
            x = np.zeros(self.env.dim)
        else:
            x = initial_x.copy()

        self.env.reset()
        for metric in self.metrics:
            metric.reset()

        history = {
            "x": [],
            "theta": [],
            "time": [],
            "loss": [],
            "best_loss": [],
            "oracle_calls": 0,
        }

        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc="Running benchmark")

        for t in iterator:
            theta = self.env.state()
            oracle_data = self.oracle.query(x)
            history["oracle_calls"] += oracle_data.get("samples_used", 1)

            x = algorithm.step(x, oracle_data)

            if t % eval_freq == 0:
                current_loss = oracle_data.get("value", 0)
                for metric in self.metrics:
                    metric.update(x, theta, time=t, loss=current_loss, best_loss=0)

            history["x"].append(x.copy())
            history["theta"].append(theta.copy())
            history["time"].append(t)
            history["loss"].append(oracle_data.get("value", 0))
            history["best_loss"].append(0)

            self.env.step()

        self.results = {
            "history": history,
            "metrics": {m.name: m.compute() for m in self.metrics},
            "final_state": {
                "x": x,
                "theta": self.env.state(),
                "final_error": np.linalg.norm(x - self.env.state()),
            },
            "config": {"n_steps": n_steps, "dim": self.env.dim},
        }

        return self.results

    def get_dataframe(self):
        history = self.results.get("history", {})
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(
            {
                "time": history["time"],
                "loss": history["loss"],
                "best_loss": history["best_loss"],
                "x_norm": [np.linalg.norm(x) for x in history["x"]],
                "theta_norm": [np.linalg.norm(theta) for theta in history["theta"]],
                "error": [
                    np.linalg.norm(x - theta)
                    for x, theta in zip(history["x"], history["theta"])
                ],
            }
        )

        for i in range(min(3, self.env.dim)):
            df[f"x_{i}"] = [x[i] for x in history["x"]]
            df[f"theta_{i}"] = [theta[i] for theta in history["theta"]]

        return df


class BatchRunner:
    def __init__(self, base_config):
        self.base_config = base_config
        self.all_results = []

    def run_grid(self, param_grid, algorithm_class, n_trials=5, n_steps=500):
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        results = []

        for params in tqdm(param_combinations, desc="Parameter grid"):
            param_dict = dict(zip(param_names, params))

            trial_results = []
            for trial in range(n_trials):
                config = self.base_config.copy()
                config.update(param_dict)
                config["seed"] = trial

                env = DynamicEnvironment(
                    drift=config["drift"], landscape=config["landscape"]
                )

                oracle = FirstOrderOracle(
                    environment=env,
                    grad_noise=config.get("grad_noise"),
                    value_noise=config.get("value_noise"),
                )

                runner = BenchmarkRunner(
                    environment=env,
                    oracle=oracle,
                    metrics=config.get("metrics", []),
                    config={"n_steps": n_steps},
                )

                algorithm = algorithm_class(
                    dim=env.dim, **config.get("algorithm_params", {})
                )
                result = runner.run(algorithm, verbose=False)

                trial_results.append(
                    {
                        "params": param_dict,
                        "trial": trial,
                        "final_error": result["final_state"]["final_error"],
                        "metrics": result["metrics"],
                        "oracle_calls": result["history"]["oracle_calls"],
                    }
                )

            avg_error = np.mean([r["final_error"] for r in trial_results])
            avg_metrics = {}
            for metric_name in trial_results[0]["metrics"].keys():
                avg_metrics[metric_name] = np.mean(
                    [r["metrics"][metric_name] for r in trial_results]
                )

            results.append(
                {
                    "params": param_dict,
                    "avg_error": avg_error,
                    "avg_metrics": avg_metrics,
                    "trials": trial_results,
                }
            )

        self.all_results = results
        return results

    def to_dataframe(self):
        rows = []
        for result in self.all_results:
            row = result["params"].copy()
            row["avg_error"] = result["avg_error"]
            for metric_name, metric_value in result["avg_metrics"].items():
                row[metric_name] = metric_value
            rows.append(row)
        return pd.DataFrame(rows)
