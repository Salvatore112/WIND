import numpy as np
from abc import ABC, abstractmethod


class Oracle(ABC):
    def __init__(self, environment, seed=None):
        self.env = environment
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def query(self, x):
        pass


class FirstOrderOracle(Oracle):
    def __init__(self, environment, grad_noise=None, value_noise=None, seed=None):
        super().__init__(environment, seed)
        self.grad_noise = grad_noise
        self.value_noise = value_noise

    def query(self, x):
        true_grad = self.env.gradient(x)
        true_value = self.env.value(x)

        noisy_grad = true_grad.copy()
        noisy_value = true_value

        if self.grad_noise:
            noisy_grad += self.grad_noise(true_grad.shape)

        if self.value_noise:
            noisy_value += self.value_noise(())

        return {
            "value": noisy_value,
            "grad": noisy_grad,
            "true_value": true_value,
            "true_grad": true_grad,
        }


class ZeroOrderOracle(Oracle):
    def __init__(
        self, environment, perturbation_scale=0.01, value_noise=None, seed=None
    ):
        super().__init__(environment, seed)
        self.perturbation_scale = perturbation_scale
        self.value_noise = value_noise

    def query(self, x):
        delta = self.rng.randn(*x.shape)
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 0:
            delta = delta / delta_norm
        else:
            delta = np.ones_like(x) / np.sqrt(len(x))

        x_plus = x + self.perturbation_scale * delta
        x_minus = x - self.perturbation_scale * delta

        f_plus = self.env.value(x_plus)
        f_minus = self.env.value(x_minus)

        if self.value_noise:
            f_plus += self.value_noise(())
            f_minus += self.value_noise(())

        grad_estimate = (f_plus - f_minus) / (2 * self.perturbation_scale) * delta

        return {
            "value": (f_plus + f_minus) / 2,
            "grad": grad_estimate,
            "true_value": self.env.value(x),
            "samples_used": 2,
        }


class HybridOracle(Oracle):
    def __init__(
        self, environment, grad_noise=None, value_noise=None, sparse_prob=0.1, seed=None
    ):
        super().__init__(environment, seed)
        self.grad_noise = grad_noise
        self.value_noise = value_noise
        self.sparse_prob = sparse_prob

    def query(self, x):
        if self.rng.rand() < self.sparse_prob:
            oracle = ZeroOrderOracle(
                self.env,
                perturbation_scale=0.01,
                value_noise=self.value_noise,
                seed=int(self.rng.rand() * 1000),
            )
            return oracle.query(x)
        else:
            oracle = FirstOrderOracle(
                self.env,
                grad_noise=self.grad_noise,
                value_noise=self.value_noise,
                seed=int(self.rng.rand() * 1000),
            )
            return oracle.query(x)
