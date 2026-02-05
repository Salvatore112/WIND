import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import ortho_group


class Landscape(ABC):
    def __init__(self, dim, seed=None):
        self.dim = dim
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def value(self, x, optimum):
        pass

    @abstractmethod
    def gradient(self, x, optimum):
        pass


class QuadraticLandscape(Landscape):
    def __init__(self, dim, condition_number=10.0, seed=None):
        super().__init__(dim, seed)
        self.condition_number = condition_number
        eigenvalues = np.linspace(1.0, condition_number, dim)
        Q = ortho_group.rvs(dim, random_state=self.rng)
        self.A = Q @ np.diag(eigenvalues) @ Q.T
        self.A_sqrt = Q @ np.diag(np.sqrt(eigenvalues)) @ Q.T

    def value(self, x, optimum):
        diff = x - optimum
        return 0.5 * diff.T @ self.A @ diff

    def gradient(self, x, optimum):
        return self.A @ (x - optimum)


class QuadraticRavineLandscape(QuadraticLandscape):
    def __init__(self, dim, condition_number=100.0, ravine_direction=None, seed=None):
        super().__init__(dim, condition_number, seed)
        if ravine_direction is None:
            self.ravine_dir = self.rng.randn(dim)
            self.ravine_dir /= np.linalg.norm(self.ravine_dir)
        else:
            self.ravine_dir = ravine_direction / np.linalg.norm(ravine_direction)

        Q, _ = np.linalg.qr(
            np.column_stack(
                [self.ravine_dir] + [self.rng.randn(dim) for _ in range(dim - 1)]
            )
        )
        eigenvalues = np.ones(dim)
        eigenvalues[0] = 1.0
        eigenvalues[1:] = np.linspace(condition_number, condition_number * 0.1, dim - 1)
        self.A = Q @ np.diag(eigenvalues) @ Q.T


class PNormLandscape(Landscape):
    def __init__(self, dim, p=3.0, seed=None):
        super().__init__(dim, seed)
        self.p = p

    def value(self, x, optimum):
        diff = x - optimum
        return np.sum(np.abs(diff) ** self.p) / self.p

    def gradient(self, x, optimum):
        diff = x - optimum
        return np.sign(diff) * (np.abs(diff) ** (self.p - 1))


class RosenbrockLandscape(Landscape):
    def __init__(self, dim, seed=None):
        super().__init__(dim, seed)
        if dim < 2:
            raise ValueError("Rosenbrock requires at least 2 dimensions")

    def value(self, x, optimum):
        z = x - optimum + 1.0
        return np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2)

    def gradient(self, x, optimum):
        z = x - optimum + 1.0
        grad = np.zeros_like(z)
        grad[:-1] = -400 * z[:-1] * (z[1:] - z[:-1] ** 2) - 2 * (1 - z[:-1])
        grad[1:] += 200 * (z[1:] - z[:-1] ** 2)
        return grad
