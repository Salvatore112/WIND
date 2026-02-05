import numpy as np
from abc import ABC, abstractmethod


class Noise(ABC):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def sample(self, shape):
        pass

    def __call__(self, shape):
        return self.sample(shape)


class GaussianNoise(Noise):
    def __init__(self, std=1.0, seed=None):
        super().__init__(seed)
        self.std = std

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape == ():
            return float(self.std * self.rng.randn())
        return self.std * self.rng.randn(*shape)


class ParetoNoise(Noise):
    def __init__(self, alpha=2.5, scale=1.0, seed=None):
        super().__init__(seed)
        self.alpha = alpha
        self.scale = scale

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape == ():
            u = self.rng.uniform()
            magnitude = (1.0 - u) ** (-1.0 / self.alpha) - 1.0
            sign = 1 if self.rng.uniform() > 0.5 else -1
            return float(self.scale * magnitude * sign)

        u = self.rng.uniform(size=shape)
        magnitude = (1.0 - u) ** (-1.0 / self.alpha) - 1.0
        signs = np.sign(self.rng.uniform(size=shape) - 0.5)
        return self.scale * magnitude * signs


class BernoulliSparseNoise(Noise):
    def __init__(self, sparsity=0.1, scale=1.0, seed=None):
        super().__init__(seed)
        self.sparsity = sparsity
        self.scale = scale

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape == ():
            if self.rng.rand() < self.sparsity:
                return float(self.scale * self.rng.randn())
            return 0.0

        mask = self.rng.binomial(1, self.sparsity, size=shape)
        values = self.rng.randn(*shape)
        return self.scale * mask * values


class QuantizationNoise(Noise):
    def __init__(self, levels=10, scale=1.0, seed=None):
        super().__init__(seed)
        self.levels = levels
        self.scale = scale

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape == ():
            continuous = self.rng.randn()
            quantized = np.round(continuous * self.levels) / self.levels
            return float(self.scale * (quantized - continuous))

        continuous = self.rng.randn(*shape)
        quantized = np.round(continuous * self.levels) / self.levels
        return self.scale * (quantized - continuous)
