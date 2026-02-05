import numpy as np
from abc import ABC, abstractmethod


class Drift(ABC):
    def __init__(self, dim, seed=None):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.reset()

    @abstractmethod
    def step(self):
        pass

    def reset(self):
        self.position = np.zeros(self.dim)
        return self.position

    def __call__(self):
        return self.step()


class StationaryDrift(Drift):
    def __init__(self, dim, seed=None):
        super().__init__(dim, seed)

    def step(self):
        return self.position


class LinearDrift(Drift):
    def __init__(self, dim, direction=None, step_size=0.01, seed=None):
        super().__init__(dim, seed)
        if direction is None:
            self.direction = self.rng.randn(dim)
            self.direction /= np.linalg.norm(self.direction)
        else:
            self.direction = np.array(direction) / np.linalg.norm(direction)
        self.step_size = step_size

    def step(self):
        self.position += self.step_size * self.direction
        return self.position


class RandomWalkDrift(Drift):
    def __init__(self, dim, step_size=0.01, seed=None):
        super().__init__(dim, seed)
        self.step_size = step_size

    def step(self):
        step = self.rng.randn(self.dim)
        step /= np.linalg.norm(step)
        self.position += self.step_size * step
        return self.position


class CyclicDrift(Drift):
    def __init__(self, dim, period=100, amplitude=1.0, seed=None):
        super().__init__(dim, seed)
        self.period = period
        self.amplitude = amplitude
        self.directions = [self.rng.randn(dim) for _ in range(3)]
        for d in self.directions:
            d /= np.linalg.norm(d)
        self.t = 0

    def step(self):
        t_norm = 2 * np.pi * self.t / self.period
        self.position = self.amplitude * (
            np.sin(t_norm) * self.directions[0]
            + np.cos(t_norm) * self.directions[1]
            + np.sin(2 * t_norm) * self.directions[2]
        )
        self.t += 1
        return self.position


class TeleportationDrift(Drift):
    def __init__(self, dim, teleport_prob=0.01, jump_scale=1.0, seed=None):
        super().__init__(dim, seed)
        self.teleport_prob = teleport_prob
        self.jump_scale = jump_scale

    def step(self):
        if self.rng.rand() < self.teleport_prob:
            self.position = self.jump_scale * self.rng.randn(self.dim)
        return self.position
