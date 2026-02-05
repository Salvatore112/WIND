import numpy as np


class DynamicEnvironment:
    def __init__(self, drift, landscape, seed=None):
        self.drift = drift
        self.landscape = landscape
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.drift.reset()
        self.time = 0
        return self.state()

    def state(self):
        return self.drift.position.copy()

    def step(self):
        self.drift.step()
        self.time += 1
        return self.state()

    def value(self, x):
        return self.landscape.value(x, self.state())

    def gradient(self, x):
        return self.landscape.gradient(x, self.state())

    def __call__(self):
        return self.step()

    def get_current_optimum(self):
        return self.state()

    @property
    def dim(self):
        return self.drift.dim
