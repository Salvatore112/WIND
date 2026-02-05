import numpy as np
from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.reset()

    @abstractmethod
    def update(self, prediction, target, **kwargs):
        pass

    def reset(self):
        self.values = []
        self.timestamps = []

    def compute(self):
        if not self.values:
            return None
        return np.mean(self.values)

    def __call__(self, prediction, target, **kwargs):
        value = self.update(prediction, target, **kwargs)
        return value


class TrackingErrorMetric(Metric):
    def __init__(self, norm=2, normalize_by_dim=False, name="TrackingError"):
        super().__init__(name)
        self.norm = norm
        self.normalize_by_dim = normalize_by_dim

    def update(self, prediction, target, **kwargs):
        error = np.linalg.norm(prediction - target, ord=self.norm)
        if self.normalize_by_dim:
            error /= len(prediction) ** (1.0 / self.norm)
        self.values.append(error)
        self.timestamps.append(kwargs.get("time", len(self.values)))
        return error


class DynamicRegretMetric(Metric):
    def __init__(self, name="DynamicRegret"):
        super().__init__(name)
        self.cumulative_loss = 0.0
        self.cumulative_best = 0.0
        self.best_loss_history = []

    def update(self, prediction, target, loss=None, best_loss=None, **kwargs):
        if loss is None:
            loss = np.linalg.norm(prediction - target)

        if best_loss is None:
            best_loss = 0.0

        self.cumulative_loss += loss
        self.cumulative_best += best_loss
        self.best_loss_history.append(best_loss)

        regret = self.cumulative_loss - self.cumulative_best
        self.values.append(regret)
        self.timestamps.append(kwargs.get("time", len(self.values)))
        return regret


class TimeToRecoveryMetric(Metric):
    def __init__(self, threshold=1e-3, name="TimeToRecovery"):
        super().__init__(name)
        self.threshold = threshold
        self.recovery_time = None

    def update(self, prediction, target, **kwargs):
        error = np.linalg.norm(prediction - target)

        if error < self.threshold and self.recovery_time is None:
            self.recovery_time = kwargs.get("time", len(self.values))

        self.values.append(error)
        self.timestamps.append(kwargs.get("time", len(self.values)))

        return self.recovery_time

    def compute(self):
        return self.recovery_time
