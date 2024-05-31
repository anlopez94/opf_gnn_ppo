from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class GenScaler(BaseEstimator, TransformerMixin):
    def __init__(self, gen_config):
        BaseEstimator.__init__(self)
        self.gen_config = gen_config

    def fit(self):
        return self

    def transform(self, ts: np.array) -> np.array:
        ts = ts.copy()
        ts_std = []
        for instance in ts:
            instance_std = []
            for gen_id, bus_id, config in enumerate(self.gen_config.items()):
                min = config["min"]
                max = config["max"]
                ts_std.append((instance[gen_id] - min) / (max - min))
            ts_std.append(instance_std)
        return np.array(ts_std)

    def inverse_transform(self, ts: np.array) -> np.array:
        ts = ts.copy()
        ts_std = []
        for instance in ts:
            instance_std = []
            for gen_id, bus_id, config in enumerate(self.gen_config.items()):
                min = config["min"]
                max = config["max"]
                instance_std.append(instance[gen_id] * (max - min) + min)
            ts_std.append(instance_std)
        return np.array(ts_std)


class RewardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, max, min):
        BaseEstimator.__init__(self)
        self.max = max
        self.min = min

    def fit(self):
        return self

    def transform(self, ts: float) -> float:
        ts_std = (ts - self.min) / (self.max - self.min)
        return ts_std

    def inverse_transform(self, ts: float) -> float:
        ts_std = ts * (self.max - self.min) + self.min
        return ts_std
