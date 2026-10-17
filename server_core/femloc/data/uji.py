"""读取原始楼层数据，分离重复观测组并拟合仅使用Support的变换。"""
import csv
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch


def load_floors(path):
    floors = {}
    with open(path, newline="") as stream:
        for row_id, row in enumerate(csv.DictReader(stream), 1):
            name = f'B{row["BUILDINGID"]}F{row["FLOOR"]}'
            floors.setdefault(name, []).append((row_id, row))
    result = {}
    for name, rows in sorted(floors.items()):
        rss = np.array([[int(row[f"WAP{i:03d}"]) for i in range(1, 521)] for _, row in rows], dtype=np.int16)
        xy = np.array([[float(row[axis]) for axis in ("LONGITUDE", "LATITUDE")] for _, row in rows])
        groups = [hashlib.sha256(r.tobytes() + y.tobytes()).hexdigest() for r, y in zip(rss, xy)]
        result[name] = {"rssi": rss, "xy": xy, "row_ids": np.array([i for i, _ in rows]), "groups": np.array(groups)}
    return result


def split_groups(groups, folds, seed):
    """留出约一个fold的观测组；同RSSI和坐标的记录不会跨分区。"""
    unique = np.unique(groups)
    held = np.random.default_rng(seed).permutation(unique)[:math.ceil(len(unique) / folds)]
    is_held = np.isin(groups, held)
    return np.flatnonzero(~is_held), np.flatnonzero(is_held)


@dataclass
class Transform:
    aps: np.ndarray
    low: float
    high: float
    origin: np.ndarray

    @classmethod
    def fit(cls, rss, xy):
        seen = rss[rss != 100]
        return cls(np.flatnonzero((rss != 100).any(axis=0)), float(seen.min() - 1),
                   float(seen.max()), xy.mean(axis=0))

    def x(self, rss, device):
        rss = rss[:, self.aps].astype(np.float32)
        rss[rss == 100] = self.low
        x = np.clip((rss - self.low) / (self.high - self.low), 0, 1) ** math.e
        return torch.tensor(x, device=device)

    def y(self, xy, device):
        return torch.tensor(xy - self.origin, dtype=torch.float32, device=device)

    def state(self):
        return {"aps": self.aps.tolist(), "low": self.low, "high": self.high, "origin": self.origin.tolist(), "power": math.e}

    def diagnostics(self, rss):
        kept = rss[:, self.aps]
        observed = kept != 100
        return {"out_of_range_observations": int(((kept < self.low) | (kept > self.high))[observed].sum()),
                "unknown_ap_observations": int((rss != 100).sum() - observed.sum())}
