"""UJI 扫描的观测统计：按坐标聚合的检出次数与检出值，以及两个与楼层无关的观测常数。

UJI 的 100 表示本次扫描没有报告该 AP。它是删失（信号低于设备能报告的水平），不是 −110 dBm 的测量值。
"""
import numpy as np

NO_DETECT = 100


def cells(floor):
    """去重后按坐标聚合：n 为每位置扫描数；k、s1、s2 为每 AP 的检出次数与检出值的一、二阶和。"""
    _, first = np.unique(floor["groups"], return_index=True)
    rssi, xy = floor["rssi"][first], floor["xy"][first]
    keys, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    det = rssi != NO_DETECT
    x = np.where(det, rssi, 0).astype(np.float64)
    k, s1, s2 = (np.zeros((len(keys), rssi.shape[1])) for _ in range(3))
    np.add.at(k, inv, det)
    np.add.at(s1, inv, x)
    np.add.at(s2, inv, x * x)
    return {"xy": keys, "n": np.bincount(inv).astype(np.float64), "k": k, "s1": s1, "s2": s2}


def fit_observation(floor_cells):
    """σ_t：同一 (位置, AP) 单元内检出值的合并标准差；
    h：强信号单元（检出均值 > −60 dBm、至少 10 次扫描）的合并检出率，即随机漏检之外的检出上限。"""
    num = sum(float((c["s2"] - c["s1"] ** 2 / np.maximum(c["k"], 1))[c["k"] >= 2].sum()) for c in floor_cells)
    den = sum(float((c["k"] - 1)[c["k"] >= 2].sum()) for c in floor_cells)
    hit = total = 0.0
    for c in floor_cells:
        strong = (c["s1"] / np.maximum(c["k"], 1) > -60) & (c["k"] >= 1) & (c["n"][:, None] >= 10)
        hit += float(c["k"][strong].sum())
        total += float(np.broadcast_to(c["n"][:, None], c["k"].shape)[strong].sum())
    return {"sigma_total": float(np.sqrt(num / den)), "h_det": hit / total}
