"""SCM-T：信号残差场校准的相邻楼层地图 + 删失（Tobit）匹配。

主线是同一个观测模型：目标层指纹 v_a(p)=u_a(p)+Δ_a(p)；AP a 被检出 ⟺ 场值高于设备阈值 θ。
  场：Δ 由 10 个锚点的信号残差经平面 GP 插值（v1，ℓ=60, λ=0.3，源端已选定，本脚本不再改）。
  匹配：候选 j 的对数似然
     score_j = −Σ_{a 检出}(x_a−V_ja)²/(2σ²) + Σ_{a 未检出} log Φ((θ−V_ja)/σ)
  权重 ∝ exp(score_j)，输出后验均值坐标。第二项是"查询没看到、候选却声称可见"的反对票。
选择量只有 (σ, θ)，只用 7 对历史相邻转移（trainingData）；官方 validation 在冻结后评价。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

from data.uji import load_floors
from scripts.evaluate_signal_calibrated_map import (FLOOR, build_map, dbm, episodes, lower_map,
                                                    mde, position_map, wknn)

PAIRS = [(f"B{b}F{f}", f"B{b}F{f + 1}") for b, top in ((0, 3), (1, 3), (2, 4)) for f in range(top - 1)]
FIELD = (60.0, 0.3)
QUAD = (8, 10)
ZERO_QUAD = (1000, 10)
SIGMAS = (12.0, 16.0, 24.0, 32.0, 48.0)
THETAS = (-90.0, -85.0, -80.0, -75.0, -70.0)
SEED = 1
EPISODES = 20
QUERY_CAP = 300
CFG = {"anchor_lookup_neighbors": 3}


def tobit_match(qraw, protos, positions, sigma, theta):
    det = (qraw != 100).astype(np.float64)
    x = dbm(qraw) * det
    d2 = (x * x).sum(1, keepdims=True) + det @ (protos * protos).T - 2 * x @ protos.T
    score = -0.5 / sigma ** 2 * d2 + (1 - det) @ log_ndtr((theta - protos) / sigma).T
    score -= score.max(1, keepdims=True)
    w = np.exp(score)
    return (w / w.sum(1, keepdims=True)) @ positions


def calibrated(old, s_rssi, s_xy):
    return build_map("scm", old, s_rssi, s_xy, FIELD, CFG)


def source_episodes(train):
    """每对单独重置随机流（与 v2/v3 相同），每对 20 个 10×3 位置隔离 episode。"""
    out = {}
    for lower, upper in PAIRS:
        old, new = lower_map(train[lower]), train[upper]
        rng = np.random.default_rng(SEED)
        eps = []
        for s, q in episodes(new["xy"], 10, EPISODES, 3, rng):
            q = rng.choice(q, min(QUERY_CAP, len(q)), replace=False)
            eps.append((calibrated(old, new["rssi"][s], new["xy"][s]), new["rssi"][q], new["xy"][q]))
        out[f"{lower}->{upper}"] = eps
    return out


def select(eps):
    grid = [(s, t) for s in SIGMAS for t in THETAS]
    per = {"quad": {p: float(np.mean([mde(wknn(q, *c, *QUAD), y) for c, q, y in e])) for p, e in eps.items()}}
    for g in grid:
        per[g] = {p: float(np.mean([mde(tobit_match(q, *c, *g), y) for c, q, y in e])) for p, e in eps.items()}
    equal = {k: float(np.mean(list(v.values()))) for k, v in per.items()}
    best = min(grid, key=equal.get)
    lopo = {}
    for held in eps:
        pick = min(grid, key=lambda g: np.mean([per[g][p] for p in eps if p != held]))
        lopo[held] = {"chosen": list(pick), "quad_m": per["quad"][held], "tobit_m": per[pick][held]}
    return best, {str(k): v for k, v in equal.items()}, lopo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    train, validation = load_floors(config["train_path"]), load_floors(config["validation_path"])

    best, equal, lopo = select(source_episodes(train))
    gains = [v["quad_m"] - v["tobit_m"] for v in lopo.values()]
    print("source: quad", round(equal["quad"], 3), "| best tobit", best, round(equal[str(best)], 3), flush=True)
    print("LOPO gain", round(float(np.mean(gains)), 3), "improved", sum(g > 0 for g in gains), "/ 7", flush=True)
    report = {"selected": {"sigma": best[0], "theta": best[1]}, "source_equal_weight": equal, "lopo": lopo}
    match = {"quad": lambda q, c: wknn(q, *c, *QUAD), "tobit": lambda q, c: tobit_match(q, *c, *best)}

    # 历史 7 层官方 validation：2×2 消融 {旧图, 校准场} × {二次, Tobit}，以及 K 曲线
    hist, kcurve = {}, {}
    for lower, upper in PAIRS:
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        own = lower_map(new)
        row = {"n": len(val["xy"]),
               "old_quad": mde(wknn(val["rssi"], *old, *ZERO_QUAD), val["xy"]),
               "old_tobit": mde(tobit_match(val["rssi"], *old, *best), val["xy"]),
               "full_quad": mde(wknn(val["rssi"], *own, *QUAD), val["xy"]),
               "full_tobit": mde(tobit_match(val["rssi"], *own, *best), val["xy"])}
        rng = np.random.default_rng(SEED + 1000)
        cands = [calibrated(old, new["rssi"][s], new["xy"][s]) for s, _ in episodes(new["xy"], 10, 20, 3, rng)]
        for name, fn in match.items():
            row[f"scm_{name}"] = float(np.mean([mde(fn(val["rssi"], c), val["xy"]) for c in cands]))
        hist[upper] = row
        rng = np.random.default_rng(SEED + 2000)
        for count in config["k_curve"]:
            cands = [calibrated(old, new["rssi"][s], new["xy"][s]) for s, _ in episodes(new["xy"], count, 20, 3, rng)]
            for name, fn in match.items():
                kcurve.setdefault(f"K{count}_{name}", {})[upper] = float(np.mean([mde(fn(val["rssi"], c), val["xy"]) for c in cands]))
        print(" ", upper, {k: round(v, 2) for k, v in row.items()}, flush=True)
    report["historical_validation"] = hist
    report["historical_k_curve"] = kcurve
    for key in ("old_quad", "old_tobit", "scm_quad", "scm_tobit", "full_quad", "full_tobit"):
        print(f"  equal {key:10s} {np.mean([r[key] for r in hist.values()]):.3f}  improved-vs-old_quad "
              f"{sum(r[key] < r['old_quad'] for r in hist.values())}/7", flush=True)
    print("  K curve:", {k: round(float(np.mean(list(v.values()))), 2) for k, v in kcurve.items()}, flush=True)

    # 附录：三个 FeMLoc 目标层（探索性），固定 manifest + 锚点置乱诊断
    targets = {}
    for name in config["targets"]:
        lower, upper = FLOOR[name]
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        ids = {int(r): i for i, r in enumerate(new["row_ids"])}
        res = {"old_quad_val": mde(wknn(val["rssi"], *old, *ZERO_QUAD), val["xy"]),
               "old_tobit_val": mde(tobit_match(val["rssi"], *old, *best), val["xy"])}
        for seed in config["manifest_seeds"]:
            man = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[name]
            s = np.array([ids[int(r)] for r in man["support"]])
            q = np.array([ids[int(r)] for r in man["unseen_position_query"]])
            cand = calibrated(old, new["rssi"][s], new["xy"][s])
            for m, fn in match.items():
                res.setdefault(f"scm_{m}_val", []).append(mde(fn(val["rssi"], cand), val["xy"]))
                res.setdefault(f"scm_{m}_internal", []).append(mde(fn(new["rssi"][q], cand), new["xy"][q]))
            _, keys, inv = position_map(new["rssi"][s], new["xy"][s])
            shuffled = [mde(match["tobit"](val["rssi"], calibrated(old, new["rssi"][s],
                            keys[np.random.default_rng(100 + r).permutation(len(keys))][inv])), val["xy"])
                        for r in range(config["shuffle_repeats"])]
            res.setdefault("scm_tobit_val_shuffled", []).append(float(np.mean(shuffled)))
        targets[name] = res
        print(" ", name, {k: round(float(np.mean(v)), 2) for k, v in res.items()}, flush=True)
    report["targets"] = targets
    (args.output / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
