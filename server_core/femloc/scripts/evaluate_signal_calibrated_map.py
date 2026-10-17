"""信号空间校准的相邻楼层虚拟地图（Signal-Calibrated Map, SCM）。

旧楼层完整地图 u_j@p_j 保留不动；目标楼层 10 个锚点给出“该处真实指纹 − 旧图在该坐标的指纹”
这一 520 维信号残差。用楼层平面坐标上的 GP 均值把残差插值到旧图每个坐标，得到目标楼层的
虚拟指纹地图，再拼上锚点自身原型，最后做 WKNN。全部超参数只在 7 对历史相邻楼层转移上选。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_adjacent_map import rss_features

FLOOR = {"B0F3": ("B0F2", "B0F3"), "B1F3": ("B1F2", "B1F3"), "B2F4": ("B2F3", "B2F4")}
NO_SIGNAL = -110.0


def dbm(rssi):
    x = rssi.astype(np.float64)
    x[x == 100] = NO_SIGNAL
    return x


def position_map(rssi, xy):
    """同坐标全部扫描的 dBm 均值；返回原型、坐标和每条扫描所属位置。"""
    keys, inverse = np.unique(xy, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    x = dbm(rssi)
    return np.stack([x[inverse == k].mean(0) for k in range(len(keys))]), keys, inverse


def wknn(query_rssi, prototypes_dbm, positions, k, beta):
    """与旧分支相同的 clip((r+110)/110) 特征；在前 k 个候选上按距离做 softmax。"""
    a = rss_features(query_rssi).astype(np.float64)
    b = np.clip((prototypes_dbm - NO_SIGNAL) / -NO_SIGNAL, 0, 1)
    d = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0)
    k = min(k, len(b))
    idx = np.argsort(d, axis=1)[:, :k]
    dd = np.take_along_axis(d, idx, axis=1)
    w = np.exp(-beta * (dd - dd[:, :1]))
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * positions[idx]).sum(1)


def lookup(xy, positions, prototypes, neighbors):
    """用锚点的已知坐标在旧图上取几何近邻指纹（反距离加权）；不读取任何 Query。"""
    d = np.sqrt(((xy[:, None] - positions[None]) ** 2).sum(-1))
    idx = np.argsort(d, axis=1)[:, :neighbors]
    w = 1 / (np.take_along_axis(d, idx, axis=1) + 1)
    w /= w.sum(1, keepdims=True)
    return (w[..., None] * prototypes[idx]).sum(1)


def gp_mean(to_xy, from_xy, values, length, noise):
    kernel = lambda a, b: np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * length ** 2))
    return kernel(to_xy, from_xy) @ np.linalg.solve(kernel(from_xy, from_xy) + noise * np.eye(len(from_xy)), values)


def build_map(variant, lower, support_rssi, support_xy, params, config):
    """只接收旧图与 Support（RSSI+坐标）；构造目标楼层的检索地图。"""
    prototypes, positions = lower
    if variant == "map":
        return prototypes, positions
    anchor, anchor_xy, _ = position_map(support_rssi, support_xy)
    if variant == "map+support":
        return np.vstack([prototypes, anchor]), np.vstack([positions, anchor_xy])
    residual = anchor - lookup(anchor_xy, positions, prototypes, config["anchor_lookup_neighbors"])
    if variant == "global_offset":
        virtual = prototypes + residual.mean(0)
    else:
        virtual = prototypes + gp_mean(positions, anchor_xy, residual, *params)
    virtual = np.clip(virtual, NO_SIGNAL, 0)
    if variant == "field_only":
        return virtual, positions
    return np.vstack([virtual, anchor]), np.vstack([positions, anchor_xy])


def mde(prediction, truth):
    return float(np.linalg.norm(prediction - truth, axis=1).mean())


def episodes(xy, count, repeats, scans, rng):
    """位置隔离：抽 count 个至少有 scans 条扫描的位置，每处 scans 条；其余全部位置为 Query。"""
    _, _, inverse = position_map(np.zeros((len(xy), 520), dtype=np.int16), xy)
    eligible = [k for k in range(inverse.max() + 1) if (inverse == k).sum() >= scans]
    for _ in range(repeats):
        chosen = rng.choice(eligible, count, replace=False)
        support = np.concatenate([rng.choice(np.flatnonzero(inverse == k), scans, replace=False) for k in chosen])
        yield support, np.flatnonzero(~np.isin(inverse, chosen))


def lower_map(floor):
    prototypes, positions, _ = position_map(floor["rssi"], floor["xy"])
    return prototypes, positions


def select(train, config):
    """7 对历史转移（下层→上层，均为 trainingData）；每对等权，Query 为上层未抽中的位置。"""
    pairs = [(f"B{b}F{f}", f"B{b}F{f + 1}") for b, top in ((0, 3), (1, 3), (2, 4)) for f in range(top - 1)]
    grid = {"map": [None], "map+support": [None],
            "scm": [(l, n) for l in config["length_scales_m"] for n in config["noise_ratios"]]}
    knn = [(k, b) for k in config["knn_k"] for b in config["knn_beta"]]
    rng = np.random.default_rng(config["source_episode_seed"])
    scores = {}
    for lower, upper in pairs:
        old, new = lower_map(train[lower]), train[upper]
        for support, query in episodes(new["xy"], config["support_positions"], config["source_episodes_per_pair"],
                                       config["scans_per_position"], rng):
            query = rng.choice(query, min(config["source_query_cap"], len(query)), replace=False)
            for variant, params_list in grid.items():
                for params in params_list:
                    mapped = build_map(variant, old, new["rssi"][support], new["xy"][support], params, config)
                    for kb in knn:
                        key = (variant, params, kb)
                        err = mde(wknn(new["rssi"][query], *mapped, *kb), new["xy"][query])
                        scores.setdefault(key, {}).setdefault(upper, []).append(err)
    per_pair = {key: {p: float(np.mean(v)) for p, v in d.items()} for key, d in scores.items()}
    equal = {key: float(np.mean(list(d.values()))) for key, d in per_pair.items()}
    chosen = {v: min((k for k in equal if k[0] == v), key=equal.get) for v in grid}
    # 留一转移审计：第 d 对的超参数只由其余 6 对选出
    audit = {}
    for _, upper in pairs:
        pick = lambda v: min((k for k in equal if k[0] == v),
                             key=lambda k: np.mean([s for p, s in per_pair[k].items() if p != upper]))
        audit[upper] = {v: per_pair[pick(v)][upper] for v in ("map", "scm")}
    return pairs, chosen, {k: equal[k] for k in chosen.values()}, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    train, validation = load_floors(config["train_path"]), load_floors(config["validation_path"])
    pairs, chosen, source_scores, audit = select(train, config)
    scm, base = chosen["scm"], chosen["map"]
    report = {"selection": {v: {"params": k[1], "knn": k[2], "source_mde_m": source_scores[k]} for v, k in chosen.items()},
              "leave_one_pair_out": audit}
    print("selection", json.dumps(report["selection"]))
    print("LOPO gain", round(np.mean([a["map"] - a["scm"] for a in audit.values()]), 3),
          "improved", sum(a["scm"] < a["map"] for a in audit.values()), "/", len(audit))

    # 历史上层楼层的官方 validation：超参数冻结，Support 仍只来自该层 trainingData
    rng = np.random.default_rng(config["source_episode_seed"] + 1000)
    source_val = {}
    for lower, upper in pairs:
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        own = lower_map(new)
        runs = [mde(wknn(val["rssi"], *build_map("scm", old, new["rssi"][s], new["xy"][s], scm[1], config), *scm[2]), val["xy"])
                for s, _ in episodes(new["xy"], config["support_positions"], config["source_episodes_per_pair"],
                                     config["scans_per_position"], rng)]
        source_val[upper] = {"n": len(val["xy"]), "map": mde(wknn(val["rssi"], *old, *base[2]), val["xy"]),
                             "scm": float(np.mean(runs)), "full_target_map": mde(wknn(val["rssi"], *own, *scm[2]), val["xy"])}
    report["source_validation"] = source_val

    variants = {"map": base, "map+support": chosen["map+support"], "global_offset": scm, "field_only": scm, "scm": scm}
    targets = {}
    rng = np.random.default_rng(config["k_curve_seed"])
    for name in config["targets"]:
        lower, upper = FLOOR[name]
        old, new, val = lower_map(train[lower]), train[upper], validation[upper]
        ids = {int(r): i for i, r in enumerate(new["row_ids"])}
        result = {"n_validation": len(val["xy"]), "full_target_map_val": mde(wknn(val["rssi"], *lower_map(new), *scm[2]), val["xy"])}
        for seed in config["manifest_seeds"]:
            manifest = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[name]
            support = np.array([ids[int(r)] for r in manifest["support"]])
            query = np.array([ids[int(r)] for r in manifest["unseen_position_query"]])
            assert not set(map(tuple, new["xy"][support])) & set(map(tuple, new["xy"][query]))
            s_rssi, s_xy = new["rssi"][support], new["xy"][support]
            for variant, (v, params, kb) in variants.items():
                mapped = build_map(variant, old, s_rssi, s_xy, params, config)
                result.setdefault(f"{variant}_val", []).append(mde(wknn(val["rssi"], *mapped, *kb), val["xy"]))
                result.setdefault(f"{variant}_internal", []).append(mde(wknn(new["rssi"][query], *mapped, *kb), new["xy"][query]))
            # 机制诊断：30 条 RSSI 和 10 个坐标都不变，只打乱“哪组扫描对应哪个坐标”
            _, keys, inverse = position_map(s_rssi, s_xy)
            shuffled = [mde(wknn(val["rssi"], *build_map("scm", old, s_rssi, keys[np.random.default_rng(100 + r).permutation(len(keys))][inverse],
                                                         scm[1], config), *scm[2]), val["xy"]) for r in range(config["shuffle_repeats"])]
            result.setdefault("scm_val_shuffled_support", []).append(float(np.mean(shuffled)))
        for count in config["k_curve"]:
            for s, q in episodes(new["xy"], count, config["k_curve_draws"], config["scans_per_position"], rng):
                mapped = build_map("scm", old, new["rssi"][s], new["xy"][s], scm[1], config)
                result.setdefault(f"K{count}_val", []).append(mde(wknn(val["rssi"], *mapped, *scm[2]), val["xy"]))
                result.setdefault(f"K{count}_internal", []).append(mde(wknn(new["rssi"][q], *mapped, *scm[2]), new["xy"][q]))
                result.setdefault(f"K{count}_internal_map", []).append(mde(wknn(new["rssi"][q], *old, *base[2]), new["xy"][q]))
        targets[name] = result
    report["targets"] = targets
    (args.output / "results.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    for p, v in source_val.items():
        print(p, {k: round(x, 2) for k, x in v.items()})
    for name, r in targets.items():
        print(name, {k: (round(float(np.mean(v)), 2) if isinstance(v, list) else round(v, 2)) for k, v in r.items()})


if __name__ == "__main__":
    main()
