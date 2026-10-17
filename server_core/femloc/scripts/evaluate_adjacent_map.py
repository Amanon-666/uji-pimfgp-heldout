"""用相邻楼层指纹图作空间参照，按 source-only 选择局部残差校正。"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.uji import load_floors


def rss_features(rows):
    x = rows.astype(np.float32).copy()
    x[x == 100] = -110
    return np.clip((x + 110) / 110, 0, 1)


def make_map(floor):
    """同一位置的全部 source 扫描形成一个平均指纹。"""
    positions, inverse = np.unique(floor["xy"], axis=0, return_inverse=True)
    features = np.zeros((len(positions), 520), dtype=np.float32)
    np.add.at(features, inverse, rss_features(floor["rssi"]))
    features /= np.bincount(inverse)[:, None]
    return positions, features


def source_predictions(floor, source_map):
    """每条扫描只匹配 source 图中的最近指纹；不接收目标标签。"""
    positions, prototypes = source_map
    x = rss_features(floor["rssi"])
    p_norm = (prototypes * prototypes).sum(axis=1)[None, :]
    predictions = []
    for block in np.array_split(x, max(1, (len(x) + 511) // 512)):
        squared = (block * block).sum(axis=1)[:, None] + p_norm - 2 * block @ prototypes.T
        predictions.append(positions[np.argmin(squared, axis=1)])
    return np.concatenate(predictions)


def observations_by_position(floor):
    """重复 RSSI/坐标只计作一个可抽取观测。"""
    positions = defaultdict(dict)
    for i, (xy, group) in enumerate(zip(floor["xy"], floor["groups"])):
        positions[tuple(xy)].setdefault(group, i)
    return {p: np.array(list(groups.values()), dtype=np.int64)
            for p, groups in sorted(positions.items())}


def episode(floor, rng, count, scans):
    groups = observations_by_position(floor)
    eligible = [p for p, rows in groups.items() if len(rows) >= scans]
    chosen = rng.choice(len(eligible), count, replace=False)
    support = np.concatenate([rng.choice(groups[eligible[j]], scans, replace=False) for j in chosen])
    chosen_positions = {eligible[j] for j in chosen}
    query = np.concatenate([rows for p, rows in groups.items() if p not in chosen_positions])
    return support, query


def correction(floor, source_xy, support, query, length_scale, noise):
    """将每个已知位置的三条误差合并，再在预测坐标空间做局部校正。"""
    anchor_xy, inverse = np.unique(floor["xy"][support], axis=0, return_inverse=True)
    anchor_source = np.zeros_like(anchor_xy)
    np.add.at(anchor_source, inverse, source_xy[support])
    anchor_source /= np.bincount(inverse)[:, None]
    residual = anchor_xy - anchor_source
    anchor_dist2 = ((anchor_source[:, None] - anchor_source[None, :]) ** 2).sum(axis=2)
    kernel = np.exp(-anchor_dist2 / (2 * length_scale ** 2))
    weights = np.linalg.solve(kernel + noise * np.eye(len(anchor_xy)), residual)
    query_dist2 = ((source_xy[query, None] - anchor_source[None, :]) ** 2).sum(axis=2)
    affinity = np.exp(-query_dist2 / (2 * length_scale ** 2))
    return source_xy[query] + affinity @ weights


def source_pairs(floors, targets):
    return [(lower, upper) for upper in sorted(floors) if upper not in targets
            for lower in [f"{upper[:2]}F{int(upper[-1]) - 1}"] if lower in floors]


def calibrate(floors, maps, config):
    """只在旧楼层转移任务上选两个校正尺度。"""
    candidates = [None] + [(ell, noise) for ell in config["length_scales_m"]
                           for noise in config["noise_ratios"]]
    rng = np.random.default_rng(config["source_episode_seed"])
    transitions = {}
    for lower, upper in source_pairs(floors, set(config["targets"])):
        floor = floors[upper]
        source_xy = source_predictions(floor, maps[lower])
        errors = {str(candidate): [] for candidate in candidates}
        for _ in range(config["source_episodes_per_pair"]):
            support, query = episode(floor, rng, config["support_positions"], config["scans_per_position"])
            truth = floor["xy"][query]
            for candidate in candidates:
                estimate = source_xy[query] if candidate is None else correction(floor, source_xy, support, query, *candidate)
                errors[str(candidate)].append(float(np.linalg.norm(estimate - truth, axis=1).mean()))
        transitions[f"{lower}->{upper}"] = {key: float(np.mean(value)) for key, value in errors.items()}
    scores = {str(candidate): float(np.mean([row[str(candidate)] for row in transitions.values()]))
              for candidate in candidates}
    best = min(candidates, key=lambda c: (scores[str(c)], c is not None,
                                           0 if c is None else -c[1], 0 if c is None else -c[0]))
    return best, scores, transitions


def evaluate(floors, maps, config, chosen):
    """直接读取 MetaLoc 分支固定的三组目标 row_id，不重新抽目标。"""
    rows = []
    for target in config["targets"]:
        lower = f"{target[:2]}F{int(target[-1]) - 1}"
        floor = floors[target]
        source_xy = source_predictions(floor, maps[lower])
        ids = {int(row): index for index, row in enumerate(floor["row_ids"])}
        for seed in config["seeds"]:
            path = Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json"
            manifest = json.loads(path.read_text())[target]
            support = np.array([ids[int(row)] for row in manifest["support"]])
            query = np.array([ids[int(row)] for row in manifest["unseen_position_query"]])
            assert len(support) == config["support_positions"] * config["scans_per_position"]
            assert len(np.unique(floor["xy"][support], axis=0)) == config["support_positions"]
            assert set(map(tuple, floor["xy"][support])).isdisjoint(map(tuple, floor["xy"][query]))
            raw = float(np.linalg.norm(source_xy[query] - floor["xy"][query], axis=1).mean())
            updated = raw if chosen is None else float(np.linalg.norm(
                correction(floor, source_xy, support, query, *chosen) - floor["xy"][query], axis=1).mean())
            source_aps = (floors[lower]["rssi"] != 100).any(axis=0)
            support_aps = (floor["rssi"][support] != 100).any(axis=0)
            rows.append({"target": target, "source": lower, "seed": seed,
                         "support_row_ids": manifest["support"], "query_row_ids": manifest["unseen_position_query"],
                         "source_map_mde_m": raw, "corrected_mde_m": updated,
                         "support_source_ap_jaccard": float((source_aps & support_aps).sum() / (source_aps | support_aps).sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adjacent_map.json")
    parser.add_argument("--output", default="outputs/adjacent_map", type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    floors = load_floors(config["data_path"])
    maps = {name: make_map(floor) for name, floor in floors.items() if name not in config["targets"]}
    chosen, scores, transitions = calibrate(floors, maps, config)
    results = evaluate(floors, maps, config, chosen)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "calibration.json").write_text(json.dumps({"chosen": chosen, "scores": scores,
                                                               "source_transitions": transitions}, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("source-selected", chosen, "scores", scores)
    for target in config["targets"]:
        rows = [r for r in results if r["target"] == target]
        print(target, "source", rows[0]["source"], "map", round(np.mean([r["source_map_mde_m"] for r in rows]), 3),
              "corrected", round(np.mean([r["corrected_mde_m"] for r in rows]), 3),
              "three_seed_values", [round(r["corrected_mde_m"], 3) for r in rows])


if __name__ == "__main__":
    main()
