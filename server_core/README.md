# server_core — 服务器真实核心模块快照（只读副本）

来源（lab-server，2026-10-17 复制，未修改）：
- `heldout/` ← `/home/panyushuo/projects/panyushuo/UJI-PIMFGP-Heldout-20261017/{pimfgp_heldout.py, diag_round2.py}`（该目录不是 git 仓库）
- `femloc/` ← `/home/panyushuo/projects/panyushuo/FeMLoc-Signal-Calibrated-Map`（git worktree，HEAD `ab1a60b17c7b85542755937c68f89d63ce26255c`，clean）
  `data/uji.py`, `models/{cross_floor,radio_map}.py`, `scripts/{evaluate_scm_tobit,evaluate_signal_calibrated_map,evaluate_adjacent_map}.py`

不含：数据（UJIIndoorLoc CSV）、`splits.json`、results、external-private、任何凭据。

mapsep.py 依赖的已确认事实：
- `tobit_match`（evaluate_scm_tobit.py）对 protos 全部 520 列打分，无 AP 筛选 → `ORIG_AP_THRESH=none`。
- `prior_map`（cross_floor.py）返回 tuple `(x[N,520], lc["xy"][N,2])` → `PRIOR_MAP_SCHEMA={"permute":[0],"fixed":[1]}`。
- `build_map('scm', ...)` 返回 `vstack([校准后旧层格点, K 锚点原型])`、`vstack([旧层坐标, 锚点坐标])`：
  候选集 C = 旧层格点 ∪ 锚点坐标。G4/SH 只作用于前 N 行（历史分区），锚点行固定；G0/G1/G2/G3b 的锚点行也取锚点原型。
- `load_floors` 的 `row_ids` 为 1 起的 CSV 数据行号（enumerate(DictReader, 1)）。
