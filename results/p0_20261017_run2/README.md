# P0 MapSep 重跑 run2（2026-10-17，lab-server）

- 运行目录：`/home/panyushuo/projects/panyushuo/UJI-P0-MapSep-20261017/run2/`（`run/` 未改动；原始 perq/null 未入库）
- 代码：`code_v2/`（= 本仓库根目录 mapsep.py / summarize.py / compare_runs.py）；core_code_version
  `femloc@ab1a60b17c7b85542755937c68f89d63ce26255c|files-sha256:89e1a5246e00ec15`，protocol_hash `f3fb01175d86`（三建筑一致）
- 设置与 run/ 相同：同 seeds（episode rng = SEED+building，SEED=1000；源选参 rng = 7）、NPERM=999、NPERM_PRE=999、每进程 3 线程
- G3 == 原版 law_other 断言：三建筑 max|Δ| = 0.0；CHECKS.txt = **VALID**

## 相对 run/ 的修改（在看到 run/ 的 INVALID 结果之后做出；均为校验/记录层修改）
判定规则**未改**：远距查询 Δ_wireless = E(G1) − E(G3) 的位置重抽样 95% CI 下界，
pass_stat: lo>0，pass_practical: lo>1 m，pass_strong: lo>2 m；跨 K 等权、建筑等权汇总方式不变；FAR=20 m 不变。
1. **manifest 一致性**：run/ 的 hash 覆盖整个 manifest，包含按建筑由源楼栋选出、本应不同的 law_field/law_match/anchor_match，导致 INVALID。
   现拆分为 `protocol` 字段（methods/nulls/shifts、nperm、nperm_pre、seed、seed_scheme、FAR、smoke、HS、GP/匹配网格、e_src、qcap、平移量、
   PRIOR_MAP_SCHEMA、ORIG_AP_THRESH(+confirmed)、core_code_version、source_sha256、mapsep_sha256）→ `protocol_hash`，跨建筑逐字段比较并重算 hash 校验；
   `per_building`（law_field、law_match、anchor_match、sel）只记录不比较。
2. **far 标记**：run/ 的 perq 写的是舍入到 0.01 m 的 d_min，summarize 用舍入值重算 far，而 null 用未舍入值 → 22 条 d_min∈[19.995,20) 的 query 两边判定不同。
   现 perq 写未舍入 d_min（repr）和显式 `is_far`（与 null 的 far 同一个变量）；summarize 只用文件中的 is_far，缺列则拒绝运行；
   n_far 配对检查保持严格，并新增 is_far 与未舍入 d_min≥FAR 的一致性检查。
3. **T5**：有效域为空的行多一个空字段（16 列 vs 表头 15 列）→ 修正，并断言每行 15 列。
修改先在 2 建筑 smoke 上检查：VALID；反向测试——只改 per_building law_match → 仍 VALID；改 protocol 字段 nperm_pre → INVALID；
翻转一个 is_far → INVALID（一致性 + n_far 配对两条都报）。

## run/ 与 run2/ 的一致性（compare_run_vs_run2.txt）
- perq 全部 75 个比较列（各方法误差、SH 列等）逐字符相同；d_min 舍入后相同；null 全部数组 max|diff| = 0；sel 相同。
- 唯一差异是 22 条边界 query 的 far 归属（b0: 3，b1: 19），进而影响少数 T4/T3 远距格子（Δ_wireless max|diff| = 0.068 m）。
- summarize 的位置 bootstrap 用单一全局 rng 流（default_rng(0)）；n_far_locations 变化（6 个格子各 −1）会使其后所有 bootstrap 抽样偏移，
  因此很多格子的 CI 端点有小差异（ci2.5 max|diff| 0.563 m，ci97.5 max|diff| 0.794 m），点估计不变。
