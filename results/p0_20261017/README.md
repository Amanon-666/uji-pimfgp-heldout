# P0 MapSep 完整运行（2026-10-17，lab-server）

- 运行目录：`/home/panyushuo/projects/panyushuo/UJI-P0-MapSep-20261017/run/`（原始 perq/null 未入库）
- 代码：mapsep.py（commit f31284f），core_code_version `femloc@ab1a60b17c7b85542755937c68f89d63ce26255c|files-sha256:4017a40c509ed209`（三个建筑一致）
- NPERM=999（post），NPERM_PRE=999；3 建筑 × K∈{3,5,10,20,30,full} × 10 draws × {internal, official}
- G3 == 原版 law_other 断言：三个建筑 max|Δ| = 0.0
- `select_b*.json`：源楼栋选参 + manifest
- `tables/D1_kde_pull_diag.csv`：事后诊断（diag_kde.py；冻结选参，重算 G1/G2 及 G1_KDE、G2_noKDE 分解；G1/G2 与主运行逐行一致，max diff 0）

## CHECKS.txt = INVALID —— 原因（未修改 summarize.py，未重跑）
1. `manifest hash 不一致`：hash 覆盖整个 manifest，其中 law_field / law_match / anchor_match 来自各建筑自己的 results/building{b}.json（源楼栋选出，按建筑本应不同）。
   方法列表、nperm、nperm_pre、schema、阈值、core_code_version 三建筑完全相同。
2. `null/perq 不对应`（4 组，共 22 条 query）：perq 的 d_min 四舍五入到 0.01 m 写出，summarize 用舍入值判 far（≥20）；null 用未舍入值。
   d_min ∈ [19.995, 20) 的 query 两边判定不同（如 B1F2 draw7 K10 internal：169 vs 188）。
3. （非 CHECK 项）T5 中有效域为空的行多一个尾随空字段（16 列 vs 表头 15 列）。
因此 T4 的 pass_* 列显示 INVALID；Δ_wireless 与 CI 照常计算。
