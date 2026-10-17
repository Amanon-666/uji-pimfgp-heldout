# uji-pimfgp-heldout

UJIIndoorLoc 少标定点（few-shot）跨楼栋 WiFi 指纹定位的研究代码（公开部分）。

本仓库用于向协作者/模型展示方法实现以便讨论。**注意：主实验的核心实现（前向信号场、路损、GP 残差、删失似然、law_other 地图迁移等）位于实验服务器上的模块中，尚未并入本仓库（取回后补充）。** 当前仓库主要包含"P0 地图迁移收益分离实验"的实验编排代码，它通过 import 复用服务器模块。

## 研究问题

跨建筑 / 跨楼层、仅有少量目标标定点（K 个）的 WiFi 指纹定位。核心争论：历史地图迁移在远离标定点处的优势，到底来自
(a) 历史 RSSI 信号知识本身，(b) 目标空间几何覆盖，还是 (c) 候选位置/先验限制。P0 实验用严格分离 + 破坏性负对照来区分这三者。

## 文件

- `mapsep.py` —— P0 分离实验主脚本。定义 G0/G1/G1orc/G2/G3/G3b/G4(R_π)/SH 各配置，复用服务器上的协议与打分函数，输出各方法后验均值定位结果。
- `summarize.py` —— 汇总脚本，生成 T0–T7 各表（位置 ID 碰撞、按 建筑×K×split 的 MDE、按到最近标定点距离分桶、配对差 + 位置聚类 bootstrap、预注册判据（按 K 与跨 K 等权）、各平移同支持域对照、混杂排查（n_det_all / n_det_used）、几何覆盖）及输入一致性校验 CHECKS.txt。
- `mock/` —— 仅为在本地跑通 `mapsep.py` 而写的**接口桩**（stub），用于展示所依赖的服务器模块的函数签名（`fit_law`、`prior_map`、`build_map`、`tobit_match`、`evaluate_scm_tobit`、`evaluate_signal_calibrated_map` 等）。**mock 产生的数字没有科学意义**，真实结果需在服务器上用真实模块运行。

## 实验配置（P0）

- **G0**：固定 RBF 核（负平方距离 softmax 归一化，断言行和=1）把 K 个锚点原型扩散到"支持点包围盒+15 m"网格，不用几何；另报 anchors / wknn。
- **G1**：同样核扩散，候选集 = 合法几何 C（同楼下层历史地图格点位置），与 law_other 同一（均匀）位置先验，不看任何历史 RSS。
  自检：build_map 之前只打乱历史指纹值（整行置换 + 逐元素打乱），断言 build_map 输出的 C 坐标完全不变；否则 G1 不是纯几何基线，停止并报告。
- **G1orc**：候选换成目标楼层自身训练参考点（**借用了测试集覆盖，仅 oracle 对照**）。
- **G2**：C 上加 KDE 先验 + 常数均值 GP 残差 + tobit 删失。
- **G3**：原版 law_other（AP 集 = 地图全部 AP）。每个 episode×split **硬断言** `np.allclose(tobit_match(qx,*cand,*law_match), G3, atol=rtol=1e-6)`，失败时打印 AP 掩码差集，不放宽容差。
- **G3_restrictedAP**：同 G3，但 AP 集限制为 K 锚点检出过的 AP（与 G1/G2 信息集一致），与 G3 分开报，不互相替代。
- **G3b**：G2 框架，GP 均值 = 校准后历史地图。
- **G4_post / G4b_post**：对**已用 K 锚点 build_map/GP 校准完的最终场**按整行指纹在同楼层 C 内置换（实现为打分矩阵列重排，位置与先验不动），默认 999 次。测的是"最终场的空间对应是否重要"。
- **G4_pre / G4b_pre**：**输入级**负对照。在 build_map('scm') 之前把历史指纹按整行（行 = 地图位置）在 C 内置换 T_π(H)，再用同一组 K 个目标锚点重新校准 F(T_π(H), S_K) 后定位；G4b_pre 每次重做 GP（`mean_s=lookup(axy,C,hist_π,3)`，固定 GP 核/正则/匹配参数，不重新寻优）。默认 199 次（`NPERM_PRE`，每次需重跑 build_map）。隔离的是"历史无线信息在 K-shot 校准之前的独立贡献"。线性对齐 law 只在源楼栋上拟合，置换在其后（固定对齐再破坏 RSS）。
- **SH**：整图共同空间平移（10/20/40 m × ±x/±y，cKDTree 最近邻重采样）。每个平移单独保存有效域掩码 ok_t（重采样距离 ≤ 2×中位格距），SH_t、G1_t、G3_t 都在同一 ok_t 候选域上重算后验（ok=False 的候选 score=−∞），分别报 SH_t−G3_t 及有效 query 数、覆盖比例、平均最近锚点距离。

## 判据（预注册）

令 Δ_wireless = E(G1) − E(G3)（远距查询 d_min ≥ 20 m 的平均定位误差，G3 更小为正）。
**主判据 PASS = 按物理位置聚类 bootstrap 的 Δ_wireless 95% 下界 > 0**（`passed = bool(lo > 0)`）。
- 口径：位置加权（先在稳定物理位置内平均再对位置等权、对位置重抽）。位置 ID：internal 用 (楼栋, 楼层, SPACEID, RELATIVEPOSITION)；official 的 SPACEID 全为 0，用原始坐标全精度字符串；T0 记录与 0.1 m 四舍五入坐标的碰撞率。
- T4 按 `split × 目标建筑 × K` 输出，并给出预先规定的跨 K 汇总：**各 K 等权**（不按扫描数加权）。
- 主报告单位 = 目标建筑；三栋合并行（建筑等权、建筑内分层重抽）只是三栋的平均，**不是"任意新建筑"的泛化区间**。
- 置换零分布（G4_post、G4_pre）的 `Δ > Q95[E(G1)−E(R_π)]` 只作 `explore_*` 探索性诊断列，**不进入 PASS**；置换比例不是严格 p 值（空间自相关破坏可交换性）。
- summarize.py 强制校验 perq 与 null 的 (building, floor, draw, K, split, n_all, n_far) 完全对应，且 manifest（方法列表、种子、置换次数、参数网格）一致；否则 PASS 写 INVALID（见 tables/CHECKS.txt）。

## 需在服务器真实模块上才能最终验证的项

- G3 == 原版 law_other 硬断言（mock 的 tobit_match AP 筛选已对齐为与 G3 等价，仅为本地跑通；真实服务器上该断言必须真的成立）。
- G4_pre 的整行置换依赖 prior_map 返回结构（需含与 C 一致的坐标数组、指纹首维 = 位置数），以及 build_map 对置换输入重新校准的真实行为。
- C 坐标对历史 RSS 不敏感的自检；row_ids 与 CSV 行号对应（取 PHONEID / 位置 ID）的断言。

## 留出协议铁律（不可违反）

- leave-one-building-out：三栋楼各留出一栋做 target，其余做 source。
- 所有超参数、地图线性对齐参数只在 **源楼栋楼层对** 上选；目标楼的 query 标签 **只用于最终打分**，绝不参与任何选择、对齐或调参。
- K ∈ {3,5,10,20,30,full}，每 K 多次抽样；内部 query split 与官方 query split 都报，正式跨楼结论以官方 split + 合法几何 C 为准。
