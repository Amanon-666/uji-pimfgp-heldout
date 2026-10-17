"""汇总 mapsep.py 输出 → tables/*.csv（numpy-only）。用法：python summarize.py <run_dir>
权重口径（表中 weighting 列标明）：
  loc  = 先在稳定物理位置 loc_id 内平均（跨 draw/扫描），再对位置等权平均；bootstrap 对位置重抽（主口径：“随机抽一个物理位置”的误差）
  scan = Σ误差/Σ条数（扫描/查询行加权）；仅作参考
主报告单位 = 目标建筑；三栋合并（target_building=all，建筑等权、建筑内分层重抽）不是“任意新建筑”的泛化区间。
预注册主判据（冻结）：远距查询（d_min≥20 m）上 Δ_wireless = E(G1) − E(G3)，lo = 位置聚类 bootstrap 双侧 95% 百分位区间的 2.5% 下界：
  pass_stat      = lo > 0
  pass_practical = lo > DELTA_MIN = 1.0 m   （主线闸门）
  pass_strong    = lo > 2.0 m
  不使用 TOST（检验的是最小效应，不是等价性）。robust_* 列为非约束性稳健性诊断。
  跨 K 汇总权重预先规定为各 K 等权（K∈{3,5,10,20,30,full} 中有远距查询者），不按扫描数加权。
  置换零分布（G4_post / G4_pre）只作探索性诊断列（explore_*），不进入 PASS；置换比例不是严格 p 值。
输入一致性校验失败（perq 与 null 分组/n_far 不对应、manifest 不一致）→ PASS 列写 INVALID。"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

RUN = Path(sys.argv[1]); TAB = RUN / 'tables'; TAB.mkdir(exist_ok=True)
EDGES = np.array([0, 5, 10, 20, 40, np.inf]); BL = ['0-5', '5-10', '10-20', '20-40', '40+']
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
DELTA_MIN = 1.0   # 预注册主线闸门（m）
DELTA_STRONG = 2.0
KORDER = ['3', '5', '10', '20', '30', 'full']
rng = np.random.default_rng(0)

# ---------- 读入 ----------
rows, hdr = [], None
for f in sorted(RUN.glob('perq_b*.csv.gz')):
    with gzip.open(f, 'rt') as fh:
        r = csv.reader(fh); h = next(r)
        assert hdr is None or h == hdr, f'{f}: 列头与其他建筑不一致'
        hdr = h; rows += list(r)
cols = {c: i for i, c in enumerate(hdr)}
col = lambda c: np.array([x[cols[c]] for x in rows])
sel = {int(f.stem.split('_b')[1]): json.loads(f.read_text()) for f in sorted(RUN.glob('select_b*.json'))}
MAN = {b: v['manifest'] for b, v in sel.items()}
FAR = float(next(iter(MAN.values()))['far'])
METH = next(iter(MAN.values()))['methods']; SHN = next(iter(MAN.values()))['shifts']
NULLS = next(iter(MAN.values()))['nulls']

B = col('building').astype(int); FL = col('floor'); DR = col('draw').astype(int); K = col('K'); SP = col('split')
LID = col('loc_id'); DM = col('d_min').astype(float); NDA = col('n_det_all').astype(int); NDU = col('n_det_used').astype(int)
PH = col('phone').astype(int); DC = col('d_C').astype(float); DO = col('d_orc').astype(float)
XY = np.column_stack([col('x').astype(float), col('y').astype(float)])
E = {m: col(m).astype(float) for m in METH}
SHV = {s: col(f'{s}:valid').astype(int).astype(bool) for s in SHN}
SHE = {s: {x: col(f'{s}:{x}').astype(float) for x in ('SH', 'G1', 'G3')} for s in SHN}
BK = np.clip(np.searchsorted(EDGES, DM, side='right') - 1, 0, 4)
FARM = DM >= FAR
KS = [k for k in KORDER if (K == k).any()]
SPL = [s for s in ['internal', 'official'] if (SP == s).any()]
BUILD = sorted(np.unique(B).tolist())
ORDER = METH


def wcsv(name, header, data):
    with open(TAB / name, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(data)


# ---------- 一致性校验 ----------
problems = []
hashes = {b: m['hash'] for b, m in MAN.items()}
if len(set(hashes.values())) != 1:
    problems.append(f'manifest hash 不一致 {hashes}')
vers = {b: m.get('core_code_version') for b, m in MAN.items()}
if None in vers.values() or len(set(vers.values())) != 1:
    problems.append(f'core_code_version 缺失或跨建筑不一致 {vers}')
for b, m in MAN.items():
    if m['methods'] != METH or m['shifts'] != SHN or m['nulls'] != NULLS:
        problems.append(f'b{b} 方法列表不一致')
grp = defaultdict(lambda: [0, 0])
for i in range(len(rows)):
    g = grp[(int(B[i]), FL[i], int(DR[i]), K[i], SP[i])]; g[0] += 1; g[1] += int(FARM[i])
nulls, seen = {}, set()
for f in sorted(RUN.glob('null_b*.npz')):
    b = int(f.stem.split('_b')[1])
    z = np.load(f, allow_pickle=False)
    meta = [json.loads(x) for x in z['meta']]
    arr = {k: z[k] for k in z.files if k != 'meta'}
    for i, mt in enumerate(meta):
        key = (int(mt['building']), mt['floor'], int(mt['draw']), str(mt['K']), mt['split'])
        seen.add(key)
        if key not in grp or grp[key] != [mt['n_all'], mt['n_far']]:
            problems.append(f'null/perq 不对应 {key}: null={mt["n_all"], mt["n_far"]} perq={grp.get(key)}')
    for nm in NULLS:
        n_exp = MAN[b]['nperm'] if 'post' in nm else MAN[b]['nperm_pre']
        if arr[f'sum_far_{nm}'].shape[1] != n_exp:
            problems.append(f'b{b} {nm} 置换次数 {arr[f"sum_far_{nm}"].shape[1]} != manifest {n_exp}')
    nulls[b] = (meta, arr)
if set(grp) != seen:
    problems.append(f'perq 有 {len(set(grp) - seen)} 组在 null 中缺失, null 有 {len(seen - set(grp))} 组 perq 中缺失')
if set(nulls) != set(BUILD) or set(MAN) != set(BUILD):
    problems.append(f'建筑集合不一致 perq={BUILD} null={sorted(nulls)} select={sorted(MAN)}')
VALID = not problems
(TAB / 'CHECKS.txt').write_text('VALID\n' if VALID else 'INVALID\n' + '\n'.join(problems) + '\n')

# ---------- 统计工具 ----------
def loc_sums(mask, v):
    u, inv = np.unique(LID[mask], return_inverse=True)
    return u, np.bincount(inv, v[mask]), np.bincount(inv).astype(float)


def boot(mask, v, nb=NB, weighting='loc'):
    """返回 (估计, ci2.5, ci97.5, n_loc, n_rows)。"""
    if not mask.any():
        return (np.nan,) * 3 + (0, 0)
    u, s, n = loc_sums(mask, v)
    idx = rng.integers(0, len(u), (nb, len(u)))
    if weighting == 'loc':
        m = s / n; est = m.mean(); bs = m[idx].mean(1)
    else:
        est = s.sum() / n.sum(); bs = s[idx].sum(1) / n[idx].sum(1)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(u), int(mask.sum())


def crossK_boot(base_mask, v, nb=NB):
    """跨 K 等权汇总（预先规定）：各 K 内位置等权均值，再对 K 等权平均；bootstrap 对位置集合联合重抽。"""
    ks = [k for k in KS if (base_mask & (K == k)).any()]
    if not ks:
        return (np.nan,) * 3 + (0, ks)
    allu = np.unique(LID[base_mask]); pos = {u: i for i, u in enumerate(allu)}
    M = np.full((len(ks), len(allu)), np.nan)
    for a, k in enumerate(ks):
        u, s, n = loc_sums(base_mask & (K == k), v)
        M[a, [pos[x] for x in u]] = s / n
    est = np.mean([np.nanmean(r) for r in M])
    idx = rng.integers(0, len(allu), (nb, len(allu)))
    bs = np.nanmean(np.stack([np.nanmean(M[a][idx], 1) for a in range(len(ks))]), 0)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(allu), ks


def bld_eq_boot(mfun, v, nb=NB, crossk=False):
    """三栋合并：建筑等权，建筑内各自位置重抽（分层）；只是三栋的平均，不是新建筑泛化区间。"""
    ests, bss = [], []
    for b in BUILD:
        m = mfun(b)
        if not m.any():
            continue
        if crossk:
            ks = [k for k in KS if (m & (K == k)).any()]
            allu = np.unique(LID[m]); pos = {u: i for i, u in enumerate(allu)}
            M = np.full((len(ks), len(allu)), np.nan)
            for a, k in enumerate(ks):
                u, s, n = loc_sums(m & (K == k), v)
                M[a, [pos[x] for x in u]] = s / n
            ests.append(np.mean([np.nanmean(r) for r in M]))
            idx = rng.integers(0, len(allu), (nb, len(allu)))
            bss.append(np.nanmean(np.stack([np.nanmean(M[a][idx], 1) for a in range(len(ks))]), 0))
        else:
            u, s, n = loc_sums(m, v); mm = s / n
            ests.append(mm.mean()); bss.append(mm[rng.integers(0, len(u), (nb, len(u)))].mean(1))
    if not ests:
        return np.nan, np.nan, np.nan
    bs = np.mean(bss, 0)
    return float(np.mean(ests)), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ep_mean(mask, m):
    key = np.char.add(FL[mask], DR[mask].astype(str))
    u, inv = np.unique(key, return_inverse=True)
    return float(np.mean(np.bincount(inv, E[m][mask]) / np.bincount(inv))) if mask.any() else np.nan


R = lambda x: round(float(x), 3) if x == x else ''

# ---------- T0 位置 ID 碰撞 ----------
t0 = []
for s in SPL:
    m = SP == s
    lid = LID[m]; rk = np.char.add(np.char.add(FL[m], ':'), np.array([f'{a:.1f},{b:.1f}' for a, b in XY[m]]))
    pairs = set(zip(lid.tolist(), rk.tolist()))
    by_r = defaultdict(set); by_l = defaultdict(set)
    for l_, r_ in pairs:
        by_r[r_].add(l_); by_l[l_].add(r_)
    t0.append([s, len(by_l), len(by_r), R(np.mean([len(v) > 1 for v in by_r.values()])),
               R(np.mean([len(v) > 1 for v in by_l.values()]))])
wcsv('T0_locid_collisions.csv', ['split', 'n_loc_id', 'n_round0.1m_coord', 'frac_coord_with_multi_locid',
                                 'frac_locid_with_multi_coord'], t0)

# ---------- T1 主表（episode 均值口径，同 §10/§11） ----------
t1 = []
for b in BUILD + ['all']:
    for s in SPL:
        for k in KS:
            m = (SP == s) & (K == k) & ((B == b) if b != 'all' else True)
            t1.append([b, s, k, 'episode'] + [R(ep_mean(m, me)) for me in ORDER])
wcsv('T1_mde.csv', ['target_building', 'split', 'K', 'weighting'] + ORDER, t1)

# ---------- T2 距离分桶 ----------
t2 = []
for b in BUILD + ['all']:
    for s in SPL:
        for k in KS + ['pooled']:
            for bi, bl in enumerate(BL):
                m = (SP == s) & (BK == bi) & ((K == k) if k != 'pooled' else True) & ((B == b) if b != 'all' else True)
                if m.any():
                    t2.append([b, s, k, bl, int(m.sum()), R(DM[m].mean()), 'scan'] + [R(E[me][m].mean()) for me in ORDER])
wcsv('T2_mde_by_dmin.csv', ['target_building', 'split', 'K', 'dmin_bucket', 'n', 'mean_dmin', 'weighting'] + ORDER, t2)

# ---------- T3 配对差 ----------
PAIRS = [('G3', 'G4_post_mean'), ('G3', 'G4_pre_mean'), ('G3', 'G2'), ('G2', 'G1'), ('G1', 'G0'), ('G3', 'G1'),
         ('G3_restrictedAP', 'G1'), ('G3', 'G3_restrictedAP'), ('G3b', 'G2'), ('G3b', 'G4b_post_mean'),
         ('G3b', 'G4b_pre_mean'), ('G1', 'anchors'), ('G0', 'anchors'), ('G1orc', 'G1')]
t3 = []
for b in BUILD:
    for s in SPL:
        for k in KS + ['pooled']:
            for sub in ['all', 'far'] + BL:
                m = (SP == s) & (B == b) & ((K == k) if k != 'pooled' else True)
                m &= FARM if sub == 'far' else (True if sub == 'all' else (BK == BL.index(sub)))
                if not m.any():
                    continue
                for a, c in PAIRS:
                    d = E[a] - E[c]
                    est, lo, hi, nl, nr = boot(m, d, 500)
                    t3.append([b, s, k, sub, f'{a}-{c}', nr, nl, 'loc', R(est), R(lo), R(hi), R(d[m].mean())])
wcsv('T3_paired_diff.csv', ['target_building', 'split', 'K', 'subset', 'pair', 'n_query', 'n_locations', 'weighting',
                            'mean_diff', 'ci2.5', 'ci97.5', 'scan_weighted_diff'], t3)

# ---------- T4 预注册判据 ----------
def null_far(b, s, k, nm):
    meta, arr = nulls[b]
    pick = np.array([mt['split'] == s and (k == 'crossK' or str(mt['K']) == k) for mt in meta])
    return arr[f'sum_far_{nm}'][pick].sum(0), sum(mt['n_far'] for mt, p in zip(meta, pick) if p)


t4 = []
TESTS = (('G3_vs_G1', 'G1', 'G3', ('G4_post', 'G4_pre')), ('G3b_vs_G2', 'G2', 'G3b', ('G4b_post', 'G4b_pre')),
         ('G3restrictedAP_vs_G1', 'G1', 'G3_restrictedAP', ()))
KTREND = [k for k in ('5', '10', '20') if k in KS]


def gate(lo, thr):
    if lo != lo:
        return 'NA'
    return 'INVALID' if not VALID else bool(lo > thr)


def sgn(x):
    return '' if x != x else ('+' if x > 0 else ('-' if x < 0 else '0'))


est_cache = {}
for s_ in SPL:  # 先算每个 建筑×K 的点估计，供稳健性列使用
    for b in BUILD:
        for k in KS:
            for tname, ref, tgt, _ in TESTS:
                m = (SP == s_) & (B == b) & FARM & (K == k)
                if m.any():
                    u, ss, nn = loc_sums(m, E[ref] - E[tgt]); est_cache[(s_, b, k, tname)] = float((ss / nn).mean())
for s in SPL:
    for b in BUILD + ['all']:
        for k in KS + ['crossK']:
            for tname, ref, tgt, nls in TESTS:
                d = E[ref] - E[tgt]
                if b == 'all':
                    mf = lambda bb: (SP == s) & (B == bb) & FARM & ((K == k) if k != 'crossK' else True)
                    est, lo, hi = bld_eq_boot(mf, d, crossk=(k == 'crossK'))
                    nl = int(sum(len(np.unique(LID[mf(bb)])) for bb in BUILD)); nr = int(sum(mf(bb).sum() for bb in BUILD))
                    wt = 'loc;building-equal' + (';K-equal' if k == 'crossK' else '')
                else:
                    m = (SP == s) & (B == b) & FARM & ((K == k) if k != 'crossK' else True)
                    if k == 'crossK':
                        est, lo, hi, nl, _ = crossK_boot(m, d); nr = int(m.sum()); wt = 'loc;K-equal'
                    else:
                        est, lo, hi, nl, nr = boot(m, d); wt = 'loc'
                if nr == 0:
                    continue
                # 稳健性（非约束）：分建筑符号/大小；K=5/10/20 趋势一致性
                if b == 'all':
                    kk = [k] if k != 'crossK' else KS
                    per_b = []
                    for bb in BUILD:
                        v = [est_cache.get((s, bb, x, tname), np.nan) for x in kk]
                        per_b.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
                    rb_signs = '/'.join(f'B{bb}:{sgn(v)}' for bb, v in zip(BUILD, per_b))
                    rb_min = R(np.nanmin(per_b)) if np.isfinite(per_b).any() else ''
                    rb_npos = int(np.sum(np.array(per_b) > 0))
                else:
                    rb_signs, rb_min, rb_npos = f'B{b}:{sgn(est)}', R(est), int(est > 0)
                bset = BUILD if b == 'all' else [b]
                trend = []
                for x in KTREND:
                    v = [est_cache.get((s, bb, x, tname), np.nan) for bb in bset]
                    trend.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
                kt_signs = '/'.join(f'K{x}:{sgn(v)}' for x, v in zip(KTREND, trend))
                kt_consistent = bool(len(trend) > 0 and np.isfinite(trend).all() and (np.all(np.array(trend) > 0) or np.all(np.array(trend) < 0)))
                ex = []
                for nm in ('post', 'pre'):
                    nmm = [x for x in nls if x.endswith(nm)]
                    if not nmm or b == 'all' or k == 'crossK':
                        ex += ['', '', '']; continue
                    m = (SP == s) & (B == b) & FARM & (K == k)
                    sfar, nfar = null_far(b, s, k, nmm[0])
                    dnull = E[ref][m].mean() - sfar / nfar
                    dw_scan = E[ref][m].mean() - E[tgt][m].mean()
                    q95 = np.percentile(dnull, 95)
                    ex += [R(np.median(dnull)), R(q95), bool(dw_scan > q95)]
                t4.append([s, b, k, tname, ref, tgt, nr, nl, wt, R(est), R(lo), R(hi),
                           gate(lo, 0.0), gate(lo, DELTA_MIN), gate(lo, DELTA_STRONG),
                           rb_signs, rb_npos, rb_min, kt_signs, kt_consistent] + ex)
wcsv('T4_preregistered_criterion.csv',
     ['split', 'target_building', 'K', 'test', 'reference', 'target', 'n_far_query', 'n_far_locations', 'weighting',
      'Delta_wireless', 'ci2.5', 'ci97.5', 'pass_stat(lo>0)', f'pass_practical(lo>{DELTA_MIN})', f'pass_strong(lo>{DELTA_STRONG})',
      'robust_building_signs', 'robust_n_buildings_pos', 'robust_min_building_est', 'robust_K5_10_20_signs',
      'robust_K5_10_20_consistent',
      'explore_post_null_median', 'explore_post_null_Q95', 'explore_post_dw_scan_gt_Q95',
      'explore_pre_null_median', 'explore_pre_null_Q95', 'explore_pre_dw_scan_gt_Q95'], t4)

# ---------- T5 共同平移（各平移各自有效域） ----------
dom = defaultdict(list)
for f in RUN.glob('shift_domain_b*.csv'):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            dom[(int(f.stem.split('_b')[1]), r['shift'])].append(float(r['frac_ok']))
t5 = []
for b in BUILD:
    for s in SPL:
        for sub in ('all', 'far'):
            base = (SP == s) & (B == b) & (FARM if sub == 'far' else True)
            for sh in SHN:
                m = base & SHV[sh]
                if not m.any():
                    t5.append([b, s, sub, sh, 0, R(0), R(np.mean(dom[(b, sh)])) if dom[(b, sh)] else ''] + [''] * 9); continue
                d = SHE[sh]['SH'] - SHE[sh]['G3']
                est, lo, hi, nl, nr = boot(m, d, 500)
                d2 = SHE[sh]['G1'] - SHE[sh]['G3']
                e2, lo2, hi2, _, _ = boot(m, d2, 500)
                t5.append([b, s, sub, sh, nr, R(m.sum() / max(base.sum(), 1)), R(np.mean(dom[(b, sh)])) if dom[(b, sh)] else '',
                           R(DM[m].mean()), R(SHE[sh]['G1'][m].mean()), R(SHE[sh]['G3'][m].mean()), R(SHE[sh]['SH'][m].mean()),
                           R(est), R(lo), R(hi), R(e2)])
wcsv('T5_shift_control.csv', ['target_building', 'split', 'subset', 'shift', 'n_valid_query', 'frac_query_valid',
                              'mean_frac_cand_ok', 'mean_dmin_valid', 'G1_t', 'G3_t', 'SH_t', 'SH_t-G3_t(loc)', 'ci2.5',
                              'ci97.5', 'G1_t-G3_t(loc)'], t5)

# ---------- T6 混杂 ----------
t6 = []
def quart(x):
    return np.searchsorted(np.percentile(x, [25, 50, 75]), x, side='right')
for b in BUILD:
    for s in SPL:
        mm = (SP == s) & (B == b)
        for fac, arr in (('n_det_all_q', quart(NDA)), ('n_det_used_q', quart(NDU)), ('phone', PH), ('floor', FL),
                         ('far', FARM.astype(int))):
            for v in np.unique(arr[mm]):
                m = mm & (arr == v)
                if m.sum() < 20:
                    continue
                t6.append([b, s, fac, v, int(m.sum()), R(DM[m].mean()), R(NDA[m].mean()), R(NDU[m].mean()), 'scan'] +
                          [R((E[a] - E[c])[m].mean()) for a, c in PAIRS[:6]] + [R(E['G3'][m].mean())])
wcsv('T6_confounders.csv', ['target_building', 'split', 'factor', 'level', 'n', 'mean_dmin', 'mean_n_det_all',
                            'mean_n_det_used', 'weighting'] + [f'{a}-{c}' for a, c in PAIRS[:6]] + ['G3'], t6)

# ---------- T7 几何覆盖 ----------
t7 = []
for b in BUILD:
    for s in SPL:
        for bi, bl in enumerate(BL + ['all']):
            m = (SP == s) & (B == b) & ((BK == bi) if bl != 'all' else True)
            if m.any():
                t7.append([b, s, bl, int(m.sum()), R(DM[m].mean()), R(DC[m].mean()), R((DC[m] > 4).mean()), R(DO[m].mean())])
wcsv('T7_geometry_coverage.csv', ['target_building', 'split', 'dmin_bucket', 'n', 'mean_dmin', 'mean_dist_to_C',
                                  'frac_dist_to_C_gt4m', 'mean_dist_to_oracle'], t7)

print('VALID' if VALID else 'INVALID:\n' + '\n'.join(problems[:20]))
print('rows', len(rows))
for x in t4:
    if x[2] == 'crossK':
        print(x[:20])
