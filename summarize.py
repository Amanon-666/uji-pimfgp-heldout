"""汇总 mapsep.py 输出 → tables/*.csv + SUMMARY 数字块（numpy-only）。用法：python summarize.py <run_dir>"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

RUN = Path(sys.argv[1]); TAB = RUN / 'tables'; TAB.mkdir(exist_ok=True)
EDGES = np.array([0, 5, 10, 20, 40, np.inf]); BL = ['0-5', '5-10', '10-20', '20-40', '40+']
FAR = 20.0; NB = 2000
rows = []; hdr = None
for f in sorted(RUN.glob('perq_b*.csv.gz')):
    with gzip.open(f, 'rt') as fh:
        r = csv.reader(fh); h = next(r); hdr = hdr or h; rows += list(r)
cols = {c: i for i, c in enumerate(hdr)}
METH = hdr[cols['sh_valid'] + 1:]
col = lambda c: np.array([x[cols[c]] for x in rows])
B = col('building').astype(int); FL = col('floor'); DR = col('draw').astype(int); K = col('K'); SP = col('split')
LID = col('loc_id'); DM = col('d_min').astype(float); ND = col('n_det').astype(int); PH = col('phone').astype(int)
DC = col('d_C').astype(float); DO = col('d_orc').astype(float)
SV = col('sh_valid').astype(int).astype(bool)
E = {m: col(m).astype(float) for m in METH}
BK = np.clip(np.searchsorted(EDGES, DM, side='right') - 1, 0, 4)
KS = [k for k in ['3', '5', '10', '20', '30', 'full'] if (K == k).any()]
SPL = [s for s in ['internal', 'official'] if (SP == s).any()]
ORDER = ['anchors', 'wknn', 'G0', 'G1', 'G1orc', 'G2', 'G3', 'G3b', 'G4mean', 'G4bmean'] + [m for m in METH if m.startswith('SH')]


def ep_mean(mask, m):
    """先按 episode（楼层×draw）求 MDE，再平均（与 §10/§11 口径一致）。"""
    key = np.char.add(FL[mask], DR[mask].astype(str))
    u, inv = np.unique(key, return_inverse=True)
    return float(np.mean(np.bincount(inv, E[m][mask]) / np.bincount(inv))) if mask.any() else np.nan


def wcsv(name, header, data):
    with open(TAB / name, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(data)


# T1 主表
t1 = []
for s in SPL:
    for k in KS:
        m = (SP == s) & (K == k)
        t1.append([s, k] + [round(ep_mean(m, me), 3) for me in ORDER])
wcsv('T1_mde.csv', ['split', 'K'] + ORDER, t1)
# T1b 分建筑
t1b = []
for b in np.unique(B):
    for s in SPL:
        for k in KS:
            m = (SP == s) & (K == k) & (B == b)
            t1b.append([b, s, k] + [round(ep_mean(m, me), 3) for me in ORDER])
wcsv('T1b_mde_by_building.csv', ['target_building', 'split', 'K'] + ORDER, t1b)
# T2 距离分桶（query 级平均）
t2 = []
for s in SPL:
    for k in KS + ['pooled']:
        for bi, bl in enumerate(BL):
            m = (SP == s) & (BK == bi) & ((K == k) if k != 'pooled' else True)
            if m.any():
                t2.append([s, k, bl, int(m.sum()), round(float(DM[m].mean()), 2)] + [round(float(E[me][m].mean()), 3) for me in ORDER])
wcsv('T2_mde_by_dmin.csv', ['split', 'K', 'dmin_bucket', 'n', 'mean_dmin'] + ORDER, t2)


# 位置聚类 bootstrap
def cboot(mask, diff, rng, nb=NB):
    lid = LID[mask]; d = diff[mask]
    u, inv = np.unique(lid, return_inverse=True)
    s = np.bincount(inv, d); n = np.bincount(inv)
    est = s.sum() / n.sum()
    idx = rng.integers(0, len(u), (nb, len(u)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(u)


PAIRS = [('G3', 'G4mean'), ('G3', 'G2'), ('G2', 'G1'), ('G1', 'G0'), ('G3', 'G1'), ('G3b', 'G2'), ('G3b', 'G4bmean'),
         ('G1', 'anchors'), ('G0', 'anchors'), ('G1orc', 'G1')]
rng = np.random.default_rng(0)
t3 = []
for s in SPL:
    for k in KS + ['pooled']:
        for sub in ['all', 'far'] + BL:
            m = (SP == s) & ((K == k) if k != 'pooled' else True)
            if sub == 'far':
                m &= DM >= FAR
            elif sub != 'all':
                m &= BK == BL.index(sub)
            if not m.any():
                continue
            for a, b in PAIRS:
                est, lo, hi, nl = cboot(m, E[a] - E[b], rng, 500)
                t3.append([s, k, sub, f'{a}-{b}', int(m.sum()), nl, round(est, 3), round(lo, 3), round(hi, 3)])
wcsv('T3_paired_diff.csv', ['split', 'K', 'subset', 'pair', 'n_query', 'n_locations', 'mean_diff', 'ci2.5', 'ci97.5'], t3)

# T4 预注册判据：Δ_wireless = E_far(G1) − E_far(G3) vs Q95[E_far(G1) − E_far(R_π)]；按目标建筑与合并
nulls = {}
for f in sorted(RUN.glob('null_b*.npz')):
    z = np.load(f, allow_pickle=True)
    nulls[int(f.stem.split('_b')[1])] = ([json.loads(x) for x in z['meta']], {k: z[k] for k in z.files if k != 'meta'})
t4 = []
for s in SPL:
    for tb in sorted(nulls) + ['all']:
        for which, g3, g4 in (('G3 vs R_pi', 'G3', 'sum_far_G4'), ('G3b vs R_pi(b)', 'G3b', 'sum_far_G4b')):
            m = (SP == s) & (DM >= FAR) & ((B == tb) if tb != 'all' else True)
            if not m.any():
                continue
            nfar = m.sum()
            eg1, eg3 = E['G1'][m].mean(), E[g3][m].mean()
            ref = E['G1'] if g3 == 'G3' else E['G2']
            eref = ref[m].mean()
            null_sum = 0
            for b2, (meta, arr) in nulls.items():
                if tb != 'all' and b2 != tb:
                    continue
                sel = np.array([x['split'] == s for x in meta])
                null_sum = null_sum + arr[g4][sel].sum(0)
            er = null_sum / nfar
            dnull = eref - er
            dw = eref - eg3
            q95 = float(np.percentile(dnull, 95))
            est, lo, hi, nl = cboot(m, ref - E[g3], rng)          # Δ_wireless 的位置聚类 bootstrap
            e2, lo2, hi2, _ = cboot(m, E['G4mean' if g3 == 'G3' else 'G4bmean'] - E[g3], rng)
            p_emp = float((np.sum(dnull >= dw) + 1) / (len(dnull) + 1))
            passed = bool(dw > q95 and lo > 0)
            t4.append([s, tb, which, 'G1' if g3 == 'G3' else 'G2', int(nfar), nl, round(float(eref), 3), round(float(eg3), 3),
                       round(float(er.mean()), 3), round(float(dw), 3), round(float(np.median(dnull)), 3), round(q95, 3),
                       round(float(lo), 3), round(float(hi), 3), round(float(e2), 3), round(float(lo2), 3), round(float(hi2), 3),
                       round(p_emp, 4), passed])
wcsv('T4_preregistered_criterion.csv',
     ['split', 'target_building', 'test', 'reference', 'n_far_query', 'n_far_locations', 'E_ref', 'E_G3', 'E_Rpi_mean',
      'Delta_wireless', 'null_median', 'null_Q95', 'Dw_locboot_ci2.5', 'Dw_locboot_ci97.5', 'Rpi_minus_G3', 'RmG3_ci2.5',
      'RmG3_ci97.5', 'perm_frac_ref', 'PASS'], t4)
# T5 共同平移（有效重叠区）
t5 = []
for s in SPL:
    for sub in ('all', 'far'):
        m = (SP == s) & SV & ((DM >= FAR) if sub == 'far' else True)
        if m.any():
            t5.append([s, sub, int(m.sum()), round(float(m.sum() / ((SP == s) & ((DM >= FAR) if sub == 'far' else True)).sum()), 3)] +
                      [round(float(E[me][m].mean()), 3) for me in ['G1', 'G3', 'G4mean'] + [x for x in METH if x.startswith('SH')]])
wcsv('T5_shift_control.csv', ['split', 'subset', 'n', 'frac_valid', 'G1', 'G3', 'G4mean'] + [x for x in METH if x.startswith('SH')], t5)
# T7 几何覆盖：query 到合法候选集 C / oracle 候选集的距离
t7 = []
for s in SPL:
    for bi, bl in enumerate(BL + ['all']):
        m = (SP == s) & ((BK == bi) if bl != 'all' else True)
        if m.any():
            t7.append([s, bl, int(m.sum()), round(float(DM[m].mean()), 2), round(float(DC[m].mean()), 2), round(float((DC[m] > 4).mean()), 3),
                       round(float(DO[m].mean()), 2)])
wcsv('T7_geometry_coverage.csv', ['split', 'dmin_bucket', 'n', 'mean_dmin', 'mean_dist_to_C', 'frac_dist_to_C_gt4m', 'mean_dist_to_oracle'], t7)
# T6 混杂：按 n_det 分位、phone、floor 的 G3−G4mean / G3−G1
t6 = []
q = np.percentile(ND, [25, 50, 75])
NDB = np.searchsorted(q, ND, side='right')
for s in SPL:
    for fac, arr, labs in (('n_det_quartile', NDB, None), ('phone', PH, None), ('floor', FL, None), ('far', (DM >= FAR).astype(int), None)):
        for v in np.unique(arr):
            m = (SP == s) & (arr == v)
            if m.sum() < 20:
                continue
            t6.append([s, fac, v, int(m.sum()), round(float(DM[m].mean()), 2), round(float(ND[m].mean()), 1)] +
                      [round(float((E[a] - E[b])[m].mean()), 3) for a, b in PAIRS[:5]] + [round(float(E['G3'][m].mean()), 3)])
wcsv('T6_confounders.csv', ['split', 'factor', 'level', 'n', 'mean_dmin', 'mean_ndet'] + [f'{a}-{b}' for a, b in PAIRS[:5]] + ['G3'], t6)
print('rows', len(rows), 'methods', METH)
for t, n in ((t1, 'T1'), (t4, 'T4')):
    print(n); [print(x) for x in t]
