"""P0 地图收益分离实验 G0–G4（建筑留出，UJIIndoorLoc）。
复用 UJI-PIMFGP-Heldout-20261017/pimfgp_heldout.py 的数据、splits、supports、目标楼层、law 拟合与 law_other 地图构造（只读导入）。
所有方法共用：同一 support、同一 query、同一候选集 C（= law_other 地图的位置 = 目标楼"下层"历史地图格点位置）、同一 tobit 打分与后验均值输出。
  G0  : 核扩散锚点原型，候选 = support 包围盒+15m 网格（无几何），均匀先验            —— 无地图基线（另报 anchors / wknn 参考）
  G1  : 核扩散锚点原型，候选 = C（合法几何），先验与 law_other 相同（均匀）            —— 纯几何
  G2  : G1 候选 + KDE 先验 + GP 残差（常数均值）+ tobit                               —— 已知有效组件
  G3  : law_other 原样（历史地图 → 源楼栋 law → scm 用 support 校准 → tobit），候选 C，均匀先验
  G3b : G2 的 GP+KDE，但 GP 均值 = G3 的校准历史地图                                   —— 历史 RSS 在 G2 之上的净贡献
  G4  : R_π = G3 但把校准后地图整行指纹在 C 内置换（固定对齐再破坏），999 次 → 经验零分布
  G4b : 同上作用于 G3b
  SH  : 二级对照：整张地图共同空间平移（NN 重采样，记录有效重叠）
超参（核带宽 h、GP ℓ/λ、匹配 σ/θ）只在源楼栋楼层对 episodes 上选；law_other 匹配参数沿用 results/ 中源楼层选出的 chosen。
目标 query 标签只用于最终打分。用法：python mapsep.py <building> ；SMOKE=1 时只跑首个目标楼层、K∈{5,20}、少量置换。
"""
import csv, gzip, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr, logsumexp

SRC = Path(os.environ.get('HELDOUT_DIR', '/home/panyushuo/projects/panyushuo/UJI-PIMFGP-Heldout-20261017'))
sys.path.insert(0, str(SRC))
os.chdir(SRC)  # pimfgp_heldout 内部相对路径
import pimfgp_heldout as H  # noqa
import diag_round2 as R2  # noqa
from pimfgp_heldout import (BUILDINGS, CFG, NO_SIGNAL, SPLITS, TRAIN, VALID, build_map, cells, dbm, episodes,
                            fit_law, kde_logprior, load_floors, pairs_of, position_map, prior_map, rss_x,
                            support_grid, supports, targets_of, wknn)
from scripts.evaluate_scm_tobit import tobit_match  # noqa
from scripts.evaluate_signal_calibrated_map import lookup  # noqa

SMOKE = os.environ.get('SMOKE', '0') == '1'
OUT = Path(os.environ.get('OUT', '/home/panyushuo/projects/panyushuo/UJI-P0-MapSep-20261017')) / ('smoke' if SMOKE else 'run')
PREV = SRC / 'results'
KS_SMOKE = {'5', '20'}
NPERM = int(os.environ.get('NPERM', '50' if SMOKE else '999'))
E_SRC = 2 if SMOKE else 6
QCAP = 60 if SMOKE else 300
HS = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
GP_GRID = R2.GP_GRID
MATCH = [tuple(m) for m in H.MATCHES]
SHIFTS = [10.0, 20.0, 40.0]
EDGES = np.array([0, 5, 10, 20, 40, np.inf]); BLAB = ['0-5', '5-10', '10-20', '20-40', '40+']
FAR = 20.0  # 预注册：远距 = d_min >= 20 m


# ---------- 打分 ----------
def tob_score(qraw, protos, sig, th):
    """tobit 对数似然矩阵 S[q,c]；protos 为 dBm，已限制在 AP 子集。"""
    det = qraw != 100
    x = dbm(qraw)
    xd = x * det
    d2 = (xd * xd).sum(1, keepdims=True) + det.astype(float) @ (protos ** 2).T - 2 * xd @ protos.T
    S = -0.5 / sig ** 2 * d2
    if th is not None:
        S = S + (~det).astype(float) @ log_ndtr((th - protos) / sig).T
    return S


def post_w(S):
    S = S - S.max(1, keepdims=True)
    W = np.exp(S)
    return W / W.sum(1, keepdims=True)


def kernel_protos(aproto, axy, C, h):
    """固定 RBF 核把锚点原型扩散到候选集（Nadaraya–Watson）；不看任何历史 RSS。"""
    w = np.exp(-cdist(C, axy, 'sqeuclidean') / (2 * h * h))
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
    return w @ aproto


def gp_field(aproto, axy, C, mean_s, mean_C, gp):
    return R2.field(aproto, axy, C, (mean_s, mean_C), gp)[0]


def err(p, t):
    return np.linalg.norm(p - t, axis=1)


# ---------- 源楼层选参 ----------
def source_select(tr, src_pairs, law, Cc, law_field, log):
    rng = np.random.default_rng(7)
    acc = {}

    def add(key, v):
        a = acc.setdefault(key, [0.0, 0]); a[0] += v; a[1] += 1

    n = 0
    for lo, up in src_pairs:
        base = prior_map(Cc[lo], law)
        fl = tr[up]
        for s, q in episodes(fl['xy'], 10, E_SRC, 3, rng):
            q = rng.choice(q, min(QCAP, len(q)), replace=False)
            sx, sy, qx, qy = fl['rssi'][s], fl['xy'][s], fl['rssi'][q], fl['xy'][q]
            aproto, axy, _ = position_map(sx, sy)
            A = np.flatnonzero((sx != 100).any(0))
            ap = aproto[:, A]
            cand = build_map('scm', base, sx, sy, law_field, CFG)
            C = cand[1]
            Gb = support_grid(sy)
            lpC = kde_logprior(C, sy)
            for h in HS:
                for nm, grid in (('G0', Gb), ('G1', C)):
                    P = kernel_protos(ap, axy, grid, h)
                    for st in MATCH:
                        add((nm, h, st), err(post_w(tob_score(qx[:, A], P, *st)) @ grid, qy).mean())
            mC = np.tile(ap.mean(0), (len(C), 1)); ms = np.tile(ap.mean(0), (len(axy), 1))
            hist = cand[0][:, A]
            hist_s = lookup(axy, C, hist, 3)
            for gp in GP_GRID:
                F2 = gp_field(ap, axy, C, ms, mC, gp)
                F3 = gp_field(ap, axy, C, hist_s, hist, gp)
                for st in MATCH:
                    add(('G2', gp, st), err(post_w(tob_score(qx[:, A], F2, *st) + lpC[None]) @ C, qy).mean())
                    add(('G3b', gp, st), err(post_w(tob_score(qx[:, A], F3, *st) + lpC[None]) @ C, qy).mean())
            n += 1
    log(f'source select episodes={n}')
    sel = {}
    for nm in ('G0', 'G1', 'G2', 'G3b'):
        keys = [k for k in acc if k[0] == nm]
        best = min(keys, key=lambda k: acc[k][0] / acc[k][1])
        sel[nm] = {'p': best[1], 'match': best[2], 'src_mde': acc[best][0] / acc[best][1]}
    return sel


# ---------- 元数据 ----------
def phone_ids():
    out = {}
    for tag, path in (('train', TRAIN), ('valid', VALID)):
        with open(path) as fh:
            r = csv.reader(fh); hdr = next(r); j = hdr.index('PHONEID')
            out[tag] = np.array([int(row[j]) for row in r])
    return out


def shift_index(C, t):
    """共同平移：候选 c 取历史地图在 c−t 处（最近格点）的指纹；返回索引与是否在有效重叠区（NN 距离 ≤ 2×中位格距）。"""
    D = cdist(C - t[None], C)
    idx = D.argmin(1)
    nn = np.sort(cdist(C, C), 1)[:, 1]
    cap = 2.0 * float(np.median(nn))
    return idx, D[np.arange(len(C)), idx] <= cap


def main(building):
    t0 = time.time(); building = int(building)
    OUT.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(building, *a, round(time.time() - t0, 1), 's', flush=True)

    tr, va = load_floors(str(TRAIN)), load_floors(str(VALID))
    splits = json.loads(SPLITS.read_text())
    prev = json.loads((PREV / f'building{building}.json').read_text())['diagnostics']['chosen']
    law_field = tuple(prev['law_other']['field']); law_match = tuple(prev['law_other']['match'])
    anchor_match = tuple(prev['anchors']['match'])
    src_b = [b for b in BUILDINGS if b != building]
    src_pairs = pairs_of(src_b)
    needed = {f for p in src_pairs for f in p} | {f'B{building}F{f}' for f in range(BUILDINGS[building])}
    Cc = {f: cells(tr[f]) for f in sorted(needed)}
    _, law = fit_law(Cc, src_pairs)
    if SMOKE:
        src_pairs = src_pairs[:3]
    sel = source_select(tr, src_pairs, law, Cc, law_field, log)
    meta = {'building': building, 'sel': sel, 'law_field': law_field, 'law_match': law_match,
            'anchor_match': anchor_match, 'nperm': NPERM, 'smoke': SMOKE, 'far': FAR}
    (OUT / f'select_b{building}.json').write_text(json.dumps(meta, indent=1, default=str))
    log('sel', json.dumps(meta['sel'], default=str))
    ph = phone_ids()
    rng = np.random.default_rng(1000 + building)
    qf = gzip.open(OUT / f'perq_b{building}.csv.gz', 'wt', newline='')
    qw = csv.writer(qf)
    METH = ['anchors', 'wknn', 'G0', 'G1', 'G1orc', 'G2', 'G3', 'G3b', 'G4mean', 'G4bmean'] + \
           [f'SH{int(s)}{d}' for s in SHIFTS for d in ('+x', '-x', '+y', '-y')]
    qw.writerow(['building', 'floor', 'draw', 'K', 'split', 'qid', 'loc_id', 'x', 'y', 'd_min', 'n_det', 'phone',
                 'd_C', 'd_orc', 'sh_valid'] + METH)
    nullrows = []  # 每个 episode×split：远距/全部查询在每次置换下的误差和
    tnames = targets_of(building)
    if SMOKE:
        tnames = tnames[:1]
    eq_chk = []
    for name in tnames:
        sp = splits[name]; new, val = tr[name], va[name]
        pos = {int(r): i for i, r in enumerate(new['row_ids'])}
        qi = np.array([pos[r] for r in sp['internal_query']])
        vrows = np.asarray(val.get('row_ids', np.arange(len(val['xy']))))
        trows = np.asarray(new['row_ids'])[qi]
        queries = {'internal': (new['rssi'][qi], new['xy'][qi], ph['train'][trows]),
                   'official': (val['rssi'], val['xy'], ph['valid'][vrows])}
        base_other = prior_map(Cc[sp['lower']], law)
        Corc = np.unique(new['xy'], axis=0)  # 借用测试覆盖：目标楼层自身全部训练参考点位置（仅作 oracle 对照）
        for draw, k, ids in supports(sp):
            if SMOKE and str(k) not in KS_SMOKE:
                continue
            s = np.array([pos[int(r)] for r in ids])
            sx, sy = new['rssi'][s], new['xy'][s]
            aproto, axy, _ = position_map(sx, sy)
            A = np.flatnonzero((sx != 100).any(0)); ap = aproto[:, A]
            cand = build_map('scm', base_other, sx, sy, law_field, CFG)
            C = cand[1]; hist = cand[0][:, A]
            Gb = support_grid(sy); lpC = kde_logprior(C, sy)
            P0 = kernel_protos(ap, axy, Gb, sel['G0']['p'])
            P1 = kernel_protos(ap, axy, C, sel['G1']['p'])
            P1o = kernel_protos(ap, axy, Corc, sel['G1']['p'])
            mC = np.tile(ap.mean(0), (len(C), 1)); ms = np.tile(ap.mean(0), (len(axy), 1))
            F2 = gp_field(ap, axy, C, ms, mC, sel['G2']['p'])
            F3 = gp_field(ap, axy, C, lookup(axy, C, hist, 3), hist, sel['G3b']['p'])
            perms = [rng.permutation(len(C)) for _ in range(NPERM)]
            shs = {}
            for sft in SHIFTS:
                for d, t in (('+x', (sft, 0)), ('-x', (-sft, 0)), ('+y', (0, sft)), ('-y', (0, -sft))):
                    shs[f'SH{int(sft)}{d}'] = shift_index(C, np.array(t, float))
            for split, (qx, qy, qph) in queries.items():
                dmin = cdist(qy, axy).min(1); far = dmin >= FAR
                ndet = (qx != 100).sum(1)
                dC = cdist(qy, C).min(1); dO = cdist(qy, Corc).min(1)
                q = qx[:, A]
                pred = {}
                pred['anchors'] = tobit_match(qx, aproto, axy, *anchor_match)
                pred['wknn'] = wknn(rss_x(sx), sy, rss_x(qx))
                pred['G0'] = post_w(tob_score(q, P0, *sel['G0']['match'])) @ Gb
                pred['G1'] = post_w(tob_score(q, P1, *sel['G1']['match'])) @ C
                pred['G1orc'] = post_w(tob_score(q, P1o, *sel['G1']['match'])) @ Corc
                pred['G2'] = post_w(tob_score(q, F2, *sel['G2']['match']) + lpC[None]) @ C
                S3 = tob_score(q, hist, *law_match)
                W3 = post_w(S3)
                pred['G3'] = W3 @ C
                ref = tobit_match(qx, *cand, *law_match)
                eq_chk.append(float(np.abs(ref - pred['G3']).max()))
                S3b = tob_score(q, F3, *sel['G3b']['match'])
                pred['G3b'] = post_w(S3b + lpC[None]) @ C
                # 置换：指纹整行在 C 内置换 → S 的列重排（位置与先验不动）
                e4 = np.zeros((NPERM, len(qy))); e4b = np.zeros((NPERM, len(qy)))
                for i, pi in enumerate(perms):
                    e4[i] = err(post_w(S3[:, pi]) @ C, qy)
                    e4b[i] = err(post_w(S3b[:, pi] + lpC[None]) @ C, qy)
                svalid = np.ones(len(qy), bool)
                for nm, (idx, ok) in shs.items():
                    pred[nm] = post_w(S3[:, idx]) @ C
                    svalid &= ok[cdist(qy, C).argmin(1)]
                E = {m: err(p, qy) for m, p in pred.items()}
                E['G4mean'] = e4.mean(0); E['G4bmean'] = e4b.mean(0)
                lid = [f'{name}:{x:.1f}:{y:.1f}' for x, y in np.round(qy, 1)]
                for j in range(len(qy)):
                    qw.writerow([building, name, draw, k, split, j, lid[j], f'{qy[j,0]:.2f}', f'{qy[j,1]:.2f}',
                                 f'{dmin[j]:.2f}', int(ndet[j]), int(qph[j]), f'{dC[j]:.2f}', f'{dO[j]:.2f}', int(svalid[j])] +
                                [f'{E[m][j]:.3f}' for m in METH])
                nullrows.append({'floor': name, 'draw': draw, 'K': str(k), 'split': split, 'n_all': len(qy),
                                 'n_far': int(far.sum()), 'sum_all_G4': e4.sum(1), 'sum_far_G4': e4[:, far].sum(1),
                                 'sum_all_G4b': e4b.sum(1), 'sum_far_G4b': e4b[:, far].sum(1)})
        log(name, 'done')
    qf.close()
    np.savez_compressed(OUT / f'null_b{building}.npz',
                        meta=np.array([json.dumps({k: v for k, v in r.items() if not k.startswith('sum_')}) for r in nullrows]),
                        **{key: np.stack([r[key] for r in nullrows]) for key in
                           ('sum_all_G4', 'sum_far_G4', 'sum_all_G4b', 'sum_far_G4b')})
    log('law_other 等价性检查 max|Δ| =', max(eq_chk) if eq_chk else None)
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
