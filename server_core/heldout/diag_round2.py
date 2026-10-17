"""第二轮诊断：RSSI 前向场外推检验 + 按最近锚点距离分层 + P×C×L 因子消融 + 固定/学习路损指数。
协议沿用 pimfgp_heldout.py（= heldout.py）：同 splits、同 supports()（KS/DRAWS）、同目标楼层、同查询。
所有超参数（GP ℓ/λ、点预测 σ、匹配 σ/θ）只在留出楼栋之外的源楼层 episodes 上选；
目标楼层 query 标签只用于最终评分。anchors/law_other 的匹配参数直接取上一轮 results/ 中
（同样只用源楼层选出的）chosen，只读不改。
用法：python diag_round2.py 0|1|2 ；SMOKE=1 走 diag_smoke/。
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pimfgp_heldout as H  # noqa: E402
from pimfgp_heldout import (BUILDINGS, CFG, D0, NO_SIGNAL, SPLITS, TRAIN, VALID, build_map, cells,  # noqa: E402
                            dbm, episodes, est_nbar, fit_law, kde_logprior, load_floors, pairs_of,
                            position_map, prior_map, rss_x, support_grid, supports, targets_of, wknn)
from scripts.evaluate_scm_tobit import tobit_match  # noqa: E402
from scripts.evaluate_signal_calibrated_map import lookup  # noqa: E402

SMOKE = H.SMOKE
OUT = HERE / ('diag_smoke' if SMOKE else 'diag')
PREV = HERE / 'results'
E_SRC = 2 if SMOKE else 6
QCAP = 60 if SMOKE else 300
THETA_R = -100.0  # RSSI 诊断用的物理检测阈值（删失 NLL）
GP_GRID = [(10.0, 0.1), (40.0, 0.3)] if SMOKE else [
    (l, lam) for l in (2.5, 5.0, 10.0, 20.0, 40.0, 80.0) for lam in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)]
SIG_PT = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]
MATCH_C1 = [tuple(m) for m in H.MATCHES]
MATCH_C0 = [(s, None) for s in ((8.0, 16.0) if SMOKE else (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0))]
NFIX = ['2.0', '2.5', '3.0', '3.5']
EDGES = np.array([0, 5, 10, 20, 40, np.inf])
BLAB = ['0-5', '5-10', '10-20', '20-40', '40+']

# 定位方法：(均值场 key, 是否 GP 残差, C 删失项, L KDE 先验)
LOC = {}
for P in (0, 1):
    for Cf in (0, 1):
        for Lf in (0, 1):
            LOC[f'G1P{P}C{Cf}L{Lf}'] = ('const' if P == 0 else 'pl:learn', True, Cf, Lf)
for Cf in (0, 1):
    for Lf in (0, 1):
        LOC[f'G0P1C{Cf}L{Lf}'] = ('pl:learn', False, Cf, Lf)
LOC['prior_only'] = ('const', False, 1, 1)
for nk in NFIX + ['supp']:
    LOC[f'G1P1C1L1@n{nk}'] = (f'pl:{nk}', True, 1, 1)
    LOC[f'G0P1C1L0@n{nk}'] = (f'pl:{nk}', False, 1, 0)

# RSSI 前向场：(均值场 key, 是否 GP)；另有 nn_anchor / law_other 两个点预测
RS = {'GP_const': ('const', True), 'GP_pl': ('pl:learn', True),
      'pl_only': ('pl:learn', False), 'const_only': ('const', False)}
for nk in NFIX + ['supp']:
    RS[f'GP_pl@n{nk}'] = (f'pl:{nk}', True)
    RS[f'pl_only@n{nk}'] = (f'pl:{nk}', False)


def nvals(nbar, aproto, axy):
    d = {'learn': nbar}
    d.update({k: float(k) for k in NFIX})
    d['supp'] = est_nbar([(aproto, axy)])
    return d


def pl_params(aproto, axy, n):
    A = aproto.shape[1]
    det = aproto > -105.0
    ok = det.any(0)
    a = np.zeros((A, 2))
    c = np.full(A, NO_SIGNAL)
    for j in np.flatnonzero(ok):
        m = det[:, j]
        y = aproto[m, j]
        p = axy[m]
        a[j] = p[int(np.argmax(y))]
        c[j] = float(np.mean(y + 10.0 * n * np.log10(np.linalg.norm(p - a[j], axis=1) + D0)))
    return a, c, ok


def pl_eval(X, a, c, ok, n):
    out = np.full((len(X), len(c)), NO_SIGNAL)
    if ok.any():
        d = np.linalg.norm(X[:, None, :] - a[None, ok, :], axis=2)
        out[:, ok] = np.clip(c[ok] - 10.0 * n * np.log10(d + D0), NO_SIGNAL, 0.0)
    return out


def means_for(aproto, axy, X, ns, keys):
    out = {}
    if 'const' in keys:
        m = aproto.mean(0)
        out['const'] = (np.tile(m, (len(axy), 1)), np.tile(m, (len(X), 1)))
    for k, n in ns.items():
        if 'pl:' + k not in keys:
            continue
        a, c, ok = pl_params(aproto, axy, n)
        out['pl:' + k] = (pl_eval(axy, a, c, ok, n), pl_eval(X, a, c, ok, n))
    return out


def field(aproto, axy, X, mean, gp):
    """均值场 + （可选）共享 RBF 核的 GP 残差；返回预测均值与预测方差（含噪声）。"""
    m_s, m_X = mean
    if gp is None:
        return m_X, None
    ell, lam = gp
    r = aproto - m_s
    Kss = np.exp(-cdist(axy, axy, 'sqeuclidean') / (2 * ell ** 2)) + lam * np.eye(len(axy))
    Kxs = np.exp(-cdist(X, axy, 'sqeuclidean') / (2 * ell ** 2))
    Kr = np.linalg.solve(Kss, r)
    mu = m_X + Kxs @ Kr
    ml = (r * Kr).sum(0) / len(axy)  # 每 AP 的 ML 幅度
    pool = float(ml[ml > 1e-6].mean()) if (ml > 1e-6).any() else 25.0
    s2 = np.maximum(0.5 * (ml + pool), 1.0)  # 向跨 AP 汇合幅度收缩，防止小 K 下过度自信
    v = np.clip(1.0 - (Kxs * np.linalg.solve(Kss, Kxs.T).T).sum(1), 0.0, None)
    var = s2[None, :] * (v[:, None] + lam)
    return np.clip(mu, NO_SIGNAL, 0.0), var


def rssi_q(mu, var, sigma, qraw):
    y = dbm(qraw)
    det = qraw != 100
    s2 = var if var is not None else np.full_like(mu, sigma ** 2)
    s = np.sqrt(s2)
    e = y - mu
    nd = 0.5 * np.log(2 * np.pi * s2) + 0.5 * e ** 2 / s2
    nc = -log_ndtr((THETA_R - mu) / s)
    return {'n_det': det.sum(1), 'n_und': (~det).sum(1), 'sum_abs': (np.abs(e) * det).sum(1),
            'sum_bias': ((mu - y) * det).sum(1), 'sum_nll_det': (nd * det).sum(1),
            'sum_nll_cen': (nc * ~det).sum(1), 'sum_cov90': ((np.abs(e) <= 1.645 * s) & det).sum(1)}


def nll_total(mu, var, sigma, qraw):
    st = rssi_q(mu, var, sigma, qraw)
    return float(st['sum_nll_det'].sum() + st['sum_nll_cen'].sum()), int(st['n_det'].sum() + st['n_und'].sum())


def softmax_pos(score, grid):
    score = score - score.max(1, keepdims=True)
    w = np.exp(score)
    return (w / w.sum(1, keepdims=True)) @ grid


def loc_sweep(qraw, protos, grid, Cf, Lf, lp, settings):
    det = qraw != 100
    x = dbm(qraw)
    if Cf:
        xd = x * det
        d2 = (xd * xd).sum(1, keepdims=True) + det.astype(float) @ (protos ** 2).T - 2 * xd @ protos.T
        und = (~det).astype(float)
    else:
        d2 = (x * x).sum(1, keepdims=True) + (protos ** 2).sum(1)[None] - 2 * x @ protos.T
    out = {}
    for sig, th in settings:
        score = -0.5 / sig ** 2 * d2
        if Cf:
            score = score + und @ log_ndtr((th - protos) / sig).T
        if Lf:
            score = score + lp[None, :]
        out[(sig, th)] = softmax_pos(score, grid)
    return out


def err(pred, truth):
    return np.linalg.norm(pred - truth, axis=1)


def add_acc(acc, key, v, n):
    a = acc.setdefault(key, [0.0, 0])
    a[0] += v
    a[1] += n


def source_select(tr, src_floors, src_pairs, nbar, law, Cc, law_field, log):
    rng = np.random.default_rng(1)
    eps = []
    for f in src_floors:
        for s, q in episodes(tr[f]['xy'], 10, E_SRC, 3, rng):
            eps.append((f, s, rng.choice(q, min(QCAP, len(q)), replace=False)))
    mean_keys = {v[0] for v in RS.values()} | {v[0] for v in LOC.values()}
    acc = {}
    t0 = time.time()
    for f, s, q in eps:
        fl = tr[f]
        sx, sy, qx, qy = fl['rssi'][s], fl['xy'][s], fl['rssi'][q], fl['xy'][q]
        aproto, axy, _ = position_map(sx, sy)
        A = np.flatnonzero((sx != 100).any(0))
        ap = aproto[:, A]
        ns = nvals(nbar, ap, axy)
        M = means_for(ap, axy, qy, ns, mean_keys)
        for mk in mean_keys:
            for gp in GP_GRID:
                mu, var = field(ap, axy, qy, M[mk], gp)
                add_acc(acc, ('gp', mk, gp), *nll_total(mu, var, None, qx[:, A]))
            for sg in SIG_PT:
                add_acc(acc, ('pt', mk, sg), *nll_total(M[mk][1], None, sg, qx[:, A]))
        mu = lookup(qy, axy, ap, 1)
        for sg in SIG_PT:
            add_acc(acc, ('pt', 'nn', sg), *nll_total(mu, None, sg, qx[:, A]))
    log(f'rssi select pass done {round(time.time() - t0, 1)}s eps={len(eps)}')
    for lo, up in src_pairs:
        base = prior_map(Cc[lo], law)
        for s, q in episodes(tr[up]['xy'], 10, E_SRC, 3, rng):
            q = rng.choice(q, min(QCAP, len(q)), replace=False)
            fl = tr[up]
            sx, sy, qx, qy = fl['rssi'][s], fl['xy'][s], fl['rssi'][q], fl['xy'][q]
            A = np.flatnonzero((sx != 100).any(0))
            cand = build_map('scm', base, sx, sy, law_field, CFG)
            mu = lookup(qy, cand[1], cand[0][:, A], 3)
            for sg in SIG_PT:
                add_acc(acc, ('pt', 'law', sg), *nll_total(mu, None, sg, qx[:, A]))
    log(f'law rssi select done {round(time.time() - t0, 1)}s')
    gp_choice, pt_choice, rssi_sel = {}, {}, {}
    for mk in mean_keys:
        best = min(GP_GRID, key=lambda g: acc[('gp', mk, g)][0] / acc[('gp', mk, g)][1])
        gp_choice[mk] = best
        rssi_sel[f'gp|{mk}'] = {'choice': list(best), 'src_nll': acc[('gp', mk, best)][0] / acc[('gp', mk, best)][1]}
    for mk in list(mean_keys) + ['nn', 'law']:
        best = min(SIG_PT, key=lambda g: acc[('pt', mk, g)][0] / acc[('pt', mk, g)][1])
        pt_choice[mk] = best
        rssi_sel[f'pt|{mk}'] = {'choice': best, 'src_nll': acc[('pt', mk, best)][0] / acc[('pt', mk, best)][1]}
    # 第二遍：定位匹配参数
    acc2 = {}
    for f, s, q in eps:
        fl = tr[f]
        sx, sy, qx, qy = fl['rssi'][s], fl['xy'][s], fl['rssi'][q], fl['xy'][q]
        aproto, axy, _ = position_map(sx, sy)
        A = np.flatnonzero((sx != 100).any(0))
        ap = aproto[:, A]
        ns = nvals(nbar, ap, axy)
        grid = support_grid(sy)
        lp = kde_logprior(grid, sy)
        M = means_for(ap, axy, grid, ns, {v[0] for v in LOC.values()})
        cache = {}
        for lm, (mk, g, Cf, Lf) in LOC.items():
            ck = (mk, g)
            if ck not in cache:
                cache[ck] = field(ap, axy, grid, M[mk], gp_choice[mk] if g else None)[0]
            settings = MATCH_C1 if Cf else MATCH_C0
            for st, pred in loc_sweep(qx[:, A], cache[ck], grid, Cf, Lf, lp, settings).items():
                add_acc(acc2, (lm, st), float(err(pred, qy).mean()), 1)
    log(f'match select done {round(time.time() - t0, 1)}s')
    match_choice = {}
    for lm, (mk, g, Cf, Lf) in LOC.items():
        settings = MATCH_C1 if Cf else MATCH_C0
        best = min(settings, key=lambda st: acc2[(lm, st)][0] / acc2[(lm, st)][1])
        match_choice[lm] = best
        rssi_sel[f'match|{lm}'] = {'choice': list(best), 'src_mde': acc2[(lm, best)][0] / acc2[(lm, best)][1]}
    return gp_choice, pt_choice, match_choice, rssi_sel


def bucket_of(d):
    return np.clip(np.searchsorted(EDGES, d, side='right') - 1, 0, len(BLAB) - 1)


def main(building):
    started = time.time()
    building = int(building)
    OUT.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(building, *a, round(time.time() - started, 1), 's', flush=True)

    tr, va = load_floors(str(TRAIN)), load_floors(str(VALID))
    splits = json.loads(SPLITS.read_text())
    prev = json.loads((PREV / f'building{building}.json').read_text())['diagnostics']['chosen']
    law_field = tuple(prev['law_other']['field'])
    law_match = tuple(prev['law_other']['match'])
    anchor_match = tuple(prev['anchors']['match'])
    src_b = [b for b in BUILDINGS if b != building]
    src_pairs = pairs_of(src_b)
    src_floors = [f'B{b}F{f}' for b in src_b for f in range(BUILDINGS[b])]
    needed = {f for p in src_pairs for f in p} | {f'B{building}F{f}' for f in range(BUILDINGS[building])}
    Cc = {f: cells(tr[f]) for f in sorted(needed)}
    obs, law = fit_law(Cc, src_pairs)
    nbar = est_nbar([position_map(tr[f]['rssi'], tr[f]['xy'])[:2] for f in src_floors])
    log('nbar', round(nbar, 3))
    gp_choice, pt_choice, match_choice, sel = source_select(tr, src_floors, src_pairs, nbar, law, Cc, law_field, log)
    meta = {'building': building, 'nbar_learned': nbar, 'law_field': law_field, 'law_match': law_match,
            'anchor_match': anchor_match, 'selection': sel, 'theta_rssi': THETA_R, 'buckets': BLAB,
            'smoke': SMOKE}
    (OUT / f'select_b{building}.json').write_text(json.dumps(meta, indent=1, default=float))
    log('selection', json.dumps({k: v['choice'] for k, v in sel.items()}, default=float))
    rrows, lrows = [], []
    loc_keys = {v[0] for v in LOC.values()}
    all_keys = loc_keys | {v[0] for v in RS.values()}
    for name in targets_of(building):
        sp = splits[name]
        new, val = tr[name], va[name]
        pos = {int(r): i for i, r in enumerate(new['row_ids'])}
        qi = np.array([pos[r] for r in sp['internal_query']])
        queries = {'internal': (new['rssi'][qi], new['xy'][qi]), 'official': (val['rssi'], val['xy'])}
        base_other = prior_map(Cc[sp['lower']], law)
        for draw, k, ids in supports(sp):
            s = np.array([pos[int(r)] for r in ids])
            sx, sy = new['rssi'][s], new['xy'][s]
            aproto, axy, _ = position_map(sx, sy)
            A = np.flatnonzero((sx != 100).any(0))
            ap = aproto[:, A]
            ns = nvals(nbar, ap, axy)
            grid = support_grid(sy)
            lp = kde_logprior(grid, sy)
            spl = list(queries)
            Xall = np.vstack([grid] + [queries[x][1] for x in spl])
            offs = np.cumsum([0, len(grid)] + [len(queries[x][1]) for x in spl])
            M = means_for(ap, axy, Xall, ns, all_keys)
            F = {}
            for mk in all_keys:
                F[(mk, False)] = (M[mk][1], None)
                F[(mk, True)] = field(ap, axy, Xall, M[mk], gp_choice[mk])
            cand_law = build_map('scm', base_other, sx, sy, law_field, CFG)
            for si, split in enumerate(spl):
                qx, qy = queries[split]
                a0, a1 = offs[si + 1], offs[si + 2]
                bk = bucket_of(cdist(qy, axy).min(1))
                base = dict(building=building, floor=name, draw=draw, K=k, split=split)

                def rec_r(method, st):
                    for b in range(len(BLAB)):
                        m = bk == b
                        if m.any():
                            rrows.append(dict(base, method=method, bucket=BLAB[b], n_q=int(m.sum()),
                                              **{kk: float(v[m].sum()) for kk, v in st.items()}))

                for rm, (mk, g) in RS.items():
                    mu, var = F[(mk, g)]
                    rec_r(rm, rssi_q(mu[a0:a1], None if var is None else var[a0:a1],
                                     None if g else pt_choice[mk], qx[:, A]))
                rec_r('nn_anchor', rssi_q(lookup(qy, axy, ap, 1), None, pt_choice['nn'], qx[:, A]))
                rec_r('law_other', rssi_q(lookup(qy, cand_law[1], cand_law[0][:, A], 3), None,
                                          pt_choice['law'], qx[:, A]))

                def rec_l(method, pred):
                    e = err(pred, qy)
                    lrows.append(dict(base, method=method, bucket='all', n=len(e), sum_err=float(e.sum()),
                                      median=float(np.median(e)), p90=float(np.percentile(e, 90))))
                    for b in range(len(BLAB)):
                        m = bk == b
                        if m.any():
                            lrows.append(dict(base, method=method, bucket=BLAB[b], n=int(m.sum()),
                                              sum_err=float(e[m].sum()), median='', p90=''))

                for lm, (mk, g, Cf, Lf) in LOC.items():
                    pred = loc_sweep(qx[:, A], F[(mk, g)][0][:len(grid)], grid, Cf, Lf, lp,
                                     [match_choice[lm]])[match_choice[lm]]
                    rec_l(lm, pred)
                rec_l('anchors', tobit_match(qx, aproto, axy, *anchor_match))
                rec_l('wknn', wknn(rss_x(sx), sy, rss_x(qx)))
                rec_l('law_other', tobit_match(qx, *cand_law, *law_match))
        log(name, 'done')
    for fn, rows in ((f'rssi_b{building}.csv', rrows), (f'loc_b{building}.csv', lrows)):
        with open(OUT / fn, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    log('wrote', len(rrows), len(lrows))


if __name__ == '__main__':
    main(sys.argv[1])
