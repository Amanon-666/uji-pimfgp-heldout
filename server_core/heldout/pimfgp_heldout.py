"""PI-MFGP★ 接入建筑留出协议：在"无地图"设定下，比较
  wknn / anchors（历史方法） vs pathloss / pathloss_kde（新增网格路损前向场）。
同时保留 law_other/law_own/copy/break_space/break_ap 作为"有地图"对照，表格自洽可比。
协议、splits、调参范围完全沿用 heldout.py；复用 FeMLoc-Signal-Calibrated-Map，不修改其代码。
用法：python pimfgp_heldout.py 0|1|2 ；SMOKE=1 走 smoke/。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr, logsumexp

ROOT = Path('/home/panyushuo/projects/panyushuo')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'FeMLoc-Signal-Calibrated-Map'))
from data.uji import load_floors  # noqa: E402
from models.cross_floor import fit_transfer, prior_map  # noqa: E402
from models.radio_map import cells, fit_observation  # noqa: E402
from scripts.evaluate_scm_tobit import tobit_match  # noqa: E402
from scripts.evaluate_signal_calibrated_map import (  # noqa: E402
    build_map, dbm, episodes, lower_map, mde, position_map)

NO_SIGNAL = -110.0
TRAIN = ROOT / 'UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv'
VALID = ROOT / 'UJIIndoorLoc/data/raw/UJIndoorLoc/validationData.csv'
SPLITS = ROOT / 'label-efficiency-20261012/splits.json'
SMOKE = os.environ.get('SMOKE') == '1'
OUT = HERE / ('smoke' if SMOKE else 'results')
BUILDINGS = {0: 4, 1: 4, 2: 5}
FIELDS = [(20.0, 0.3), (40.0, 0.3)] if SMOKE else [
    (ell, lam) for ell in (20.0, 40.0, 60.0, 100.0) for lam in (0.1, 0.3, 1.0, 3.0)]
MATCHES = [(12.0, -85.0)] if SMOKE else [
    (sig, th) for sig in (8.0, 12.0, 16.0, 24.0) for th in (-90.0, -85.0, -80.0, -75.0)]
CFG = {'anchor_lookup_neighbors': 3}
EPISODES = 2 if SMOKE else 20
QUERY_CAP = 40 if SMOKE else 300
KS = [5] if SMOKE else [3, 5, 10, 20, 30, 'full']
DRAWS = 1 if SMOKE else 10
GRID_STEP = 8.0 if SMOKE else 4.0
GRID_PAD = 15.0
GRID_CAP = 400 if SMOKE else 1600
D0 = 1.0


def pairs_of(buildings):
    out = []
    for b in buildings:
        for f in range(BUILDINGS[b] - 1):
            out.append((f'B{b}F{f}', f'B{b}F{f + 1}'))
    return out


def targets_of(building):
    names = [f'B{building}F{f}' for f in range(1, BUILDINGS[building])]
    return names[:1] if SMOKE else names


def rss_x(rssi):
    return np.where(rssi == 100, 0.0, (rssi.astype(np.float64) + 105.0) / 105.0)


def wknn(sx, sy, qx, neighbors=3):
    d = cdist(qx, sx)
    k = min(neighbors, len(sx))
    nearest = np.argpartition(d, k - 1, axis=1)[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(d, nearest, axis=1), 1e-8)
    return (sy[nearest] * w[..., None]).sum(1) / w.sum(1, keepdims=True)


# ---------- 新增：网格路损前向场 ----------

def est_nbar(proto_xy_list):
    """在源楼栋（非留出）上，用固定截距的汇合回归估计对数距离路损指数 n̄（AP 无关共享先验）。"""
    num = den = 0.0
    for proto, pxy in proto_xy_list:
        det = proto > -105.0
        for j in range(proto.shape[1]):
            m = det[:, j]
            if int(m.sum()) < 3:
                continue
            yj = proto[m, j]
            aj = pxy[m][int(np.argmax(yj))]
            xj = np.log10(np.linalg.norm(pxy[m] - aj, axis=1) + D0)
            xc = xj - xj.mean()
            yc = yj - yj.mean()
            num += float((xc * yc).sum())
            den += float((xc * xc).sum())
    slope = num / den if den > 0 else -20.0
    return float(np.clip(-slope / 10.0, 1.5, 5.0))


def support_grid(sy, step=GRID_STEP, pad=GRID_PAD, cap=GRID_CAP):
    lo = sy.min(0) - pad
    hi = sy.max(0) + pad
    while True:
        xs = np.arange(lo[0], hi[0] + step, step)
        ys = np.arange(lo[1], hi[1] + step, step)
        if len(xs) * len(ys) <= cap or step > 40:
            break
        step *= 1.5
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def pathloss_map(sx, sy, nbar, grid):
    """每个 AP：锚点上估最强位置 a_j 与截距 c_j（斜率固定 -10·n̄），在网格上给出前向场原型。"""
    aproto, axy, _ = position_map(sx, sy)
    A = aproto.shape[1]
    protos = np.full((grid.shape[0], A), NO_SIGNAL)
    det = aproto > -105.0
    for j in range(A):
        m = det[:, j]
        if int(m.sum()) < 1:
            continue
        yj = aproto[m, j]
        pj = axy[m]
        aj = pj[int(np.argmax(yj))]
        cj = float(np.mean(yj + 10.0 * nbar * np.log10(np.linalg.norm(pj - aj, axis=1) + D0)))
        gd = np.linalg.norm(grid - aj, axis=1)
        protos[:, j] = np.clip(cj - 10.0 * nbar * np.log10(gd + D0), NO_SIGNAL, 0.0)
    return protos, grid


def kde_logprior(grid, sy):
    if len(sy) > 1:
        dd = cdist(sy, sy)
        np.fill_diagonal(dd, np.inf)
        h = max(6.0, float(np.median(dd.min(1))))
    else:
        h = 10.0
    d2 = cdist(grid, sy) ** 2
    return logsumexp(-0.5 * d2 / h ** 2, axis=1)


def tobit_match_prior(qraw, protos, positions, sigma, theta, logprior):
    det = (qraw != 100).astype(np.float64)
    x = dbm(qraw) * det
    d2 = (x * x).sum(1, keepdims=True) + det @ (protos * protos).T - 2 * x @ protos.T
    score = -0.5 / sigma ** 2 * d2 + (1 - det) @ log_ndtr((theta - protos) / sigma).T
    score = score + logprior[None, :]
    score -= score.max(1, keepdims=True)
    w = np.exp(score)
    return (w / w.sum(1, keepdims=True)) @ positions


# ---------- 以下与 heldout.py 一致 ----------

def mirror(base):
    protos, xy = base
    c = xy.mean(0)
    return protos, (xy - c) * np.array([-1.0, 1.0]) + c


def permute_cells(lc, perm):
    out = dict(lc)
    for key in ('k', 's1', 's2'):
        out[key] = lc[key][:, perm]
    return out


def ap_detected(floor):
    return (floor['rssi'] != 100).any(0)


def overlap(a, b):
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return {'intersection': inter, 'union': union, 'jaccard': inter / union if union else None}


def fit_law(C, pairs):
    floors = sorted({f for pair in pairs for f in pair})
    obs = fit_observation([C[f] for f in floors])
    law = fit_transfer([(C[a], C[b]) for a, b in pairs], obs)
    return obs, law


def law_record(law):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in law.items() if k in ('bbar', 'rho', 'theta', 's', 'sigma_total', 'h',
                                             'n_cells', 'n_link_cells', 'n_groups', 'link_nll')}


def select(tr, C, pairs, law):
    rng = np.random.default_rng(1)
    acc = {}
    t0 = time.time()
    for lo, up in pairs:
        base = prior_map(C[lo], law)
        new = tr[up]
        for s, q in episodes(new['xy'], 10, EPISODES, 3, rng):
            q = rng.choice(q, min(QUERY_CAP, len(q)), replace=False)
            qx, qy, sx, sy = new['rssi'][q], new['xy'][q], new['rssi'][s], new['xy'][s]
            anchors = position_map(sx, sy)[:2]
            for match in MATCHES:
                acc.setdefault(('anchors', None, match), []).append(mde(tobit_match(qx, *anchors, *match), qy))
            for field in FIELDS:
                cand = build_map('scm', base, sx, sy, field, CFG)
                for match in MATCHES:
                    acc.setdefault(('law_other', field, match), []).append(
                        mde(tobit_match(qx, *cand, *match), qy))
        print('select', lo, up, 'seconds', round(time.time() - t0, 1), flush=True)
    chosen = {}
    for name in ('anchors', 'law_other'):
        keys = [k for k in acc if k[0] == name]
        best = min(keys, key=lambda k: float(np.mean(acc[k])))
        chosen[name] = {'field': None if best[1] is None else list(best[1]), 'match': list(best[2]),
                        'source_mde': float(np.mean(acc[best])), 'n': len(acc[best])}
    return chosen


def supports(sp):
    draws = sp['draws'][:DRAWS]
    for d, draw in enumerate(draws):
        for k in KS:
            if k == 'full':
                continue
            yield d, k, np.array([r for g in draw[:k] for r in g])
    if 'full' in KS:
        yield 0, 'full', np.array(sp['full_support'])


def score(pred, truth):
    err = np.linalg.norm(pred - truth, axis=1)
    return {'n': int(len(err)), 'mde': float(err.mean()), 'median': float(np.median(err)),
            'p90': float(np.percentile(err, 90))}


def main(building):
    started = time.time()
    building = int(building)
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = load_floors(str(TRAIN)), load_floors(str(VALID))
    splits = json.loads(SPLITS.read_text())
    source_buildings = [b for b in BUILDINGS if b != building]
    source_pairs = pairs_of(source_buildings)
    needed = {f for pair in source_pairs for f in pair}
    needed |= {f'B{building}F{f}' for f in range(BUILDINGS[building])}
    C = {f: cells(tr[f]) for f in sorted(needed)}
    print(building, 'fit source law', source_pairs, flush=True)
    obs, law = fit_law(C, source_pairs)
    src_floors = [f'B{b}F{f}' for b in source_buildings for f in range(BUILDINGS[b])]
    nbar = est_nbar([position_map(tr[f]['rssi'], tr[f]['xy'])[:2] for f in src_floors])
    print(building, 'nbar', round(nbar, 3), flush=True)
    own_pairs = pairs_of([building])
    chosen = select(tr, C, source_pairs, law)
    print(building, 'chosen', chosen, 'seconds', round(time.time() - started, 1), flush=True)
    perm = np.random.default_rng(0).permutation(tr['B0F0']['rssi'].shape[1])
    detected = {b: np.logical_or.reduce([ap_detected(tr[f'B{b}F{f}']) for f in range(BUILDINGS[b])])
                for b in BUILDINGS}
    diag = {
        'building_ap_jaccard': {
            f'{a}-{b}': overlap(detected[a], detected[b]) for a in BUILDINGS for b in BUILDINGS if a < b},
        'source_law': law_record(law), 'observation': obs, 'chosen': chosen,
        'source_pairs': source_pairs, 'nbar': nbar,
        'grid': {'step': GRID_STEP, 'pad': GRID_PAD, 'cap': GRID_CAP}}
    rows = []
    own_cache = {}
    field = tuple(chosen['law_other']['field'])
    match = tuple(chosen['law_other']['match'])
    anchor_match = tuple(chosen['anchors']['match'])
    for name in targets_of(building):
        sp = splits[name]
        new, val = tr[name], va[name]
        assert sorted(val['row_ids'].tolist()) == sorted(sp['official_query'])
        pos = {int(r): i for i, r in enumerate(new['row_ids'])}
        qi = np.array([pos[r] for r in sp['internal_query']])
        queries = {'internal': (new['rssi'][qi], new['xy'][qi]), 'official': (val['rssi'], val['xy'])}
        lower_name = sp['lower']
        base_other = prior_map(C[lower_name], law)
        kept = [p for p in own_pairs if name not in p]
        key = tuple(kept)
        if key not in own_cache:
            print(building, name, 'refit own law without', name, kept, flush=True)
            own_cache[key] = fit_law(C, kept)[1]
        base_own = prior_map(C[lower_name], own_cache[key])
        base_copy = lower_map(tr[lower_name])
        base_space = mirror(base_other)
        base_ap = prior_map(permute_cells(C[lower_name], perm), law)
        bases = {'law_other': base_other, 'law_own': base_own, 'copy': base_copy,
                 'break_space': base_space, 'break_ap': base_ap}
        lower_det, this_det = ap_detected(tr[lower_name]), ap_detected(new)
        diag.setdefault('floor_ap', {})[name] = {
            'lower': lower_name, 'overlap_with_lower': overlap(lower_det, this_det),
            'own_pairs': kept, 'own_law': law_record(own_cache[key])}

        def add(method, draw, k, split, pred, truth):
            rows.append(dict(building=building, floor=name, method=method, draw=draw, K=k,
                             split=split, **score(pred, truth)))

        for split, (qx, qy) in queries.items():
            for method, base in bases.items():
                add(method, 0, 0, split, tobit_match(qx, *base, *match), qy)
        for draw, k, ids in supports(sp):
            s = np.array([pos[int(r)] for r in ids])
            sx, sy = new['rssi'][s], new['xy'][s]
            maps = {m: build_map('scm', base, sx, sy, field, CFG) for m, base in bases.items()}
            maps['anchors'] = position_map(sx, sy)[:2]
            grid = support_grid(sy)
            pl = pathloss_map(sx, sy, nbar, grid)
            lp = kde_logprior(grid, sy)
            for split, (qx, qy) in queries.items():
                for method, cand in maps.items():
                    use = anchor_match if method == 'anchors' else match
                    add(method, draw, k, split, tobit_match(qx, *cand, *use), qy)
                add('pathloss', draw, k, split, tobit_match(qx, *pl, *anchor_match), qy)
                add('pathloss_kde', draw, k, split,
                    tobit_match_prior(qx, pl[0], pl[1], anchor_match[0], anchor_match[1], lp), qy)
                add('wknn', draw, k, split, wknn(rss_x(sx), sy, rss_x(qx)), qy)
        print(building, name, 'done', round(time.time() - started, 1), 's', 'grid', len(grid), flush=True)
    payload = dict(building=building, smoke=SMOKE, seconds=time.time() - started,
                   diagnostics=diag, rows=rows)
    path = OUT / f'building{building}.json'
    path.write_text(json.dumps(payload, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)))
    print('wrote', path, 'rows', len(rows), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
