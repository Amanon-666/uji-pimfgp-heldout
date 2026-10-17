"""P0 地图收益分离实验 G0–G4（建筑留出，UJIIndoorLoc）。
复用 UJI-PIMFGP-Heldout-20261017/pimfgp_heldout.py 的数据、splits、supports、目标楼层、law 拟合与 law_other 地图构造（只读导入）。
共用：同一 support、同一 query、同一候选集 C（= law_other 地图位置 = 目标楼"下层"历史地图格点位置）、同一 tobit 打分、后验均值输出。

  G0            核扩散锚点原型，候选 = support 包围盒+15m 网格（无几何），均匀先验 —— 无地图基线（另报 anchors / wknn）
  G1            核扩散锚点原型，候选 = C（合法几何），均匀先验（与 law_other 同）  —— 纯几何，不看任何历史 RSS
  G1orc         同 G1，候选换成目标楼层自身全部训练参考点（借用测试覆盖，仅 oracle 对照）
  G2            C + KDE 先验 + GP 残差（常数均值）+ tobit                        —— 已知有效组件
  G3            原版 law_other（历史地图→源楼栋 law→scm 用 K 锚点校准→tobit），AP 集 = 地图全部 AP；
                硬断言与 tobit_match(qx, *cand, *law_match) 逐点一致
  G3_restrictedAP 同 G3 但 AP 集限制为 K 锚点检出过的 AP（与 G1/G2 信息集一致；与 G3 分开报，不互相替代）
  G3b           G2 框架，GP 均值 = 校准后历史地图（AP 集 = K 锚点检出 AP）
  G4_post       校准后的最终场按整行指纹在 C 内置换（S 列重排），NPERM 次 —— 测"最终场的空间对应是否重要"
  G4b_post      同上作用于 G3b 的最终场
  G4_pre        输入级：build_map 之前把历史指纹按整行在 C 内置换 T_π(H)，再用同一 K 锚点 F(T_π(H),S_K) 重新校准再定位，
                NPERM_PRE 次 —— 在保留源楼栋拟合的跨楼层 law 的前提下，检验"特定空间↔RSS 对应"是否在 K-shot 校准之前就有增益
                （不是"有/无全部历史无线知识"的对照）。post 与 pre 是两种不同干预，都不是"错的"，也不声称二者等价；
                每个 episode 独立抽置换，构成复合随机化参照分布（不是同一张图置换 NPERM_PRE 次）。
  G4b_pre       同上：hist_π=hist[π] → mean_s=lookup(axy,C,hist_π,3) → gp_field(固定 GP 参数) → 固定匹配参数
  SH            二级对照：整张地图共同平移 t（10/20/40 m × ±x/±y），最近邻重采样；每个 t 单独有效域掩码 ok_t，
                SH_t、G1_t、G3_t 都在同一 ok_t 候选域上重算（ok=False 的候选 score=-inf）
线性对齐（law）只在源楼栋楼层对上拟合，置换在其后进行（"固定对齐再破坏 RSS"）。
超参（核带宽 h、GP ℓ/λ、匹配 σ/θ）只在源楼栋楼层对 episodes 上选；law_other 匹配参数沿用 results/ 中源楼层选出的 chosen。
目标 query 标签只用于最终打分。用法：python mapsep.py <building>；SMOKE=1 只跑首个目标楼层、K∈{5,20}、少量置换。
"""
import csv, gzip, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import log_ndtr, logsumexp

SRC = Path(os.environ.get('HELDOUT_DIR', '/home/panyushuo/projects/panyushuo/UJI-PIMFGP-Heldout-20261017'))
sys.path.insert(0, str(SRC))
os.chdir(SRC)
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
NPERM_PRE = int(os.environ.get('NPERM_PRE', '20' if SMOKE else '199'))
SEED = 1000
E_SRC = 2 if SMOKE else 6
QCAP = 60 if SMOKE else 300
HS = [3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
GP_GRID = R2.GP_GRID
MATCH = [tuple(m) for m in H.MATCHES]
SHIFTS = [10.0, 20.0, 40.0]
SHDIR = (('+x', (1, 0)), ('-x', (-1, 0)), ('+y', (0, 1)), ('-y', (0, -1)))
SHN = [f'SH{int(s)}{d}' for s in SHIFTS for d, _ in SHDIR]
FAR = 20.0  # 预注册：远距 = d_min >= 20 m
EQ_TOL = 1e-6
METH = ['anchors', 'wknn', 'G0', 'G1', 'G1orc', 'G2', 'G3', 'G3_restrictedAP', 'G3b',
        'G4_post_mean', 'G4b_post_mean', 'G4_pre_mean', 'G4b_pre_mean']
NULLS = ['G4_post', 'G4b_post', 'G4_pre', 'G4b_pre']


# ---------- 打分 ----------
def tob_score(qraw, protos, sig, th):
    """tobit 对数似然矩阵 S[q,c]；qraw 与 protos 的列须对应同一 AP 集。"""
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
    """固定 RBF 核把锚点原型扩散到候选集（Nadaraya–Watson，负平方距离 softmax 归一化）；不看任何历史 RSS。"""
    L = -cdist(C, axy, 'sqeuclidean') / (2 * h * h)
    w = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    assert np.allclose(w.sum(1), 1.0, atol=1e-10), 'RBF 权重行和 != 1'
    return w @ aproto


def gp_field(aproto, axy, C, mean_s, mean_C, gp):
    return R2.field(aproto, axy, C, (mean_s, mean_C), gp)[0]


def err(p, t):
    return np.linalg.norm(p - t, axis=1)


# ---------- 历史地图整行置换（build_map 之前） ----------
# prior_map 返回结构必须显式声明：PERMUTE = 随"行=地图位置"整行置换的 RSS/信号字段；FIXED = 保持不动的几何字段。
# 任何未声明、且首维 = 位置数的数组 → 直接报错（逼迫在真实服务器上显式声明 prior_map 的真实结构）。
# 默认值对应本仓库 mock 的 prior_map（dict: 'p' 指纹, 'xy' 坐标）。
# TODO(服务器)：读真实 pimfgp_heldout.prior_map 后在此（或用环境变量 PRIOR_MAP_SCHEMA='{"permute":[..],"fixed":[..]}'）声明。
PRIOR_MAP_SCHEMA = json.loads(os.environ.get('PRIOR_MAP_SCHEMA', '{"permute": ["p"], "fixed": ["xy"]}'))


def _fields(base):
    if isinstance(base, dict):
        return list(base.items())
    if isinstance(base, (tuple, list)):
        return list(enumerate(base))
    raise TypeError(f'未知的 prior_map 返回类型 {type(base)}，需在 PRIOR_MAP_SCHEMA 中显式声明')


def _rebuild(base, items):
    return dict(items) if isinstance(base, dict) else type(base)(v for _, v in sorted(items, key=lambda kv: kv[0]))


def _key(k):
    return k if isinstance(k, str) else int(k)


def schema_check(base):
    """校验 prior_map 输出与声明的 schema 一致；返回 (位置数 N, 几何字段值)。"""
    perm = [_key(x) for x in PRIOR_MAP_SCHEMA['permute']]; fixed = [_key(x) for x in PRIOR_MAP_SCHEMA['fixed']]
    items = dict(_fields(base))
    for k in perm + fixed:
        if k not in items:
            raise RuntimeError(f'PRIOR_MAP_SCHEMA 声明的字段 {k!r} 不在 prior_map 输出中（实际字段 {list(items)}）')
    pos = np.asarray(items[fixed[0]])
    n = len(pos)
    for k, v in items.items():
        if k in perm:
            if not (isinstance(v, np.ndarray) and v.shape[0] == n):
                raise RuntimeError(f'置换字段 {k!r} 的首维 != 位置数 {n}')
        elif k not in fixed and isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            raise RuntimeError(f'prior_map 字段 {k!r} 首维 = 位置数 {n} 但未在 PRIOR_MAP_SCHEMA 的 permute/fixed 中声明，拒绝猜测')
    return n, pos


def permute_rows(base, pi):
    """按白名单整行置换 RSS/信号字段；几何字段不动；未声明的 N 维数组报错。"""
    n, _ = schema_check(base)
    assert len(pi) == n and np.array_equal(np.sort(pi), np.arange(n)), 'π 不是 0..N-1 的置换'
    perm = {_key(x) for x in PRIOR_MAP_SCHEMA['permute']}
    return _rebuild(base, [(k, (v[pi] if k in perm else v)) for k, v in _fields(base)])


def rss_field(base):
    return np.asarray(dict(_fields(base))[_key(PRIOR_MAP_SCHEMA['permute'][0])], float)


def base_positions(base, C):
    """几何字段必须与 build_map 候选坐标 C 一致（确认"行=位置"）；否则停止。"""
    _, pos = schema_check(base)
    if not (pos.shape == C.shape and np.allclose(pos, C)):
        raise RuntimeError('prior_map 几何字段与 build_map 候选坐标 C 不一致；G4_pre 的整行置换无法确认，停止（需人工决策）')
    return pos


def check_C_invariant(base, C, sx, sy, law_field, rng):
    """自检：只打乱历史指纹值（白名单字段的整行置换 + 逐元素打乱），build_map 的 C 坐标必须完全不变。"""
    n, _ = schema_check(base)
    perm = {_key(x) for x in PRIOR_MAP_SCHEMA['permute']}
    for mode in ('row', 'entry'):
        if mode == 'row':
            b2 = permute_rows(base, rng.permutation(n))
        else:
            b2 = _rebuild(base, [(k, (rng.permutation(np.asarray(v).ravel()).reshape(np.shape(v)) if k in perm else v))
                                 for k, v in _fields(base)])
        c2 = build_map('scm', b2, sx, sy, law_field, CFG)
        if not (c2[1].shape == C.shape and np.array_equal(c2[1], C)):
            raise RuntimeError(f'[停止条件2] build_map 的候选坐标随历史 RSS 改变（mode={mode}）：G1 不是纯几何基线，需人工决策')


# ---------- G3 等价性诊断 ----------
# 原版 tobit_match 的 AP 掩码阈值：None = 假设"丢弃在全部候选上恒为 NO_SIGNAL 的 AP"（与 G3 全 AP 打分数学等价）。
# TODO(服务器)：读真实 scripts/evaluate_scm_tobit.tobit_match 的 AP 筛选规则，把阈值填到 ORIG_AP_THRESH，并置
#               ORIG_AP_THRESH_CONFIRMED=1（环境变量）；非 SMOKE 正式运行在未确认时拒绝启动。
ORIG_AP_THRESH = None if os.environ.get('ORIG_AP_THRESH', '') == '' else float(os.environ['ORIG_AP_THRESH'])
ORIG_AP_THRESH_CONFIRMED = os.environ.get('ORIG_AP_THRESH_CONFIRMED', '0') == '1'


def g3_equiv_diag(qx, histF, C, law_match, pred_full):
    """按原版阈值构造 AP 掩码，量化被丢弃 AP 对打分/定位的影响（总是记录）。"""
    keep = (histF > (NO_SIGNAL if ORIG_AP_THRESH is None else ORIG_AP_THRESH)).any(0)
    drop = ~keep
    nd = int(drop.sum())
    nonconst = int((np.ptp(histF[:, drop], axis=0) > 1e-10).sum()) if nd else 0
    if nd:
        Sd = tob_score(qx[:, drop], histF[:, drop], *law_match)
        rng_q = Sd.max(1) - Sd.min(1)
    else:
        rng_q = np.zeros(len(qx))
    p_keep = post_w(tob_score(qx[:, keep], histF[:, keep], *law_match)) @ C
    return {'orig_ap_thresh': 'NO_SIGNAL' if ORIG_AP_THRESH is None else ORIG_AP_THRESH, 'n_ap_map': int(histF.shape[1]),
            'n_dropped': nd, 'n_dropped_nonconst': nonconst, 'Sdrop_range_max': float(rng_q.max()),
            'Sdrop_range_median': float(np.median(rng_q)), 'loc_diff_keep_vs_full_max': float(np.abs(p_keep - pred_full).max())}


# ---------- 代码版本 ----------
def core_code_version():
    import inspect, subprocess
    files = {SRC / 'pimfgp_heldout.py', SRC / 'diag_round2.py', SRC / 'scripts/evaluate_scm_tobit.py',
             SRC / 'scripts/evaluate_signal_calibrated_map.py', Path(__file__).resolve()}
    for fn in (build_map, prior_map, tobit_match, lookup, fit_law, cells, R2.field):
        try:
            files.add(Path(inspect.getsourcefile(fn)).resolve())
        except TypeError:
            pass
    hashes = {}
    for f in sorted(files):
        if f.exists():
            hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps(sorted((Path(k).name, v) for k, v in hashes.items())).encode()).hexdigest()
    sha = None
    try:
        r = subprocess.run(['git', '-C', str(SRC), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            sha = r.stdout.strip()
            d = subprocess.run(['git', '-C', str(SRC), 'status', '--porcelain'], capture_output=True, text=True, timeout=10)
            if d.stdout.strip():
                sha += '+dirty:' + combined[:12]
    except Exception:
        pass
    return {'core_code_version': sha or ('sha256:' + combined), 'source_sha256': {Path(k).name: v for k, v in hashes.items()}}


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
def csv_meta():
    """PHONEID 与稳定物理位置 ID：training 用 (FLOOR,BUILDING,SPACEID,RELATIVEPOSITION)（数据划分阶段的物理位置标识）；
    validation 的 SPACEID/RELPOS 全为 0，只能用原始坐标全精度字符串。"""
    out = {}
    for tag, path in (('train', TRAIN), ('valid', VALID)):
        with open(path) as fh:
            r = csv.reader(fh); hdr = next(r); ix = {c: hdr.index(c) for c in hdr[520:]}
            rows = [(row[ix['PHONEID']], row[ix['FLOOR']], row[ix['BUILDINGID']], row[ix['SPACEID']],
                     row[ix['RELATIVEPOSITION']], row[ix['LONGITUDE']], row[ix['LATITUDE']]) for row in r]
        a = np.array(rows, dtype=object)
        ph = a[:, 0].astype(int); fl = a[:, 1].astype(int); bd = a[:, 2].astype(int)
        if tag == 'train':
            lid = np.array([f'B{b}F{f}:S{s}:R{p}' for b, f, s, p in zip(bd, fl, a[:, 3], a[:, 4])])
        else:
            lid = np.array([f'B{b}F{f}:xy{x}:{y}' for b, f, x, y in zip(bd, fl, a[:, 5], a[:, 6])])
        out[tag] = {'phone': ph, 'floor': fl, 'building': bd, 'lid': lid,
                    'xy': np.column_stack([a[:, 5].astype(float), a[:, 6].astype(float)])}
    return out


def check_rows(meta, rows, name):
    b, f = int(name[1]), int(name[3:])
    assert (meta['building'][rows] == b).all() and (meta['floor'][rows] == f).all(), \
        f'{name}: row_ids 与 CSV 行号不对应（楼栋/楼层不符），无法取 PHONEID/位置 ID'


def shift_ok(C, t, tree, cap):
    d, idx = tree.query(C - t[None])
    return idx, d <= cap


def manifest(sel, law_field, law_match, anchor_match):
    m = {'methods': METH, 'nulls': NULLS, 'shifts': SHN, 'nperm': NPERM, 'nperm_pre': NPERM_PRE, 'seed': SEED,
         'far': FAR, 'smoke': SMOKE, 'hs': HS, 'gp_grid': [list(g) for g in GP_GRID], 'match_grid': [list(x) for x in MATCH],
         'law_field': list(law_field), 'law_match': list(law_match), 'anchor_match': list(anchor_match),
         'prior_map_schema': PRIOR_MAP_SCHEMA, 'orig_ap_thresh': ORIG_AP_THRESH,
         'orig_ap_thresh_confirmed': ORIG_AP_THRESH_CONFIRMED}
    m.update(core_code_version())
    m['hash'] = hashlib.sha1(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return m


def main(building):
    t0 = time.time(); building = int(building)
    if not SMOKE and not ORIG_AP_THRESH_CONFIRMED:
        raise RuntimeError('正式运行前必须在服务器上确认原版 tobit_match 的 AP 筛选阈值（设 ORIG_AP_THRESH 与 ORIG_AP_THRESH_CONFIRMED=1）')
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
    man = manifest(sel, law_field, law_match, anchor_match)
    meta = {'building': building, 'sel': sel, 'manifest': man}
    log('sel', json.dumps(sel, default=str))
    cm = csv_meta()
    rng = np.random.default_rng(SEED + building)
    qf = gzip.open(OUT / f'perq_b{building}.csv.gz', 'wt', newline='')
    qw = csv.writer(qf)
    shcols = [f'{sh}:{x}' for sh in SHN for x in ('valid', 'SH', 'G1', 'G3')]
    qw.writerow(['building', 'floor', 'draw', 'K', 'split', 'qid', 'loc_id', 'x', 'y', 'd_min', 'n_det_all', 'n_det_used',
                 'phone', 'd_C', 'd_orc'] + METH + shcols)
    nullrows, eq_max, shinfo, eqdiag, g4diag = [], 0.0, [], [], []

    def write_diag():
        for fn, rows in ((f'g3_equiv_b{building}.csv', eqdiag), (f'g4pre_diag_b{building}.csv', g4diag)):
            if rows:
                with open(OUT / fn, 'w', newline='') as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    tnames = targets_of(building)[:1] if SMOKE else targets_of(building)
    for name in tnames:
        sp = splits[name]; new, val = tr[name], va[name]
        pos = {int(r): i for i, r in enumerate(new['row_ids'])}
        qi = np.array([pos[r] for r in sp['internal_query']])
        trows = np.asarray(new['row_ids'])[qi]
        vrows = np.asarray(val['row_ids'])
        check_rows(cm['train'], trows, name); check_rows(cm['valid'], vrows, name)
        queries = {'internal': (new['rssi'][qi], new['xy'][qi], cm['train']['phone'][trows], cm['train']['lid'][trows]),
                   'official': (val['rssi'], val['xy'], cm['valid']['phone'][vrows], cm['valid']['lid'][vrows])}
        base_other = prior_map(Cc[sp['lower']], law)
        Corc = np.unique(new['xy'], axis=0)  # 借用测试覆盖：目标楼层自身全部训练参考点（仅 oracle 对照）
        checked = False
        for draw, k, ids in supports(sp):
            if SMOKE and str(k) not in KS_SMOKE:
                continue
            s = np.array([pos[int(r)] for r in ids])
            sx, sy = new['rssi'][s], new['xy'][s]
            aproto, axy, _ = position_map(sx, sy)
            A = np.flatnonzero((sx != 100).any(0)); ap = aproto[:, A]
            cand = build_map('scm', base_other, sx, sy, law_field, CFG)
            C = cand[1]
            base_positions(base_other, C)
            if not checked:
                check_C_invariant(base_other, C, sx, sy, law_field, np.random.default_rng(5))
                checked = True
            histF = np.asarray(cand[0], float)        # 地图全部 AP（原版 law_other 的信息集）
            hist = histF[:, A]                         # K 锚点检出 AP
            Gb = support_grid(sy); lpC = kde_logprior(C, sy)
            P0 = kernel_protos(ap, axy, Gb, sel['G0']['p'])
            P1 = kernel_protos(ap, axy, C, sel['G1']['p'])
            P1o = kernel_protos(ap, axy, Corc, sel['G1']['p'])
            mC = np.tile(ap.mean(0), (len(C), 1)); ms = np.tile(ap.mean(0), (len(axy), 1))
            F2 = gp_field(ap, axy, C, ms, mC, sel['G2']['p'])
            gp3 = sel['G3b']['p']
            F3 = gp_field(ap, axy, C, lookup(axy, C, hist, 3), hist, gp3)
            perms = [rng.permutation(len(C)) for _ in range(NPERM)]
            pperm = [rng.permutation(len(C)) for _ in range(NPERM_PRE)]
            # G4_pre：先置换输入，再用同一 K 锚点重新校准（每次 build_map / gp_field，参数固定）
            pre_full, pre_b = [], []
            for pi in pperm:
                cp = build_map('scm', permute_rows(base_other, pi), sx, sy, law_field, CFG)
                assert np.array_equal(cp[1], C), 'G4_pre: 置换后 build_map 候选坐标变化'
                hp = np.asarray(cp[0], float)
                pre_full.append(hp)
                # G4b_pre：hist_π = 置换后重新 scm 校准的地图（限 K 锚点检出 AP），每次重做 GP（固定 GP 核/正则，不重新寻优），匹配参数固定。
                hpa = hp[:, A]
                pre_b.append(gp_field(ap, axy, C, lookup(axy, C, hpa, 3), hpa, gp3))
            # G4_pre 诊断（一次确定性置换 π0 + 全部 pre 置换的汇总）
            pi0 = np.random.default_rng(12345).permutation(len(C))
            b0 = rss_field(base_other)
            c0 = np.asarray(build_map('scm', permute_rows(base_other, pi0), sx, sy, law_field, CFG)[0], float)

            def calib(Hcal, Hin):  # scm 校准量的代理：校准后 − 校准前（逐 AP 平均偏移 + 整体幅度）
                d = Hcal - Hin
                return d.mean(0), float(np.abs(d).mean())

            def anc_res(Hcal):  # 目标锚点处残差（只计锚点检出条目）
                m = lookup(axy, C, Hcal[:, A], 1); det = ap > NO_SIGNAL
                return float(np.abs((m - ap)[det]).mean()) if det.any() else float('nan')

            off0, mag0 = calib(histF, b0); offp, magp = calib(c0, b0[pi0])
            F3p0 = gp_field(ap, axy, C, lookup(axy, C, c0[:, A], 3), c0[:, A], gp3)
            g4diag.append(dict(floor=name, draw=draw, K=str(k), n_cand=len(C),
                               calib_mag_orig=mag0, calib_mag_perm=magp,
                               calib_offset_diff_max=float(np.abs(off0 - offp).max()),
                               anchor_resid_orig=anc_res(histF), anchor_resid_perm_pi0=anc_res(c0),
                               anchor_resid_perm_mean=float(np.nanmean([anc_res(h) for h in pre_full])) if pre_full else float('nan'),
                               gp_corr_orig=float(np.abs(F3 - hist).mean()), gp_corr_perm_pi0=float(np.abs(F3p0 - c0[:, A]).mean()),
                               equivariance_maxdiff=float(np.abs(c0 - histF[pi0]).max())))
            tree = cKDTree(C); tree_o = cKDTree(Corc)
            nn = tree.query(C, k=2)[0][:, 1]
            cap = 2.0 * float(np.median(nn))
            shm = {}
            for sft in SHIFTS:
                for d, v in SHDIR:
                    nm = f'SH{int(sft)}{d}'
                    idx, ok = shift_ok(C, sft * np.array(v, float), tree, cap)
                    shm[nm] = (idx, ok)
                    shinfo.append(dict(floor=name, draw=draw, K=str(k), shift=nm, n_cand=len(C), frac_ok=float(ok.mean())))
            for split, (qx, qy, qph, qlid) in queries.items():
                dmin = cdist(qy, axy).min(1); far = dmin >= FAR
                q = qx[:, A]
                nall = (qx != 100).sum(1); nused = (q != 100).sum(1)
                pred = {}
                pred['anchors'] = tobit_match(qx, aproto, axy, *anchor_match)
                pred['wknn'] = wknn(rss_x(sx), sy, rss_x(qx))
                pred['G0'] = post_w(tob_score(q, P0, *sel['G0']['match'])) @ Gb
                S1 = tob_score(q, P1, *sel['G1']['match'])
                pred['G1'] = post_w(S1) @ C
                pred['G1orc'] = post_w(tob_score(q, P1o, *sel['G1']['match'])) @ Corc
                pred['G2'] = post_w(tob_score(q, F2, *sel['G2']['match']) + lpC[None]) @ C
                S3 = tob_score(qx, histF, *law_match)  # 全 AP：与原版 tobit_match 同信息集
                pred['G3'] = post_w(S3) @ C
                ref = tobit_match(qx, *cand, *law_match)
                delta = float(np.max(np.abs(ref - pred['G3'])))
                dg = g3_equiv_diag(qx, histF, C, law_match, pred['G3'])
                eqdiag.append(dict(floor=name, draw=draw, K=str(k), split=split, max_delta_vs_tobit_match=delta, **dg))
                if not np.allclose(ref, pred['G3'], atol=EQ_TOL, rtol=EQ_TOL):
                    m_full = set(np.flatnonzero((histF > NO_SIGNAL).any(0)).tolist())
                    m_res = set(A.tolist())
                    print('[G3 等价性失败] 诊断：', json.dumps(dg), flush=True)
                    print('AP 掩码差集：地图有信号而 K 锚点未检出', sorted(m_full - m_res)[:50],
                          '；K 锚点检出而地图无信号', sorted(m_res - m_full)[:50], flush=True)
                    write_diag()
                assert np.allclose(ref, pred['G3'], atol=EQ_TOL, rtol=EQ_TOL), f"G3 != law_other: max delta={delta}"
                eq_max = max(eq_max, delta)
                pred['G3_restrictedAP'] = post_w(tob_score(q, hist, *law_match)) @ C
                S3b = tob_score(q, F3, *sel['G3b']['match'])
                pred['G3b'] = post_w(S3b + lpC[None]) @ C
                E = {m: err(p, qy) for m, p in pred.items()}
                nul = {}
                nul['G4_post'] = np.stack([err(post_w(S3[:, pi]) @ C, qy) for pi in perms])
                nul['G4b_post'] = np.stack([err(post_w(S3b[:, pi] + lpC[None]) @ C, qy) for pi in perms])
                nul['G4_pre'] = np.stack([err(post_w(tob_score(qx, hp, *law_match)) @ C, qy) for hp in pre_full])
                nul['G4b_pre'] = np.stack([err(post_w(tob_score(q, fb, *sel['G3b']['match']) + lpC[None]) @ C, qy)
                                           for fb in pre_b])
                for nm in NULLS:
                    E[nm + '_mean'] = nul[nm].mean(0)
                dC, qnear = tree.query(qy); dO = tree_o.query(qy)[0]
                shE = {}
                for nm, (idx, ok) in shm.items():
                    neg = np.where(ok, 0.0, -np.inf)[None]
                    shE[nm] = (ok[qnear], err(post_w(S3[:, idx] + neg) @ C, qy), err(post_w(S1 + neg) @ C, qy),
                               err(post_w(S3 + neg) @ C, qy))
                for j in range(len(qy)):
                    row = [building, name, draw, k, split, j, qlid[j], f'{qy[j,0]:.2f}', f'{qy[j,1]:.2f}', f'{dmin[j]:.2f}',
                           int(nall[j]), int(nused[j]), int(qph[j]), f'{dC[j]:.2f}',
                           f'{dO[j]:.2f}'] + [f'{E[m][j]:.3f}' for m in METH]
                    for nm in SHN:
                        v, a, b1, b3 = shE[nm]
                        row += [int(v[j]), f'{a[j]:.3f}', f'{b1[j]:.3f}', f'{b3[j]:.3f}']
                    qw.writerow(row)
                nr = {'building': building, 'floor': name, 'draw': int(draw), 'K': str(k), 'split': split,
                      'n_all': int(len(qy)), 'n_far': int(far.sum())}
                for nm in NULLS:
                    nr[f'sum_all_{nm}'] = nul[nm].sum(1); nr[f'sum_far_{nm}'] = nul[nm][:, far].sum(1)
                nullrows.append(nr)
        log(name, 'done')
    qf.close()
    keys = [k for k in nullrows[0] if k.startswith('sum_')]
    np.savez_compressed(OUT / f'null_b{building}.npz',
                        meta=np.array([json.dumps({k: v for k, v in r.items() if not k.startswith('sum_')}) for r in nullrows]),
                        **{key: np.stack([r[key] for r in nullrows]) for key in keys})
    with open(OUT / f'shift_domain_b{building}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(shinfo[0])); w.writeheader(); w.writerows(shinfo)
    write_diag()
    meta['law_other_equiv_max_delta'] = eq_max
    meta['elapsed_s'] = round(time.time() - t0, 1)
    (OUT / f'select_b{building}.json').write_text(json.dumps(meta, indent=1, default=str))
    log('G3 == law_other 断言全部通过，max|Δ| =', eq_max)
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
