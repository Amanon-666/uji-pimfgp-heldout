"""事后诊断（不改协议、不选参）：G2 远距变差是否因 KDE 先验把预测拉向锚点。
复用 run/select_b*.json 中已冻结的源楼栋选参，重算 G1、G2，并加两个分解：G2_noKDE（GP 场、无 KDE 先验）、G1_KDE（核扩散 + KDE 先验）。
输出每 query：真值/各预测到最近锚点的距离、误差、锚点行后验质量。用法：python diag_kde.py <run_dir>"""
import csv, gzip, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
sys.argv, RUN = sys.argv[:1], Path(sys.argv[1]).resolve()
import mapsep as M

def main():
    tr, va = M.load_floors(str(M.TRAIN)), M.load_floors(str(M.VALID))
    splits = json.loads(M.SPLITS.read_text())
    out = gzip.open(RUN / 'diag_kde_perq.csv.gz', 'wt', newline=''); w = csv.writer(out)
    meths = ['G1', 'G1_KDE', 'G2_noKDE', 'G2']
    w.writerow(['building', 'floor', 'draw', 'K', 'split', 'd_min_true'] + [f'{m}:{x}' for m in meths for x in ('err', 'pred_dmin', 'anchor_mass')])
    for b in M.BUILDINGS:
        sel = json.loads((RUN / f'select_b{b}.json').read_text())['sel']
        law_field = tuple(json.loads((M.PREV / f'building{b}.json').read_text())['diagnostics']['chosen']['law_other']['field'])
        src_pairs = M.pairs_of([x for x in M.BUILDINGS if x != b])
        needed = {f for p in src_pairs for f in p} | {f'B{b}F{f}' for f in range(M.BUILDINGS[b])}
        Cc = {f: M.cells(tr[f]) for f in sorted(needed)}
        _, law = M.fit_law(Cc, src_pairs)
        for name in M.targets_of(b):
            sp = splits[name]; new, val = tr[name], va[name]
            pos = {int(r): i for i, r in enumerate(new['row_ids'])}
            qi = np.array([pos[r] for r in sp['internal_query']])
            Q = {'internal': (new['rssi'][qi], new['xy'][qi]), 'official': (val['rssi'], val['xy'])}
            base = M.prior_map(Cc[sp['lower']], law)
            for draw, k, ids in M.supports(sp):
                s = np.array([pos[int(r)] for r in ids]); sx, sy = new['rssi'][s], new['xy'][s]
                aproto, axy, _ = M.position_map(sx, sy)
                A = np.flatnonzero((sx != 100).any(0)); ap = aproto[:, A]
                C = M.build_map('scm', base, sx, sy, law_field, M.CFG)[1]
                nh = M.base_positions(base, C, axy)
                lpC = M.kde_logprior(C, sy)
                P1 = M.with_anchor_rows(M.kernel_protos(ap, axy, C, sel['G1']['p']), ap, nh)
                mC = np.tile(ap.mean(0), (len(C), 1)); ms = np.tile(ap.mean(0), (len(axy), 1))
                F2 = M.with_anchor_rows(M.gp_field(ap, axy, C, ms, mC, sel['G2']['p']), ap, nh)
                for split, (qx, qy) in Q.items():
                    q = qx[:, A]
                    S1 = M.tob_score(q, P1, *sel['G1']['match']); S2 = M.tob_score(q, F2, *sel['G2']['match'])
                    Ws = {'G1': M.post_w(S1), 'G1_KDE': M.post_w(S1 + lpC[None]),
                          'G2_noKDE': M.post_w(S2), 'G2': M.post_w(S2 + lpC[None])}
                    dt = cdist(qy, axy).min(1)
                    cols = []
                    for m in meths:
                        p = Ws[m] @ C
                        cols += [M.err(p, qy), cdist(p, axy).min(1), Ws[m][:, nh:].sum(1)]
                    for j in range(len(qy)):
                        w.writerow([b, name, draw, k, split, f'{dt[j]:.2f}'] + [f'{c[j]:.3f}' for c in cols])
        print('building', b, 'done', flush=True)
    out.close()

if __name__ == '__main__':
    main()
