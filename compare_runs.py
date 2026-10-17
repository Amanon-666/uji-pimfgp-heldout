"""run/ 与 run2/ 逐行比较（只读）。用法：python compare_runs.py <run_dir> <run2_dir>"""
import csv, gzip, sys
import numpy as np
A, B = sys.argv[1], sys.argv[2]
for b in range(3):
    ra = list(csv.reader(gzip.open(f'{A}/perq_b{b}.csv.gz', 'rt'))); rb = list(csv.reader(gzip.open(f'{B}/perq_b{b}.csv.gz', 'rt')))
    ha, hb = ra[0], rb[0]
    assert len(ra) == len(rb), (b, len(ra), len(rb))
    ia = {c: i for i, c in enumerate(ha)}; ib = {c: i for i, c in enumerate(hb)}
    common = [c for c in ha if c in ib and c != 'd_min']
    nd = {c: 0 for c in common}; maxnum = 0.0
    for x, y in zip(ra[1:], rb[1:]):
        for c in common:
            if x[ia[c]] != y[ib[c]]:
                nd[c] += 1
                try: maxnum = max(maxnum, abs(float(x[ia[c]]) - float(y[ib[c]])))
                except ValueError: pass
    dmin_a = np.array([float(x[ia['d_min']]) for x in ra[1:]]); dmin_b = np.array([float(y[ib['d_min']]) for y in rb[1:]])
    far_b = np.array([int(y[ib['is_far']]) for y in rb[1:]]).astype(bool)
    print(f'b{b}: rows={len(ra)-1} cols_compared={len(common)} cols_with_diff={ {c: n for c, n in nd.items() if n} } max_abs_numeric_diff={maxnum}')
    print(f'    d_min: max|round(run2,2)-run|={np.abs(np.round(dmin_b, 2) - dmin_a).max():.4g}; '
          f'far(run, rounded>=20) vs is_far(run2) mismatches={int(((dmin_a >= 20) != far_b).sum())}')
    za, zb = np.load(f'{A}/null_b{b}.npz'), np.load(f'{B}/null_b{b}.npz')
    for k in sorted(set(za.files) - {'meta'}):
        if k.startswith('sum_all'):
            print(f'    null {k}: max|diff|={np.abs(za[k] - zb[k]).max():.3g}')
    fa = [k for k in za.files if k.startswith('sum_far')]
    print('    null sum_far max|diff|:', {k: float(np.abs(za[k] - zb[k]).max()) for k in fa})
