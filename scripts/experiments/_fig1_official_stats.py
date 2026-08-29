# -*- coding: utf-8 -*-
"""Paired-t statistics for the official L-BFGS-B Fig1 re-run (Table S1 cross-check)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
CP = os.path.join(PKG, 'data', 'fig1_official.json')
d = json.load(open(CP, encoding='utf-8'))
cells = d['cells']

S0P = (0.2, 0.4, 0.8)
NS = (200, 300, 600)


def paired(a, b):
    dv = np.array(a) - np.array(b)
    n = len(dv)
    md = float(dv.mean())
    sd = float(dv.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else float('inf')
    t = md / se if se > 0 else 0.0
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        p = float('nan')
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=md, scale=se) if (n > 1 and se > 0) else (float('nan'), float('nan'))
    cd = md / sd if sd > 0 else float('nan')
    return md, sd, t, p, lo, hi, cd


print('=== OFFICIAL L-BFGS-B d=50 Fig1 (10 seeds) paired t-test ===')
print('%-10s %5s | %-14s %-14s %-12s %10s %8s %8s' % (
    'density', 'n', 'base(mean±sd)', 'per(mean±sd)', 'gain(mean±sd)', 't', 'CI', "Cohen's d"))
print('-' * 96)
all_pos = []
for s0p in S0P:
    for n in NS:
        rows = [cells['c%s_n%d_s%d' % (s0p, n, s)] for s in range(10)]
        base = [r['baseline']['f1'] for r in rows]
        per = [r['percluster']['f1'] for r in rows]
        md, sd, t, p, lo, hi, cd = paired(per, base)
        npos = sum(1 for b, k in zip(base, per) if k > b)
        all_pos.append((npos, s0p, n, md))
        print('%-10s %5d | %.3f±%.3f  %.3f±%.3f  %+.3f±%.3f  %.2f  %s  %.2f (p=%.2g, pos=%d/10)' % (
            s0p, n, np.mean(base), np.std(base), np.mean(per), np.std(per),
            md, sd, t, '[%+.3f, %+.3f]' % (lo, hi), cd, p, npos))

tot = sum(1 for npos, *_ in all_pos if npos >= 5)
print('\nTotal configs with majority-positive gain:', tot, '/', len(all_pos))
print('Mean gain across 9 configs: %+.3f' % np.mean([x[3] for x in all_pos]))
