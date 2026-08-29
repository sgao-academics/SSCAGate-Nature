# -*- coding: utf-8 -*-
"""Verify the numbers shown in the regenerated figures match the manuscript text.

This is a PATH-PATCHED copy of the original `_verify_fig_numbers.py` that was
hard-coded to the authors' local data directory. Here it reads the bundled
result files from `replication/data/` (../../data from this script's directory),
so the check runs inside the standalone replication package.
"""
import json, os, re
import numpy as np
from collections import defaultdict

# Resolve the bundled data dir: <pkg>/scripts/fig_generators -> <pkg>/data
base = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data")
)

checks = []
def check(name, got, expect_str, tol=0.02):
    ok = abs(got - float(re.search(r'[-+]?\d+\.?\d*', expect_str).group())) <= tol
    checks.append((name, got, expect_str, ok))

# ---- Fig2a : synthetic F1 gain range ----
cp = json.load(open(os.path.join(base, 'overnight_checkpoint.json'), encoding='utf-8'))
gains = []
for p, n in [(0.2,200),(0.2,300),(0.2,600),(0.4,200),(0.4,300),(0.4,600),
             (0.8,200),(0.8,300),(0.8,600)]:
    cfg = 's0p%g_n%d' % (p, n)
    b = np.mean([cp['grid/%s/seed%d/baseline' % (cfg, s)]['f1'] for s in range(10)])
    pc = np.mean([cp['grid/%s/seed%d/percluster' % (cfg, s)]['f1'] for s in range(10)])
    gains.append(pc - b)
check('Fig2a gain min', min(gains), '+0.083')
check('Fig2a gain max', max(gains), '+0.182')
# Fig2b 16-cancer percluster mean F1
cpC = json.load(open(os.path.join(base, 'overnight_checkpoint_cancers.json'), encoding='utf-8'))
per = defaultdict(list)
for k, v in cpC.items():
    if v['exp'] == 'percluster' and v['method'] == 'percluster':
        per[v['cancer']].append(v['f1'])
check('Fig2b 16-cancer percluster mean', np.mean([np.mean(x) for x in per.values()]), '0.17')

# ---- Fig3a : soft n/d=1 edges 124 f1 0.148 ; baseline 13 / 0.329 ----
d2 = json.load(open(os.path.join(base, 'synth_v2_d50.json'), encoding='utf-8'))
r = d2['results']['1.0']
check('Fig3a soft edges nd=1', r['soft']['edges'], '124')
check('Fig3a soft f1 nd=1', r['soft']['f1'], '0.148')
check('Fig3a base edges nd=1', r['base']['edges'], '13')
check('Fig3a base f1 nd=1', r['base']['f1'], '0.329')

# ---- Fig3b : TCGA soft 92.7 edges prec 0.062 ; base 12.7 / 0.512 ----
d3 = json.load(open(os.path.join(base, 'tcga_implant_BRCA_n200.json'), encoding='utf-8'))['results']
check('Fig3b soft edges', d3['soft']['pred_edges'], '92.7', 0.05)
check('Fig3b soft prec', d3['soft']['precision'], '0.062', 0.005)
check('Fig3b base edges', d3['base']['pred_edges'], '12.7', 0.05)
check('Fig3b base prec', d3['base']['precision'], '0.512', 0.005)

# ---- Fig3c : edge delta at n/d=0.5 -> +131 ; at 16 -> +1 ----
d4 = json.load(open(os.path.join(base, 'spurious_phase_reproduction.json'), encoding='utf-8'))['results']
check('Fig3c edge delta @0.5', d4['0.5']['edge_delta'], '131', 2)
check('Fig3c edge delta @16', d4['16.0']['edge_delta'], '1', 2)

# ---- Fig3d : DAGMA 150/6 f1 0.40->0.07 ; GOLEM 134 f1 0.06 ----
e1 = json.load(open(os.path.join(base, 'e1_checkpoint.json'), encoding='utf-8'))
grp = defaultdict(list)
for k, v in e1.items():
    m = re.match(r'(\w+)\|nd([\d.]+)\|s\d+\|(\w+)', k)
    if m:
        grp[(m.group(1), m.group(2), m.group(3))].append(v)
check('Fig3d DAGMA soft edges', np.mean([x['edges'] for x in grp[('dagma','0.5','soft')]]), '150', 3)
check('Fig3d DAGMA base edges', np.mean([x['edges'] for x in grp[('dagma','0.5','base')]]), '6', 2)
check('Fig3d DAGMA soft f1', np.mean([x['f1'] for x in grp[('dagma','0.5','soft')]]), '0.066', 0.01)
check('Fig3d GOLEM base edges', np.mean([x['edges'] for x in grp[('golem','0.5','base')]]), '134', 3)

# ---- Fig2d : partitioners ----
cmj = json.load(open(os.path.join(base, 'cluster_method_comparison.json'), encoding='utf-8'))['results']
check('Fig2d global f1', cmj['base']['f1'], '0.273', 0.01)
check('Fig2d kmeans f1', cmj['kmeans']['f1'], '0.407', 0.01)
check('Fig2d spectral f1', cmj['spectral']['f1'], '0.283', 0.01)
check('Fig2d oracle f1', cmj['oracle']['f1'], '0.664', 0.01)

print('%-40s %-12s %-14s %s' % ('CHECK', 'GOT', 'EXPECT', 'OK'))
print('-' * 74)
for name, got, expect, ok in checks:
    print('%-40s %-12.3f %-14s %s' % (name, got, expect, 'PASS' if ok else '**FAIL**'))
nbad = sum(1 for *_, ok in checks if not ok)
print('\nTotal: %d checks, %d FAIL' % (len(checks), nbad))
