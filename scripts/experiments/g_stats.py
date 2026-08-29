"""
Experiment G (post-processing): complete statistical reporting for Table S7.

From the grid per-seed data (overnight_checkpoint.json, 10 seeds x 9 conditions
x 3 methods), compute for the per-cluster gain over baseline:
  - paired t-test: t statistic, degrees of freedom, exact two-sided p
  - 95% confidence interval of the mean gain
  - Cohen's d (paired) effect size

Also emit the per-seed mean + std for Fig1 error bars (base/percluster/oracle).
Output written to g_stats_out.txt (ASCII-safe).
"""
import os, json, math
import numpy as np
from scipy import stats

# Path-patched for the standalone replication package: resolve relative to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, os.pardir, os.pardir))
SRC = os.path.join(PKG, 'data', 'overnight_checkpoint.json')
OUT = os.path.join(HERE, 'g_stats_out.txt')

order = [(0.2, 200), (0.2, 300), (0.2, 600),
         (0.4, 200), (0.4, 300), (0.4, 600),
         (0.8, 200), (0.8, 300), (0.8, 600)]

cp = json.load(open(SRC, encoding='utf-8'))
lines = []

lines.append('=== Table S7 complete statistics (paired t-test, 10 seeds) ===')
lines.append('%-12s %-6s %-14s %-10s %-6s %-14s %-8s' % (
    'condition', 'n', 'gain (m+-sd)', 't', 'df', '95% CI', "Cohen's d"))

fig1 = {}  # config -> {method: (mean, std)}

for (s0p, n) in order:
    cfg = f's0p{s0p}_n{n}'
    base = [cp[f'grid/{cfg}/seed{s}/baseline']['f1'] for s in range(10)]
    pcl = [cp[f'grid/{cfg}/seed{s}/percluster']['f1'] for s in range(10)]
    orc = [cp[f'grid/{cfg}/seed{s}/oracle']['f1'] for s in range(10)]

    diff = [p - b for p, b in zip(pcl, base)]
    m = np.mean(diff)
    sd = np.std(diff, ddof=1)
    t = m / (sd / math.sqrt(10))
    df = 9
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    ci_half = stats.t.ppf(0.975, df) * sd / math.sqrt(10)
    d = m / sd if sd > 0 else float('nan')

    lines.append('%-12s %-6d %+8.3f+-%.3f %8.3f %-6d [%+0.3f, %+0.3f] %8.2f' % (
        cfg, n, m, sd, t, df, m - ci_half, m + ci_half, d))

    fig1[cfg] = {
        'base': (float(np.mean(base)), float(np.std(base, ddof=1))),
        'percluster': (float(np.mean(pcl)), float(np.std(pcl, ddof=1))),
        'oracle': (float(np.mean(orc)), float(np.std(orc, ddof=1))),
    }

lines.append('')
lines.append('=== Fig1 error-bar data (mean, std) ===')
lines.append(json.dumps(fig1, indent=1))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('WROTE', OUT)
print('\n'.join(lines))
