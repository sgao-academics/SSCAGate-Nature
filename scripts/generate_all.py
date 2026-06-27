#!/usr/bin/env python3
"""
Master reproduction script for SSCAGate Nature paper.
Usage:
    python scripts/generate_all.py --quick       # Regenerate figures from pre-computed data (~5 min)
    python scripts/generate_all.py --quick --verify  # Verify all tables against data
    python scripts/generate_all.py --full        # Re-run experiments + figures (~2-4h, GPU required)

Requirements: pip install torch numpy pandas matplotlib scipy scikit-learn pillow
"""
import json, os, sys, argparse, time, warnings, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PKG_ROOT, 'results')
MAIN_FIG_DIR = os.path.join(PKG_ROOT, 'main_figure')
ED_FIG_DIR = os.path.join(PKG_ROOT, 'ED_figure')
SCRIPTS_DIR = os.path.join(PKG_ROOT, 'scripts')

os.makedirs(MAIN_FIG_DIR, exist_ok=True)
os.makedirs(ED_FIG_DIR, exist_ok=True)

EXPECTED_STATS = {
    'cross_cancer_spearman_r': 0.827,
    'within_cancer_spearman_r': 0.793,
    'scrna_spearman_r': 0.820,
    'critical_n_exponent': 0.902,
    'critical_n_r2': 0.985,
}

def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def verify_results():
    """Check pre-computed results against expected statistics."""
    print('='*60)
    print('Verifying pre-computed results')
    print('='*60)
    all_pass = True

    # 1. Cross-cancer phase from mega_33_full.json
    mega = load_json('mega_33_full.json')
    if mega:
        n_values, v3_values = [], []
        for cancer, data in mega.items():
            if isinstance(data, dict) and 'n' in data and 'v3_advantage' in data:
                n_values.append(data['n'])
                v3_values.append(data['v3_advantage'])
        if len(n_values) >= 10:
            from scipy.stats import spearmanr
            r, p = spearmanr(n_values, v3_values)
            expected = EXPECTED_STATS['cross_cancer_spearman_r']
            status = '[PASS]' if abs(r - expected) < 0.02 else '[FAIL]'
            print(f'  Cross-cancer r={r:.3f} (expected {expected}) {status}')
            if abs(r - expected) >= 0.02:
                all_pass = False

    # 2. Within-cancer from within_cancer_resample.json (dict with spearman_r key)
    within = load_json('within_cancer_resample.json')
    if within and isinstance(within, dict) and 'spearman_r' in within:
        r = within['spearman_r']
        expected = EXPECTED_STATS['within_cancer_spearman_r']
        status = '[PASS]' if abs(r - expected) < 0.02 else '[FAIL]'
        print(f'  Within-cancer r={r:.3f} (expected {expected}) {status}')
        if abs(r - expected) >= 0.02:
            all_pass = False

    # 3. scRNA
    scrna = load_json('scrna_phase_summary.json')
    if scrna and isinstance(scrna, dict) and 'r' in scrna:
        r = scrna['r']
        expected = EXPECTED_STATS['scrna_spearman_r']
        status = '[PASS]' if abs(r - expected) < 0.02 else '[FAIL]'
        print(f'  scRNA r={r:.3f} (expected {expected}) {status}')
        if abs(r - expected) >= 0.02:
            all_pass = False

    # 4. Critical n
    crit = load_json('critical_n_fit.json')
    if crit:
        llf = crit.get('loglog_fit', {})
        coef = float(llf.get('alpha', llf.get('α', llf.get('coef', 0))))
        exp_val = float(llf.get('beta', llf.get('β', llf.get('exp', 0))))
        r2 = float(llf.get('R2', 0))
        print(f'  n_crit = {coef:.3f} * d^{exp_val:.3f}, R2={r2:.3f}')
        exp_expected = EXPECTED_STATS['critical_n_exponent']
        r2_expected = EXPECTED_STATS['critical_n_r2']
        exp_ok = abs(exp_val - exp_expected) < 0.02
        r2_ok = abs(r2 - r2_expected) < 0.02
        status = '[PASS]' if (exp_ok and r2_ok) else '[FAIL]'
        print(f'  Expected exponent={exp_expected}, R2={r2_expected} {status}')
        if not (exp_ok and r2_ok):
            all_pass = False

    print(f'\nOverall: {"[PASS] ALL CHECKS PASSED" if all_pass else "[FAIL] SOME CHECKS FAILED"}')
    return all_pass

def generate_figures():
    """Generate all figures by calling gen scripts."""
    print('\n' + '='*60)
    print('Generating figures from pre-computed data')
    print('='*60)

    scripts = [
        ('gen_fig1.py', 'Fig1 composite (5 panels)'),
        ('gen_fig2.py', 'Fig2 universal scaling'),
        ('gen_ed_fig1.py', 'ED Figure 1: K-sweep'),
        ('gen_ed_fig2.py', 'ED Figure 2: Phase transition + robustness'),
        ('gen_ed_fig3.py', 'ED Figure 3: Cross-method validation'),
        ('gen_ed_fig4.py', 'ED Figure 4: Hyperparameter sensitivity'),
        ('gen_graphical_abstract.py', 'Graphical Abstract'),
    ]

    for script, desc in scripts:
        path = os.path.join(SCRIPTS_DIR, script)
        if os.path.exists(path):
            print(f'\n--- {desc} ---')
            r = subprocess.run([sys.executable, path], cwd=PKG_ROOT,
                             capture_output=True, text=True)
            if r.stdout:
                print(r.stdout[-500:])
            if r.returncode != 0:
                print(f'[WARN] Non-zero exit: {r.returncode}')
                if r.stderr:
                    print(r.stderr[-300:])
        else:
            print(f'[SKIP] {script} not found')

    print(f'\nFigures saved in: {MAIN_FIG_DIR}/  and  {ED_FIG_DIR}/')

def run_experiments():
    """Re-run experiments with real NOTEARS (GPU recommended)."""
    print('='*60)
    print('Full experiment mode: this will take 2-4 hours')
    print('='*60)
    print('Step 1: Download TCGA data (if needed)')
    print('  python scripts/download_tcga.py')

    print('\nAvailable experiment scripts:')
    experiments = [
        '_ksweep.py', '_burn_b100.py', '_resample_phase.py',
        'run_hyperparam_sweep.py', '_dagma_phase_test.py', '_bench_dagma_ed.py',
    ]
    for exp in experiments:
        path = os.path.join(SCRIPTS_DIR, exp)
        exists = os.path.exists(path)
        print(f'  [{'EXISTS' if exists else 'MISSING'}] {exp}')

    print('\nRun individual experiments as needed, then call --quick to regenerate figures.')

def main():
    parser = argparse.ArgumentParser(description='SSCAGate Nature paper reproduction')
    parser.add_argument('--quick', action='store_true', help='Regenerate figures from pre-computed data')
    parser.add_argument('--verify', action='store_true', help='Verify tables against data sources')
    parser.add_argument('--full', action='store_true', help='Re-run all experiments (requires GPU + TCGA data)')
    args = parser.parse_args()

    t0 = time.time()

    if args.full:
        print('Full reproduction mode: running experiments...')
        print('First download TCGA data: python scripts/download_tcga.py')
        run_experiments()

    if args.verify:
        ok = verify_results()
        if not ok:
            print('\n[WARN] Some results deviate from expected values!')

    if args.quick or args.full:
        generate_figures()

    if not (args.quick or args.verify or args.full):
        parser.print_help()
        print('\nQuick start: python scripts/generate_all.py --quick --verify')

    print(f'\nTotal time: {time.time() - t0:.1f}s')

if __name__ == '__main__':
    main()
