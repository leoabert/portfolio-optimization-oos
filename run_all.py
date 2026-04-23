#!/usr/bin/env python3
"""
run_all.py - Master Execution Script
=====================================
Runs all thesis analyses in correct order.
Runtime: ~5-10 minutes depending on hardware.
"""

import subprocess
import sys
import os
import time

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ensure output directories exist
os.makedirs('./output/tables', exist_ok=True)
os.makedirs('./output/figures', exist_ok=True)

LOG_FILE = './output/_run_log.txt'

scripts = [
    ("01_build_dataset.py", "Building dataset"),
    ("02_backtest.py", "Running backtests"),
    ("03_regime_comparison.py", "Regime comparison & Table 2"),
    ("04_robustness_regressions.py", "Regressions (DCEQ ~ real rate)"),
    ("05_robustness_bootstrap.py", "Bootstrap, HHI scatter"),
    ("05b_subperiod_bootstrap.py", "Subperiod bootstrap CI"),
    ("05c_sharpe_tests.py", "Sharpe ratio equality tests"),
    ("06_descriptive_stats.py", "Descriptive statistics"),
    ("07_export_table1.py", "Table 1 (full sample)"),
    ("08_robustness_tables.py", "Tables 3-5 (robustness)"),
    ("09_thesis_figures.py", "Figures 1-5 (publication)"),
]

# Force UTF-8 in subprocesses (Windows cp1252 can't handle Unicode symbols)
env = os.environ.copy()
env['PYTHONUTF8'] = '1'

def log(msg, logf):
    """Print and log simultaneously."""
    print(msg, flush=True)
    logf.write(msg + '\n')
    logf.flush()

with open(LOG_FILE, 'w', encoding='utf-8') as logf:
    log("=" * 70, logf)
    log("MASTER EXECUTION: Running All Thesis Scripts", logf)
    log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", logf)
    log("=" * 70, logf)

    failed = []
    passed = []
    timings = {}

    for i, (script, desc) in enumerate(scripts, 1):
        log(f"\n[{i}/{len(scripts)}] {desc} ({script})...", logf)

        if not os.path.exists(script):
            log(f"  SKIPPED: {script} not found", logf)
            continue

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True,
                encoding='utf-8', errors='replace', env=env,
                timeout=900  # 15-minute timeout per script
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            log(f"  TIMEOUT after {elapsed:.0f}s", logf)
            failed.append(script)
            timings[script] = elapsed
            continue
        elapsed = time.time() - t0
        timings[script] = elapsed

        if result.returncode != 0:
            log(f"  FAIL ({elapsed:.1f}s)", logf)
            err_msg = result.stderr[:1000] if result.stderr else "No error message"
            log(f"  ERROR:\n{err_msg}", logf)
            failed.append(script)
        else:
            log(f"  OK ({elapsed:.1f}s)", logf)
            passed.append(script)
            # Log last 5 lines of stdout for key info
            stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
            for line in stdout_lines[-5:]:
                log(f"    {line}", logf)

    log("\n" + "=" * 70, logf)
    log("SUMMARY", logf)
    log("=" * 70, logf)
    log(f"Passed: {len(passed)}/{len(scripts)}", logf)
    for s in passed:
        log(f"  OK  {s} ({timings[s]:.1f}s)", logf)
    if failed:
        log(f"\nFailed: {len(failed)}/{len(scripts)}", logf)
        for s in failed:
            log(f"  FAIL  {s}", logf)
    else:
        log("\nALL ANALYSES COMPLETE - READY FOR THESIS", logf)

    # List output files (including subdirectories)
    output_dir = './output/'
    if os.path.exists(output_dir):
        all_files = []
        for root, dirs, flist in os.walk(output_dir):
            for f in sorted(flist):
                if f.startswith('_'):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, output_dir)
                sz = os.path.getsize(full)
                all_files.append((rel, sz))
        log(f"\nOutput files ({len(all_files)}):", logf)
        for rel, sz in sorted(all_files):
            log(f"  {sz:>8} bytes  {rel}", logf)

    log(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}", logf)
    total_time = sum(timings.values())
    log(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)", logf)
    log("=" * 70, logf)
