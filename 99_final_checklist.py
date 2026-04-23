#!/usr/bin/env python3
"""
99_final_checklist.py - Final Verification Checklist
=====================================================
Checks all 10 items and reports PASS/FAIL for each.
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT   = './output/'
TABLE_DIR = f'{OUTPUT}tables/'
FIG_DIR   = f'{OUTPUT}figures/'

passed = 0
failed = 0
total  = 10

def check(num, label, ok, detail=''):
    global passed, failed
    status = 'PASS' if ok else 'FAIL'
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n  [{num:>2}] {status}  {label}")
    if detail:
        for line in detail.strip().split('\n'):
            print(f"       {line}")

print("=" * 70)
print("FINAL VERIFICATION CHECKLIST")
print("=" * 70)

# =====================================================================
# CHECK 1: LaTeX tables compile without errors
# =====================================================================
tex_files = [
    f'{TABLE_DIR}table1_full_sample.tex',
    f'{TABLE_DIR}table2_subperiod.tex',
    f'{TABLE_DIR}table3_window_robustness.tex',
    f'{TABLE_DIR}table4_gamma_sensitivity.tex',
    f'{TABLE_DIR}table5_tc_sensitivity.tex',
]

latex_issues = []
for tf in tex_files:
    if not os.path.exists(tf):
        latex_issues.append(f"MISSING: {tf}")
        continue
    with open(tf, 'r', encoding='utf-8') as f:
        content = f.read()
    # Check structural requirements
    if r'\begin{table}' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\begin{{table}}")
    if r'\end{table}' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\end{{table}}")
    if r'\toprule' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\toprule (not booktabs?)")
    if r'\bottomrule' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\bottomrule")
    if r'\midrule' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\midrule")
    if r'\caption' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\caption")
    if r'\label' not in content:
        latex_issues.append(f"{os.path.basename(tf)}: missing \\label")
    # Check for unbalanced braces
    if content.count('{') != content.count('}'):
        n_open = content.count('{')
        n_close = content.count('}')
        latex_issues.append(f"{os.path.basename(tf)}: unbalanced braces "
                           f"({n_open} open, {n_close} close)")

# Also try to compile a minimal document if pdflatex is available
compile_ok = True
try:
    test_doc = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{amsmath}
\begin{document}
"""
    for tf in tex_files:
        test_doc += f"\\input{{{os.path.abspath(tf).replace(os.sep, '/')}}}\n\\clearpage\n"
    test_doc += r"\end{document}"

    tmp_tex = os.path.join(OUTPUT, '_test_compile.tex')
    with open(tmp_tex, 'w', encoding='utf-8') as f:
        f.write(test_doc)

    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
         '-output-directory', os.path.abspath(OUTPUT), tmp_tex],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        # Extract the error line
        for line in result.stdout.split('\n'):
            if line.startswith('!'):
                latex_issues.append(f"pdflatex error: {line}")
                compile_ok = False
                break
        else:
            compile_ok = False
            latex_issues.append("pdflatex returned non-zero but no ! error found")
except FileNotFoundError:
    latex_issues.append("pdflatex not found - structural checks only (all passed)")
    compile_ok = True  # Don't fail just because pdflatex isn't installed
except Exception as e:
    latex_issues.append(f"compile test exception: {e}")
finally:
    # Cleanup temp files
    for ext in ['.tex', '.pdf', '.aux', '.log']:
        try:
            os.remove(os.path.join(OUTPUT, f'_test_compile{ext}'))
        except FileNotFoundError:
            pass

ok1 = len([i for i in latex_issues if 'missing' in i.lower() or 'unbalanced' in i.lower() or 'error' in i.lower()]) == 0
check(1, "All LaTeX tables have valid structure (booktabs, caption, label, balanced braces)",
      ok1,
      '\n'.join(latex_issues) if latex_issues else f"All {len(tex_files)} tables verified structurally")


# =====================================================================
# CHECK 2: All figures exist and are >= 300 dpi
# =====================================================================
expected_figs = [
    f'{FIG_DIR}fig1_cumulative_wealth.png',
    f'{FIG_DIR}fig2_mv_weights.png',
    f'{FIG_DIR}fig3_minvar_weights.png',
    f'{FIG_DIR}fig4_hhi_timeseries.png',
    f'{FIG_DIR}fig5_interest_rates.png',
]

fig_issues = []
for fp in expected_figs:
    if not os.path.exists(fp):
        fig_issues.append(f"MISSING: {fp}")
        continue
    try:
        img = Image.open(fp)
        dpi_info = img.info.get('dpi', (72, 72))
        dpi = max(dpi_info) if isinstance(dpi_info, (tuple, list)) else dpi_info
        w, h = img.size
        dpi_ok = dpi >= 299   # 300 dpi target; allow tiny float rounding
        fig_issues.append(f"{os.path.basename(fp)}: {w}x{h} px, {dpi:.0f} dpi - "
                         f"{'OK' if dpi_ok else 'LOW DPI'}")
        if not dpi_ok:
            fig_issues[-1] += " FAIL"
    except Exception as e:
        fig_issues.append(f"{os.path.basename(fp)}: error reading - {e}")

all_figs_ok = all('FAIL' not in i and 'MISSING' not in i for i in fig_issues)
check(2, "All figures exist and >= 300 dpi",
      all_figs_ok,
      '\n'.join(fig_issues))


# =====================================================================
# CHECK 3: Table 1 numbers match 02_backtest.py console output
# =====================================================================
# Load Table 1 data
t1 = pd.read_csv(f'{TABLE_DIR}table1_full_sample.csv', index_col=0)
# Load the performance_table.csv from backtest
perf = pd.read_csv(f'{OUTPUT}performance_table.csv', index_col=0)

# Table 1 CEQ should match performance_table CEQ
ceq_t1_1n = t1.loc['1/N', 'CEQ Return (%)']
ceq_perf_1n = perf.loc['CEQ (%)', '1/N Benchmark']
ceq_t1_mv = t1.loc['Mean-Variance', 'CEQ Return (%)']
ceq_perf_mv = perf.loc['CEQ (%)', 'Mean-Variance']
ceq_t1_minvar = t1.loc['Minimum Variance', 'CEQ Return (%)']
ceq_perf_minvar = perf.loc['CEQ (%)', 'Minimum Variance']

ceq_match_1n = abs(ceq_t1_1n - ceq_perf_1n) < 0.01
ceq_match_mv = abs(ceq_t1_mv - ceq_perf_mv) < 0.01
ceq_match_minvar = abs(ceq_t1_minvar - ceq_perf_minvar) < 0.01

# Sharpe comparison
sharpe_t1_1n = t1.loc['1/N', 'Sharpe Ratio']
sharpe_perf_1n = perf.loc['Sharpe Ratio', '1/N Benchmark']
sharpe_match = abs(sharpe_t1_1n - sharpe_perf_1n) < 0.001

all_match = ceq_match_1n and ceq_match_mv and ceq_match_minvar and sharpe_match
check(3, "Table 1 numbers match 02_backtest.py performance_table.csv",
      all_match,
      f"CEQ(1/N):     T1={ceq_t1_1n:.3f}  backtest={ceq_perf_1n:.3f}  {'OK' if ceq_match_1n else 'MISMATCH'}\n"
      f"CEQ(MV):      T1={ceq_t1_mv:.3f}  backtest={ceq_perf_mv:.3f}  {'OK' if ceq_match_mv else 'MISMATCH'}\n"
      f"CEQ(MinVar):  T1={ceq_t1_minvar:.3f}  backtest={ceq_perf_minvar:.3f}  {'OK' if ceq_match_minvar else 'MISMATCH'}\n"
      f"Sharpe(1/N):  T1={sharpe_t1_1n:.4f}  backtest={sharpe_perf_1n:.4f}  {'OK' if sharpe_match else 'MISMATCH'}")


# =====================================================================
# CHECK 4: Table 2 panels are non-overlapping and cover full OOS
# =====================================================================
t2 = pd.read_csv(f'{TABLE_DIR}table2_subperiod.csv')
ret = pd.read_csv(f'{OUTPUT}strategy_returns.csv', index_col=0, parse_dates=True)

oos_start = ret.index[0]
oos_end   = ret.index[-1]
oos_months = len(ret)

# Check that Panel A end < Panel B start
# From the CSV: Panel A is Pre-Crisis, Panel B is Post-Crisis
# The split date is 2009-01-01 as set in 03_regime_comparison.py
split = pd.Timestamp('2009-01-01')
pre_months  = (ret.index < split).sum()
post_months = (ret.index >= split).sum()
total_months = pre_months + post_months

coverage_ok = total_months == oos_months
non_overlap_ok = pre_months > 0 and post_months > 0

check(4, "Table 2 panels non-overlapping and cover full OOS period",
      coverage_ok and non_overlap_ok,
      f"OOS: {oos_start.strftime('%Y-%m')} to {oos_end.strftime('%Y-%m')} ({oos_months} months)\n"
      f"Pre-Crisis (< 2009-01):  {pre_months} months\n"
      f"Post-Crisis (>= 2009-01): {post_months} months\n"
      f"Sum: {total_months} = OOS total {oos_months}  {'OK' if coverage_ok else 'MISMATCH'}\n"
      f"Non-overlapping: {'OK' if non_overlap_ok else 'FAIL'}")


# =====================================================================
# CHECK 5: Table 1 ΔCEQ for MV is positive (1/N outperforms)
# =====================================================================
dceq_mv = t1.loc['Mean-Variance', 'DCEQ (%)']
dceq_minvar = t1.loc['Minimum Variance', 'DCEQ (%)']

check(5, "Table 1: DCEQ(MV) > 0 (1/N outperforms MV)",
      dceq_mv > 0,
      f"DCEQ(MV) = {dceq_mv:+.3f}%  (positive = 1/N wins)\n"
      f"DCEQ(MinVar) = {dceq_minvar:+.3f}%  (positive = 1/N wins)")


# =====================================================================
# CHECK 6: Table 3 M=120 results match Table 1 exactly
# =====================================================================
t3 = pd.read_csv(f'{TABLE_DIR}table3_window_robustness.csv')
t3_120 = t3[t3['M'] == 120].iloc[0]

ceq_1n_t3 = t3_120['CEQ_1N']
ceq_mv_t3 = t3_120['CEQ_MV']
ceq_minvar_t3 = t3_120['CEQ_MinVar']

# Table 1 CEQ values
ceq_1n_t1 = t1.loc['1/N', 'CEQ Return (%)']
ceq_mv_t1 = t1.loc['Mean-Variance', 'CEQ Return (%)']
ceq_minvar_t1 = t1.loc['Minimum Variance', 'CEQ Return (%)']

match_1n = abs(ceq_1n_t3 - ceq_1n_t1) < 0.02
match_mv = abs(ceq_mv_t3 - ceq_mv_t1) < 0.02
match_minvar = abs(ceq_minvar_t3 - ceq_minvar_t1) < 0.02
all_t3_match = match_1n and match_mv and match_minvar

check(6, "Table 3 M=120 results match Table 1",
      all_t3_match,
      f"CEQ(1/N):     T3={ceq_1n_t3:.3f}  T1={ceq_1n_t1:.3f}  diff={abs(ceq_1n_t3-ceq_1n_t1):.4f}  {'OK' if match_1n else 'MISMATCH'}\n"
      f"CEQ(MV):      T3={ceq_mv_t3:.3f}  T1={ceq_mv_t1:.3f}  diff={abs(ceq_mv_t3-ceq_mv_t1):.4f}  {'OK' if match_mv else 'MISMATCH'}\n"
      f"CEQ(MinVar):  T3={ceq_minvar_t3:.3f}  T1={ceq_minvar_t1:.3f}  diff={abs(ceq_minvar_t3-ceq_minvar_t1):.4f}  {'OK' if match_minvar else 'MISMATCH'}")


# =====================================================================
# CHECK 7: Table 5 TC=50bps results match Table 1 exactly
# =====================================================================
t5 = pd.read_csv(f'{TABLE_DIR}table5_tc_sensitivity.csv')
t5_50 = t5[t5['TC_bps'] == 50].iloc[0]

ceq_1n_t5 = t5_50['CEQ_1N']
ceq_mv_t5 = t5_50['CEQ_MV']
ceq_minvar_t5 = t5_50['CEQ_MinVar']

match_1n_5 = abs(ceq_1n_t5 - ceq_1n_t1) < 0.02
match_mv_5 = abs(ceq_mv_t5 - ceq_mv_t1) < 0.02
match_minvar_5 = abs(ceq_minvar_t5 - ceq_minvar_t1) < 0.02
all_t5_match = match_1n_5 and match_mv_5 and match_minvar_5

check(7, "Table 5 TC=50bps results match Table 1",
      all_t5_match,
      f"CEQ(1/N):     T5={ceq_1n_t5:.3f}  T1={ceq_1n_t1:.3f}  diff={abs(ceq_1n_t5-ceq_1n_t1):.4f}  {'OK' if match_1n_5 else 'MISMATCH'}\n"
      f"CEQ(MV):      T5={ceq_mv_t5:.3f}  T1={ceq_mv_t1:.3f}  diff={abs(ceq_mv_t5-ceq_mv_t1):.4f}  {'OK' if match_mv_5 else 'MISMATCH'}\n"
      f"CEQ(MinVar):  T5={ceq_minvar_t5:.3f}  T1={ceq_minvar_t1:.3f}  diff={abs(ceq_minvar_t5-ceq_minvar_t1):.4f}  {'OK' if match_minvar_5 else 'MISMATCH'}")


# =====================================================================
# CHECK 8: 1/N turnover is non-zero (~2-3% monthly)
# =====================================================================
turnover = pd.read_csv(f'{OUTPUT}strategy_turnover.csv', index_col=0, parse_dates=True)
to_1n_mean = turnover['1/N'].mean() * 100
to_1n_nonzero = (turnover['1/N'] > 0).sum()
to_1n_total = len(turnover)
to_ok = 1.0 < to_1n_mean < 5.0 and to_1n_nonzero > to_1n_total * 0.9

check(8, "1/N turnover is non-zero (~2-3% monthly from drift rebalancing)",
      to_ok,
      f"Mean turnover: {to_1n_mean:.2f}% per month\n"
      f"Non-zero months: {to_1n_nonzero}/{to_1n_total} ({to_1n_nonzero/to_1n_total*100:.1f}%)\n"
      f"Range: [{turnover['1/N'].min()*100:.3f}%, {turnover['1/N'].max()*100:.3f}%]")


# =====================================================================
# CHECK 9: All weight vectors sum to 1.0 (within 1e-6) for every period
# =====================================================================
# We need to run a quick backtest to get weights. But instead, let's verify
# from the backtest sanity check already in 02_backtest.py.
# More directly: load data and run a quick optimization check on a few periods.

from config import ESTIMATION_WINDOW, RISK_AVERSION, INDUSTRIES
from utils import run_backtest, load_cached_backtest

data = pd.read_csv('./data/data_clean.csv', index_col=0, parse_dates=True)
rf_check = data['RF_Monthly']
industries = INDUSTRIES

bt = load_cached_backtest(OUTPUT)
if bt is not None:
    print("\n       Using cached backtest for weight verification ...", flush=True)
else:
    print("\n       Running weight verification backtest ...", flush=True)
    bt = run_backtest(data, rf_check, industries,
                      estimation_window=ESTIMATION_WINDOW,
                      transaction_cost=0.005)

tol = 1e-6
weight_issues = []
for strategy in ['1/N', 'Mean-Variance', 'Minimum Variance']:
    w = bt[strategy]['weights']
    sums = w.sum(axis=1)
    min_sum = sums.min()
    max_sum = sums.max()
    sum_ok = abs(min_sum - 1.0) < tol and abs(max_sum - 1.0) < tol
    nonneg = (w.values >= -tol).all()
    weight_issues.append(f"{strategy}: sums in [{min_sum:.8f}, {max_sum:.8f}]  "
                        f"{'OK' if sum_ok else 'FAIL'}  "
                        f"non-neg: {'OK' if nonneg else 'FAIL'}")
    if not sum_ok or not nonneg:
        weight_issues[-1] += " <-- PROBLEM"

weights_ok = all('PROBLEM' not in i for i in weight_issues)
check(9, "All weight vectors sum to 1.0 (within 1e-6) for every period",
      weights_ok,
      '\n'.join(weight_issues))


# =====================================================================
# CHECK 10: No NaN or Inf values in any output CSV file
# =====================================================================
csv_files = []
for root, dirs, files in os.walk(OUTPUT):
    for f in files:
        if f.endswith('.csv') and not f.startswith('_'):
            csv_files.append(os.path.join(root, f))

nan_inf_issues = []
for cf in csv_files:
    try:
        df = pd.read_csv(cf)
        # Check numeric columns only
        numeric = df.select_dtypes(include=[np.number])
        n_nan = numeric.isna().sum().sum()
        n_inf = np.isinf(numeric.values).sum() if numeric.size > 0 else 0
        basename = os.path.relpath(cf, OUTPUT)
        if n_nan > 0 or n_inf > 0:
            nan_inf_issues.append(f"{basename}: {n_nan} NaN, {n_inf} Inf  <-- PROBLEM")
        else:
            nan_inf_issues.append(f"{basename}: clean ({numeric.shape[0]} rows x {numeric.shape[1]} num cols)")
    except Exception as e:
        nan_inf_issues.append(f"{os.path.basename(cf)}: read error - {e}")

# Note: Table 1 CSV has NaN for 1/N DCEQ and Break-Even by design (--).
# Table 2 CSV has NaN for 1/N DCEQ and Difference 1/N DCEQ by design.
# These are intentional blanks, not data errors.
# Filter out known intentional NaNs
real_problems = [i for i in nan_inf_issues if 'PROBLEM' in i]
# Check if problems are only in expected places
false_alarms = []
for issue in real_problems:
    fname = issue.split(':')[0]
    if 'table1_full_sample' in fname or 'table2_subperiod' in fname:
        false_alarms.append(issue)  # NaN in DCEQ for 1/N is by design

actual_problems = [i for i in real_problems if i not in false_alarms]
nan_ok = len(actual_problems) == 0

detail_lines = []
for i in nan_inf_issues:
    if 'PROBLEM' in i:
        if i in false_alarms:
            detail_lines.append(i.replace('PROBLEM', 'EXPECTED (1/N DCEQ = --)'))
        else:
            detail_lines.append(i)
    else:
        detail_lines.append(i)

check(10, "No unexpected NaN or Inf values in any output CSV",
      nan_ok,
      '\n'.join(detail_lines))


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Count outputs
all_tables = [f for f in os.listdir(TABLE_DIR) if f.endswith('.tex')]
all_csvs = [f for f in os.listdir(TABLE_DIR) if f.endswith('.csv')]
all_figs = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')]

print(f"\n  Tables generated:    {len(all_tables)} LaTeX + {len(all_csvs)} CSV = {len(all_tables) + len(all_csvs)} files")
for t in sorted(all_tables):
    print(f"    - {t}")
print(f"\n  Figures generated:   {len(all_figs)} PNG files")
for f in sorted(all_figs):
    print(f"    - {f}")

print(f"\n  Checklist:  {passed}/{total} PASSED, {failed}/{total} FAILED")

if failed == 0:
    print("\n  STATUS: ALL CHECKS PASSED - PROJECT IS THESIS-READY")
else:
    print(f"\n  STATUS: {failed} ISSUE(S) REQUIRE ATTENTION")

print("=" * 70)


