#!/usr/bin/env python3
"""
07_export_table1.py - Publication-Quality LaTeX Table Export
==============================================================
Produces Table 1 (full-sample out-of-sample performance) for the thesis.

Inputs:  output/strategy_returns.csv, output/strategy_turnover.csv, data/data_clean.csv
Outputs: output/tables/table1_full_sample.csv, output/tables/table1_full_sample.tex
"""

import os
import numpy as np
import pandas as pd
from config import (OUTPUT_DIR, ESTIMATION_WINDOW, RISK_AVERSION,
                    TRANSACTION_COST)

print("=" * 70)
print("GENERATING PUBLICATION-QUALITY LaTeX TABLE 1")
print("=" * 70)

# =====================================================================
# 1. Load backtest outputs
# =====================================================================
returns = pd.read_csv(f'{OUTPUT_DIR}strategy_returns.csv',
                      index_col=0, parse_dates=True)
turnover = pd.read_csv(f'{OUTPUT_DIR}strategy_turnover.csv',
                       index_col=0, parse_dates=True)
data = pd.read_csv('./data/data_clean.csv',
                    index_col=0, parse_dates=True)
rf = data['RF_Monthly'].reindex(returns.index)

date_start = returns.index[0].strftime('%Y-%m')
date_end = returns.index[-1].strftime('%Y-%m')
n_months = len(returns)

print(f"  Period: {date_start} to {date_end} ({n_months} months)")

# =====================================================================
# 2. Compute all metrics from scratch (canonical source of truth)
# =====================================================================
strategies = ['1/N', 'Mean-Variance', 'Minimum Variance']
rows = []

for s in strategies:
    r = returns[s]
    t = turnover[s]
    excess = r - rf

    mean_excess_ann = excess.mean() * 12 * 100
    vol_ann = excess.std() * np.sqrt(12) * 100
    std_val = excess.std()
    sharpe_ann = (excess.mean() / std_val) * np.sqrt(12) if std_val > 0 else 0.0
    ceq_ann = (excess.mean() - (RISK_AVERSION / 2) * excess.var()) * 12 * 100
    turnover_monthly = t.mean() * 100  # already a fraction, convert to %

    rows.append({
        'Strategy': s,
        'mean_excess': mean_excess_ann,
        'vol': vol_ann,
        'sharpe': sharpe_ann,
        'ceq': ceq_ann,
        'turnover': turnover_monthly,
        '_ceq_raw': ceq_ann,           # keep for ΔCEQ calc
    })

df = pd.DataFrame(rows).set_index('Strategy')

# Welfare cost ΔCEQ = CEQ(1/N) - CEQ(strategy)
ceq_1n = df.loc['1/N', '_ceq_raw']
df['dceq'] = ceq_1n - df['_ceq_raw']
df.loc['1/N', 'dceq'] = np.nan  # blank for benchmark row

# =====================================================================
# 3. Build the CSV
# =====================================================================
table_dir = f'{OUTPUT_DIR}tables/'
os.makedirs(table_dir, exist_ok=True)

csv_df = df[['mean_excess', 'vol', 'sharpe', 'ceq', 'dceq',
             'turnover']].copy()
csv_df.columns = [
    'Mean Excess Return (%)',
    'Volatility (%)',
    'Sharpe Ratio',
    'CEQ Return (%)',
    'DCEQ (%)',
    'Turnover (%)',
]

csv_path = f'{table_dir}table1_full_sample.csv'
csv_df.to_csv(csv_path)
print(f"  CSV saved: {csv_path}")

# =====================================================================
# 4. Build the LaTeX by hand (full control over formatting)
# =====================================================================

def fmt2(v):
    """Format to 2 decimal places."""
    return f'{v:.2f}'

def fmt3(v):
    """Format to 3 decimal places."""
    return f'{v:.3f}'

def fmt0(v):
    """Format to 0 decimal places (for bps)."""
    return f'{v:.0f}'

def fmtcell(v, formatter, blank_nan=True):
    """Format a cell; return '--' for NaN."""
    if blank_nan and (v is None or (isinstance(v, float) and np.isnan(v))):
        return '--'
    return formatter(v)

# Year strings for caption
yr_start = returns.index[0].strftime('%Y')
yr_end = returns.index[-1].strftime('%Y')
mo_start = returns.index[0].strftime('%B %Y')
mo_end = returns.index[-1].strftime('%B %Y')

latex_lines = []
latex_lines.append(r'\begin{table}[t]')
latex_lines.append(r'  \centering')
latex_lines.append(r'  \caption[Out-of-sample performance of portfolio strategies]'
                   r'{Out-of-sample performance of portfolio strategies. '
                   f'The sample covers {mo_start} to {mo_end} '
                   r'using Fama--French 10 Industry Portfolios with a rolling '
                   f'estimation window of $M={ESTIMATION_WINDOW}$ months and '
                   r'risk aversion $\gamma=' + str(RISK_AVERSION) + r'$.}')
latex_lines.append(r'  \label{tab:full_sample}')
latex_lines.append(r'  \small')
latex_lines.append(r'  \resizebox{\textwidth}{!}{%')
latex_lines.append(r'  \begin{tabular}{l *{6}{r}}')
latex_lines.append(r'    \toprule')
latex_lines.append(r'    & \multicolumn{1}{c}{Mean Exc.}')
latex_lines.append(r'    & \multicolumn{1}{c}{Volatility}')
latex_lines.append(r'    & \multicolumn{1}{c}{Sharpe}')
latex_lines.append(r'    & \multicolumn{1}{c}{CEQ}')
latex_lines.append(r'    & \multicolumn{1}{c}{$\Delta$CEQ}')
latex_lines.append(r'    & \multicolumn{1}{c}{Turnover} \\')
latex_lines.append(r'    Strategy')
latex_lines.append(r'    & \multicolumn{1}{c}{Ret.\ (\%)}')
latex_lines.append(r'    & \multicolumn{1}{c}{(\%)}')
latex_lines.append(r'    & \multicolumn{1}{c}{Ratio}')
latex_lines.append(r'    & \multicolumn{1}{c}{(\%)}')
latex_lines.append(r'    & \multicolumn{1}{c}{(\%)}')
latex_lines.append(r'    & \multicolumn{1}{c}{(\%)} \\')
latex_lines.append(r'    \midrule')

# Display names for strategies
display = {
    '1/N': r'$1/N$',
    'Mean-Variance': 'Mean--Variance',
    'Minimum Variance': 'Minimum--Variance',
}

for s in strategies:
    row = df.loc[s]
    cells = [
        display[s],
        fmt2(row['mean_excess']),
        fmt2(row['vol']),
        fmt3(row['sharpe']),
        fmt2(row['ceq']),
        fmtcell(row['dceq'], fmt2),
        fmt2(row['turnover']),
    ]
    latex_lines.append('    ' + ' & '.join(cells) + r' \\')

latex_lines.append(r'    \bottomrule')
latex_lines.append(r'  \end{tabular}%')
latex_lines.append(r'  }')
latex_lines.append('')
latex_lines.append(r'  \smallskip')
latex_lines.append(r'  \begin{minipage}{\textwidth}')
latex_lines.append(r'    \footnotesize')
latex_lines.append(r'    \textit{Notes.} '
                   r'All values are annualised except turnover (monthly average). '
                   r'Mean excess return, volatility, and CEQ are computed on '
                   r'excess returns $r_{p,t} - r_{f,t}$. '
                   r'Welfare cost is $\Delta\text{CEQ} = '
                   r'\text{CEQ}_{1/N} - \text{CEQ}_{\text{strategy}}$; '
                   r'positive values indicate $1/N$ outperformance. '
                   f'Baseline TC = {TRANSACTION_COST * 10000:.0f}\\,bps.')
latex_lines.append(r"    \textit{Source:} Author's calculations based on data from"
                   r" Kenneth R.\ French's Data Library.")
latex_lines.append(r'  \end{minipage}')
latex_lines.append(r'\end{table}')

latex_code = '\n'.join(latex_lines) + '\n'

latex_code = latex_code.replace('\xa0', ' ')
tex_path = f'{table_dir}table1_full_sample.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(latex_code)
print(f"  TeX saved: {tex_path}")
# LaTeX export (disabled for portable submission):
# sync_to_latex(tex_path, LATEX_TABLES_DIR)

# Also keep the old-format table for backward compatibility
old_perf = pd.read_csv(f'{OUTPUT_DIR}performance_table.csv', index_col=0)
# Escape %, _, & in column/index names for LaTeX safety
def _latex_escape_name(s):
    return s.replace('%', r'\%').replace('_', r'\_').replace('&', r'\&')
old_perf.index = [_latex_escape_name(i) for i in old_perf.index]
old_perf.columns = [_latex_escape_name(c) for c in old_perf.columns]
old_latex = old_perf.T.to_latex(
    float_format="%.3f",
    caption=f"Performance Comparison: Net Returns After "
            f"{TRANSACTION_COST*10000:.0f}bps Transaction Costs "
            f"({yr_start}--{yr_end})",
    label="tab:performance",
    column_format="l" + "r" * len(old_perf.index),
    escape=False
)
old_latex = old_latex.replace('\xa0', ' ')
with open(f'{OUTPUT_DIR}performance_table.tex', 'w', encoding='utf-8') as f:
    f.write(old_latex)
print(f"  Legacy table updated: {OUTPUT_DIR}performance_table.tex")

# =====================================================================
# 5. Print preview
# =====================================================================
print("\n" + "=" * 70)
print("TABLE 1 PREVIEW (LaTeX)")
print("=" * 70)
print(latex_code)

print("=" * 70)
print("TABLE 1 DATA")
print("=" * 70)
print(csv_df.round(3).to_string())

print("\n" + "=" * 70)
print("LaTeX TABLE EXPORT COMPLETE")
print("=" * 70)
