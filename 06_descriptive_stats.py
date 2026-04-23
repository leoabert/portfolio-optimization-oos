"""
06_descriptive_stats.py - Descriptive Statistics
==================================================
Computes annualized return, volatility, and Sharpe ratio for each of the
10 Fama-French Industry Portfolios, split by pre-/post-crisis subperiods.

Inputs:  data/data_clean.csv
Outputs: output/table1_descriptive_stats.tex
"""

import pandas as pd
import numpy as np
from config import INDUSTRIES, SPLIT_DATE, OUTPUT_DIR

print("="*70)
print("DESCRIPTIVE STATISTICS: Industry Returns & Regime Characteristics")
print("="*70)

# Load data
data = pd.read_csv('./data/data_clean.csv', index_col=0, parse_dates=True)
industries = INDUSTRIES
rf = data['RF_Monthly']

# Full sample
print("\n--- FULL SAMPLE (1963-08 to 2024-12) ---")
full_stats = data[industries].describe().T
full_stats['Mean (Ann %)'] = full_stats['mean'] * 12 * 100
full_stats['Std (Ann %)'] = full_stats['std'] * np.sqrt(12) * 100
# Sharpe: use excess return std (proper definition)
excess_full = data[industries].subtract(rf, axis=0)
full_stats['Sharpe'] = (excess_full.mean() / excess_full.std()) * np.sqrt(12)

table1_full = full_stats[['Mean (Ann %)', 'Std (Ann %)', 'Sharpe', 'min', 'max']]
print(table1_full.round(2))

# By regime
pre = data.index < SPLIT_DATE
post = data.index >= SPLIT_DATE

_split = pd.Timestamp(SPLIT_DATE)
_pre_end = (_split - pd.DateOffset(months=1)).strftime('%Y-%m')
_post_start = _split.strftime('%Y-%m')
_data_end = data.index[-1].strftime('%Y-%m')

print(f"\n--- PRE-CRISIS (1963-08 to {_pre_end}) ---")
pre_data = data.loc[pre, industries]
pre_stats = pre_data.describe().T
pre_stats['Mean (Ann %)'] = pre_stats['mean'] * 12 * 100
pre_stats['Std (Ann %)'] = pre_stats['std'] * np.sqrt(12) * 100
pre_stats['Sharpe'] = ((pre_data.subtract(rf[pre], axis=0)).mean() / (pre_data.subtract(rf[pre], axis=0)).std()) * np.sqrt(12)
print(pre_stats[['Mean (Ann %)', 'Std (Ann %)', 'Sharpe']].round(2))

print(f"\n--- POST-CRISIS ({_post_start} to {_data_end}) ---")
post_data = data.loc[post, industries]
post_stats = post_data.describe().T
post_stats['Mean (Ann %)'] = post_stats['mean'] * 12 * 100
post_stats['Std (Ann %)'] = post_stats['std'] * np.sqrt(12) * 100
post_stats['Sharpe'] = ((post_data.subtract(rf[post], axis=0)).mean() / (post_data.subtract(rf[post], axis=0)).std()) * np.sqrt(12)
print(post_stats[['Mean (Ann %)', 'Std (Ann %)', 'Sharpe']].round(2))

# Correlation matrix
print("\n--- CORRELATION MATRIX (Full Sample) ---")
corr_matrix = data[industries].corr()
print(corr_matrix.round(2))

# Average correlation by regime
pre_corr = data.loc[pre, industries].corr()
post_corr = data.loc[post, industries].corr()

upper_tri_pre = pre_corr.values[np.triu_indices_from(pre_corr.values, k=1)]
upper_tri_post = post_corr.values[np.triu_indices_from(post_corr.values, k=1)]

print(f"\n--- AVERAGE PAIRWISE CORRELATION ---")
print(f"Pre-Crisis:  {upper_tri_pre.mean():.4f}")
print(f"Post-Crisis: {upper_tri_post.mean():.4f}")
print(f"Change:      {upper_tri_post.mean() - upper_tri_pre.mean():+.4f}")

# Export to LaTeX (escape %, _, & for LaTeX safety)
table1_latex = table1_full.copy()
def _latex_escape_name(s):
    return s.replace('%', r'\%').replace('_', r'\_').replace('&', r'\&')
table1_latex.columns = [_latex_escape_name(c) for c in table1_latex.columns]
table1_latex.index = [_latex_escape_name(i) for i in table1_latex.index]
latex_str = table1_latex.to_latex(float_format="%.2f", escape=False)
latex_str = latex_str.replace('\xa0', ' ')
# Replace text hyphens with LaTeX math minus signs for negative numbers
latex_str = latex_str.replace(' -0.', ' $-$0.')
# Build complete table environment
tex_lines = []
tex_lines.append(r'\begin{table}[t]')
tex_lines.append(r'  \centering')
tex_lines.append(r'  \caption[Descriptive statistics for the Fama-French 10 Industry Portfolios]'
                 r'{Descriptive statistics for the Fama-French 10~Industry Portfolios.'
                 r' The sample covers the period from August~1963 to December~2025'
                 r' (749~monthly observations).}')
tex_lines.append(r'  \label{tab:descriptive_stats}')
tex_lines.append(r'  \small')
tex_lines.append(latex_str)
tex_lines.append(r'')
tex_lines.append(r'  \smallskip')
tex_lines.append(r'  \begin{minipage}{\textwidth}')
tex_lines.append(r'    \footnotesize')
tex_lines.append(r'    \textit{Notes.} Mean returns and volatilities are annualised.'
                 r' Sharpe ratios are computed as the ratio of annualised mean'
                 r' excess return to annualised volatility. Monthly minimum and'
                 r' maximum refer to simple returns.')
tex_lines.append(r"    \textit{Source:} Author's calculations based on data from"
                 r" Kenneth R.\ French's Data Library.")
tex_lines.append(r'  \end{minipage}')
tex_lines.append(r'\end{table}')
full_tex = '\n'.join(tex_lines) + '\n'
with open(f'{OUTPUT_DIR}table1_descriptive_stats.tex', 'w', encoding='utf-8') as f:
    f.write(full_tex)
print(f"\n✓ LaTeX table saved: {OUTPUT_DIR}table1_descriptive_stats.tex")
# LaTeX export (disabled for portable submission):
# sync_to_latex(f'{OUTPUT_DIR}table1_descriptive_stats.tex', LATEX_TABLES_DIR)

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS COMPLETE")
print("="*70)
