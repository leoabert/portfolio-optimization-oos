#!/usr/bin/env python3
"""
03_regime_comparison.py - Subperiod Analysis & Table 2
=======================================================
Splits the OOS period at January 2009 and produces:
  - Console diagnostics (regime characteristics, betas, etc.)
  - Table 2 LaTeX (03_Output/tables/table2_subperiod.tex)
  - Table 2 CSV   (03_Output/tables/table2_subperiod.csv)
"""

import os
import pandas as pd
import numpy as np
import warnings
from config import (RISK_AVERSION, INDUSTRIES, OUTPUT_DIR, ESTIMATION_WINDOW,
                    TRANSACTION_COST)
from utils import latex_minus

warnings.filterwarnings('ignore')

SPLIT_DATE = '2009-01-01'   # Pre-crisis: before 2009-01; Post-crisis: 2009-01 onward

print("=" * 70)
print("REGIME COMPARISON & TABLE 2: Subperiod Analysis")
print("=" * 70)

# =====================================================================
# 1. Load pre-computed backtest data
# =====================================================================
returns  = pd.read_csv(f'{OUTPUT_DIR}strategy_returns.csv',  index_col=0, parse_dates=True)
turnover = pd.read_csv(f'{OUTPUT_DIR}strategy_turnover.csv', index_col=0, parse_dates=True)
data     = pd.read_csv('./data/data_clean.csv',           index_col=0, parse_dates=True)
rf       = data['RF_Monthly'].reindex(returns.index)

strategies = ['1/N', 'Mean-Variance', 'Minimum Variance']

pre_mask  = returns.index < SPLIT_DATE
post_mask = returns.index >= SPLIT_DATE

pre_start  = returns.index[pre_mask][0].strftime('%Y-%m')
pre_end    = returns.index[pre_mask][-1].strftime('%Y-%m')
post_start = returns.index[post_mask][0].strftime('%Y-%m')
post_end   = returns.index[post_mask][-1].strftime('%Y-%m')

print(f"\n  Split date:   {SPLIT_DATE}")
print(f"  Pre-Crisis:   {pre_start} to {pre_end}  ({pre_mask.sum()} months)")
print(f"  Post-Crisis:  {post_start} to {post_end} ({post_mask.sum()} months)")

# =====================================================================
# 2. Metric computation helper
# =====================================================================
def compute_panel_metrics(mask):
    """Compute all Table 2 metrics for strategies in a given subperiod."""
    rows = {}
    for s in strategies:
        r = returns[s][mask]
        t = turnover[s][mask]
        rf_sub = rf[mask]
        excess = r - rf_sub

        mean_excess_ann = excess.mean() * 12 * 100
        std_val         = excess.std()
        vol_ann         = std_val * np.sqrt(12) * 100
        sharpe_ann      = (excess.mean() / std_val) * np.sqrt(12) if std_val > 0 else 0.0
        ceq_ann         = (excess.mean() - (RISK_AVERSION / 2) * excess.var()) * 12 * 100
        turnover_mo     = t.mean() * 100

        rows[s] = {
            'mean_excess': mean_excess_ann,
            'vol':         vol_ann,
            'sharpe':      sharpe_ann,
            'ceq':         ceq_ann,
            'turnover':    turnover_mo,
        }

    df = pd.DataFrame(rows).T
    # DCEQ = CEQ(1/N) - CEQ(strategy)
    ceq_1n = df.loc['1/N', 'ceq']
    df['dceq'] = ceq_1n - df['ceq']
    df.loc['1/N', 'dceq'] = np.nan
    return df


pre_df  = compute_panel_metrics(pre_mask)
post_df = compute_panel_metrics(post_mask)

# Difference row (Post - Pre) for each strategy
diff_df = post_df - pre_df   # NaN propagates for 1/N dceq, which is correct

# =====================================================================
# 3. Console output — performance panels
# =====================================================================
col_labels = ['Mean Exc. Ret. (%)', 'Volatility (%)', 'Sharpe',
              'CEQ (%)', 'DCEQ (%)', 'Turnover (%)']

def print_panel(label, df):
    display = df.copy()
    display.columns = col_labels
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print('=' * 70)
    print(display.round(3).to_string())

print_panel(f"PANEL A: Pre-Crisis ({pre_start} to {pre_end})", pre_df)
print_panel(f"PANEL B: Post-Crisis ({post_start} to {post_end})", post_df)
print_panel("DIFFERENCE (Post - Pre)", diff_df)

# =====================================================================
# 4. Interest rate context
# =====================================================================
print("\n" + "=" * 70)
print("INTEREST RATE ENVIRONMENT")
print("=" * 70)

data_oos = data.reindex(returns.index)

for label, mask in [("Pre-Crisis", pre_mask), ("Post-Crisis", post_mask)]:
    rf_sub   = data_oos['RF_Monthly'][mask]
    real_sub = data_oos['Real_Rate'][mask]
    print(f"\n  {label}:")
    print(f"    Avg 1-Month T-Bill (ann.):  {rf_sub.mean() * 12 * 100:>6.2f}%")
    print(f"    Avg Real Rate (ann.):       {real_sub.mean() * 12 * 100:>6.2f}%")

# =====================================================================
# 5. Regime characteristics (correlation, vol dispersion)
# =====================================================================
print("\n" + "=" * 70)
print("REGIME CHARACTERISTICS")
print("=" * 70)

industries = INDUSTRIES

def calc_regime_stats(mask):
    subset = data.loc[returns.index[mask], industries]
    corr_matrix = subset.corr()
    upper = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    vols = subset.std() * np.sqrt(12)
    return {
        'Avg Correlation':     upper.mean(),
        'Avg Volatility (%)':  vols.mean() * 100,
        'Vol Dispersion (%)':  vols.std() * 100,
    }

pre_stats  = calc_regime_stats(pre_mask)
post_stats = calc_regime_stats(post_mask)
stats_df = pd.DataFrame({
    'Pre-Crisis':  pre_stats,
    'Post-Crisis': post_stats,
    'Change': {k: post_stats[k] - pre_stats[k] for k in pre_stats},
})
print(stats_df.round(4).to_string())

# =====================================================================
# 6. Min-Var beta analysis
# =====================================================================
print("\n" + "=" * 70)
print("MINIMUM VARIANCE BETA ANALYSIS")
print("=" * 70)

# Use locally cached Mkt-RF (saved by 01_build_dataset.py) — no network needed
if 'Mkt_RF' in data.columns and data['Mkt_RF'].notna().sum() > 0:
    mkt_ret = data['Mkt_RF'].reindex(returns.index)
else:
    # Fallback: download if local column is missing (e.g. old data_clean.csv)
    import pandas_datareader.data as web
    import datetime
    ff_factors = web.DataReader('F-F_Research_Data_Factors', 'famafrench',
                                datetime.datetime(1963, 7, 1),
                                datetime.datetime(2025, 12, 31))[0]
    ff_factors = ff_factors / 100.0
    ff_factors.index = ff_factors.index.to_timestamp()
    mkt_ret = ff_factors['Mkt-RF'].reindex(returns.index)

def calc_beta(portfolio_ret, market_ret, mask):
    p = portfolio_ret[mask].values
    m = market_ret[mask].values
    cov_pm = np.cov(p, m)[0, 1]
    var_m  = np.var(m, ddof=1)
    return cov_pm / var_m if var_m > 0 else np.nan

for label, mask in [("Pre-Crisis", pre_mask), ("Post-Crisis", post_mask)]:
    beta_1n     = calc_beta(returns['1/N'], mkt_ret, mask)
    beta_minvar = calc_beta(returns['Minimum Variance'], mkt_ret, mask)
    mkt_ann     = mkt_ret[mask].mean() * 12 * 100
    print(f"\n  {label}:")
    print(f"    Market Exc. Return (ann.): {mkt_ann:>6.2f}%")
    print(f"    Beta(1/N):                 {beta_1n:>6.4f}")
    print(f"    Beta(MinVar):              {beta_minvar:>6.4f}")

# =====================================================================
# 7. Build & save Table 2 (CSV + LaTeX)
# =====================================================================
print("\n" + "=" * 70)
print("EXPORTING TABLE 2")
print("=" * 70)

table_dir = f'{OUTPUT_DIR}tables/'
os.makedirs(table_dir, exist_ok=True)

# --- CSV ---
csv_rows = []
for panel_label, panel_df in [('Pre-Crisis', pre_df), ('Post-Crisis', post_df),
                               ('Difference', diff_df)]:
    for s in strategies:
        row = panel_df.loc[s].to_dict()
        row['Panel']    = panel_label
        row['Strategy'] = s
        csv_rows.append(row)

csv_out = pd.DataFrame(csv_rows)[['Panel', 'Strategy', 'mean_excess', 'vol',
                                   'sharpe', 'ceq', 'dceq', 'turnover']]
csv_out.columns = ['Panel', 'Strategy', 'Mean Excess Return (%)',
                   'Volatility (%)', 'Sharpe Ratio', 'CEQ Return (%)',
                   'DCEQ (%)', 'Turnover (%)']
csv_path = f'{table_dir}table2_subperiod.csv'
csv_out.to_csv(csv_path, index=False)
print(f"  CSV saved: {csv_path}")

# --- LaTeX ---
display_name = {
    '1/N': r'$1/N$',
    'Mean-Variance': 'Mean--Variance',
    'Minimum Variance': 'Minimum--Variance',
}

def fmt2(v):
    if isinstance(v, float) and np.isnan(v):
        return '--'
    return latex_minus(f'{v:.2f}')

def fmt3(v):
    if isinstance(v, float) and np.isnan(v):
        return '--'
    return latex_minus(f'{v:.3f}')

def latex_data_row(strategy, row):
    """Format one strategy's metrics as a LaTeX table row."""
    return (f'    {display_name[strategy]}'
            f' & {fmt2(row["mean_excess"])}'
            f' & {fmt2(row["vol"])}'
            f' & {fmt3(row["sharpe"])}'
            f' & {fmt2(row["ceq"])}'
            f' & {fmt2(row["dceq"])}'
            f' & {fmt2(row["turnover"])}'
            r' \\')

mo_start_full = returns.index[0].strftime('%B %Y')
mo_end_full   = returns.index[-1].strftime('%B %Y')

lines = []
lines.append(r'\begin{table}[t]')
lines.append(r'  \centering')
lines.append(r'  \caption[Subperiod performance comparison]'
             r'{Subperiod performance comparison. '
             f'Panel~A covers the pre-crisis period ({pre_start} to {pre_end}) and '
             f'Panel~B covers the post-crisis low interest rate period '
             f'({post_start} to {post_end}). '
             r'The split date is January 2009. '
             r'Estimation window $M=' + str(ESTIMATION_WINDOW) + r'$ months, '
             r'$\gamma=' + str(RISK_AVERSION) + r'$.}')
lines.append(r'  \label{tab:subperiod}')
lines.append(r'  \small')
lines.append(r'  \resizebox{\textwidth}{!}{%')
lines.append(r'  \begin{tabular}{l *{6}{r}}')
lines.append(r'    \toprule')
# Header row 1 — metric names
lines.append(r'    & \multicolumn{1}{c}{Mean Exc.}'
             r'    & \multicolumn{1}{c}{Volatility}'
             r'    & \multicolumn{1}{c}{Sharpe}'
             r'    & \multicolumn{1}{c}{CEQ}'
             r'    & \multicolumn{1}{c}{$\Delta$CEQ}'
             r'    & \multicolumn{1}{c}{Turnover} \\')
# Header row 2 — units
lines.append(r'    Strategy'
             r'    & \multicolumn{1}{c}{Ret.\ (\%)}'
             r'    & \multicolumn{1}{c}{(\%)}'
             r'    & \multicolumn{1}{c}{Ratio}'
             r'    & \multicolumn{1}{c}{(\%)}'
             r'    & \multicolumn{1}{c}{(\%)}'
             r'    & \multicolumn{1}{c}{(\%)} \\')
lines.append(r'    \midrule')

# Panel A
lines.append(f'    \\multicolumn{{7}}{{l}}'
             f'{{\\textit{{Panel A: Pre-Crisis ({pre_start} to {pre_end})}}}} \\\\')
lines.append(r'    \addlinespace[2pt]')
for s in strategies:
    lines.append(latex_data_row(s, pre_df.loc[s]))
lines.append(r'    \addlinespace[6pt]')

# Panel B
lines.append(f'    \\multicolumn{{7}}{{l}}'
             f'{{\\textit{{Panel B: Post-Crisis ({post_start} to {post_end})}}}} \\\\')
lines.append(r'    \addlinespace[2pt]')
for s in strategies:
    lines.append(latex_data_row(s, post_df.loc[s]))
lines.append(r'    \addlinespace[6pt]')

# Difference (Post - Pre)
lines.append(r'    \multicolumn{7}{l}{\textit{Difference (Post $-$ Pre)}} \\')
lines.append(r'    \addlinespace[2pt]')
for s in strategies:
    lines.append(latex_data_row(s, diff_df.loc[s]))

lines.append(r'    \bottomrule')
lines.append(r'  \end{tabular}%')
lines.append(r'  }')
lines.append('')
lines.append(r'  \smallskip')
lines.append(r'  \begin{minipage}{\textwidth}')
lines.append(r'    \footnotesize')
lines.append(r'    \textit{Notes.} '
             r'All values are annualised except turnover (monthly average). '
             r'Metrics are computed on excess returns $r_{p,t} - r_{f,t}$. '
             r'$\Delta\text{CEQ} = \text{CEQ}_{1/N} - \text{CEQ}_{\text{strategy}}$; '
             r'positive values indicate $1/N$ outperformance.')
lines.append(r"    \textit{Source:} Author's calculations based on data from"
             r" Kenneth R.\ French's Data Library.")
lines.append(r'  \end{minipage}')
lines.append(r'\end{table}')

latex_code = '\n'.join(lines) + '\n'

latex_code = latex_code.replace('\xa0', ' ')
tex_path = f'{table_dir}table2_subperiod.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(latex_code)
print(f"  TeX saved: {tex_path}")
# LaTeX export (disabled for portable submission):
# sync_to_latex(tex_path, LATEX_TABLES_DIR)

# =====================================================================
# 8. Print LaTeX preview
# =====================================================================
print("\n" + "=" * 70)
print("TABLE 2 PREVIEW (LaTeX)")
print("=" * 70)
print(latex_code)

print("=" * 70)
print("REGIME COMPARISON COMPLETE")
print("=" * 70)
