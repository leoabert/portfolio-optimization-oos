#!/usr/bin/env python3
"""
05b_subperiod_bootstrap.py - Subperiod Bootstrap Inference
============================================================
Block bootstrap confidence intervals for ΔCEQ in each subperiod
(pre-crisis / post-crisis), 10,000 replications.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import (OUTPUT_DIR, RISK_AVERSION, SPLIT_DATE,
                    BOOTSTRAP_ITERATIONS)
from utils import calculate_ceq, latex_minus

N_BOOT = BOOTSTRAP_ITERATIONS
BLOCK_SIZE = 12

print("=" * 60)
print("SUBPERIOD BOOTSTRAP CONFIDENCE INTERVALS")
print(f"  Replications: {N_BOOT:,}, Block size: {BLOCK_SIZE} months")
print("=" * 60)

# =====================================================================
# 1. Load data
# =====================================================================
returns = pd.read_csv(f'{OUTPUT_DIR}strategy_returns.csv',
                      index_col=0, parse_dates=True)
data = pd.read_csv('./data/data_clean.csv',
                    index_col=0, parse_dates=True)
rf = data['RF_Monthly'].reindex(returns.index)

strategies = ['Mean-Variance', 'Minimum Variance']
strategy_labels = {'Mean-Variance': 'MV', 'Minimum Variance': 'MinVar'}

pre_mask  = returns.index < SPLIT_DATE
post_mask = returns.index >= SPLIT_DATE

pre_start = returns.index[pre_mask][0].strftime('%Y-%m')
pre_end   = returns.index[pre_mask][-1].strftime('%Y-%m')
post_start = returns.index[post_mask][0].strftime('%Y-%m')
post_end   = returns.index[post_mask][-1].strftime('%Y-%m')

print(f"  Pre-Crisis:  {pre_start} to {pre_end}  ({pre_mask.sum()} months)")
print(f"  Post-Crisis: {post_start} to {post_end} ({post_mask.sum()} months)")

# =====================================================================
# 2. Bootstrap helper
# =====================================================================
def ceq_boot(r, rf_b):
    """Compute annualised CEQ (%) from arrays."""
    excess = r - rf_b
    return (excess.mean() - (RISK_AVERSION / 2) * np.var(excess, ddof=1)) * 12 * 100


# =====================================================================
# 3. Run bootstrap for each subperiod
# =====================================================================
np.random.seed(42)

results = []

for label, mask in [('Pre-Crisis', pre_mask), ('Post-Crisis', post_mask)]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    ret_1n     = returns['1/N'][mask].values
    rf_sub     = rf[mask].values
    n_obs      = mask.sum()
    n_blocks   = n_obs // BLOCK_SIZE

    # Point estimates
    ceq_1n_point = ceq_boot(ret_1n, rf_sub)

    for s in strategies:
        ret_s = returns[s][mask].values
        ceq_s_point = ceq_boot(ret_s, rf_sub)
        delta_point = ceq_1n_point - ceq_s_point

        delta_boot = []
        for b in range(N_BOOT):
            if (b + 1) % 2000 == 0:
                print(f"    {strategy_labels[s]}: iteration {b+1:,}/{N_BOOT:,}")

            block_idx = np.random.choice(n_blocks, size=n_blocks, replace=True)
            boot_idx = []
            for bi in block_idx:
                boot_idx.extend(range(bi * BLOCK_SIZE,
                                      min((bi + 1) * BLOCK_SIZE, n_obs)))
            boot_idx = boot_idx[:n_obs]

            ceq_1n_b = ceq_boot(ret_1n[boot_idx], rf_sub[boot_idx])
            ceq_s_b  = ceq_boot(ret_s[boot_idx],  rf_sub[boot_idx])
            delta_boot.append(ceq_1n_b - ceq_s_b)

        ci = np.percentile(delta_boot, [2.5, 97.5])
        sig = (ci[0] > 0) or (ci[1] < 0)  # CI excludes zero

        results.append({
            'Subperiod':     label,
            'Comparison':    f'DCEQ({strategy_labels[s]})',
            'PointEstimate': delta_point,
            'CI_Lower':      ci[0],
            'CI_Upper':      ci[1],
            'Significant':   sig,
        })

        print(f"    DCEQ({strategy_labels[s]}): {delta_point:+.2f}%  "
              f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]  "
              f"{'Significant' if sig else 'Not significant'}")

# =====================================================================
# 4. Export CSV
# =====================================================================
df = pd.DataFrame(results)
csv_path = f'{OUTPUT_DIR}subperiod_bootstrap_ci.csv'
df.to_csv(csv_path, index=False)
print(f"\n  CSV saved: {csv_path}")

# =====================================================================
# 5. Export LaTeX table
# =====================================================================
table_dir = f'{OUTPUT_DIR}tables/'
os.makedirs(table_dir, exist_ok=True)

lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'  \centering')
lines.append(r'  \caption[Bootstrap confidence intervals for subperiod welfare costs]'
             r'{Bootstrap confidence intervals for subperiod welfare costs. '
             r'Block bootstrap with $b=12$ month blocks and $B=\text{10,000}$ replications. '
             r'Point estimates and 95\% confidence intervals for '
             r'$\Delta\text{CEQ} = \text{CEQ}_{1/N} - \text{CEQ}_{\text{strategy}}$ '
             r'are reported for each subperiod. '
             r'Risk aversion $\gamma=' + str(RISK_AVERSION) + r'$.}')
lines.append(r'  \label{tab:subperiod_bootstrap}')
lines.append(r'  \addcontentsline{loa}{appendixentry}{\protect\numberline{\thetable}Bootstrap confidence intervals for subperiod welfare costs}')
lines.append(r'  \small')
lines.append(r'  \begin{tabular}{l l r r r c}')
lines.append(r'    \toprule')
lines.append(r'    & & Point & \multicolumn{2}{c}{95\% CI} & \\')
lines.append(r'    \cmidrule(lr){4-5}')
lines.append(r'    Subperiod & Comparison & Est.\ (\%) & Lower (\%) & Upper (\%) & Significant \\')
lines.append(r'    \midrule')

latex_comp = {'DCEQ(MV)': r'$\Delta$CEQ(MV)', 'DCEQ(MinVar)': r'$\Delta$CEQ(MinVar)'}
for i, row in df.iterrows():
    sig_str = 'Yes' if row['Significant'] else 'No'
    comp_label = latex_comp.get(row['Comparison'], row['Comparison'])
    # Add addlinespace between subperiods
    if i == 2:
        lines.append(r'    \addlinespace')
    pt  = latex_minus(f'{row["PointEstimate"]:+.2f}')
    clo = latex_minus(f'{row["CI_Lower"]:+.2f}')
    chi = latex_minus(f'{row["CI_Upper"]:+.2f}')
    lines.append(f'    {row["Subperiod"]} & {comp_label} '
                 f'& {pt} '
                 f'& {clo} '
                 f'& {chi} '
                 f'& {sig_str} \\\\')

lines.append(r'    \bottomrule')
lines.append(r'  \end{tabular}')
lines.append('')
lines.append(r'  \smallskip')
lines.append(r'  \begin{minipage}{\textwidth}')
lines.append(r'    \footnotesize')
lines.append(r'    \textit{Notes.} '
             r'Block bootstrap confidence intervals for the welfare cost '
             r'$\Delta\text{CEQ}$ in each subperiod. '
             r'Positive values indicate $1/N$ outperformance. '
             r'The pre-crisis period covers ' + pre_start + ' to ' + pre_end +
             r' and the post-crisis period covers ' + post_start + ' to ' + post_end + '. '
             r"``Significant'' indicates that the 95\% CI excludes zero.")
lines.append(r"    \textit{Source:} Author's calculations based on data from"
             r" Kenneth R.\ French's Data Library.")
lines.append(r'  \end{minipage}')
lines.append(r'\end{table}')

latex_code = '\n'.join(lines) + '\n'
latex_code = latex_code.replace('\xa0', ' ')

tex_path = f'{table_dir}table_subperiod_bootstrap.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(latex_code)
print(f"  TeX saved: {tex_path}")
# LaTeX export (disabled for portable submission):
# sync_to_latex(tex_path, LATEX_TABLES_DIR)

# =====================================================================
# 6. Summary
# =====================================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(df.to_string(index=False))

print("\n" + "=" * 60)
print("SUBPERIOD BOOTSTRAP COMPLETE")
print("=" * 60)
