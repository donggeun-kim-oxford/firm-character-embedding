
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from tabulate import tabulate

# ─── CONFIG ─────────────────────────────────────────────────────────────
STACK_CSV        = "embed_port_output/grid_search_results_tc.csv"
DESC_STAT_DIR    = "analysis_output/descrip_stats"
os.makedirs(DESC_STAT_DIR, exist_ok=True)

PORTFOLIOS = ["gross", "net"]

# Load full grid for filtering
grid_df_full = pd.read_csv(STACK_CSV)
FILTERS = {
    "all":        pd.Series(True, index=grid_df_full.index),
}


# ─── DESCRIPTIVE STATS LaTeX ────────────────────────────────────────────────
bench_cols = ['sharpe_gross', 'sharpe_net']
desc = grid_df_full[bench_cols].describe().T.round(2)
desc.rename(index={
    "sharpe_gross": "Sharpe Before TC",
    "sharpe_net":    "Sharpe After TC",
}, inplace=True)
desc.index.name = "\\textbf{Portfolio}"
desc.columns = [f"\\textbf{{{c.capitalize()}}}" for c in desc.columns]

latex_table = tabulate(desc, headers="keys", tablefmt="latex")
out_tex = os.path.join(DESC_STAT_DIR, "desc_stats_tc.tex")
with open(out_tex, "w") as f:
    f.write(latex_table)

print("LaTeX table saved to", out_tex)