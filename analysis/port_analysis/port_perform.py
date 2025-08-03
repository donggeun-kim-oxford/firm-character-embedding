#!/usr/bin/env python3
"""
Generate distribution, cumulative equity-difference, embedding distribution,
and absolute equity plots for Embed vs. classical portfolios, filtered by hyperparameters,
and then produce a LaTeX table of descriptive stats.

Reads:      new_results/grid_search_results_stack.csv
Monthly performance files: new_monthly_perf/
Writes:     new_plots/dist_<portfolio>_<filter>.png,
            new_plots/dist_embedding_<filter>.png,
            new_plots/cumul_<portfolio>_<filter>.png,
            new_plots/equity_embedding_<filter>.png,
            new_plots/desc_stats.tex
Portfolios compared: Benchmark (ew), rf, cat, light
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from tabulate import tabulate

# ─── CONFIG ─────────────────────────────────────────────────────────────
STACK_CSV        = "embed_port_output/grid_search_results_combined.csv"
MONTHLY_PERF_DIR = "embed_port_output/combined_monthly_perf"
PLOTS_DIR        = "analysis_output/port_perform_plots"
DESC_STAT_DIR    = "analysis_output/descrip_stats"
os.makedirs(PLOTS_DIR, exist_ok=True)

PORTFOLIOS = ["Benchmark", "rf", "cat", "light"]

# Load full grid for filtering
grid_df_full = pd.read_csv(STACK_CSV)
FILTERS = {
    "all":        pd.Series(True, index=grid_df_full.index),
}

# ─── MAIN LOOP ────────────────────────────────────────────────────────────
for filt_name, mask in FILTERS.items():
    sub_grid = grid_df_full.loc[mask]
    if sub_grid.empty:
        print(f"Skipping filter '{filt_name}' (no rows match)")
        continue

    final_diff = {p: [] for p in PORTFOLIOS}
    equity_diff_curves = {p: [] for p in PORTFOLIOS}
    embedding_finals = []
    embedding_curves = []

    # Collect data
    for idx, row in sub_grid.iterrows():
        i, n, k = idx+1, int(row["n_estimators"]), int(row["knn_neighbors"])
        pca = int(row["pca_components"]) if pd.notna(row["pca_components"]) else "None"
        cq, thr = float(row["clip_quantile"]), float(row["threshold_pos"])
        base = f"{i:03d}_RF{n}_K{k}_PCA{pca}_CQ{cq:.2f}_THR{thr:.2f}"
        perf_path = os.path.join(MONTHLY_PERF_DIR, f"{base}_perf.csv")
        if not os.path.exists(perf_path):
            continue

        df = pd.read_csv(perf_path, parse_dates=["Month"])
        df["Month"] = df["Month"].dt.to_period("M")
        df.set_index("Month", inplace=True)

        # Embedding-only equity
        if "Equity_Portfolio" in df.columns:
            vals = df["Equity_Portfolio"].values
            embedding_curves.append((df.index.to_timestamp(), vals))
            embedding_finals.append(vals[-1])

        # Differences vs. each benchmark
        for p in PORTFOLIOS:
            if f"Equity_{p}" not in df or "Equity_Portfolio" not in df:
                continue
            emb, bench = df["Equity_Portfolio"], df[f"Equity_{p}"]
            final_diff[p].append(emb.iloc[-1] - bench.iloc[-1])
            equity_diff_curves[p].append((emb.index.to_timestamp(),
                                          emb.values, bench.values))

    # 1) Distribution of final differences
    for p in PORTFOLIOS:
        diffs = np.array(final_diff[p])
        if diffs.size == 0:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(diffs, bins=20, color="cornflowerblue",
                 edgecolor="black", alpha=0.7)
        plt.xlabel("Excess Cumulative Return")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.5)
        out_dist = os.path.join(PLOTS_DIR, f"dist_{p}_{filt_name}.png")
        plt.tight_layout(); plt.savefig(out_dist); plt.close()
        # t-test
        if diffs.size > 1:
            t_stat, p_two = ttest_1samp(diffs, 0.0, nan_policy="omit")
            p_one = p_two/2 if t_stat>0 else 1 - p_two/2
            print(f"{p} [{filt_name}]: t={t_stat:.3f}, p(one-tailed)={p_one:.6f}")

    # 1b) Distribution of embedding final returns
    if embedding_finals:
        plt.figure(figsize=(8, 5))
        plt.hist(embedding_finals, bins=20, color="seagreen",
                 edgecolor="black", alpha=0.7)
        plt.xlabel("Cumulative Return")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.5)
        out_emb = os.path.join(PLOTS_DIR, f"dist_embedding_{filt_name}.png")
        plt.tight_layout(); plt.savefig(out_emb); plt.close()

    # 2) Equity-difference curves
    for p in PORTFOLIOS:
        curves = equity_diff_curves[p]
        if not curves:
            continue
        plt.figure(figsize=(10, 6))
        cmap = plt.cm.viridis
        for j, (dates, emb, bench) in enumerate(curves):
            diff_ts = emb - bench
            color = cmap(j / (len(curves)-1) if len(curves)>1 else 0.5)
            plt.plot(dates, diff_ts, color=color, alpha=0.8)
        plt.axhline(0, color="black", ls="--")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returne")
        plt.grid(True, linestyle="--", alpha=0.5)
        out_cumul = os.path.join(PLOTS_DIR, f"cumul_{p}_{filt_name}.png")
        plt.tight_layout(); plt.savefig(out_cumul); plt.close()

    # 3) Absolute embedding equity curves
    if embedding_curves:
        plt.figure(figsize=(10, 6))
        cmap = plt.cm.plasma
        for j, (dates, vals) in enumerate(embedding_curves):
            color = cmap(j / (len(embedding_curves)-1) if len(embedding_curves)>1 else 0.5)
            plt.plot(dates, vals, color=color, alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True, linestyle="--", alpha=0.5)
        out_eq = os.path.join(PLOTS_DIR, f"equity_embedding_{filt_name}.png")
        plt.tight_layout(); plt.savefig(out_eq); plt.close()

# ─── DESCRIPTIVE STATS LaTeX ────────────────────────────────────────────────
bench_cols = ['embed_sharpe', 'ew_sharpe', 'rf_sharpe',
              'cat_sharpe', 'light_sharpe']
desc = grid_df_full[bench_cols].describe().T.round(2)
desc.rename(index={
    "embed_sharpe": "Embedding",
    "ew_sharpe":    "Equal-Weight",
    "rf_sharpe":    "Random Forest",
    "cat_sharpe":   "CatBoost",
    "light_sharpe":"LightGBM"
}, inplace=True)
desc.index.name = "\\textbf{Portfolio}"
desc.columns = [f"\\textbf{{{c.capitalize()}}}" for c in desc.columns]

latex_table = tabulate(desc, headers="keys", tablefmt="latex")
out_tex = os.path.join(DESC_STAT_DIR, "desc_stats.tex")
with open(out_tex, "w") as f:
    f.write(latex_table)

print("LaTeX table saved to", out_tex)