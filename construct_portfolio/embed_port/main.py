#!/usr/bin/env python3


import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import glob

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from construct_portfolio.embed_port.utils import (
    load_and_clean,
    build_monthly_returns,
    summarize_series
)

from construct_portfolio.config import (
    CONFIG,
    ORIGINAL_MONTHLY_RETURNS_DIR,
    ORIGINAL_MONTHLY_PERF_DIR,
    ORIGINAL_GRID_EMBEDDING
)

MONTHLY_IN = ORIGINAL_MONTHLY_RETURNS_DIR
PERF_OUT   = ORIGINAL_MONTHLY_PERF_DIR
GRID_CSV   = ORIGINAL_GRID_EMBEDDING

os.makedirs(CONFIG["MONTHLY_RETURNS_DIR"], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)
os.makedirs(PERF_OUT, exist_ok=True)
os.makedirs(os.path.dirname(GRID_CSV), exist_ok=True)



# ─── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=CONFIG["LOG_FILE"],
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    
    print("[START] Loading & cleaning TRAIN/TEST sets")
    Xtr, ytr, _   = load_and_clean(CONFIG["TRAIN_FEAT"], CONFIG["TRAIN_TARG"])
    Xte, yte, dte = load_and_clean(CONFIG["TEST_FEAT"], CONFIG["TEST_TARG"])

    results = []
    combo_id = 0
    total = (
        len(CONFIG["N_EST_LIST"])
        * len(CONFIG["KNN_K_LIST"])
        * len(CONFIG["PCA_COMPONENTS"])
        * len(CONFIG["CLIP_QUANTILES"])
        * len(CONFIG["EDGE_THRESHOLDS"])
    )
    print(f"[LOOP] {total} combos\n")

    for n_est in CONFIG["N_EST_LIST"]:
        for k in CONFIG["KNN_K_LIST"]:
            for pca_n in CONFIG["PCA_COMPONENTS"]:
                # 1) PCA
                if pca_n is not None:
                    pca      = PCA(n_components=pca_n)
                    Xtr_proc = pca.fit_transform(Xtr)
                    Xte_proc = pca.transform(Xte)
                else:
                    Xtr_proc, Xte_proc = Xtr, Xte

                # 2) RF + KNN
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(Xtr_proc, ytr)
                pred_tr_rf = rf.predict(Xtr_proc)

                knn = KNeighborsRegressor(
                    n_neighbors=k,
                    n_jobs=-1
                )
                knn.fit(Xtr_proc, pred_tr_rf)
                pred_te = knn.predict(Xte_proc)

                for cq in CONFIG["CLIP_QUANTILES"]:
                    if cq > 0:
                        low, high = np.quantile(yte, cq), np.quantile(yte, 1-cq)
                        mask_clip = (yte >= low) & (yte <= high)
                    else:
                        mask_clip = np.ones_like(yte, bool)

                    yte_c, dte_c, pred_c = (
                        yte[mask_clip], dte[mask_clip], pred_te[mask_clip]
                    )

                    for thr in CONFIG["EDGE_THRESHOLDS"]:
                        combo_id += 1
                        port_s, bench_s = build_monthly_returns(
                            dte_c, yte_c, pred_c, thr
                        )
                        mu_t, sd_t, sp_t = summarize_series(port_s)
                        mu_b, sd_b, sb_t = summarize_series(bench_s)

                        # 3) append grid results
                        results.append({
                            "n_estimators":      n_est,
                            "knn_neighbors":     k,
                            "rolling_window":    "",
                            "pca_components":    pca_n or "",
                            "max_short_frac":    CONFIG["MAX_SHORT_FRAC"],
                            "max_weight":        CONFIG["MAX_WEIGHT"],
                            "clip_quantile":     cq,
                            "edge_threshold":    thr,
                            "mean_test":         mu_t,
                            "std_test":          sd_t,
                            "sharpe_test":       sp_t,
                            "mean_bench_test":   mu_b,
                            "std_bench_test":    sd_b,
                            "sharpe_bench_test": sb_t,
                        })

                        # ─── SAVE RAW MONTHLY RETURNS ───────────────────────────
                        monthly_df = pd.DataFrame({
                            "Month":                     port_s.index,
                            "Monthly_Return_Portfolio":  port_s.values,
                            "Monthly_Return_Benchmark":  bench_s.values
                        })
                        fname = (
                            f"{combo_id:03d}_RF{n_est}_K{k}"
                            f"_PCA{pca_n or 'None'}_CQ{cq:.2f}_THR{thr:.2f}.csv"
                        )
                        monthly_df.to_csv(
                            os.path.join(CONFIG["MONTHLY_RETURNS_DIR"], fname),
                            index=False
                        )

    # 4) save grid CSV
    df = pd.DataFrame(results)
    cols = [
        "n_estimators","knn_neighbors","rolling_window",
        "pca_components","max_short_frac","max_weight",
        "clip_quantile","edge_threshold","mean_test","std_test",
        "sharpe_test","mean_bench_test","std_bench_test",
        "sharpe_bench_test"
    ]
    
    df.to_csv(GRID_CSV, index=False, columns=cols, float_format="%.15g")
    print("\n[DONE] grid_search_results.csv")



    # GENERATE MONTHLY PERFORMANCE STATISTICS
    grid = pd.read_csv(GRID_CSV)

    for path in sorted(glob.glob(os.path.join(MONTHLY_IN, "*.csv"))):
        fname = os.path.basename(path).replace(".csv","")
        print(f"[PROCESS] {fname}")

        df = pd.read_csv(path)
        port_rets  = df["Monthly_Return_Portfolio"].to_numpy()
        bench_rets = df["Monthly_Return_Benchmark"].to_numpy()

        # equity curves
        equity_p = (1 + port_rets).cumprod()
        equity_b = (1 + bench_rets).cumprod()

        # running stats
        run_std_p    = pd.Series(port_rets).expanding(min_periods=1).std(ddof=1).values
        run_mean_p   = pd.Series(port_rets).expanding(min_periods=1).mean().values
        run_sharpe_p = run_mean_p / run_std_p * np.sqrt(12)

        run_std_b    = pd.Series(bench_rets).expanding(min_periods=1).std(ddof=1).values
        run_mean_b   = pd.Series(bench_rets).expanding(min_periods=1).mean().values
        run_sharpe_b = run_mean_b / run_std_b * np.sqrt(12)

        # overall Sharpe
        mu_p, sd_p = port_rets.mean(), port_rets.std(ddof=1)
        ov_sp_p = (mu_p/sd_p)*np.sqrt(12) if sd_p!=0 else np.nan

        mu_b, sd_b = bench_rets.mean(), bench_rets.std(ddof=1)
        ov_sp_b = (mu_b/sd_b)*np.sqrt(12) if sd_b!=0 else np.nan


        if not ((np.isnan(ov_sp_p) and np.isnan(run_sharpe_p[-1])) 
                or np.isclose(run_sharpe_p[-1], ov_sp_p, atol=1e-8)):
            raise AssertionError(
                f"Running Sharpe mismatch for {fname}: "
                f"last_running={run_sharpe_p[-1]:.6f} vs overall={ov_sp_p:.6f}"
            )

        # parse hyperparams from filename
        parts    = fname.split("_")
        n_est    = int(parts[1].lstrip("RF"))
        knn_k    = int(parts[2].lstrip("K"))
        pca_n    = None if parts[3]=="PCANone" else int(parts[3].lstrip("PCA"))
        clip_q   = float(parts[4].lstrip("CQ"))
        edge_thr = float(parts[5].lstrip("THR"))

        # assert final Sharpe matches original grid
        mask = (
            (grid.n_estimators   == n_est) &
            (grid.knn_neighbors  == knn_k) &
            (grid.pca_components.fillna(-1)==(-1 if pca_n is None else pca_n)) &
            (np.isclose(grid.clip_quantile, clip_q)) &
            (np.isclose(grid.edge_threshold, edge_thr))
        )
        sub = grid.loc[mask]
        if not sub.empty:
            orig_sp = sub["sharpe_test"].iloc[0]
            if not ((np.isnan(orig_sp) and np.isnan(ov_sp_p)) 
                    or np.isclose(orig_sp, ov_sp_p, atol=1e-8)):
                raise AssertionError(
                    f"Sharpe mismatch {fname}: "
                    f"recomputed={ov_sp_p:.6f} vs original={orig_sp:.6f}"
                )

        # write out performance CSV
        perf_df = pd.DataFrame({
            "Month":                    df["Month"],
            "Monthly_Return_Portfolio": port_rets,
            "Equity_Portfolio":         equity_p,
            "Run_STD_Portfolio":        run_std_p,
            "Run_Sharpe_Portfolio":     run_sharpe_p,
            "Monthly_Return_Benchmark": bench_rets,
            "Equity_Benchmark":         equity_b,
            "Run_STD_Benchmark":        run_std_b,
            "Run_Sharpe_Benchmark":     run_sharpe_b,
        })
        out_path = os.path.join(PERF_OUT, fname + "_perf.csv")
        perf_df.to_csv(out_path, index=False)
        print(f"[Wrote perf] {out_path}")

        print("✔ All done.")
if __name__ == "__main__":
    main()
    