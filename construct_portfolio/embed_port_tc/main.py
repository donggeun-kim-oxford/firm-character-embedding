#!/usr/bin/env python3
"""
grid_search_embedding_hope7_d.py

… [your header here, unchanged] …
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# ─── CONFIG ────────────────────────────────────────────────────────────────────

from construct_portfolio.embed_port_tc.utils import (
    load_and_clean,
    build_monthly_returns,
    summarize_series
)

from construct_portfolio.config import (
    CONFIG,
    TC_MONTHLY_RETURNS_DIR,
    TC_RESULTS_CSV
)

TC_LOG_FILE = CONFIG["TC_LOG_FILE"]
os.makedirs(TC_MONTHLY_RETURNS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TC_LOG_FILE), exist_ok=True)

# ─── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=TC_LOG_FILE,
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)



def main():
    print('[START] Loading & cleaning datasets')
    Xtr, ytr, _, _ = load_and_clean(CONFIG["TRAIN_FEAT"], CONFIG["TRAIN_TARG"])
    Xte, yte, dte, tte = load_and_clean(CONFIG["TEST_FEAT"], CONFIG["TEST_TARG"])

    results = []
    combo_id = 0
    total = (len(CONFIG["N_EST_LIST"])
             *len(CONFIG["KNN_K_LIST"])
             *len(CONFIG["PCA_COMPONENTS"])
             *len(CONFIG["CLIP_QUANTILES"])
             *len(CONFIG["EDGE_THRESHOLDS"]))
    print(f'[LOOP] {total} combos')

    for n_est in CONFIG["N_EST_LIST"]:
        for k in CONFIG["KNN_K_LIST"]:
            for pca_n in CONFIG["PCA_COMPONENTS"]:
                # PCA
                if pca_n:
                    pca = PCA(n_components=pca_n)
                    Xtr_p = pca.fit_transform(Xtr)
                    Xte_p = pca.transform(Xte)
                else:
                    Xtr_p, Xte_p = Xtr, Xte

                # RF → KNN
                rf = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
                rf.fit(Xtr_p, ytr)
                pred_rf = rf.predict(Xtr_p)

                knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
                knn.fit(Xtr_p, pred_rf)
                pred_te = knn.predict(Xte_p)

                for cq in CONFIG["CLIP_QUANTILES"]:
                    if cq>0:
                        low, high = np.quantile(yte, cq), np.quantile(yte,1-cq)
                        mask_c = (yte>=low)&(yte<=high)
                    else:
                        mask_c = np.ones_like(yte, bool)
                    y_c = yte[mask_c]
                    d_c = dte[mask_c]
                    t_c = tte[mask_c]
                    p_c = pred_te[mask_c]

                    for thr in CONFIG["EDGE_THRESHOLDS"]:
                        combo_id += 1
                        port_s, bench_s, to_s, c_s = build_monthly_returns(d_c, t_c, y_c, p_c, thr)
                        # net series
                        net_s = port_s - to_s*CONFIG["TRANSACTION_COST"]

                        mu_g, sd_g, sp_g = summarize_series(port_s)
                        mu_n, sd_n, sp_n = summarize_series(net_s)

                        results.append({
                            'n_estimators':   n_est,
                            'knn_neighbors':  k,
                            'pca_components': pca_n or '',
                            'clip_quantile':  cq,
                            'edge_threshold': thr,
                            'mean_gross':     mu_g,
                            'sharpe_gross':   sp_g,
                            'mean_net':       mu_n,
                            'sharpe_net':     sp_n,
                            'avg_turnover':   to_s.mean(),
                            'avg_cost':      c_s.mean()
                        })

                        # save monthly CSV
                        df_out = pd.DataFrame({
                            'Month':    port_s.index,
                            'Gross':    port_s.values,
                            'Net':      net_s.values,
                            'Turnover': to_s.values,
                            'Cost':     c_s.values
                        })
                        fname = f"{combo_id:03d}_RF{n_est}_K{k}_PCA{pca_n or 'None'}_CQ{cq:.2f}_THR{thr:.2f}.csv"
                        df_out.to_csv(os.path.join(TC_MONTHLY_RETURNS_DIR, fname), index=False)

    pd.DataFrame(results).to_csv(TC_RESULTS_CSV, index=False, float_format='%.15g')
    print('[DONE] grid_search_results.csv')

if __name__=='__main__':
    try:
        main()
    except Exception:
        logging.exception('Unhandled exception occurred')
        raise