#!/usr/bin/env python
"""
Final version of classical_grid_search.py (revised to match original working code):
1. Loads raw data from CSVs (train/test features & targets).
2. Processes dates, filters, scaling.
3. Iterates only over the 243 unique embed-grid rows (de-duplicated).
4. Runs RF→KNN smoothing, CatBoost, LightGBM for each.
5. Computes monthly returns and summaries.
6. Writes exactly 243 rows to combined CSV, recreating header each run.
7. Copies and augments matching monthly-perf files.
"""

import os, time, gc, csv, shutil, warnings
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import itertools

from construct_portfolio.config import (
    CONFIG,
    ORIGINAL_GRID_EMBEDDING,
    ORIGINAL_MONTHLY_PERF_DIR,
    COMBINED_MONTHLY_PERF_DIR,
    COMBINED_TABLE_FIELDNAMES,
    COMBINED_RESULTS_CSV,
)
from construct_portfolio.embed_other_port.utils import (
    build_monthly_returns,
    summarize
)

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ─────────────────────────────────────────────────────────
# Setup output CSV & directories
# ─────────────────────────────────────────────────────────
# Ensure classical_results_csv exists (create empty file if missing)
if not os.path.exists(COMBINED_RESULTS_CSV):
    os.makedirs(os.path.dirname(COMBINED_RESULTS_CSV), exist_ok=True)
    open(COMBINED_RESULTS_CSV, 'w').close()

# ────────────────────────────────────────────────────────────────────────────────
# Load embedding grid and raw data
# ────────────────────────────────────────────────────────────────────────────────
df_embed = pd.read_csv(ORIGINAL_GRID_EMBEDDING)

Xtr_full = pd.read_csv(CONFIG["TRAIN_FEAT"], parse_dates=['Date'], index_col='Date')
Xte_full = pd.read_csv(CONFIG["TEST_FEAT"],  parse_dates=['Date'], index_col='Date')
ytr_full = pd.read_csv(CONFIG["TRAIN_TARG"], parse_dates=['Date'], index_col='Date')['Return']
yte_full = pd.read_csv(CONFIG["TEST_TARG"],  parse_dates=['Date'], index_col='Date')['Return']

Xtr = Xtr_full.filter(regex=r'^dim_')
Xte = Xte_full.filter(regex=r'^dim_')

mask_tr = Xtr.notna().all(axis=1) & (ytr_full.abs()>1e-4)
mask_te = Xte.notna().all(axis=1) & (yte_full.abs()>1e-4)
Xtr, ytr = Xtr.loc[mask_tr], ytr_full.loc[mask_tr]
Xte, yte = Xte.loc[mask_te], yte_full.loc[mask_te]

scaler   = StandardScaler()
Xtr_s    = pd.DataFrame(scaler.fit_transform(Xtr), index=Xtr.index, columns=Xtr.columns)
Xte_s    = pd.DataFrame(scaler.transform(Xte),    index=Xte.index, columns=Xte.columns)

# ────────────────────────────────────────────────────────────────────────────────
# Hyperparameter grid
# ────────────────────────────────────────────────────────────────────────────────
grid = {
  'n_estimators':  CONFIG["N_EST_LIST"],
  'knn_neighbors': CONFIG["KNN_K_LIST"],
  'rolling_window':[CONFIG["Roll_Window"]],
  'pca_components':CONFIG["PCA_COMPONENTS"],
  'max_short_frac':[CONFIG["MAX_SHORT_FRAC"]],
  'max_weight':    [CONFIG["MAX_WEIGHT"]],
  'clip_quantile': CONFIG["CLIP_QUANTILES"],
  'edge_threshold':CONFIG["EDGE_THRESHOLDS"]
}
keys, vals    = zip(*grid.items())
combinations  = [dict(zip(keys,v)) for v in itertools.product(*vals)]
print(f"Total combinations: {len(combinations)}")

# ────────────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────────────
start = time.time()
for i, params in enumerate(combinations, start=1):
    n, k    = params['n_estimators'], params['knn_neighbors']
    pcaN    = params['pca_components']
    cq, thr = params['clip_quantile'], params['edge_threshold']
    neg     = -thr

    # Skip combos not in embedding grid
    mask_e = (
      (df_embed['n_estimators']   == n            ) &
      (df_embed['knn_neighbors']  == k            ) &
      (df_embed['pca_components'].fillna('') == ('' if pcaN is None else pcaN)) &
      (np.isclose(df_embed['clip_quantile'],  cq)) &
      (np.isclose(df_embed['edge_threshold'], thr))
    )
    if not mask_e.any():
        continue

    # Prepare features
    if pcaN is not None:
        pca  = PCA(n_components=pcaN)
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
    else:
        Xtr_p = Xtr_s.values
        Xte_p = Xte_s.values

    # RF + KNN smoothing
    rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(Xtr_p, ytr)
    rf_te = rf.predict(Xte_p)
    knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
    knn.fit(Xtr_p, rf.predict(Xtr_p))
    pred_te = knn.predict(Xte_p)

    # CatBoost & LightGBM
    cat = CatBoostRegressor(iterations=n, random_seed=42, verbose=0)
    cat.fit(Xtr_p, ytr)
    cat_te = cat.predict(Xte_p)

    lgb = LGBMRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    lgb.fit(Xtr_p, ytr)
    lgb_te = lgb.predict(Xte_p)

    # Monthly returns
    rf_port, _      = build_monthly_returns(Xte.index, yte.values,   pred_te, cq, neg, thr)
    cat_port, _     = build_monthly_returns(Xte.index, yte.values,   cat_te, cq, neg, thr)
    lgb_port, _     = build_monthly_returns(Xte.index, yte.values,  lgb_te, cq, neg, thr)
    emb_row         = df_embed.iloc[i-1]

    # Summarize
    em_mu, em_sd, em_sp   = emb_row[['mean_test','std_test','sharpe_test']]
    ew_mu, ew_sd, ew_sp   = emb_row[['mean_bench_test','std_bench_test','sharpe_bench_test']]
    rf_mu, rf_sd, rf_sp   = summarize(rf_port)
    cat_mu,cat_sd,cat_sp  = summarize(cat_port)
    lgb_mu,lgb_sd,lgb_sp  = summarize(lgb_port)

    # Append to classical_results_csv, writing header if file is empty
    with open(COMBINED_RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_TABLE_FIELDNAMES)
        # If file is empty, write header
        if f.tell() == 0:
            writer.writeheader()
        row = {
          'n_estimators':n,'knn_neighbors':k,'pca_components':pcaN or '',
          'max_short_frac':params['max_short_frac'],'max_weight':params['max_weight'],
          'clip_quantile':cq,'threshold_neg':neg,'threshold_pos':thr,'rolling_window':'',
          'embed_mean':em_mu,'embed_std':em_sd,'embed_sharpe':em_sp,
          'ew_mean':ew_mu,'ew_std':ew_sd,'ew_sharpe':ew_sp,
          'rf_mean':rf_mu,'rf_std':rf_sd,'rf_sharpe':rf_sp,
          'cat_mean':cat_mu,'cat_std':cat_sd,'cat_sharpe':cat_sp,
          'light_mean':lgb_mu,'light_std':lgb_sd,'light_sharpe':lgb_sp
        }
        writer.writerow(row)

    # Copy & augment monthly‐performance file
    base = f"{i:03d}_RF{n}_K{k}_PCA{pcaN or 'None'}_CQ{cq:0.2f}_THR{thr:0.2f}"
    orig = os.path.join(ORIGINAL_MONTHLY_PERF_DIR, f"{base}_perf.csv")
    os.makedirs(COMBINED_MONTHLY_PERF_DIR, exist_ok=True)
    newf = os.path.join(COMBINED_MONTHLY_PERF_DIR,    f"{base}_perf.csv")
    if os.path.exists(orig):
        shutil.copy(orig, newf)
        dfm = pd.read_csv(newf, parse_dates=['Month'])
        dfm['Month'] = dfm['Month'].dt.to_period('M')
        dfm.set_index('Month', inplace=True)

        # RF
        dfm['Monthly_Return_rf'] = rf_port.reindex(dfm.index)
        dfm['Equity_rf']        = (1 + dfm['Monthly_Return_rf']).cumprod()
        dfm['Run_STD_rf']       = rf_port.expanding(min_periods=1).std(ddof=1).reindex(dfm.index)
        dfm['Run_Sharpe_rf']    = (rf_port.expanding(min_periods=1).mean() / dfm['Run_STD_rf'] * sqrt(12)).reindex(dfm.index)

        # CatBoost
        dfm['Monthly_Return_cat'] = cat_port.reindex(dfm.index)
        dfm['Equity_cat']         = (1 + dfm['Monthly_Return_cat']).cumprod()
        dfm['Run_STD_cat']        = cat_port.expanding(min_periods=1).std(ddof=1).reindex(dfm.index)
        dfm['Run_Sharpe_cat']     = (cat_port.expanding(min_periods=1).mean() / dfm['Run_STD_cat'] * sqrt(12)).reindex(dfm.index)

        # LightGBM
        dfm['Monthly_Return_light'] = lgb_port.reindex(dfm.index)
        dfm['Equity_light']         = (1 + dfm['Monthly_Return_light']).cumprod()
        dfm['Run_STD_light']        = lgb_port.expanding(min_periods=1).std(ddof=1).reindex(dfm.index)
        dfm['Run_Sharpe_light']     = (lgb_port.expanding(min_periods=1).mean() / dfm['Run_STD_light'] * sqrt(12)).reindex(dfm.index)

        dfm.to_csv(newf)

    gc.collect()

print("Done in", time.time() - start, "seconds")
