#!/usr/bin/env python3
# code11_combined_neighbor_analysis.py

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from analysis.embed_analysis.config import (
    CONFIG,
    EMB_PATH,
    RET_JSON,
    TRAIN_FILES_JSON,
    TEST_FILES_JSON,
    GICS_HIERARCHY,
    RAW_DATA_PATTERN,
    MEAN_TIME_DIFF_OUTPUT_DIR,
)

from analysis.embed_analysis.utils import (
    extract_date_set,
    load_memmap,
    parse_gics_hierarchy,
    parse_date,
    build_ticker_gics_map,
)
# ─── CONFIG ────────────────────────────────────────────────────────────────
os.makedirs(MEAN_TIME_DIFF_OUTPUT_DIR, exist_ok=True)

RAW_CHUNK_SIZE   = CONFIG["RAW_CHUNK_SIZE"]
K                = CONFIG["K_Neighbors"]


# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    # 1) load embeddings & returns
    X_all      = load_memmap(EMB_PATH)
    row_to_ret = json.load(open(RET_JSON))
    keys       = list(row_to_ret.keys())
    rets_all   = np.array([row_to_ret[k] for k in keys], dtype=float)
    dates_all  = np.array([parse_date(k) for k in keys])

    # 2) train/test split
    tr_dates = extract_date_set(TRAIN_FILES_JSON)
    te_dates = extract_date_set(TEST_FILES_JSON)
    ds_str   = dates_all.astype('datetime64[D]').astype(str)
    is_tr    = np.isin(ds_str, list(tr_dates))
    is_te    = np.isin(ds_str, list(te_dates))

    X_tr, X_te         = X_all[is_tr], X_all[is_te]
    ret_tr, ret_te     = rets_all[is_tr], rets_all[is_te]
    dates_tr, dates_te = dates_all[is_tr], dates_all[is_te]
    keys_tr = [keys[i] for i, v in enumerate(is_tr) if v]
    keys_te = [keys[i] for i, v in enumerate(is_te) if v]

    # 3) drop outliers
    mask_tr = ((ret_tr < -1e-5) | (ret_tr > 1e-5)) & (ret_tr > -0.5) & (ret_tr < 0.5)
    X_tr, ret_tr, dates_tr, keys_tr = (
        X_tr[mask_tr], ret_tr[mask_tr], dates_tr[mask_tr],
        [k for k, m in zip(keys_tr, mask_tr) if m]
    )
    mask_te = ((ret_te < -1e-5) | (ret_te > 1e-5)) & (ret_te > -0.5) & (ret_te < 0.5)
    X_te, ret_te, dates_te, keys_te = (
        X_te[mask_te], ret_te[mask_te], dates_te[mask_te],
        [k for k, m in zip(keys_te, mask_te) if m]
    )

    # 4) scale & fit KNN
    scaler      = StandardScaler()
    X_tr_s      = scaler.fit_transform(X_tr)
    X_te_s      = scaler.transform(X_te)
    nbrs        = NearestNeighbors(n_neighbors=K).fit(X_tr_s)
    idxs_tr_all = nbrs.kneighbors(X_tr_s, n_neighbors=K+1, return_distance=False)[:,1:]
    idxs_te     = nbrs.kneighbors(X_te_s,  n_neighbors=K,   return_distance=False)

    # 5) mean time-difference helper
    def mean_time_diff(idxs, neighbor_dates, target_dates=None):
        if target_dates is None:
            target_dates = neighbor_dates
        diffs = np.abs(neighbor_dates[idxs] - target_dates[:,None])
        # diffs is array of numpy.timedelta64; convert with .astype or .days
        return np.array([d.astype('timedelta64[D]').astype(int).mean() for d in diffs])

    mtd_tr = mean_time_diff(idxs_tr_all, dates_tr)
    mtd_te = mean_time_diff(idxs_te,      dates_tr, dates_te)

    # 6) assemble per-sample DataFrame
    df_tr = pd.DataFrame({
        'set':       'train',
        'key':       keys_tr,
        'date':      dates_tr,
        'time_diff': mtd_tr
    })
    df_te = pd.DataFrame({
        'set':       'test',
        'key':       keys_te,
        'date':      dates_te,
        'time_diff': mtd_te
    })
    df = pd.concat([df_tr, df_te], ignore_index=True)
    df['ticker'] = df['key'].apply(lambda k: k.split('_')[-1])

    # 7) map GICS
    tg_map = build_ticker_gics_map(RAW_DATA_PATTERN, RAW_CHUNK_SIZE)
    g_all  = df['ticker'].map(tg_map).fillna('')
    df['sector_code']         = g_all.str[:2]
    df['industry_group_code'] = g_all.str[:4]
    df['industry_code']       = g_all.str[:6]
    df['subindustry_code']    = g_all.str[:8]
    sec_map, grp_map, ind_map, sub_map = parse_gics_hierarchy(GICS_HIERARCHY)
    df['sector']         = df['sector_code'].map(sec_map).fillna('Unknown')
    df['industry_group'] = df['industry_group_code'].map(grp_map).fillna('Unknown')
    df['industry']       = df['industry_code'].map(ind_map).fillna('Unknown')
    df['subindustry']    = df['subindustry_code'].map(sub_map).fillna('Unknown')

    df.to_csv(os.path.join(MEAN_TIME_DIFF_OUTPUT_DIR, "knn_time_diff_per_sample.csv"), index=False)
    print("Saved per-sample → knn_time_diff_per_sample.csv")

    # 8) summary by GICS level
    levels = [
      ('sector',         'sector_code',         'sector'),
      ('industry_group', 'industry_group_code', 'industry_group'),
      ('industry',       'industry_code',       'industry'),
      ('subindustry',    'subindustry_code',    'subindustry'),
    ]
    all_summary = []
    for lvl, code_col, name_col in levels:
        for subset in ['train','test']:
            subdf = df[(df['set']==subset) &
                       (df[code_col] != '') &
                       (df[name_col] != 'Unknown')]
            grp = subdf.groupby(code_col)['time_diff'] \
                       .agg(mean_td='mean', std_td='std', count='count') \
                       .reset_index()
            name_map = dict(subdf[[code_col,name_col]].drop_duplicates().values)
            grp[name_col] = grp[code_col].map(name_map)
            grp['level'] = lvl
            grp['set']   = subset
            all_summary.append(grp[['set','level',code_col,name_col,'mean_td','std_td','count']])
    summary = pd.concat(all_summary, ignore_index=True)
    summary.to_csv(os.path.join(MEAN_TIME_DIFF_OUTPUT_DIR, "knn_time_diff_summary_by_level.csv"), index=False)
    print("Saved summary → knn_time_diff_summary_by_level.csv")

    # 9) time-series of mean time-diff per month
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
    ts_rows = []
    for subset in ['train','test']:
        for lvl in ['sector','industry_group','industry','subindustry']:
            tmp = (
                df[(df['set']==subset) & (df[lvl] != 'Unknown')]
                  .groupby('month')['time_diff']
                  .mean()
                  .rename(lvl)
                  .reset_index()
            )
            tmp['level'] = lvl
            tmp['set']   = subset
            ts_rows.append(tmp)
    ts = pd.concat(ts_rows, ignore_index=True)
    ts.to_csv(os.path.join(MEAN_TIME_DIFF_OUTPUT_DIR, "knn_time_diff_timeseries.csv"), index=False)
    print("Saved time-series → knn_time_diff_timeseries.csv")

    # 10a) TRAIN raw
    plt.figure(figsize=(14,6))
    for lvl, color in zip(['sector','industry_group','industry','subindustry'],
                          ['C0','C1','C2','C3']):
        sub_ts = ts[(ts['set']=='train') & (ts['level']==lvl)]
        plt.plot(sub_ts['month'], sub_ts[lvl],
                 label=lvl.replace('_',' ').title(), color=color)
    plt.title("Train: Mean Neighbor Time-Difference by GICS Level")
    plt.ylabel("Mean |Δ days|")
    plt.xlabel("Month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MEAN_TIME_DIFF_OUTPUT_DIR, "train_time_diff_timeseries.png"))
    plt.close()
    print("Saved plot → train_time_diff_timeseries.png")

    # 10b) TEST raw
    plt.figure(figsize=(14,6))
    for lvl, color in zip(['sector','industry_group','industry','subindustry'],
                          ['C0','C1','C2','C3']):
        sub_ts = ts[(ts['set']=='test') & (ts['level']==lvl)]
        plt.plot(sub_ts['month'], sub_ts[lvl],
                 label=lvl.replace('_',' ').title(), color=color)
    plt.title("Test: Mean Neighbor Time-Difference by GICS Level (Raw)")
    plt.ylabel("Mean |Δ days|")
    plt.xlabel("Month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MEAN_TIME_DIFF_OUTPUT_DIR, "test_time_diff_raw.png"))
    plt.close()
    print("Saved plot → test_time_diff_raw.png")

    # 10c) TEST demeaned vs training-end gap
    train_end_month = pd.to_datetime(dates_tr.max()).to_period('M').to_timestamp()
    # baseline gap in days for each test-month
    baseline = {
        m: (m - train_end_month).days
        for m in ts[ts['set']=='test']['month'].unique()
    }

    plt.figure(figsize=(14,6))
    for lvl, color in zip(['sector','industry_group','industry','subindustry'],
                          ['C0','C1','C2','C3']):
        sub_ts = ts[(ts['set']=='test') & (ts['level']==lvl)]
        demeaned = sub_ts[lvl] - sub_ts['month'].map(baseline)
        plt.plot(sub_ts['month'], demeaned,
                 label=lvl.replace('_',' ').title() + " (demeaned)", color=color)
    plt.title("Test: Mean Neighbor Time-Difference by GICS Level (Demeaned)")
    plt.ylabel("Mean |Δ days| minus gap to train-end")
    plt.xlabel("Month")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MEAN_TIME_DIFF_OUTPUT_DIR, "test_time_diff_demeaned.png"))
    plt.close()
    print("Saved plot → test_time_diff_demeaned.png")


if __name__ == "__main__":
    main()