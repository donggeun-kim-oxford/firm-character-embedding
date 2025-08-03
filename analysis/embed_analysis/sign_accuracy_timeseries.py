#!/usr/bin/env python3
# code10_sign_accuracy_over_time_with_baseline_and_shading.py

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
    DATE_JSON,
    TRAIN_FILES_JSON,
    TEST_FILES_JSON,
    GICS_HIERARCHY,
    RAW_DATA_PATTERN,
    SIGN_TIME_OUTPUT_DIR,
)

from analysis.embed_analysis.utils import (
    extract_date_set,
    load_memmap,
    parse_gics_hierarchy,
    build_ticker_gics_map,
    sign
)

# ─── CONFIG ────────────────────────────────────────────────────────────────
os.makedirs(SIGN_TIME_OUTPUT_DIR, exist_ok=True)

RAW_CHUNK_SIZE   = CONFIG["RAW_CHUNK_SIZE"]
K = CONFIG["K_Neighbors"]  # number of neighbors

# ─── ECONOMIC REGIMES ──────────────────────────────────────────────────────
REGIMES = [
    ("Dotcom Bust", pd.Timestamp("2000-03-10"), pd.Timestamp("2002-10-31"), "#36dbef"),
    ("Global Financial Crisis", pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30"), "#6baed6"),
    ("COVID Crash", pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-07"), "#08519c"),
]

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    # 1) Load embeddings, returns, and dates
    X_all       = load_memmap(EMB_PATH)
    row_to_ret  = json.load(open(RET_JSON))
    row_to_date = json.load(open(DATE_JSON))
    keys        = list(row_to_ret.keys())

    rets_all = np.array([row_to_ret[k] for k in keys], dtype=float)
    dates     = np.array([pd.to_datetime(row_to_date[k]) for k in keys])
    tickers   = np.array([k.split('_')[-1] for k in keys])

    # 2) Train/test split
    tr_dates = extract_date_set(TRAIN_FILES_JSON)
    te_dates = extract_date_set(TEST_FILES_JSON)
    ds_str   = dates.astype('datetime64[D]').astype(str)
    is_tr    = np.isin(ds_str, list(tr_dates))
    is_te    = np.isin(ds_str, list(te_dates))

    X_tr, X_te         = X_all[is_tr], X_all[is_te]
    ret_tr, ret_te     = rets_all[is_tr], rets_all[is_te]
    tick_tr, tick_te   = tickers[is_tr], tickers[is_te]
    dates_tr, dates_te = dates[is_tr], dates[is_te]

    # 3) Remove outliers
    mask_tr = ((ret_tr < -1e-5) | (ret_tr > 1e-5)) & (ret_tr > -0.5) & (ret_tr < 0.5)
    X_tr, ret_tr, tick_tr, dates_tr = X_tr[mask_tr], ret_tr[mask_tr], tick_tr[mask_tr], dates_tr[mask_tr]
    mask_te = ((ret_te < -1e-5) | (ret_te > 1e-5)) & (ret_te > -0.5) & (ret_te < 0.5)
    X_te, ret_te, tick_te, dates_te = X_te[mask_te], ret_te[mask_te], tick_te[mask_te], dates_te[mask_te]

    # 4) Scale & KNN fit
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    nbrs   = NearestNeighbors(n_neighbors=K).fit(X_tr_s)
    idxs_tr_all = nbrs.kneighbors(X_tr_s, K+1, return_distance=False)[:, 1:]
    idxs_te      = nbrs.kneighbors(X_te_s,  K,   return_distance=False)

    # 5) Predicted sign & correctness (embedding)
    pred_tr = sign(ret_tr[idxs_tr_all].mean(axis=1))
    corr_tr = (pred_tr == sign(ret_tr)).astype(int)
    pred_te = sign(ret_tr[idxs_te].mean(axis=1))
    corr_te = (pred_te == sign(ret_te)).astype(int)

    # 6) Assemble DataFrame, include returns for baseline
    df_tr = pd.DataFrame({
        'set':      'train',
        'date':     dates_tr,
        'ticker':   tick_tr,
        'correct':  corr_tr,
        'return':   ret_tr
    })
    df_te = pd.DataFrame({
        'set':      'test',
        'date':     dates_te,
        'ticker':   tick_te,
        'correct':  corr_te,
        'return':   ret_te
    })
    df = pd.concat([df_tr, df_te], ignore_index=True)

    # 7) Map tickers → GICS codes & names
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

    # 8) Bin to month (true monthly dates)
    df['month'] = df['date'].values.astype('datetime64[M]')

    # 9) Compute “always-long” baseline correctness per sample
    df['baseline_correct'] = (df['return'] > 0).astype(int)

    # 10) Overall monthly sign accuracy (train/test separately)
    overall_rows = []
    for subset in ['train', 'test']:
        tmp = (
            df[df['set'] == subset]
              .groupby('month')['correct']
              .mean()
              .reset_index(name='accuracy_knn')
        )
        base = (
            df[df['set'] == subset]
              .groupby('month')['baseline_correct']
              .mean()
              .reset_index(name='accuracy_baseline')
        )
        merged = pd.merge(tmp, base, on='month')
        merged['set'] = subset
        overall_rows.append(merged)

    overall_ts = pd.concat(overall_rows, ignore_index=True)
    overall_ts.to_csv(os.path.join(SIGN_TIME_OUTPUT_DIR, 'overall_monthly_sign_accuracy.csv'), index=False)

    # 11) Mean accuracy by calendar month across years (Jan–Dec), embedding & baseline
    df['calendar_month'] = df['date'].dt.month
    moy_rows = []
    for subset in ['train', 'test']:
        emb = (
            df[df['set'] == subset]
              .groupby('calendar_month')['correct']
              .mean()
              .reset_index(name='accuracy_knn')
        )
        bl = (
            df[df['set'] == subset]
              .groupby('calendar_month')['baseline_correct']
              .mean()
              .reset_index(name='accuracy_baseline')
        )
        merged_moy = pd.merge(emb, bl, on='calendar_month')
        merged_moy['set'] = subset
        moy_rows.append(merged_moy)

    moy_ts = pd.concat(moy_rows, ignore_index=True)
    moy_ts.to_csv(os.path.join(SIGN_TIME_OUTPUT_DIR, 'accuracy_by_month_of_year.csv'), index=False)

    # 12) Plot overall monthly sign accuracy (no titles, lighter colors), WITH REGIME SHADING

    for subset in ['train', 'test']:
        subdf = overall_ts[overall_ts['set'] == subset]
        fig, ax = plt.subplots(figsize=(10, 5))

        # Determine x-limits for shading
        xmin = subdf['month'].min()
        xmax = subdf['month'].max()

        # Shade economic regimes
        for label, start, end, color in REGIMES:
            s = max(start, xmin)
            e = min(end,   xmax)
            if s < e:
                ax.axvspan(s, e, color=color, alpha=0.3, label=label)

        # Plot KNN vs. baseline lines
        ax.plot(subdf['month'], subdf['accuracy_knn'],
                color='#4c72b0', linewidth=1.5, label='Embedding-KNN')
        ax.plot(subdf['month'], subdf['accuracy_baseline'],
                color='#55a868', linewidth=1.5, linestyle='--', label='Always-Long Baseline')
        ax.axhline(0.5, color='black', linestyle='--', linewidth=1)

        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Mean Sign Accuracy", fontsize=12)
        ax.set_ylim(0.1, 0.9)  # <-- limit y-axis to 0.7
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

        # De-duplicate regime labels in legend
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_handles, new_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                new_handles.append(h)
                new_labels.append(l)
                seen.add(l)
        ax.legend(new_handles, new_labels, fontsize=10, loc='upper left', frameon=False)

        plt.tight_layout()
        fn = f"{subset}_overall_monthly_sign_accuracy_with_regimes.png"
        plt.savefig(os.path.join(SIGN_TIME_OUTPUT_DIR, fn))
        plt.close(fig)

    # 13) Plot month-of-year accuracy (grouped bar charts, no titles, y-axis ≤ 0.7)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bar_width = 0.35

    for subset in ['train', 'test']:
        subdf = moy_ts[moy_ts['set'] == subset].copy()
        # Ensure all 12 months are present
        subdf = subdf.set_index('calendar_month').reindex(np.arange(1, 13), fill_value=0).reset_index()
        inds = np.arange(1, 13)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(inds - bar_width/2,
               subdf['accuracy_knn'].values,
               width=bar_width,
               color='#4c72b0', alpha=0.7,
               label='Embedding-KNN')
        ax.bar(inds + bar_width/2,
               subdf['accuracy_baseline'].values,
               width=bar_width,
               color='#55a868', alpha=0.7,
               label='Always-Long Baseline')

        ax.set_xticks(inds)
        ax.set_xticklabels(month_names, fontsize=10)
        ax.axhline(0.5, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel("Mean Sign Accuracy", fontsize=12)
        ax.set_xlabel("Calendar Month", fontsize=12)
        ax.set_ylim(0, 0.7)  # <-- limit y-axis to 0.7
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10, frameon=False)

        plt.tight_layout()
        fn = f"{subset}_accuracy_by_month_of_year.png"
        plt.savefig(os.path.join(SIGN_TIME_OUTPUT_DIR, fn))
        plt.close(fig)

    # 14) Colored-regime plots per GICS level (no titles, unchanged)
    levels = ['sector', 'industry_group', 'industry', 'subindustry']
    for subset in ['train', 'test']:
        fig, ax = plt.subplots(figsize=(14, 6))
        subdf = overall_ts[overall_ts['set'] == subset]
        xmin = subdf['month'].min()
        xmax = subdf['month'].max()

        # Shade economic regimes
        for label, start, end, color in REGIMES:
            s = max(start, xmin)
            e = min(end, xmax)
            if s < e:
                ax.axvspan(s, e, color=color, alpha=0.3, label=label)

        # Plot mean sign accuracy by GICS level each month
        for lvl, color in zip(levels, ['C0', 'C1', 'C2', 'C3']):
            tmp = (
                df[(df['set'] == subset) & (df[lvl] != 'Unknown')]
                  .groupby('month')['correct']
                  .mean()
            )
            ax.plot(tmp.index, tmp.values, label=lvl.replace('_', ' ').title(), color=color)

        ax.axhline(0.5, color='red', linestyle='--', linewidth=1)
        handles, labels = ax.get_legend_handles_labels()
        seen, new_h, new_l = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                new_h.append(h)
                new_l.append(l)
                seen.add(l)
        ax.legend(new_h, new_l, loc='upper left', fontsize='small', frameon=False)
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Mean Sign Accuracy", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(xmin, xmax)
        plt.tight_layout()
        fn = f"{subset}_accuracy_time_series_with_regimes.png"
        plt.savefig(os.path.join(SIGN_TIME_OUTPUT_DIR, fn))
        plt.close(fig)

if __name__ == "__main__":
    main()