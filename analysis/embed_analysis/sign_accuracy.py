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
    SIGN_OUTPUT_DIR,
)

from analysis.embed_analysis.utils import (
    extract_date_set,
    load_memmap,
    parse_gics_hierarchy,
    parse_date,
    build_ticker_gics_map,
    sign
)
# ─── CONFIG ───────────────────────────────────────────────────────────────
os.makedirs(SIGN_OUTPUT_DIR, exist_ok=True)
RAW_CHUNK_SIZE = CONFIG["RAW_CHUNK_SIZE"]
K = CONFIG["K_Neighbors"]

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    # 1) Load embeddings & returns
    X_all      = load_memmap(EMB_PATH)
    row_to_ret = json.load(open(RET_JSON))

    keys        = list(row_to_ret.keys())
    rets_all    = np.array([row_to_ret[k] for k in keys], dtype=float)
    dates_all   = np.array([parse_date(k) for k in keys])
    tickers_all = np.array([k.split('_')[-1] for k in keys])

    # 2) Split train/test by dates
    train_ds = extract_date_set(TRAIN_FILES_JSON)
    test_ds  = extract_date_set(TEST_FILES_JSON)
    ds_str   = dates_all.astype('datetime64[D]').astype(str)
    is_tr    = np.isin(ds_str, list(train_ds))
    is_te    = np.isin(ds_str, list(test_ds))

    X_tr, X_te         = X_all[is_tr], X_all[is_te]
    ret_tr, ret_te     = rets_all[is_tr], rets_all[is_te]
    tick_tr, tick_te   = tickers_all[is_tr], tickers_all[is_te]
    dates_tr, dates_te = dates_all[is_tr], dates_all[is_te]

    # 3) Filter outliers
    mask_tr = ((ret_tr < -1e-5) | (ret_tr > 1e-5)) & (ret_tr > -0.5) & (ret_tr < 0.5)
    X_tr, ret_tr, tick_tr, dates_tr = X_tr[mask_tr], ret_tr[mask_tr], tick_tr[mask_tr], dates_tr[mask_tr]
    mask_te = ((ret_te < -1e-5) | (ret_te > 1e-5)) & (ret_te > -0.5) & (ret_te < 0.5)
    X_te, ret_te, tick_te, dates_te = X_te[mask_te], ret_te[mask_te], tick_te[mask_te], dates_te[mask_te]

    # 4) Standardize & fit KNN
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    nbrs   = NearestNeighbors(n_neighbors=K).fit(X_tr_s)

    # 5) Predict neighbors
    idxs_te     = nbrs.kneighbors(X_te_s, return_distance=False)
    idxs_tr_all = nbrs.kneighbors(X_tr_s, K+1, return_distance=False)
    idxs_tr     = idxs_tr_all[:, 1:]  # drop self

    # 6) Predicted sign & correctness
    pred_te   = sign(ret_tr[idxs_te].mean(axis=1))
    actual_te = sign(ret_te)
    corr_te   = (pred_te == actual_te).astype(int)

    pred_tr   = sign(ret_tr[idxs_tr].mean(axis=1))
    actual_tr = sign(ret_tr)
    corr_tr   = (pred_tr == actual_tr).astype(int)

    # 7) Build DataFrame
    df_tr = pd.DataFrame({'set':'train','date':dates_tr,'ticker':tick_tr,'correct':corr_tr})
    df_te = pd.DataFrame({'set':'test', 'date':dates_te, 'ticker':tick_te, 'correct':corr_te})
    df = pd.concat([df_tr, df_te], ignore_index=True)
    df.to_csv(os.path.join(SIGN_OUTPUT_DIR, "knn_sign_accuracy_per_sample.csv"), index=False)

    # 8) Map GICS codes
    tg_map = build_ticker_gics_map(RAW_DATA_PATTERN, RAW_CHUNK_SIZE)
    full_g = df['ticker'].map(tg_map).fillna('')
    df['sector_code']         = full_g.str[:2]
    df['industry_group_code'] = full_g.str[:4]
    df['industry_code']       = full_g.str[:6]
    df['subindustry_code']    = full_g.str[:8]
    sec_map, grp_map, ind_map, sub_map = parse_gics_hierarchy(GICS_HIERARCHY)
    df['sector']         = df['sector_code'].map(sec_map)
    df['industry_group'] = df['industry_group_code'].map(grp_map)
    df['industry']       = df['industry_code'].map(ind_map)
    df['subindustry']    = df['subindustry_code'].map(sub_map)

    # 9) Aggregate by GICS level & set
    levels = [
        ('sector','sector_code','sector'),
        ('industry_group','industry_group_code','industry_group'),
        ('industry','industry_code','industry'),
        ('subindustry','subindustry_code','subindustry')
    ]
    all_summaries = []
    for lvl, code_col, name_col in levels:
        for subset in ['train','test']:
            subdf = df[(df['set']==subset) & df[name_col].notna()]
            grp = (
                subdf
                .groupby(code_col)['correct']
                .agg(sign_accuracy='mean', n_samples='count')
                .reset_index()
            )
            grp = grp[grp['n_samples']>0]
            grp[name_col] = grp[code_col].map(dict(subdf[[code_col,name_col]].drop_duplicates().values))
            grp['level'], grp['set'] = lvl, subset
            all_summaries.append(grp[[ 'set','level',code_col,name_col,'sign_accuracy','n_samples' ]])
    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(os.path.join(SIGN_OUTPUT_DIR, "gics_sign_accuracy_summary.csv"), index=False)

    # 10) Per-level descriptive stats CSV
    for subset in ['train','test']:
        stats = (
            summary[summary['set']==subset]
            .groupby('level')['sign_accuracy']
            .describe()[['count','mean','std','min','25%','50%','75%','max']]
            .round(4)
        )
        stats.to_csv(os.path.join(SIGN_OUTPUT_DIR, f"sign_accuracy_by_level_stats_{subset}.csv"))

    # 11) Bar-plots by level & set, and save the DataFrame for each
    for subset in ['train','test']:
        for lvl, _, name_col in levels:
            sub = summary[(summary['set']==subset) & (summary['level']==lvl)]
            if sub.empty:
                continue
            # Save DataFrame used for plot
            sub.to_csv(os.path.join(SIGN_OUTPUT_DIR, f"{subset}_{lvl}_sign_accuracy.csv"), index=False)

            # Plot
            plt.figure(figsize=(12,6))
            plt.bar(sub[name_col], sub['sign_accuracy'], color='C0')
            plt.axhline(0.5, color='red', linestyle='--', linewidth=1)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Accuracy")
            plt.ylim(0,1)
            plt.tight_layout()
            plt.savefig(os.path.join(SIGN_OUTPUT_DIR, f"{subset}_{lvl}_sign_accuracy.png"))
            plt.close()

    # 12) Overall distribution histograms, and save the series for each
    for subset in ['train','test']:
        data = summary[summary['set']==subset]['sign_accuracy']
        # Save the raw data
        data_df = data.reset_index(drop=True).to_frame("sign_accuracy")
        data_df.to_csv(os.path.join(SIGN_OUTPUT_DIR, f"{subset}_overall_sign_accuracy_distribution.csv"), index=False)

        # Plot histogram
        plt.figure(figsize=(10,6))
        plt.hist(data, bins=20, density=True, alpha=0.7)
        plt.axhline(0.5, color='red', ls='--', lw=1)
        plt.xlabel("Sign Accuracy")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(SIGN_OUTPUT_DIR, f"{subset}_overall_sign_accuracy_distribution.png"))
        plt.close()

    # 13) Overall descriptive stats CSV (already saved in step 10 but repeat if needed)
    for subset in ['train','test']:
        data = summary[summary['set']==subset]['sign_accuracy']
        stats = data.describe()[['count','mean','std','min','25%','50%','75%','max']].round(4)
        stats.to_csv(os.path.join(SIGN_OUTPUT_DIR, f"overall_sign_accuracy_stats_{subset}.csv"))

    print(f"All CSVs and plots saved under '{SIGN_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()