
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
    COMPANY_JSON,
    DATE_JSON,
    TRAIN_FILES_JSON,
    TEST_FILES_JSON,
    NEIGHBOR_OUTPUT_DIR,
    GICS_HIERARCHY,
    RAW_DATA_PATTERN
)
from analysis.embed_analysis.utils import (
    extract_date_set,
    load_memmap,
    parse_key,
    build_ticker_industry_map,
    parse_gics_hierarchy
)
# ─── Config ────────────────────────────────────────────────────────────────
os.makedirs(NEIGHBOR_OUTPUT_DIR, exist_ok=True)
K_Neighbors = CONFIG["K_Neighbors"]
RAW_CHUNK_SIZE = CONFIG["RAW_CHUNK_SIZE"]
# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    # 1) Load embeddings & metadata
    X_all = load_memmap(EMB_PATH)
    with open(RET_JSON)     as f: row_to_ret     = json.load(f)
    with open(COMPANY_JSON) as f: row_to_company = json.load(f)
    with open(DATE_JSON)    as f: row_to_date    = json.load(f)

    keys      = list(row_to_ret.keys())
    parsed    = [parse_key(k) for k in keys]
    dates     = np.array([row_to_date[k] for k in keys], dtype='datetime64[ns]')
    companies = np.array([c for _,c,_ in parsed])
    tickers   = np.array([t for _,_,t in parsed])

    # 2) Train/test split by date
    train_ds = extract_date_set(TRAIN_FILES_JSON)
    test_ds  = extract_date_set(TEST_FILES_JSON)
    ds_str   = dates.astype('datetime64[D]').astype(str)
    is_train = np.isin(ds_str, list(train_ds))
    is_test  = np.isin(ds_str, list(test_ds))

    X_tr, X_te         = X_all[is_train], X_all[is_test]
    comp_tr, comp_te   = companies[is_train], companies[is_test]
    tick_tr, tick_te   = tickers[is_train], tickers[is_test]

    # 3) Build ticker → GICS map
    ind_map = build_ticker_industry_map(RAW_DATA_PATTERN, RAW_CHUNK_SIZE)
    g_tr = np.array([ind_map.get(t, '') for t in tick_tr])
    g_te = np.array([ind_map.get(t, '') for t in tick_te])

    # 4) Extract codes at each level
    sec_tr      = np.array([c[:2] for c in g_tr]);      sec_te      = np.array([c[:2] for c in g_te])
    group_tr    = np.array([c[:4] for c in g_tr]);      group_te    = np.array([c[:4] for c in g_te])
    industry_tr = np.array([c[:6] for c in g_tr]);      industry_te = np.array([c[:6] for c in g_te])
    sub_tr      =            g_tr.copy();                sub_te      =            g_te.copy()

    # 5) Normalize embeddings
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 6) KNN fit & query
    nbrs = NearestNeighbors(n_neighbors=K_Neighbors).fit(X_tr_s)
    idxs = nbrs.kneighbors(X_te_s, return_distance=False)

    # fraction helper
    def frac_match(arr_tr, arr_te):
        return (arr_tr[idxs] == arr_te[:, None]).mean(axis=1)

    # 7) Build main DataFrame
    df = pd.DataFrame({
        'date':               dates[is_test].astype(str),
        'company':            comp_te,
        'ticker':             tick_te,
        'gics_code':          g_te,
        'frac_same_company':  frac_match(tick_tr,     tick_te),
        'frac_same_sector':   frac_match(sec_tr,      sec_te),
        'frac_same_group':    frac_match(group_tr,    group_te),
        'frac_same_industry': frac_match(industry_tr, industry_te),
        'frac_same_sub':      frac_match(sub_tr,      sub_te),
    })

    # 8) Validation
    for lvl in ('sector','group','industry','sub'):
        if (df['frac_same_company'] > df[f'frac_same_{lvl}']).any():
            raise RuntimeError(f"Company fraction exceeds {lvl} fraction!")

    df.to_csv(os.path.join(NEIGHBOR_OUTPUT_DIR, "knn_neighbor_stats.csv"), index=False)
    print("Saved knn_neighbor_stats.csv")

    # 9) Build GICS maps
    sector_map, group_map, industry_map, subind_map = parse_gics_hierarchy(GICS_HIERARCHY)

    # 10) Annotate with words
    df['sector']         = df['gics_code'].str[:2].map(sector_map).fillna('N/A')
    df['industry_group'] = df['gics_code'].str[:4].map(group_map).fillna('N/A')
    df['industry']       = df['gics_code'].str[:6].map(industry_map).fillna('N/A')
    df['sub_industry']   = df['gics_code'].str[:8].map(subind_map).fillna('N/A')

    # 11) Aggregate stats per level
    levels = [
        ('sector',         'frac_same_sector'),
        ('industry_group', 'frac_same_group'),
        ('industry',       'frac_same_industry'),
        ('sub_industry',   'frac_same_sub'),
    ]
    out = []
    for lvl, metric in levels:
        grp = df.groupby(lvl).agg(
            frac_same_company=('frac_same_company','mean'),
            level_frac=(metric,      'mean')
        ).reset_index().rename(columns={lvl:'name'})
        grp['level'] = lvl
        out.append(grp)

    summary = pd.concat(out, ignore_index=True)
    summary.to_csv(os.path.join(NEIGHBOR_OUTPUT_DIR, "gics_level_summary.csv"), index=False)
    print("Saved gics_level_summary.csv")

    # 12) Plot each level, sub_industry extra wide
    for lvl, metric in levels:
        sub = summary[summary['level']==lvl].sort_values('level_frac', ascending=False)
        width = 40 if lvl=='sub_industry' else 12
        plt.figure(figsize=(width,6))
        x = np.arange(len(sub))
        plt.bar(x-0.2, sub['frac_same_company'], 0.4, label='Same Company')
        plt.bar(x+0.2, sub['level_frac'],      0.4, label=lvl.replace('_',' ').title())
        plt.xticks(x, sub['name'], rotation=45, ha='right')
        plt.xlabel(lvl.replace('_',' ').title())
        plt.ylabel("Mean fraction")
        plt.title(f"Neighbors: Company vs {lvl.replace('_',' ').title()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(NEIGHBOR_OUTPUT_DIR, f"{lvl}_compare.png"))
        plt.close()
        print(f"Saved {lvl}_compare.png")

if __name__=="__main__":
    main()