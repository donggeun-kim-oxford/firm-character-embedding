import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib
from math import sqrt

def get_numeric_cols(sample_file):
    sample_df = pd.read_csv(sample_file, nrows=1000, low_memory=False)
    numeric_cols = [c for c in sample_df.columns if c.startswith('dim_') and pd.api.types.is_numeric_dtype(sample_df[c])]
    return numeric_cols

def adjust_predictions(preds, min_avg=0.01, factor=100):
    if np.mean(np.abs(preds)) < min_avg:
        return preds * factor
    return preds

def clip_and_filter_predictions(preds, clip_quantile, threshold_neg, threshold_pos):
    if np.ptp(preds) < 1e-4:
        return np.where(preds > 0, 1, np.where(preds < 0, -1, 0))
    lower = np.quantile(preds, clip_quantile)
    upper = np.quantile(preds, 1 - clip_quantile)
    preds_clipped = np.clip(preds, lower, upper)
    return np.array([-1 if p < threshold_neg else (1 if p > threshold_pos else 0) for p in preds_clipped])

def build_monthly_returns_incremental(test_files, preds_func, clip_quantile, threshold_neg, threshold_pos, max_short_frac, max_weight, chunksize=10000):
    port_all = []
    months_all = []
    test_rows = 0
    for file in test_files:
        if not os.path.isfile(file):
            continue
        for chunk in pd.read_csv(file, chunksize=chunksize, low_memory=False):
            if 'date' in chunk.columns:
                chunk['date'] = pd.to_datetime(chunk['date'])
                chunk = chunk.dropna(subset=['date'])
                chunk = chunk.set_index('date', drop=True)
            else:
                continue
            X_te = chunk.filter(regex=r'^dim_')
            y_te = chunk['ret_exc_lead1m']
            tickers = chunk['ticker']
            dates = chunk.index
            mask_te = X_te.notna().all(axis=1)
            X_te, y_te, tickers, dates = X_te.loc[mask_te], y_te.loc[mask_te], tickers.loc[mask_te], dates[mask_te]
            if len(X_te) == 0 or X_te.shape[1] == 0:
                continue
            X_te = X_te.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            test_rows += len(X_te)
            preds = adjust_predictions(preds_func(X_te.values))
            signals = clip_and_filter_predictions(preds, clip_quantile, threshold_neg, threshold_pos)
            df = pd.DataFrame({'date': dates, 'R': y_te, 'signal': signals, 'ticker': tickers})
            df['Month'] = df['date'].dt.to_period('M')
            for mon, grp in df.groupby('Month'):
                r = grp['R'].values
                sig = grp['signal'].values
                mask = sig != 0
                if mask.sum() > 0:
                    w = sig[mask] / mask.sum()
                    w = np.clip(w, -max_weight, max_weight)
                    w_short = w[w < 0]
                    short_frac = -w_short.sum() if len(w_short) > 0 else 0
                    if short_frac > max_short_frac:
                        scale = max_short_frac / short_frac
                        w[w < 0] *= scale
                    if np.sum(np.abs(w)) > 0:
                        w /= np.sum(np.abs(w))
                    port_all.append(np.dot(w, r[mask]))
                else:
                    port_all.append(0.0)
                months_all.append(mon)
    print(f"Total test rows used: {test_rows}")
    idx = pd.Index(months_all, name='Month')
    return pd.Series(port_all, index=idx)

def compute_perf(returns):
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    m = returns.mean()
    s = returns.std()
    sh = (m / s * sqrt(12)) if s != 0 else 0.0
    return m, s, sh

def get_row_key(row):
    return (row['n_estimators'], row['pca_components'], row['max_short_frac'], row['max_weight'],
            row['clip_quantile'], row['threshold_neg'], row['threshold_pos'], row['rolling_window'])