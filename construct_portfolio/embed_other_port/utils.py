#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd


def clip_and_signal(preds, clip_q, thr_neg, thr_pos):
    if np.ptp(preds) < 1e-4:
        return np.sign(preds)
    lo, hi = np.quantile(preds, clip_q), np.quantile(preds, 1-clip_q)
    clipped = np.clip(preds, lo, hi)
    return np.array([-1 if p < thr_neg else (1 if p > thr_pos else 0) for p in clipped])

def build_monthly_returns(dates, rets, preds, clip_q, thr_neg, thr_pos):
    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'R': rets, 'P': preds})
    df['Month'] = df['Date'].dt.to_period('M')
    port, bench, months = [], [], []
    for mon, grp in df.groupby('Month'):
        sig = clip_and_signal(grp['P'].values, clip_q, thr_neg, thr_pos)
        r   = grp['R'].values
        mask = sig != 0
        if mask.sum()>0:
            w = sig[mask] / mask.sum()
            port.append(np.dot(w, r[mask]))
        else:
            port.append(0.0)
        bench.append(r.mean() if len(r)>0 else 0.0)
        months.append(mon)
    idx = pd.Index(months, name='Month')
    return pd.Series(port,  index=idx), pd.Series(bench, index=idx)

def summarize(s):
    mu = s.mean()
    sd = s.std(ddof=1)
    return mu, sd, (mu/sd*np.sqrt(12) if sd!=0 else np.nan)

