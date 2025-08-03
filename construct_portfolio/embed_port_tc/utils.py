import pandas as pd
import numpy as np
from construct_portfolio.config import CONFIG

RETURN_THRESH = CONFIG["RETURN_THRESH"] 
TRANSACTION_COST = CONFIG["TRANSACTION_COST"]

class LoggerWriter:
    def __init__(self, level_func): self.level_func = level_func
    def write(self, message):
        msg = message.rstrip()
        if msg: self.level_func(msg)
    def flush(self): pass



def load_and_clean(path_X, path_y):
    print(f"[LOAD] {path_X}")
    dfX = pd.read_csv(path_X, parse_dates=["Date"])
    dfY = pd.read_csv(path_y, parse_dates=["Date"])  # Date, Ticker, Return

    dims    = dfX.filter(regex=r"^dim_")
    tickers = dfY['Ticker']
    returns = dfY['Return']

    mask1 = np.isfinite(dims).all(axis=1)
    X1 = dims.loc[mask1].values.astype('float32')
    y1 = returns.loc[mask1].values
    d1 = dfX.loc[mask1, 'Date'].values
    t1 = tickers.loc[mask1].values

    keep = np.abs(y1) > RETURN_THRESH
    return X1[keep], y1[keep], d1[keep], t1[keep]


def build_monthly_returns(dates, tickers, rets, preds, edge_thr, transaction_cost=TRANSACTION_COST):
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Ticker': tickers,
        'R': rets,
        'P': preds
    })
    df['Month'] = df['Date'].dt.to_period('M')

    months, port, bench = [], [], []
    turnover, costs = [], []

    # previous carried-forward weights (fractions)
    prev_w = pd.Series(dtype=float)

    for mon, grp in df.groupby('Month'):
        grp = grp.reset_index(drop=True)
        # signals
        mask = np.abs(grp['P']) > edge_thr
        tick_sel = grp.loc[mask, 'Ticker']
        r_sel    = grp.loc[mask, 'R']
        M        = len(r_sel)

        # new target fractions
        if M > 0:
            new_w = pd.Series(np.sign(grp.loc[mask, 'P'].values)/M,
                              index=tick_sel.values)
        else:
            new_w = pd.Series(dtype=float)

        # carry-forward old weights by returns and renormalize to sum to 1
        all_t = new_w.index.union(prev_w.index)
        r_all = grp.set_index('Ticker')['R'].reindex(all_t, fill_value=0.0)
        prev_expanded = prev_w.reindex(all_t, fill_value=0.0)

        if not prev_expanded.empty and prev_expanded.sum()!=0:
            carry_unnorm = prev_expanded * (1 + r_all)
            carried_w = carry_unnorm / carry_unnorm.sum()
        else:
            carried_w = prev_expanded

        # align new and carried weights
        new_aligned = new_w.reindex(all_t, fill_value=0.0)
        old_aligned = carried_w.reindex(all_t, fill_value=0.0)

        # turnover as sum of absolute weight shifts
        trn = (new_aligned.subtract(old_aligned)).abs().sum()
        cost = trn * transaction_cost

        # portfolio return and benchmark
        port_ret  = (new_aligned * r_all).sum()
        bench_ret = grp['R'].mean() if len(grp)>0 else 0.0

        # record
        months.append(mon.to_timestamp())
        port.append(port_ret)
        bench.append(bench_ret)
        turnover.append(trn)
        costs.append(cost)

        # next iteration
        prev_w = new_aligned

    idx = pd.Index(months, name='Month')
    return (
        pd.Series(port,     index=idx),
        pd.Series(bench,    index=idx),
        pd.Series(turnover, index=idx),
        pd.Series(costs,    index=idx)
    )


def summarize_series(s):
    mu, sd = s.mean(), s.std(ddof=1)
    sharpe = (mu/sd)*np.sqrt(12) if sd!=0 else np.nan
    return mu, sd, sharpe