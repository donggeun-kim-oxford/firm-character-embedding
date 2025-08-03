import numpy as np
from construct_portfolio.config import CONFIG
import pandas as pd

class LoggerWriter:
    """Redirect writes to a logging function."""
    def __init__(self, level_func):
        self.level_func = level_func
    def write(self, message):
        message = message.rstrip()
        if message:
            self.level_func(message)
    def flush(self):
        pass

def load_and_clean(path_X, path_y, RETURN_THRESH=CONFIG["RETURN_THRESH"]):
    print(f"[LOAD]  {path_X}")
    dfX = pd.read_csv(path_X, parse_dates=["Date"])
    sY  = pd.read_csv(path_y, parse_dates=["Date"])["Return"]

    dims = dfX.filter(regex=r"^dim_")
    mask1 = np.isfinite(dims).all(axis=1)
    X1, y1, dates1 = (
        dims.loc[mask1].values.astype("float32"),
        sY.loc[mask1].values,
        dfX.loc[mask1, "Date"].values
    )
    mask2 = np.abs(y1) > RETURN_THRESH
    return X1[mask2], y1[mask2], dates1[mask2]

def build_monthly_returns(dates, rets, preds, edge_thr):
    df = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "R":    rets,
        "P":    preds
    })
    df["Month"] = df["Date"].dt.to_period("M")

    port, bench, months = [], [], []
    for mon, grp in df.groupby("Month"):
        p, r = grp["P"].values, grp["R"].values
        mask = np.abs(p) > edge_thr
        M = mask.sum()
        port.append(
            np.dot(np.sign(p[mask]) / M, r[mask]) if M > 0 else 0.0
        )
        bench.append(r.mean() if len(r) > 0 else 0.0)
        months.append(mon.to_timestamp())

    idx = pd.Index(months, name="Month")
    return pd.Series(port, index=idx), pd.Series(bench, index=idx)

def summarize_series(s):
    mu, sd = s.mean(), s.std(ddof=1)
    sharpe = (mu/sd)*np.sqrt(12) if sd != 0 else np.nan
    return mu, sd, sharpe
