import glob
import re
import json
import logging
import pandas as pd

from sklearn.preprocessing import StandardScaler
import joblib
import os
from embed_train.config import CONFIG

# Chronological train-test split

def generate_train_test_split_chronological(config, test_size=0.2):
    all_files = glob.glob(config["file_pattern"])
    if len(all_files) == 0:
        logging.error(f"No data files found => pattern {config['file_pattern']}")
        exit(1)
    def extract_start_date(filename):
        match = re.search(r"(\d{4}-\d{2}-\d{2})_to_", filename)
        return pd.to_datetime(match.group(1)) if match else pd.Timestamp("1900-01-01")
    sorted_files = sorted(all_files, key=extract_start_date)
    test_count = int(test_size * len(sorted_files))
    config["train_files"] = sorted_files[:-test_count]
    config["test_files"]  = sorted_files[-test_count:]
    with open(config["train_json"], "w") as f:
        json.dump(config["train_files"], f, indent=2)
    with open(config["test_json"], "w") as f:
        json.dump(config["test_files"], f, indent=2)
    logging.info(f"Chronological split => {len(config['train_files'])} train, {len(config['test_files'])} test")

# Verify existence

def verify_files_exist(file_list):
    missing = [f for f in file_list if not os.path.isfile(f)]
    if missing:
        logging.error("Missing files:\n" + "\n".join(missing))
        exit(1)

# Outlier & NaN filtering

def find_outliers_zscore(df, numeric_cols, z_thresh=2.0):
    out_idx = set()
    for c in numeric_cols:
        if c not in df.columns:
            continue
        col_vals = df[c].dropna()
        if len(col_vals) <= 1:
            continue
        mean_, std_ = col_vals.mean(), col_vals.std(ddof=0)
        if std_ == 0:
            continue
        z_scores = (df[c] - mean_).abs() / std_
        out_idx.update(df.index[z_scores > z_thresh])
    return out_idx

def find_rows_with_too_many_nans(df, numeric_cols, nan_fraction=0.5):
    too_many = set()
    max_nans = int(len(numeric_cols) * nan_fraction)
    for idx, row in df.iterrows():
        active = [c for c in numeric_cols if c in df.columns]
        if row[active].isna().sum() > max_nans:
            too_many.add(idx)
    return too_many

# Dictionary functions

def build_dict_for_col(files, col_name, out_json, has_unk=True):
    token_to_id = {"UNK": 0} if has_unk else {}
    for f in files:
        if not os.path.isfile(f):
            continue
        for chunk in pd.read_csv(f, chunksize=CONFIG["chunksize"], low_memory=False):
            if col_name not in chunk.columns:
                continue
            chunk = chunk.dropna(subset=[col_name])
            for val in chunk[col_name].unique():
                s = str(val)
                if s not in token_to_id:
                    token_to_id[s] = len(token_to_id)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(token_to_id, f, indent=2)
    logging.info(f"Built dict for {col_name} => {out_json}")
    return token_to_id

def load_or_init_dict(filename: str):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        return {"UNK": 0}

def save_dict(d: dict, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(d, f, indent=2)

def get_or_add_key(token_dict, key):
    if key in token_dict:
        return token_dict[key]
    nxt = max(token_dict.values()) + 1
    token_dict[key] = nxt
    return nxt

# Scaler Functions
def partial_fit_chunk_with_fallback(scaler, df_chunk, numeric_cols_chunk, global_numeric_cols):
    try:
        scaler.partial_fit(df_chunk[numeric_cols_chunk])
        return numeric_cols_chunk
    except ValueError as e:
        msg = str(e)
        logging.warning(f"partial_fit error => {e}")
        if "feature names" not in msg:
            return numeric_cols_chunk
        survived = []
        for c in numeric_cols_chunk:
            try:
                scaler.partial_fit(df_chunk[[c]])
                survived.append(c)
            except Exception as e2:
                logging.warning(f"Skipping column => {c}")
                if c in global_numeric_cols:
                    global_numeric_cols.remove(c)
        if survived:
            try:
                scaler.partial_fit(df_chunk[survived])
            except Exception as e3:
                logging.warning(f"Still fails => skip entire chunk => {e3}")
                return []
        return survived


def compute_and_save_scaler(file_list, numeric_cols, scaler_path):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = StandardScaler()
    for f in file_list:
        logging.info(f"[Scaler Fit] => {f}")
        if not os.path.isfile(f):
            logging.warning(f"File not found => {f}")
            continue
        for chunk in pd.read_csv(f, chunksize=CONFIG["chunksize"], low_memory=False):
            if chunk.empty:
                continue
            active_cols = [c for c in numeric_cols if c in chunk.columns]
            if not active_cols:
                continue
            for c in active_cols:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            chunk[active_cols] = chunk[active_cols].fillna(0.0)
            partial_fit_chunk_with_fallback(scaler, chunk, active_cols, numeric_cols)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved => {scaler_path}")
    return scaler