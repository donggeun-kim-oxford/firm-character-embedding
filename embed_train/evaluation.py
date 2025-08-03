import torch
import torch.nn.functional as F
from torch.amp import autocast
import os
from embed_train.config import CONFIG
from embed_train.model.multitask import contrastive_triplet_loss, add_gaussian_noise
from embed_train.utils.preprocess_utils import find_outliers_zscore, find_rows_with_too_many_nans
import pandas as pd
import random
import numpy as np
from typing import List
import sklearn

###############################################################################
# EVALUATION FUNCTIONS
###############################################################################
def evaluate_masked(
    files: List[str],
    model: torch.nn.Module,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str]
) -> float:
    """
    Evaluate masked auto-encoding task performance on test set.
    Computes MSE over masked positions.

    Returns
    -------
    float
        Average MSE for masked positions.
    """
    device = CONFIG["device"]
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad(), torch.amp.autocast(device_type=CONFIG["device"]):
        for f in files:
            if not os.path.isfile(f):
                continue
            for df in pd.read_csv(f, chunksize=CONFIG["chunksize"], low_memory=False):
                if "date" not in df.columns or "gics" not in df.columns:
                    continue
                df = df.dropna(subset=["date", "gics"])
                if df.empty:
                    continue
                df = df.sort_values("date")
                active_cols = [c for c in numeric_cols if c in df.columns]
                for c in active_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                out_idx = find_outliers_zscore(df, active_cols, CONFIG["z_thresh"])
                nan_idx = find_rows_with_too_many_nans(df, active_cols, CONFIG["nan_fraction"])
                skip_idx = out_idx.union(nan_idx)
                df = df.drop(index=skip_idx)
                if df.empty:
                    continue
                df[active_cols] = df[active_cols].fillna(0.0)
                try:
                    df[active_cols] = scaler.transform(df[active_cols])
                except:
                    continue
                rows = df.to_dict("records")
                batch_size = CONFIG["batch_size"]
                i_row = 0
                while i_row < len(rows):
                    batch_data = rows[i_row:i_row+batch_size]
                    i_row += batch_size
                    numeric_mat = []
                    maskpos_mat = []
                    for r in batch_data:
                        arr = []
                        maskpos = []
                        for c in active_cols:
                            val = r.get(c, 0.0)
                            if random.random() < CONFIG["mask_prob"]:
                                arr.append(0.0)
                                maskpos.append(val)
                            else:
                                arr.append(val)
                                maskpos.append(None)
                        numeric_mat.append(arr)
                        maskpos_mat.append(maskpos)
                    numeric_tensor = torch.tensor(numeric_mat, dtype=torch.float, device=device)
                    numeric_tensor = add_gaussian_noise(numeric_tensor, CONFIG["noise_std"])
                    pred = model.forward_masked(numeric_tensor)
                    B, C = pred.shape
                    for ib in range(B):
                        for ic in range(C):
                            if maskpos_mat[ib][ic] is not None:
                                total_loss += (pred[ib, ic] - maskpos_mat[ib][ic]) ** 2
                                total_count += 1
    return total_loss / total_count if total_count > 0 else 0.0

def evaluate_next_forecast(
    files: List[str],
    model: torch.nn.Module,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str]
) -> float:
    """
    Evaluate next-row forecasting performance on a dataset.
    Computes the mean squared error (MSE) across all forecasted pairs.

    Parameters
    ----------
    files : list of str
        List of paths to CSV files containing sequential numeric data.
    model : torch.nn.Module
        The trained SAINT model with a `forward_nextpredict` method.
    scaler : sklearn.preprocessing._Scaler
        Fitted scaler for numeric features.
    numeric_cols : list of str
        Names of numeric feature columns to use for forecasting.

    Returns
    -------
    float
        Average MSE of next-row forecasts over all file segments.
    """
    device = CONFIG["device"]
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    with torch.no_grad(), torch.amp.autocast(device_type=CONFIG["device"]):
        for f in files:
            if not os.path.isfile(f):
                continue
            for df in pd.read_csv(f, chunksize=CONFIG["chunksize"], low_memory=False):
                if "date" not in df.columns or "gics" not in df.columns:
                    continue
                df = df.dropna(subset=["date", "gics"])
                if df.empty or df.shape[0] < 2:
                    continue
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                df = df.sort_values("date")
                if df.shape[0] < 2:
                    continue
                active_cols = [c for c in numeric_cols if c in df.columns]
                for c in active_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                out_idx = find_outliers_zscore(df, active_cols, CONFIG["z_thresh"])
                nan_idx = find_rows_with_too_many_nans(df, active_cols, CONFIG["nan_fraction"])
                skip_idx = out_idx.union(nan_idx)
                df = df.drop(index=skip_idx)
                if df.shape[0] < 2:
                    continue
                df[active_cols] = df[active_cols].fillna(0.0)
                try:
                    df[active_cols] = scaler.transform(df[active_cols])
                except:
                    continue
                data_arr = df[active_cols].values.astype("float32")
                if data_arr.shape[0] < 2:
                    continue
                X = torch.tensor(data_arr[:-1], dtype=torch.float, device=device)
                Y = torch.tensor(data_arr[1:], dtype=torch.float, device=device)
                X = add_gaussian_noise(X, CONFIG["noise_std"])
                predY = model.forward_nextpredict(X)
                loss = F.mse_loss(predY, Y)
                total_loss += loss.item() * X.shape[0]
                total_pairs += X.shape[0]
    return total_loss / total_pairs if total_pairs > 0 else 0.0

def evaluate_contrast(
    files: List[str],
    model: torch.nn.Module,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str]
) -> float:
    """
    Evaluate contrastive triplet loss on embeddings over a test set.
    Returns the average contrastive loss across all examples.

    Parameters
    ----------
    files : list of str
        List of paths to CSV files containing sequential numeric data.
    model : torch.nn.Module
        The trained SAINT model with an embedding extraction method.
    scaler : sklearn.preprocessing._Scaler
        Fitted scaler for numeric features.
    numeric_cols : list of str
        Names of numeric feature columns to use for embedding.

    Returns
    -------
    float
        Average contrastive triplet loss over all test segments.
    """
    device = CONFIG["device"]
    model.eval()
    total_loss = 0.0
    total_count = 0
    margin = CONFIG["margin"]
    with torch.no_grad(), torch.amp.autocast(device_type=CONFIG["device"]):
        for f in files:
            if not os.path.isfile(f):
                continue
            for df in pd.read_csv(f, chunksize=CONFIG["chunksize"], low_memory=False):
                if "date" not in df.columns or "gics" not in df.columns:
                    continue
                df = df.dropna(subset=["date", "gics"])
                if df.empty or df.shape[0] < 2:
                    continue
                df = df.sort_values("date")
                active_cols = [c for c in numeric_cols if c in df.columns]
                for c in active_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                out_idx = find_outliers_zscore(df, active_cols, CONFIG["z_thresh"])
                nan_idx = find_rows_with_too_many_nans(df, active_cols, CONFIG["nan_fraction"])
                skip_idx = out_idx.union(nan_idx)
                df = df.drop(index=skip_idx)
                if df.shape[0] < 2:
                    continue
                df[active_cols] = df[active_cols].fillna(0.0)
                try:
                    df[active_cols] = scaler.transform(df[active_cols])
                except:
                    continue
                data_arr = df[active_cols].values.astype("float32")
                if data_arr.shape[0] < 2:
                    continue
                N = data_arr.shape[0]
                anchor = torch.tensor(data_arr, dtype=torch.float, device=device)
                positive = anchor.clone()
                indices = np.arange(N)
                neg_indices = np.array([np.random.choice(indices[indices != i]) for i in range(N)])
                negative = torch.tensor(data_arr[neg_indices], dtype=torch.float, device=device)
                embA = model.get_embedding(anchor)
                embP = model.get_embedding(positive)
                embN = model.get_embedding(negative)
                loss_c = contrastive_triplet_loss(embA, embP, embN, margin)
                total_loss += loss_c.item() * N
                total_count += N
    return total_loss / total_count if total_count > 0 else 0.0