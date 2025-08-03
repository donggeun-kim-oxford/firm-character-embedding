import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import os
from embed_train.config import CONFIG
from embed_train.model.multitask import contrastive_triplet_loss, add_gaussian_noise
from embed_train.utils import  find_outliers_zscore, find_rows_with_too_many_nans
import random
import pandas as pd
import numpy as np
import sklearn
from typing import List, Tuple

###############################################################################
# TRAINING: Masked Reconstruction
###############################################################################
def train_masked(
    files: List[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str]
) -> float:
    """
    Perform masked-value reconstruction training.

    Iterates over input CSV files, chunks and rows, applies random mask to numeric features,
    computes MSE loss on masked positions, and backpropagates.

    Parameters
    ----------
    files : list of str
        Paths to CSV files containing time-series data.
    model : torch.nn.Module
        The SAINT multi-task model.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    scaler : sklearn Scaler
        Fitted scaler for numeric features.
    numeric_cols : list of str
        Names of numeric feature columns.

    Returns
    -------
    float
        Average MSE loss over all masked entries in this epoch.
    """
    device = CONFIG["device"]
    model.train()
    scaler_amp = GradScaler()
    accum_steps = CONFIG["accum_steps"]
    global_batch = 0
    mask_updates = 0
    total_loss_sum = 0.0
    total_count = 0
    for f in files:
        if mask_updates >= CONFIG["max_mask_updates"]:
            break
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
            for c_ in active_cols:
                df[c_] = pd.to_numeric(df[c_], errors="coerce")
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
            i_row = 0
            while i_row < len(rows) and mask_updates < CONFIG["max_mask_updates"]:
                batch_data = rows[i_row: i_row + CONFIG["batch_size"]]
                i_row += CONFIG["batch_size"]
                if not batch_data:
                    break
                numeric_mat = []
                maskpos_mat = []
                for r in batch_data:
                    arr = []
                    maskpos = []
                    for c_ in active_cols:
                        val = r.get(c_, 0.0)
                        if random.random() < CONFIG["mask_prob"]:
                            arr.append(0.0)
                            maskpos.append(val)
                        else:
                            arr.append(val)
                            maskpos.append(None)
                    numeric_mat.append(arr)
                    maskpos_mat.append(maskpos)
                numeric_mat = torch.tensor(numeric_mat, dtype=torch.float, device=device)
                numeric_mat = add_gaussian_noise(numeric_mat, CONFIG["noise_std"])
                with torch.amp.autocast(device_type=CONFIG["device"]):
                    pred = model.forward_masked(numeric_mat)
                    masked_vals = []
                    masked_preds = []
                    B, C = pred.shape
                    for ib in range(B):
                        for ic in range(C):
                            if maskpos_mat[ib][ic] is not None:
                                masked_vals.append(maskpos_mat[ib][ic])
                                masked_preds.append(pred[ib, ic])
                    if masked_vals:
                        mv_t = torch.tensor(masked_vals, dtype=torch.float, device=device)
                        mp_t = torch.stack(masked_preds, dim=0)
                        loss_mse = F.mse_loss(mp_t, mv_t)
                    else:
                        loss_mse = torch.tensor(0.0, device=device)
                scaler_amp.scale(loss_mse).backward()
                if (global_batch + 1) % accum_steps == 0:
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    optimizer.zero_grad()
                mask_updates += 1
                global_batch += 1
                total_loss_sum += loss_mse.item() * len(batch_data)
                total_count += len(batch_data)
                if mask_updates >= CONFIG["max_mask_updates"]:
                    break
    if global_batch % accum_steps != 0:
        scaler_amp.step(optimizer)
        scaler_amp.update()
        optimizer.zero_grad()
    return total_loss_sum / total_count if total_count > 0 else 0.0

###############################################################################
# TRAINING: Next-Row Forecasting (Vectorized)
###############################################################################
def train_next_forecast(
    files: List[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str]
) -> float:
    """
    Train model to forecast next-row values for numeric features.

    For each sequence in the files, constructs input X from all but last row, and target Y as the next row.
    Computes MSE loss between model predictions and true Y.

    Parameters
    ----------
    files : list of str
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scaler : sklearn Scaler
    numeric_cols : list of str

    Returns
    -------
    float
        Average forecast MSE across all row-pairs.
    """
    device = CONFIG["device"]
    model.train()
    scaler_amp = GradScaler()
    total_loss_sum = 0.0
    total_pairs = 0
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
            with torch.amp.autocast(device_type=CONFIG["device"]):
                predY = model.forward_nextpredict(X)
                loss = F.mse_loss(predY, Y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad()
            total_loss_sum += loss.item() * X.shape[0]
            total_pairs += X.shape[0]
    return total_loss_sum / total_pairs if total_pairs > 0 else 0.0

###############################################################################
# TRAINING: Contrastive Triplet Loss (Vectorized)
###############################################################################
def train_contrast(
    files: List[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str]
) -> float:
    """
    Train model using contrastive triplet loss on embeddings.

    Constructs anchor, positive, and negative examples by sampling rows and noisy copies,
    then computes triplet loss with margin from CONFIG.

    Parameters
    ----------
    files : list of str
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scaler : sklearn Scaler
    numeric_cols : list of str

    Returns
    -------
    float
        Average triplet loss over all examples.
    """
    device = CONFIG["device"]
    model.train()
    scaler_amp = GradScaler()
    total_loss_sum = 0.0
    total_count = 0
    margin = CONFIG["margin"]
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
            anchor = add_gaussian_noise(anchor, CONFIG["noise_std"])
            positive = add_gaussian_noise(positive, CONFIG["noise_std"])
            negative = add_gaussian_noise(negative, CONFIG["noise_std"])
            with torch.amp.autocast(device_type=CONFIG["device"]):
                embA = model.get_embedding(anchor)
                embP = model.get_embedding(positive)
                embN = model.get_embedding(negative)
                loss_contrast = contrastive_triplet_loss(embA, embP, embN, margin)
            scaler_amp.scale(loss_contrast).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad()
            total_loss_sum += loss_contrast.item() * N
            total_count += N
    return total_loss_sum / total_count if total_count > 0 else 0.0

def train_alternative_multitask(
    files: List[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: sklearn.preprocessing.scale,
    numeric_cols: List[str],
    mask_loss: bool = True,
    next_loss: bool = True,
    contrast_loss: bool = True
) -> Tuple[float, float, float]:
    """
    Combined multi-task training wrapper.

    Executes masked, next-row, and contrastive phases based on flags.

    Parameters
    ----------
    files : list of str
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    scaler : sklearn.preprocessing._Scaler
    numeric_cols : list of str
    mask_loss : bool, optional
        If False, skip masked phase and set its loss to 0.
    next_loss : bool, optional
        If False, skip forecasting phase and set its loss to 0.
    contrast_loss : bool, optional
        If False, skip contrastive phase and set its loss to 0.

    Returns
    -------
    masked_mse : float
    next_mse : float
    contrast_val : float
    """
    # Masked autoencoding
    if mask_loss:
        masked_mse = train_masked(files, model, optimizer, scaler, numeric_cols)
    else:
        masked_mse = 0.0

    # Next-row forecasting
    if next_loss:
        next_mse = train_next_forecast(files, model, optimizer, scaler, numeric_cols)
    else:
        next_mse = 0.0

    # Contrastive triplet
    if contrast_loss:
        contrast_val = train_contrast(files, model, optimizer, scaler, numeric_cols)
    else:
        contrast_val = 0.0

    return masked_mse, next_mse, contrast_val