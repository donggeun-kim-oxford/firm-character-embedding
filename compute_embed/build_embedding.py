#!/usr/bin/env python

import os
import logging
import pandas as pd
import torch
from compute_embed.config import CONFIG
from compute_embed.utils import normalize_company_name

def build_row_embeddings(all_csv_files, model, numeric_cols, scaler, nan_fraction=0.6):
    """
    Builds row-level embeddings with 'ret_exc_lead1m' masked to 0.0.

    Parameters:
        all_csv_files (list): List of CSV file paths to process.
        model (nn.Module): The trained embedding model.
        numeric_cols (list): List of numeric column names.
        scaler (StandardScaler or None): Scaler for numeric data.
        nan_fraction (float): Maximum allowed fraction of NaNs in numeric_cols.

    Returns:
        row_to_emb (dict): Mapping from row_key to embedding vector.
        row_to_ret (dict): Mapping from row_key to actual 'ret_exc_lead1m' value.
        row_to_company (dict): Mapping from row_key to 'company_name'.
        row_to_rvol (dict): Mapping from row_key to 'rvol_21d'.
        row_to_date (dict): Mapping from row_key to 'date'.
    """
    device = CONFIG["device"]
    row_to_emb = {}
    row_to_ret = {}
    row_to_company = {}
    row_to_rvol = {}
    row_to_date = {}
    processed_keys = set()  # To track processed row_keys and eliminate duplicates

    for csv_ in all_csv_files:
        logging.info(f"[RowEmb] => {csv_}")
        if not os.path.isfile(csv_):
            logging.warning(f"File not found => {csv_}")
            continue

        chunk_iter = pd.read_csv(csv_, chunksize=CONFIG["chunk_size"], low_memory=False)
        for chunk_i, df in enumerate(chunk_iter, start=1):
            logging.info(f"   chunk={chunk_i}, shape={df.shape}")
            required_columns = set(["date", "ticker", "company_name", "ret_exc_lead1m"]) | set(numeric_cols)
            missing_required = required_columns - set(df.columns)
            if missing_required:
                logging.warning(f"Missing required columns {missing_required} in chunk => filling with default values.")
                # Efficiently add missing columns with default values
                default_values = {}
                for c in missing_required:
                    if c == "date":
                        default_values[c] = pd.NaT
                    elif c in ["ticker", "company_name"]:
                        default_values[c] = "UNK"
                    else:
                        default_values[c] = 0.0
                df = df.assign(**default_values)

            try:
                # Convert 'date' to datetime
                df.loc[:,"date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                if df.empty:
                    logging.warning("All 'date' values are NaT => skipping chunk.")
                    continue
                df = df.sort_values("date")

                # Handle 'ret_exc_lead1m'
                df["ret_exc_lead1m"] = pd.to_numeric(df["ret_exc_lead1m"], errors="coerce")
                num_ret_nans = df["ret_exc_lead1m"].isna().sum()
                logging.debug(f"   'ret_exc_lead1m' has {num_ret_nans} NaNs in this chunk.")
                # Fill NaNs with 0.0 (masking)
                df["ret_exc_lead1m"] = df["ret_exc_lead1m"].fillna(0.0)

                # Convert all expected numeric columns to numeric dtype
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

                # Identify rows based on NaN fraction using assign to prevent fragmentation
                df = df.assign(nan_fraction=df[numeric_cols].isna().sum(axis=1) / len(numeric_cols))
                df_masked = df[df['nan_fraction'] <= nan_fraction].copy()

                logging.info(f"   Rows to process as masked: {len(df_masked)}")

                if df_masked.empty:
                    logging.warning("No valid rows after filtering NaN fraction.")
                    continue

                # Store actual 'ret_exc_lead1m' before masking
                df_masked = df_masked.assign(ret_exc_lead1m_actual=df_masked['ret_exc_lead1m'])

                # Mask 'ret_exc_lead1m' by setting it to 0.0
                df_masked = df_masked.assign(ret_exc_lead1m=0.0)

                # Fill remaining NaNs with 0.0
                df_masked[numeric_cols] = df_masked[numeric_cols].fillna(0.0)
                
                # Apply scaler if available
                if scaler is not None:
                    try:
                        df_masked[numeric_cols] = scaler.transform(df_masked[numeric_cols])
                    except Exception as e:
                        logging.warning(f"Skipping chunk due to scaler transformation error: {e}")
                        df_masked = pd.DataFrame()  # Empty DataFrame

                if df_masked.empty:
                    continue

                # Normalize company names
                df_masked['company_name'] = df_masked['company_name'].astype(str).apply(normalize_company_name)

                # Remove duplicate (date, company_name, ticker) combinations
                df_masked['unique_key'] = df_masked.apply(lambda row: f"{row['date'].strftime('%Y-%m-%d')}_{row['company_name']}_{row['ticker']}", axis=1)
                df_masked = df_masked[~df_masked['unique_key'].isin(processed_keys)]
                df_masked = df_masked.drop_duplicates(subset=['unique_key'])
                processed_keys.update(df_masked['unique_key'].tolist())

                if df_masked.empty:
                    logging.info("No new unique rows to process after removing duplicates.")
                    continue

                # Convert to numpy array for batch processing
                numeric_array = df_masked[numeric_cols].values.astype('float32')  # Shape: (num_rows, num_features)
                numeric_tensor = torch.tensor(numeric_array, dtype=torch.float, device=device)

                # Compute embeddings in batches to optimize GPU usage
                batch_size = CONFIG['batch_size']  # Adjust based on your GPU memory
                num_rows = numeric_tensor.size(0)
                num_batches = (num_rows + batch_size - 1) // batch_size

                for b in range(num_batches):
                    start = b * batch_size
                    end = min((b + 1) * batch_size, num_rows)
                    batch_tensor = numeric_tensor[start:end]
                    with torch.no_grad():
                        embeddings = model.row_embedding(batch_tensor)  # Shape: (batch_size, embed_dim)
                    embeddings_np = embeddings.cpu().numpy()

                    # Iterate over the batch and store embeddings
                    for i in range(end - start):
                        row_ = df_masked.iloc[start + i]
                        date_str = row_["date"].strftime("%Y-%m-%d")
                        company_name = row_.get("company_name", "UNK")
                        ticker_val = row_.get("ticker", "UNK")
                        key_str = f"{date_str}_{company_name}_{ticker_val}"  # Unique key without suffix

                        # Ensure uniqueness
                        if key_str in row_to_emb:
                            logging.warning(f"Duplicate key detected after processing: {key_str}. Skipping.")
                            continue

                        # Store the embedding
                        row_to_emb[key_str] = embeddings_np[i]

                        # Store actual 'ret_exc_lead1m', 'rvol_21d', 'company_name', and 'date' separately
                        ret_val = row_["ret_exc_lead1m_actual"]
                        row_to_ret[key_str] = float(ret_val) if ret_val is not None else None
                        row_to_company[key_str] = company_name
                        row_to_date[key_str] = date_str

                        # Log some non-zero ret_exc_lead1m and rvol_21d values for debugging
                        if float(ret_val) != 0.0 and len(row_to_emb) <= 10:
                            logging.debug(f"Non-zero ret_exc_lead1m: {key_str} -> ret: {ret_val}")

            except Exception as e:
                logging.error(f"Error processing chunk {chunk_i} in file {csv_}: {e}")
                continue  # Skip to the next chunk

    # Ensure that the function always returns the dictionaries
    return row_to_emb, row_to_ret, row_to_company, row_to_date

