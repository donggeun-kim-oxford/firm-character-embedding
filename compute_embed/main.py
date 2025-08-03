import logging
import os
import joblib
import glob
import pandas as pd
import torch
from embed_train.model.multitask import MultiTaskSAINT
from compute_embed.utils import process_data, print_sample_and_shape
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore")

from compute_embed.utils import (
    setup_logging,
    save_json,
    load_json,
    verify_csv_structure,
    extract_date_range
)
from compute_embed.config import CONFIG
from embed_train.config import CONFIG as MODEL_CONFIG
from compute_embed.build_embedding import build_row_embeddings


def main():
    # Enable detailed logging for troubleshooting
    setup_logging(debug=True)  # Set to False once debugging is complete
    logging.info("Compute Row-Level Embeddings.")

    # 1) Load dictionaries if needed (optional)
    char_dict = load_json(CONFIG["char_dict_json"])
    ticker_dict = load_json(CONFIG["ticker_dict_json"])
    gics_dict = load_json(CONFIG["gics_dict_json"])

    # 2) Load scaler if available
    if os.path.isfile(CONFIG["scaler_path"]):
        scaler = joblib.load(CONFIG["scaler_path"])
        logging.info(f"Loaded scaler from {CONFIG['scaler_path']}")
    else:
        logging.warning("No scaler found => using None")
        scaler = None

    # 3) Gather CSV files
    all_csv_files = glob.glob(CONFIG["file_pattern"])
    if not all_csv_files:
        logging.error("No CSV files found => aborting.")
        return

    # 4) Dynamically define expected_numeric_cols based on data files
    logging.info("Determining 'expected_numeric_cols' dynamically from data files.")
    # Read the first CSV file to determine column positions
    sample_csv = all_csv_files[0]

    df_sample = pd.read_csv(sample_csv, nrows=1, low_memory=False)


    all_columns = df_sample.columns.tolist()

    # Define non-numerical columns
    exclude_cols = {"gics", "ticker", "date", "company_name"}

    # Identify numeric columns excluding the specified columns
    numeric_cols = [c for c in all_columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_sample[c])]
    logging.debug(f"Numeric Columns ({len(numeric_cols)}): {numeric_cols}")

    # Log data types
    logging.debug("Column Data Types:")
    logging.debug(df_sample.dtypes)

    # Verify the number of numeric columns
    expected_num_cols = 437 
    if len(numeric_cols) != expected_num_cols:
        # Identify missing numeric columns
        missing_columns = set()
        for col in all_columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_sample[col]) and col not in numeric_cols:
                missing_columns.add(col)
        logging.error(f"'expected_numeric_cols' contains {len(numeric_cols)} columns instead of {expected_num_cols}.")
        logging.error(f"Missing numeric columns: {missing_columns}")

        # If missing_columns is empty but count is off, check for unintended inclusions
        if not missing_columns:
            # Identify any unintended columns included in numeric_cols
            unintended_columns = set()
            for col in numeric_cols:
                if col not in all_columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df_sample[col]):
                    unintended_columns.add(col)
            if unintended_columns:
                logging.error(f"Unintended columns detected in numeric_cols: {unintended_columns}")
                for col in unintended_columns:
                    numeric_cols.remove(col)
                    logging.debug(f"Manually removed unintended column '{col}' from numeric_cols.")
            else:
                logging.error("Count mismatch without missing or unintended columns. Please verify 'expected_num_cols'.")
            # Attempt to manually add missing columns if necessary
            for missing_col in missing_columns:
                if missing_col in all_columns:
                    # Attempt to convert to numeric
                    df_sample[missing_col] = pd.to_numeric(df_sample[missing_col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df_sample[missing_col]):
                        numeric_cols.append(missing_col)
                        logging.debug(f"Manually added missing column '{missing_col}' to numeric_cols.")

        # Re-verify the number of columns after manual addition
        if len(numeric_cols) != expected_num_cols:
            logging.error(f"After attempting to add missing columns, 'expected_numeric_cols' contains {len(numeric_cols)} columns.")
            return
        else:
            logging.info(f"'expected_numeric_cols' successfully identified with {len(numeric_cols)} columns.")
    else:
        logging.info(f"'expected_numeric_cols' successfully identified with {len(numeric_cols)} columns.")

    # 5) Verify CSV structures
    verify_csv_structure(all_csv_files, numeric_cols)

    # 6) Initialize the model with the same architecture as training
    logging.info("Initializing the model with the specified architecture.")
    n_num_feats = len(numeric_cols)
    model = MultiTaskSAINT(
        n_numeric_features=n_num_feats,
        embed_dim=MODEL_CONFIG["saint_embed_dim"],
        num_layers=MODEL_CONFIG["saint_num_layers"],
        num_heads=MODEL_CONFIG["saint_num_heads"],
        ff_dim=MODEL_CONFIG["saint_ff_dim"],
        dropout=MODEL_CONFIG["dropout"]
    ).to(MODEL_CONFIG["device"])
    # 7) Load the checkpoint
    try:
        ckpt = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"])
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        logging.info(f"Successfully loaded checkpoint from {CONFIG['checkpoint_path']}")
    except RuntimeError as e:
        logging.error(f"RuntimeError while loading state_dict: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        return

    # 8) Build Row Embeddings for All Data 
    # -- caching logic for build_row_embeddings --
    emb_path  = CONFIG["embed_save_path_all"]
    ret_path  = CONFIG["row_to_ret_save_path_all"]
    comp_path = CONFIG["row_to_company_save_path_all"]
    date_path = CONFIG["row_to_date_save_path_all"]
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)

    if all(os.path.exists(p) for p in (emb_path, ret_path, comp_path, date_path)):
        logging.info("Found cached embeddings & metadata, loading from disk…")
        emb_array        = np.load(emb_path)
        row_to_ret_all     = load_json(ret_path)
        row_to_company_all = load_json(comp_path)
        row_to_date_all    = load_json(date_path)
        # reconstruct embeddings dict
        keys = list(row_to_ret_all.keys())
        row_to_emb_all = {k: emb_array[i] for i, k in enumerate(keys)}
    else:
        logging.info("No cache found, computing embeddings from scratch…")
        # 8) Build Row Embeddings for All Data 
        row_to_emb_all, row_to_ret_all, row_to_company_all, row_to_date_all = build_row_embeddings(
            all_csv_files=all_csv_files,
            model=model,
            numeric_cols=numeric_cols,
            scaler=scaler,
            nan_fraction=CONFIG["nan_fraction"]
        )

        embeddings = np.array(list(row_to_emb_all.values()), dtype='float32')
        np.save(emb_path, embeddings)
        logging.info(f"Saved embeddings to {emb_path}")

        save_json(row_to_company_all, comp_path)
        logging.info(f"Saved company names to {comp_path}")

        save_json(row_to_ret_all, ret_path)
        logging.info(f"Saved actual ret_exc_lead1m to {ret_path}")

        save_json(row_to_date_all, date_path)
        logging.info(f"Saved dates to {date_path}")

    # Check if build_row_embeddings returned the expected dictionaries
    if row_to_emb_all is None or row_to_ret_all is None or row_to_company_all is None or row_to_date_all is None:
        logging.error("build_row_embeddings returned None. Exiting.")
        return
    
    
    embeddings = np.array(list(row_to_emb_all.values()), dtype='float32')  # Shape: (num_rows, embed_dim)
    np.save(CONFIG["embed_save_path_all"], embeddings)
    logging.info(f"Saved embeddings to {CONFIG['embed_save_path_all']}")

    # 2. Save associated data
    save_json(row_to_company_all, CONFIG["row_to_company_save_path_all"])
    logging.info(f"Saved company names to {CONFIG['row_to_company_save_path_all']}")

    save_json(row_to_ret_all, CONFIG["row_to_ret_save_path_all"])
    logging.info(f"Saved actual ret_exc_lead1m to {CONFIG['row_to_ret_save_path_all']}")

    save_json(row_to_date_all, CONFIG["row_to_date_save_path_all"])
    logging.info(f"Saved dates to {CONFIG['row_to_date_save_path_all']}")


    # Load train and test file lists
    with open(CONFIG["train_files_json"], "r") as f:
        train_files = json.load(f)

    with open(CONFIG["test_files_json"], "r") as f:
        test_files = json.load(f)

    
    train_dates = set()
    test_dates = set()

    for file in train_files:
        start_date, end_date = extract_date_range(file)
        if start_date and end_date:
            train_dates.update(pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d"))

    for file in test_files:
        start_date, end_date = extract_date_range(file)
        if start_date and end_date:
            test_dates.update(pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d"))


    # Run the data processing
    process_data(CONFIG["embed_save_path_all"], row_to_ret_all, train_dates, test_dates)
   

if __name__ == "__main__":
    main()