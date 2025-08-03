"""
Runs the full SAINT multi-task training pipeline.

    Steps:
    1. Set up logging.
    2. Generate chronological train/test split.
    3. Shuffle and verify data files.
    4. Build categorical dictionaries for `gics`, `ticker`, and other chars.
    5. Detect numeric columns and fit a scaler.
    6. Instantiate the MultiTaskSAINT model, optimizer, and scheduler.
    7. Initiates Training over epochs:
       - Run masked, next-row, and contrastive training phases.
       - Evaluate on test set.
       - Adjust learning rate and save checkpoints.
       - Early stop if no improvement.
    8. Save performance history.
"""
import os
import json
import logging
import random
import pandas as pd
import torch.optim as optim
import joblib

from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from embed_train.utils.logging_utils import setup_logging
from embed_train.utils.checkpoint_utils import save_checkpoint
from embed_train.utils.preprocess_utils import (
    generate_train_test_split_chronological,
    verify_files_exist,
    build_dict_for_col,
    compute_and_save_scaler,
    load_or_init_dict,
    save_dict
)
from embed_train.training import (
    train_alternative_multitask
)
from embed_train.evaluation import (
    evaluate_masked,
    evaluate_next_forecast,
    evaluate_contrast
)
from embed_train.model.multitask import MultiTaskSAINT 
from embed_train.config import (                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    CONFIG,
    MODEL_CHECKPOINT_DIR,
    SCALER_PATH,
    EMBED_TRAIN_LOG,
    CHAR_DICT_JSON,
    TICKER_DICT_JSON,
    GICS_DICT_JSON,
    TRAIN_TEST_DIR
)

def main():
    
    setup_logging()
    logging.info("SAINT multi-task training starting.")
    # Use chronological split to avoid leakage.
    os.makedirs(TRAIN_TEST_DIR, exist_ok=True)
    generate_train_test_split_chronological(CONFIG, test_size=0.2)
    # Randomly shuffle the training files.
    random.seed(CONFIG['random_seed'])
    train_files = CONFIG["train_files"]
    random.shuffle(train_files)
    CONFIG["train_files"] = train_files

    verify_files_exist(CONFIG["train_files"])
    verify_files_exist(CONFIG["test_files"])

    # Build dictionaries for categorical columns.
    for col in CONFIG["cat_cols"]:
        if col == "gics":
            build_dict_for_col(CONFIG["train_files"], col, GICS_DICT_JSON, has_unk=True)
        elif col == "ticker":
            build_dict_for_col(CONFIG["train_files"], col, TICKER_DICT_JSON, has_unk=True)
        else:
            build_dict_for_col(CONFIG["train_files"], col, CHAR_DICT_JSON, has_unk=True)

    # Identify numeric columns.
    sample_file = CONFIG["train_files"][0]
    sample_df = pd.read_csv(sample_file, nrows=1000, low_memory=False)
    skip_cols = {"date", "gics", "ticker"}
    numeric_cols = [c for c in sample_df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(sample_df[c])]
    logging.info(f"Detected numeric_cols => {numeric_cols}")

    # Build and save scaler.
    compute_and_save_scaler(CONFIG["train_files"], numeric_cols, SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Initialize model & optimizer.
    n_num_feats = len(numeric_cols)
    model = MultiTaskSAINT(
        n_numeric_features=n_num_feats,
        embed_dim=CONFIG["saint_embed_dim"],
        num_layers=CONFIG["saint_num_layers"],
        num_heads=CONFIG["saint_num_heads"],
        ff_dim=CONFIG["saint_ff_dim"],
        dropout=CONFIG["dropout"]
    ).to(CONFIG["device"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=CONFIG["scheduler_factor"],
                                     patience=CONFIG["scheduler_patience"], verbose=True)
    start_epoch, start_phase = 0, "none"

    perf_history = {
        "masked_mse_train": [],
        "masked_mse_test": [],
        "next_mse_train": [],
        "next_mse_test": [],
        "contrast_loss_train": [],
        "contrast_loss_test": []
    }
    best_test_loss = float("inf")
    no_improve_count = 0

    for ep in range(start_epoch, CONFIG["epochs"]):
        eidx = ep + 1
        logging.info(f"=== EPOCH {eidx}/{CONFIG['epochs']} ===")
        masked_train, next_train, contrast_train = train_alternative_multitask(
            CONFIG["train_files"], model, optimizer,
            scaler, numeric_cols, 
            mask_loss=CONFIG["mask_loss"],
            next_loss=CONFIG["next_loss"],
            contrast_loss=CONFIG["contrast_loss"]
        )
        masked_test = evaluate_masked(CONFIG["test_files"], model, scaler, numeric_cols)
        next_test = evaluate_next_forecast(CONFIG["test_files"], model, scaler, numeric_cols)
        contrast_test = evaluate_contrast(CONFIG["test_files"], model, scaler, numeric_cols)
        perf_history["masked_mse_train"].append(float(masked_train))
        perf_history["masked_mse_test"].append(float(masked_test))
        perf_history["next_mse_train"].append(float(next_train))
        perf_history["next_mse_test"].append(float(next_test))
        perf_history["contrast_loss_train"].append(float(contrast_train))
        perf_history["contrast_loss_test"].append(float(contrast_test))
        # For scheduling, combine next and contrast losses.
        combined_test_val = float(masked_test + next_test + contrast_test)
        lr_scheduler.step(combined_test_val)
        if combined_test_val < best_test_loss:
            best_test_loss = combined_test_val
            no_improve_count = 0
            save_checkpoint(model, optimizer, eidx, "best", MODEL_CHECKPOINT_DIR, name_prefix="checkpoint_saint")
        else:
            no_improve_count += 1
            logging.info(f"No improvement => {no_improve_count}/{CONFIG['early_stopping_patience']}")
        save_checkpoint(model, optimizer, eidx, "all", MODEL_CHECKPOINT_DIR, name_prefix="checkpoint_saint")
        save_dict(load_or_init_dict(CHAR_DICT_JSON), CHAR_DICT_JSON)
        save_dict(load_or_init_dict(TICKER_DICT_JSON),TICKER_DICT_JSON)
        save_dict(load_or_init_dict(GICS_DICT_JSON), GICS_DICT_JSON)
        # Convert perf_history values to floats for JSON serialization.
        perf_history_serializable = {k: [float(v) for v in vals] for k, vals in perf_history.items()}
        os.makedirs(os.path.dirname(EMBED_TRAIN_LOG), exist_ok=True)
        with open(EMBED_TRAIN_LOG, "w") as f:
            json.dump(perf_history_serializable, f, indent=2)
        logging.info(
            f"Epoch {eidx}: masked_train={masked_train:.4f}, masked_test={masked_test:.4f}, " +
            f"next_train={next_train:.4f}, next_test={next_test:.4f}, " +
            f"contrast_train={contrast_train:.4f}, contrast_test={contrast_test:.4f}"
        )
        if no_improve_count > CONFIG["early_stopping_patience"]:
            logging.info("Early stopping triggered => break.")
            break

    logging.info("Training complete. See perf_history_saint.json & train_saint.log.")

if __name__ == '__main__':
    setup_logging()
    main()