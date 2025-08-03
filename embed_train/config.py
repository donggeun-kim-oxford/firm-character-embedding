
import os
OUTPUT_ROOT = "output"
MODEL_CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "model_checkpoint")
EMBED_TRAIN_LOG = os.path.join(OUTPUT_ROOT, "embed_train_loss_log", "embed_train_loss.json")
EMBED_TRAIN_OTHERS = os.path.join(OUTPUT_ROOT, "embed_train_others")
CHAR_DICT_JSON = os.path.join(EMBED_TRAIN_OTHERS, "char2id_saint.json")
TICKER_DICT_JSON =  os.path.join(EMBED_TRAIN_OTHERS, "ticker2id_saint.json")
GICS_DICT_JSON =os.path.join(EMBED_TRAIN_OTHERS, "gics2id_saint.json")
SCALER_PATH = os.path.join(EMBED_TRAIN_OTHERS, "scaler_saint.pkl")
TRAIN_TEST_DIR = os.path.join(OUTPUT_ROOT, "train_test_split")

CONFIG = {
    "file_pattern": "../data/retrieved_data_*",
    "train_json": os.path.join(TRAIN_TEST_DIR, "train_files.json"),
    "test_json":  os.path.join(TRAIN_TEST_DIR, "test_files.json"),
    # Dictionary file paths:
    
    "cat_cols": ["gics", "ticker"],
    # SAINT hyperparameters:
    "saint_embed_dim": 256,
    "saint_num_layers": 4,
    "saint_num_heads": 4,
    "saint_ff_dim": 512,
    "dropout": 0.2,
    # Training parameters:
    "epochs": 5000,
    "batch_size": 128,
    "mask_prob": 0.2,
    "max_mask_updates": 100,
    "max_next_updates": 100,
    "max_contrast_updates": 300,
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",
    "chunksize": 10000,
    "accum_steps": 1,
    "lambda_masked": 0.5,
    "lambda_next": 0.3,
    "lambda_contrast": 0.2,
    "noise_std": 0.01,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 40,
    "margin": 0.1,
    "z_thresh": 2.0,
    "nan_fraction": 0.5,
    "nan_char_threshold": 0.8,
    "random_seed" : 42,
    
    # Multi-task configuration
    "mask_loss": True, 
    "next_loss": True,
    "contrast_loss": False
}