###############################################################################
# CONFIGURATION
###############################################################################
import os 
OUTPUT_ROOT = "output"
TRAIN_TEST_DIR = os.path.join(OUTPUT_ROOT, "train_test_split")
OTHER_DIR = os.path.join(OUTPUT_ROOT, "embed_train_others")
EMBED_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "embeddings")
CONFIG = {

    "file_pattern": "../data/retrieved_data_*",  # Update this pattern as needed
    "train_files_json": os.path.join(TRAIN_TEST_DIR,"train_files.json"),  # Path to training files JSON
    "test_files_json": os.path.join(TRAIN_TEST_DIR,"test_files.json"),    # Path to test files JSON
    "chunk_size": 100_000,
    "batch_size": 1024,
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",

    # Path to the existing checkpoint (ensure it matches the model architecture below)
    "checkpoint_path": os.path.join(OUTPUT_ROOT,
                                    "model_checkpoint", 
                                    "checkpoint_saint_epoch_79_best.pt"),

    # If using dictionary files (optional)
    "char_dict_json": os.path.join(OTHER_DIR, "char2id_saint.json"),
    "ticker_dict_json": os.path.join(OTHER_DIR, "ticker2id_saint.json"),
    "gics_dict_json": os.path.join(OTHER_DIR, "gics2id_saint.json"),

    # Where your StandardScaler is saved
    "scaler_path": os.path.join(OTHER_DIR, "scaler_saint.pkl"),

    # Outlier + row-nan thresholds
    "z_thresh": 3.0,
    "nan_fraction": 0.8,

    # Output paths
    "embed_save_path_all": os.path.join(EMBED_OUTPUT_DIR,
                                        "row_to_emb.npy"),  # .npy for NumPy array
    "row_to_company_save_path_all": os.path.join(EMBED_OUTPUT_DIR,
                                                 "row_to_company.json"),
    "row_to_ret_save_path_all": os.path.join(EMBED_OUTPUT_DIR,
                                             "row_to_ret.json"),
    "row_to_date_save_path_all": os.path.join(EMBED_OUTPUT_DIR,
                                              "row_to_date.json"),  # To store dates
    
    "train_features_path": os.path.join(TRAIN_TEST_DIR, "train_features.csv"),
    "train_targets_path" : os.path.join(TRAIN_TEST_DIR, "train_targets.csv"),
    "test_features_path" : os.path.join(TRAIN_TEST_DIR,"test_features.csv"),
    "test_targets_path"  : os.path.join(TRAIN_TEST_DIR, "test_targets.csv"),

}