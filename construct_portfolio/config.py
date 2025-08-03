import os

###############################################################################
# PATHS & FILENAMES
###############################################################################
RESULTS_DIR = "output"
TRAIN_TEST_DIR = os.path.join(RESULTS_DIR, "train_test_split")
GRID_SEARCH_DIR = os.path.join(RESULTS_DIR, "embed_port_output")
ORIGINAL_GRID_EMBEDDING = os.path.join(GRID_SEARCH_DIR, "grid_search_results.csv")
ORIGINAL_MONTHLY_RETURNS_DIR = os.path.join(GRID_SEARCH_DIR, "monthly_returns")
ORIGINAL_MONTHLY_PERF_DIR = os.path.join(GRID_SEARCH_DIR, "monthly_performance")
COMBINED_MONTHLY_PERF_DIR = os.path.join(GRID_SEARCH_DIR, "combined_monthly_perf")
RAW_MONTHLY_PERF_DIR = os.path.join(GRID_SEARCH_DIR, "raw_monthly_perf")
COMBINED_RESULTS_CSV = os.path.join(GRID_SEARCH_DIR, "grid_search_results_combined.csv")
RAW_RESULTS_CSV = os.path.join(GRID_SEARCH_DIR, "grid_search_results_raw.csv")
RAW_INTERIM_DIR = os.path.join(GRID_SEARCH_DIR, "raw_interim")

TC_MONTHLY_RETURNS_DIR = os.path.join(GRID_SEARCH_DIR, "tc_monthly_returns")
TC_RESULTS_CSV = os.path.join(GRID_SEARCH_DIR, "grid_search_results_tc.csv")
SCALER_PATH = os.path.join(RESULTS_DIR, "embed_train_others","scaler_saint.pkl")
###############################################################################
# OUTPUT CSV HEADER
###############################################################################
COMBINED_TABLE_FIELDNAMES = [
    'n_estimators','knn_neighbors','pca_components','max_short_frac','max_weight',
    'clip_quantile','threshold_neg','threshold_pos','rolling_window',
    'embed_mean','embed_std','embed_sharpe',
    'ew_mean','ew_std','ew_sharpe',
    'rf_mean','rf_std','rf_sharpe',
    'cat_mean','cat_std','cat_sharpe',
    'light_mean','light_std','light_sharpe'
]

###############################################################################
# TRAIN/TEST DATA FILES & LOGGING
###############################################################################
CONFIG = {
    "TRAIN_FEAT": os.path.join(TRAIN_TEST_DIR , "train_features.csv"),
    "TRAIN_TARG": os.path.join(TRAIN_TEST_DIR , "train_targets.csv"),
    "TEST_FEAT" : os.path.join(TRAIN_TEST_DIR ,"test_features.csv"),
    "TEST_TARG" : os.path.join(TRAIN_TEST_DIR ,  "test_targets.csv"),
    "TRAIN_JSON": os.path.join(TRAIN_TEST_DIR , "train_files.json"),
    "TEST_JSON":  os.path.join(TRAIN_TEST_DIR , "test_files.json"),
    "LOG_FILE"  : os.path.join(GRID_SEARCH_DIR, "grid_search.log"),
    "TC_LOG_FILE"  : os.path.join(GRID_SEARCH_DIR, "tc_grid_search.log"),
    "MONTHLY_RETURNS_DIR": os.path.join(GRID_SEARCH_DIR, "monthly_returns"),

    # Pre-filter threshold
    "RETURN_THRESH": 1e-4,

    # Hyperparameter grid
    "N_EST_LIST"     : [100, 200, 300],
    "KNN_K_LIST"     : [5, 10, 20],
    "PCA_COMPONENTS" : [None, 100, 200],
    "CLIP_QUANTILES" : [0.0, 0.01, 0.05],
    "EDGE_THRESHOLDS": [0.0, 0.01, 0.05],

    # Fixed parameters
    "MAX_SHORT_FRAC" : 0.3,
    "MAX_WEIGHT"     : 0.1,
    "Roll_Window"    : None,

    "TRANSACTION_COST": 1e-4
}