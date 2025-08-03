import os
import json
import logging
import numpy as np
import re
import pandas as pd
import csv
from compute_embed.config import CONFIG

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle NumPy data types.
    Converts np.float32 and similar types to native Python types.
    """
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
def setup_logging(debug=False):
    """
    Sets up logging configuration.
    If debug is True, sets logging level to DEBUG for detailed logs.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("compute_embed.log"),
            logging.StreamHandler()
        ]
    )

def save_json(data, filename):
    """
    Saves data to a JSON file using the custom NumpyEncoder.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

def load_json(path_):
    """
    Loads a JSON file. If the file doesn't exist, returns {'UNK': 0} and logs a warning.
    """
    if not os.path.isfile(path_):
        logging.warning(f"Dict file not found => {path_}, returning {{'UNK':0}}")
        return {"UNK": 0}
    with open(path_, "r") as f:
        return json.load(f)

def verify_csv_structure(all_csv_files, expected_columns):
    """
    Verifies that all CSV files have the expected columns.
    Logs any discrepancies found.
    """
    for csv_ in all_csv_files:
        try:
            df = pd.read_csv(csv_, nrows=1, low_memory=False)
            cols = df.columns.tolist()
            missing_cols = set(expected_columns) - set(cols)
            if missing_cols:
                logging.error(f"File {csv_} is missing columns: {missing_cols}")
        except Exception as e:
            logging.error(f"Error reading CSV file {csv_}: {e}")

def normalize_company_name(name):
    normalized_name = re.sub(r'_\d+$', '', name)
    return normalized_name.strip()


def process_data(embeddings_path, row_to_ret, train_dates, test_dates,
                 output_feature_train=CONFIG["train_features_path"],  
                 output_feature_test=CONFIG["test_features_path"], 
                 output_target_train=CONFIG["train_targets_path"],
                 output_target_test=CONFIG["test_targets_path"],
                 chunk_size=10000):
        # Load the shape of the .npy file first
        with open(embeddings_path, "rb") as f:
            shape = np.load(f).shape  # This gets the correct shape

        num_rows, embedding_dim = shape  # Unpack shape
        
        # Now, create a memory map with correct shape
        data_iterator = np.memmap(embeddings_path, dtype="float32", mode="r", shape=(num_rows, embedding_dim))

        # Initialize CSV headers
        feature_columns = ["Date", "Ticker"] + [f"dim_{i}" for i in range(embedding_dim)]
        target_columns = ["Date", "Ticker", "Return"]

        for file in [output_feature_train, output_feature_test, output_target_train, output_target_test]:
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(feature_columns if "features" in file else target_columns)

        batch_size = chunk_size

        with open(output_feature_train, "a", newline="") as f_train, \
            open(output_feature_test, "a", newline="") as f_test, \
            open(output_target_train, "a", newline="") as f_train_target, \
            open(output_target_test, "a", newline="") as f_test_target:

            writer_train = csv.writer(f_train)
            writer_test = csv.writer(f_test)
            writer_train_target = csv.writer(f_train_target)
            writer_test_target = csv.writer(f_test_target)

            for idx, (key, embedding) in enumerate(zip(row_to_ret.keys(), data_iterator)):
                match = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", key)
                if match:
                    date, ticker = match.groups()
                    ret_value = row_to_ret[key]

                    row_X = [date, ticker] + list(np.round(embedding, 5))
                    row_F = [date, ticker, ret_value]

                    if date in train_dates:
                        writer_train.writerow(row_X)
                        writer_train_target.writerow(row_F)
                    elif date in test_dates:
                        writer_test.writerow(row_X)
                        writer_test_target.writerow(row_F)

                    # Process in batches
                    if idx % batch_size == 0 and idx > 0:
                        print(f"Processed {idx} rows...")

        print("Processing complete. Data saved to:")
        print(f"- {output_feature_train}, {output_feature_test}")
        print(f"- {output_target_train}, {output_target_test}")

# Function to print a sample chunk and dimensions
def print_sample_and_shape(file_path, sample_size=5):
    try:
        # Read a small sample (first `sample_size` rows)
        df_sample = pd.read_csv(file_path, nrows=sample_size)
        df_full = pd.read_csv(file_path, usecols=[0])  # Read only first column to get number of rows
        
        print(f"\n===== {file_path} =====")
        print(f"Shape: {df_full.shape[0]} rows × {len(df_sample.columns)} columns")
        print(df_sample)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Extract date ranges from filenames
def extract_date_range(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.groups()
    return None, None