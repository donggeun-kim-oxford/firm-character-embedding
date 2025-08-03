
import os, json, itertools, time, gc, csv, shutil, warnings
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from construct_portfolio.config import (

    CONFIG,
    ORIGINAL_GRID_EMBEDDING,
    RAW_MONTHLY_PERF_DIR,
    RAW_RESULTS_CSV,
    ORIGINAL_MONTHLY_PERF_DIR,
    RAW_INTERIM_DIR

)

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ----------------------------
# Setup folder paths
# ----------------------------
original_grid_embedding = ORIGINAL_GRID_EMBEDDING 
original_monthly_perf_dir = ORIGINAL_MONTHLY_PERF_DIR
new_monthly_perf_dir = RAW_MONTHLY_PERF_DIR 
os.makedirs(new_monthly_perf_dir, exist_ok=True)
raw_results_dir = RAW_INTERIM_DIR
os.makedirs(raw_results_dir, exist_ok=True)


# We'll use a master file only for reference; interim results are saved per grid combination.

    

# Fieldnames for classical results.
RAW_FIELDNAMES = [
    'n_estimators', 'knn_neighbors', 'pca_components', 'max_short_frac', 'max_weight',
    'clip_quantile', 'threshold_neg', 'threshold_pos', 'rolling_window',
    'embed_mean', 'embed_std', 'embed_sharpe',
    'ew_mean', 'ew_std', 'ew_sharpe',
    'rf_raw_mean', 'rf_raw_std', 'rf_raw_sharpe',
    'cat_raw_mean', 'cat_raw_std', 'cat_raw_sharpe',
    'light_raw_mean', 'light_raw_std', 'light_raw_sharpe'
]
raw_results_csv = RAW_RESULTS_CSV 
if not os.path.exists(raw_results_csv):
    os.makedirs(os.path.dirname(raw_results_csv), exist_ok=True)
    with open(raw_results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
# =============================================================================
# Utility Functions
# =============================================================================
def iter_chunks(file_list, chunksize=10000):
    for file in file_list:
        if os.path.isfile(file):
            try:
                for chunk in pd.read_csv(file, chunksize=chunksize, low_memory=False):
                    yield chunk
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"File not found: {file}")

def process_chunk(chunk, categorical_cols):
    if 'date' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
    chunk = chunk.dropna(subset=['date'])
    chunk = chunk.set_index('date', drop=False)
    for col in categorical_cols:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype(str)
    return chunk

def adjust_predictions(preds, min_avg=0.01, factor=100):
    if np.mean(np.abs(preds)) < min_avg:
        return preds * factor
    return preds

def clip_and_filter_predictions(preds, clip_quantile, threshold_neg, threshold_pos):
    if np.ptp(preds) < 1e-4:
        return np.where(preds > 0, 1, np.where(preds < 0, -1, 0))
    lower = np.quantile(preds, clip_quantile)
    upper = np.quantile(preds, 1 - clip_quantile)
    preds_clipped = np.clip(preds, lower, upper)
    return np.array([ -1 if p < threshold_neg else (1 if p > threshold_pos else 0) for p in preds_clipped ])

def build_portfolio_df_online(index, y, preds, tickers, clip_quantile, threshold_neg, threshold_pos):
    signals = clip_and_filter_predictions(preds, clip_quantile, threshold_neg, threshold_pos)
    df = pd.DataFrame(index=index)
    df['ticker'] = tickers
    df['y_return'] = y
    df['pred'] = preds
    df['signal'] = signals
    df = df.sort_index()
    df['year_month'] = pd.to_datetime(df.index).to_period('M')
    return df

def compute_portfolio_returns_online(df, return_col, signal_col, ticker_col='ticker',
                                    max_short_frac=1.0, max_weight=1.0, clip_quantile=0.01):
    agg_df = df.groupby(['year_month', ticker_col]).agg(
        last_signal=(signal_col, 'last'),
        last_return=(return_col, 'last')
    ).reset_index()
    def monthly_return(group):
        lr = group['last_return']
        lower_bound = lr.quantile(clip_quantile)
        upper_bound = lr.quantile(1 - clip_quantile)
        clipped = lr.clip(lower=lower_bound, upper=upper_bound)
        longs = group[group['last_signal'] == 1]
        shorts = group[group['last_signal'] == -1]
        if len(longs) == 0 and len(shorts) == 0:
            return 0.0
        weight_long = (1 - max_short_frac)
        weight_short = max_short_frac
        long_weight = weight_long / len(longs) if len(longs) > 0 else 0.0
        short_weight = -weight_short / len(shorts) if len(shorts) > 0 else 0.0
        weights = group['last_signal'].apply(lambda s: long_weight if s == 1 else (short_weight if s == -1 else 0.0))
        weights = weights.clip(-max_weight, max_weight)
        total_abs = weights.abs().sum()
        if total_abs == 0:
            return 0.0
        return (weights * clipped).sum() / total_abs
    monthly_returns = agg_df.groupby('year_month').apply(monthly_return)
    return monthly_returns.sort_index()

# =============================================================================
# Part B: Load File Lists
# =============================================================================
with open(CONFIG["TRAIN_JSON"], "r") as f:
    train_files = json.load(f)
with open(CONFIG["TEST_JSON"], "r") as f:
    test_files = json.load(f)
print(f"Found {len(train_files)} training files and {len(test_files)} testing files.")

# =============================================================================
# Part C: Build Dictionaries for Categorical Columns
# =============================================================================
def build_dict_for_col(file_list, col_name, chunksize=10000):
    token_to_id = {"UNK": 0}
    for file in file_list:
        try:
            for chunk in pd.read_csv(file, usecols=[col_name], chunksize=chunksize, low_memory=False):
                chunk = chunk.dropna(subset=[col_name])
                for val in chunk[col_name].unique():
                    s = str(val)
                    if s not in token_to_id:
                        token_to_id[s] = len(token_to_id)
        except Exception as e:
            print(f"Error processing {file} for column {col_name}: {e}")
    return token_to_id

ticker_dict = build_dict_for_col(train_files, "ticker")
gics_exists = any("gics" in pd.read_csv(f, nrows=1).columns for f in train_files if os.path.isfile(f))
if gics_exists:
    gics_dict = build_dict_for_col(train_files, "gics")

# =============================================================================
# Part D: Define Categorical and Predictor Columns
# =============================================================================
categorical_cols = ["ticker"]
if gics_exists:
    categorical_cols.append("gics")
def get_predictor_cols(df):
    return [col for col in df.columns if col not in ['ret_exc_lead1m', 'date']]

# =============================================================================
# Part E: Load Full Raw Data for Classical Models (Chunked)
# =============================================================================
def load_full_data(file_list, chunksize=10000):
    df_list = []
    for file in file_list:
        try:
            for chunk in pd.read_csv(file, chunksize=chunksize, low_memory=False):
                chunk = process_chunk(chunk, categorical_cols)
                df_list.append(chunk)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    if df_list:
        return pd.concat(df_list, ignore_index=False)
    else:
        return pd.DataFrame()

print("Loading full training data...")
train_df = load_full_data(train_files, chunksize=10000)
print(f"Combined raw training data shape: {train_df.shape}")
print("Loading full test data...")
test_df = load_full_data(test_files, chunksize=10000)
print(f"Combined raw testing data shape: {test_df.shape}")

for col in get_predictor_cols(train_df):
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

train_df['ret_exc_lead1m'] = pd.to_numeric(train_df['ret_exc_lead1m'], errors='coerce')
test_df['ret_exc_lead1m'] = pd.to_numeric(test_df['ret_exc_lead1m'], errors='coerce')
train_df = train_df.dropna(subset=['ret_exc_lead1m'])
test_df = test_df.dropna(subset=['ret_exc_lead1m'])

train_df['ticker'] = train_df['ticker'].map(lambda x: ticker_dict.get(x, 0))
test_df['ticker'] = test_df['ticker'].map(lambda x: ticker_dict.get(x, 0))
if gics_exists:
    train_df['gics'] = train_df['gics'].map(lambda x: gics_dict.get(x, 0))
    test_df['gics'] = test_df['gics'].map(lambda x: gics_dict.get(x, 0))

# =============================================================================
# Part F: Scaling
# =============================================================================
scaler = StandardScaler()
X_train = train_df[get_predictor_cols(train_df)].astype(float)
X_test = test_df[get_predictor_cols(test_df)].astype(float)
y_train = train_df['ret_exc_lead1m'].astype(float)
y_test = test_df['ret_exc_lead1m'].astype(float)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

test_index = pd.to_datetime(test_df.index)
test_tickers = test_df['ticker']

# =============================================================================
# Part G: Classical Grid Search Loop for Classical Models
# =============================================================================
grid = {
'n_estimators':  CONFIG["N_EST_LIST"],
'knn_neighbors': CONFIG["KNN_K_LIST"],
'rolling_window':[CONFIG["Roll_Window"]],
'pca_components':CONFIG["PCA_COMPONENTS"],
'max_short_frac':[CONFIG["MAX_SHORT_FRAC"]],
'max_weight':    [CONFIG["MAX_WEIGHT"]],
'clip_quantile': CONFIG["CLIP_QUANTILES"],
'edge_threshold':CONFIG["EDGE_THRESHOLDS"]
}
keys, values = zip(*grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Total grid combinations: {len(combinations)}")

df_embed_orig = pd.read_csv(original_grid_embedding)

# Rebuild master CSV from existing interim files
existing_df = pd.DataFrame(columns=RAW_FIELDNAMES)
for j in range(len(combinations)):
    interim = os.path.join(raw_results_dir, f"grid_search_results_raw_{j+1}.csv")
    if os.path.exists(interim):
        df_int = pd.read_csv(interim)
        existing_df = pd.concat([existing_df, df_int], ignore_index=True)
        print(f"Appended {interim} to master dataframe")
existing_df.to_csv(raw_results_csv, index=False)
print("Rebuilt master CSV from interim files")

results_list = []
start_time = time.time()

for i, params in enumerate(combinations):
    interim_file = os.path.join(raw_results_dir, f"grid_search_results_raw_{i+1}.csv")
    if os.path.exists(interim_file):
        print(f"Combination {i+1} already computed (interim file exists: {interim_file}); skipping...")
        continue

    print(f"Evaluating grid combination {i+1}/{len(combinations)}: {params}")
    
    # Enforce: threshold_neg = - threshold_pos
    threshold_pos = params['edge_threshold']
    threshold_neg = -threshold_pos  # Override any value provided in params
    # Use the rest of the parameters as is.
    clip_quantile = params['clip_quantile']
    max_short_frac = params['max_short_frac']
    max_weight = params['max_weight']
    
    if params['pca_components'] is None:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    else:
        pca = PCA(n_components=int(params['pca_components']))
        X_train_final = pd.DataFrame(pca.fit_transform(X_train_scaled), index=X_train_scaled.index)
        X_test_final = pd.DataFrame(pca.transform(X_test_scaled), index=X_test_scaled.index)
    
    rf_model = RandomForestRegressor(n_estimators=int(params['n_estimators']), random_state=42, n_jobs=-1)
    rf_model.fit(X_train_final, y_train)
    rf_preds = rf_model.predict(X_test_final)
    rf_preds = adjust_predictions(rf_preds)
    
    cat_model = CatBoostRegressor(iterations=int(params['n_estimators']), random_seed=42, verbose=0)
    cat_model.fit(X_train_final, y_train)
    cat_preds = cat_model.predict(X_test_final)
    cat_preds = adjust_predictions(cat_preds)
    
    light_model = LGBMRegressor(n_estimators=int(params['n_estimators']), random_state=42, n_jobs=-1)
    light_model.fit(X_train_final, y_train)
    light_preds = light_model.predict(X_test_final)
    light_preds = adjust_predictions(light_preds)
    
    def build_portfolio_from_preds(preds):
        df_port = build_portfolio_df_online(test_index, y_test, preds, test_tickers,
                                            clip_quantile, threshold_neg, threshold_pos)
        return compute_portfolio_returns_online(df_port, 'y_return', 'signal',
                                                max_short_frac=max_short_frac, max_weight=max_weight,
                                                clip_quantile=clip_quantile)
    
    rf_returns = build_portfolio_from_preds(rf_preds)
    cat_returns = build_portfolio_from_preds(cat_preds)
    light_returns = build_portfolio_from_preds(light_preds)
    
    def compute_perf(returns):
        m = returns.mean()
        s = returns.std()
        sh = (m / s * sqrt(12)) if s != 0 else np.nan
        return m, s, sh
    
    rf_mean, rf_std, rf_sharpe = compute_perf(rf_returns)
    cat_mean, cat_std, cat_sharpe = compute_perf(cat_returns)
    light_mean, light_std, light_sharpe = compute_perf(light_returns)
    
    if i < len(df_embed_orig):
        embed_mean = df_embed_orig.iloc[i]['mean_test']
        embed_std = df_embed_orig.iloc[i]['std_test']
        embed_sharpe = df_embed_orig.iloc[i]['sharpe_test']
        ew_mean = df_embed_orig.iloc[i]['mean_bench_test']
        ew_std = df_embed_orig.iloc[i]['std_bench_test']
        ew_sharpe = df_embed_orig.iloc[i]['sharpe_bench_test']
    else:
        embed_mean, embed_std, embed_sharpe = np.nan, np.nan, np.nan
        ew_mean, ew_std, ew_sharpe = np.nan, np.nan, np.nan
    
    def fmt(val):
        return "" if pd.isnull(val) or val is None else val
    result = {
        'n_estimators': fmt(params.get('n_estimators')),
        'knn_neighbors': fmt(params.get('knn_neighbors')),
        'pca_components': fmt(params.get('pca_components')),
        'max_short_frac': fmt(params.get('max_short_frac')),
        'max_weight': fmt(params.get('max_weight')),
        'clip_quantile': fmt(params.get('clip_quantile')),
        'threshold_neg': fmt(threshold_neg),  # use updated value
        'threshold_pos': fmt(threshold_pos),
        'rolling_window': fmt(params.get('rolling_window')),
        'embed_mean': embed_mean,
        'embed_std': embed_std,
        'embed_sharpe': embed_sharpe,
        'ew_mean': ew_mean,
        'ew_std': ew_std,
        'ew_sharpe': ew_sharpe,
        'rf_raw_mean': rf_mean, 'rf_raw_std': rf_std, 'rf_raw_sharpe': rf_sharpe,
        'cat_raw_mean': cat_mean, 'cat_raw_std': cat_std, 'cat_raw_sharpe': cat_sharpe,
        'light_raw_mean': light_mean, 'light_raw_std': light_std, 'light_raw_sharpe': light_sharpe,
    }
    results_list.append(result)
    
    # Save interim result to a separate file for this grid combination.
    with open(interim_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerow(result)
    print(f"Saved interim result to {interim_file}")
    
    # Append new result to master CSV in append mode.
    with open(raw_results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
        writer.writerow(result)
    
    # --- Part G: Augment Monthly Performance for This Grid Combination ---
    orig_mp_file = os.path.join(original_monthly_perf_dir, f"monthly_perf_hp_{i+1}.csv")
    new_mp_file = os.path.join(new_monthly_perf_dir, f"monthly_perf_hp_{i+1}.csv")
    if os.path.exists(orig_mp_file):
        shutil.copy(orig_mp_file, new_mp_file)
        print(f"Copied {orig_mp_file} to {new_mp_file}")
        df_mp = pd.read_csv(new_mp_file, index_col=0)
        try:
            df_mp.index = pd.to_datetime(df_mp.index).to_period('M')
        except Exception as e:
            print(f"Error converting index in {new_mp_file}: {e}")
        rf_monthly_returns = compute_portfolio_returns_online(
            build_portfolio_df_online(test_index, y_test, rf_preds, test_tickers,
                                    clip_quantile, threshold_neg, threshold_pos),
            'y_return', 'signal', max_short_frac=max_short_frac, max_weight=max_weight, clip_quantile=clip_quantile)
        rf_monthly_equity = (1 + rf_monthly_returns).cumprod()
        
        cat_monthly_returns = compute_portfolio_returns_online(
            build_portfolio_df_online(test_index, y_test, cat_preds, test_tickers,
                                    clip_quantile, threshold_neg, threshold_pos),
            'y_return', 'signal', max_short_frac=max_short_frac, max_weight=max_weight, clip_quantile=clip_quantile)
        cat_monthly_equity = (1 + cat_monthly_returns).cumprod()
        
        light_monthly_returns = compute_portfolio_returns_online(
            build_portfolio_df_online(test_index, y_test, light_preds, test_tickers,
                                    clip_quantile, threshold_neg, threshold_pos),
            'y_return', 'signal', max_short_frac=max_short_frac, max_weight=max_weight, clip_quantile=clip_quantile)
        light_monthly_equity = (1 + light_monthly_returns).cumprod()
        
        df_mp['Monthly_Return_rf_raw'] = rf_monthly_returns.reindex(df_mp.index)
        df_mp['Equity_rf_raw'] = rf_monthly_equity.reindex(df_mp.index)
        rf_std_month = rf_monthly_returns.expanding(min_periods=1).std().reindex(df_mp.index)
        rf_mean_month = rf_monthly_returns.expanding(min_periods=1).mean().reindex(df_mp.index)
        df_mp['Run_STD_rf_raw'] = rf_std_month
        df_mp['Run_Sharpe_rf_raw'] = (rf_mean_month / rf_std_month * sqrt(12))
        
        df_mp['Monthly_Return_cat_raw'] = cat_monthly_returns.reindex(df_mp.index)
        df_mp['Equity_cat_raw'] = cat_monthly_equity.reindex(df_mp.index)
        cat_std_month = cat_monthly_returns.expanding(min_periods=1).std().reindex(df_mp.index)
        cat_mean_month = cat_monthly_returns.expanding(min_periods=1).mean().reindex(df_mp.index)
        df_mp['Run_STD_cat_raw'] = cat_std_month
        df_mp['Run_Sharpe_cat_raw'] = (cat_mean_month / cat_std_month * sqrt(12))
        
        df_mp['Monthly_Return_light_raw'] = light_monthly_returns.reindex(df_mp.index)
        df_mp['Equity_light_raw'] = light_monthly_equity.reindex(df_mp.index)
        light_std_month = light_monthly_returns.expanding(min_periods=1).std().reindex(df_mp.index)
        light_mean_month = light_monthly_returns.expanding(min_periods=1).mean().reindex(df_mp.index)
        df_mp['Run_STD_light_raw'] = light_std_month
        df_mp['Run_Sharpe_light_raw'] = (light_mean_month / light_std_month * sqrt(12))
        
        df_mp.to_csv(new_mp_file)
        print(f"Augmented monthly performance file saved to {new_mp_file}")
    else:
        print(f"Original monthly file {orig_mp_file} not found.")
    
    gc.collect()
    print(f"Completed grid combination {i+1}")

end_time = time.time()
print(f"Grid search complete. Raw results saved in {raw_results_csv}")
print(f"Total evaluation time: {end_time - start_time:.2f} seconds")