# Embed: SAINT Multi-Task Embedding and Portfolio Construction

An end-to-end workflow for training, evaluating, and using multi-task embeddings from tablular timeseries data, constructed using PyTorch and Scikit-learn.

## Project Structure

- **embed/**
  - **embed_train/** *(Train SAINT multi-task model)*
    - `main.py` - Main script to run training pipeline
    - `training.py` - Training functions
    - `evaluation.py` - Evaluation functions 
    - `model/` - Model definitions and related classes
    - `utils/` - Utility scripts and helper functions
    - `config.py` - Configuration settings for training

  - **compute_embed/** *(Compute and save embeddings)*
    - `main.py` - Script to compute embeddings
    - `build_embedding.py` - Core embedding computation 
    - `config.py` - Configuration settings for embeddings

  - **construct_portfolio/** *(Portfolio construction scripts)*
    - `embed_port/` - Portfolio using computed embeddings
	- `embed_port_tc/` - Portfolio using computed embeddings after transaction cost`	
	- `embed_other_port/` -  Classical ML Portfolio using computed embeddings as features (RF, CatBoost, LGBM)
    - `raw_other_port/` - Classical ML Portfolio using raw features (RF, CatBoost, LGBM)
    - `config.py` - Configuration for portfolio construction

  - **analysis/** *(Analysis and visualization scripts)*
    - `embed_analysis/` - Analyze trained embedding
    - `port_analysis/` - Analyze portfolio performance

- `README.md` - Project documentation


## Data Setup

The project expects CSV batch files stored separately in a sibling directory: ../data

Example filenames:
- `retrieved_data_batch_1_2000-01-01_to_2000-03-31.csv`
- `retrieved_data_batch_96_2023-10-01_to_2023-12-31.csv`

The files contain extensive financial and accounting features necessary for the SAINT model, such as returns, fundamental ratios, market data, and company identifiers.

---

## Installation

You can run all scripts **from the project root** without changing directories.

1. **Clone your GitHub repo (make sure branch is `main`)**  

   ```bash
   git clone git@github.com:donggeun‑kim‑oxford/firm‑character‑embedding.git
   cd firm-character-embedding
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```


## Usage

All scripts must be executed from the project root without changing directories:

1. Training SAINT embeddings
	```bash
	python embed_train/main.py
	```

2. Computing Embeddings
	```bash
	python compute_embed/main.py
	```

3. Portfolio Construction

   * Construct portfolio using embeddings:

     ```bash
     python -m construct_portfolio.embed_port.main
     ```

   * Construct portfolio using raw financial features (RandomForest, CatBoost, LightGBM):

     ```bash
     python -m construct_portfolio.raw_other_port.main
     ```

   * Construct portfolio using ML models trained on embeddings as features (RandomForest, CatBoost, LightGBM):

     ```bash
     python -m construct_portfolio.embed_other_port.main
     ```

   * Construct portfolio using embeddings with transaction cost adjustments:

     ```bash
     python -m construct_portfolio.embed_port_tc.main
     ```

4. Embedding and Portfolio Performance Analysis

   * Embedding Analysis:

     * Analyze the average temporal difference between target embeddings and their closest neighbors:

       ```bash
       python -m analysis.embed_analysis.mean_time_diff
       ```
     * Evaluate the fraction of closest neighbors sharing the same GICS industry classification:

       ```bash
       python -m analysis.embed_analysis.neighbor_analysis
       ```
     * Visualize SAINT model training and test loss over epochs:

       ```bash
       python -m analysis.embed_analysis.plot_train_loss
       ```
     * Analyze the time series of prediction sign accuracy using closest neighbors:

       ```bash
       python -m analysis.embed_analysis.sign_accuracy_timeseries
       ```
     * Assess the average prediction sign accuracy from closest neighbors:

       ```bash
       python -m analysis.embed_analysis.sign_accuracy
       ```

   * Portfolio Performance Analysis:

     * Compare embedding-based portfolio performance against ML-based portfolios using raw features:

       ```bash
       python -m analysis.port_analysis.port_perform_raw
       ```
     * Compare embedding-based portfolio performance against ML-based portfolios trained on embeddings:

       ```bash
       python -m analysis.port_analysis.port_perform_raw
       ```
     * Evaluate embedding-based portfolio performance before and after incorporating transaction costs:

       ```bash
       python -m analysis.port_analysis.tc_analysis
       ```
