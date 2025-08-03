import json
import matplotlib.pyplot as plt
import os
import matplotlib
from analysis.embed_analysis.config import (
    TRAIN_LOSS_OUTPUT_DIR,
    PERF_JSON
)

# Set a professional style
plt.style.use('seaborn-v0_8-muted')
matplotlib.rcParams.update({
    'figure.figsize': (8, 5),
    'figure.dpi': 100,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.grid': True
})

def plot_losses_from_file(json_path, output_path):
    """
    Reads loss history from a JSON file and plots train vs. test curves
    for each task. Any task whose training loss is zero throughout is skipped.
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r') as f:
        loss_data = json.load(f)

    tasks = [key[:-6] for key in loss_data.keys() if key.endswith('_train')]
    os.makedirs(output_path, exist_ok=True)
    for task in tasks:
        train = loss_data.get(f"{task}_train")
        test  = loss_data.get(f"{task}_test")

        if train is None or test is None or all(v == 0.0 for v in train):
            continue

        plt.figure()
        plt.plot(train, label='Train Loss', color='#1f77b4', marker='o')  # blue
        plt.plot(test, label='Test Loss', color='#ff7f0e', marker='s')    # orange
        
        plt.title(f"{task.replace('_', ' ').title()} Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{task}.png"))
        plt.show()

if __name__ == "__main__":
    
    plot_losses_from_file(PERF_JSON, TRAIN_LOSS_OUTPUT_DIR)