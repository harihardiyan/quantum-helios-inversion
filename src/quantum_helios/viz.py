import matplotlib.pyplot as plt
import numpy as np
import os

def plot_diagnostic_results(logs, y_true, y_pred, labels, save_dir="results/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Loss Curve (Supervised vs Physics)
    plt.figure(figsize=(10, 5))
    plt.plot(logs['epoch'], logs['sup_loss'], label='Supervised (Data) Loss', lw=2)
    plt.plot(logs['epoch'], logs['phys_loss'], label='Physics (Residual) Loss', lw=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Diagnostic Training Evolution')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)
    plt.close()

    # 2. Parity Plots (Predicted vs Ground Truth)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, lab in enumerate(labels):
        ax = axes[i]
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, edgecolors='k')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2) # Ideal line
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Parity: {lab}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/parity_plots.png", dpi=300)
    plt.close()
