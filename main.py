#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for Differentiable Quantum-Informed Neural Inversion.
Author: Hari Hardiyan <lorozloraz@gmail.com>
License: MIT
"""

import os
import math
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np
import pandas as pd

# Import modular components from src layout
from quantum_helios.physics import physics_residuals, R_SUN
from quantum_helios.model import PIQNN, OutputLogRangeAdaptor
from quantum_helios.utils import (
    StandardScalerJAX, 
    synthesize_dataset_np, 
    train_test_split_jax,
    calculate_metrics
)
from quantum_helios.viz import plot_diagnostic_results

# Set double precision for physical integrity
jax.config.update("jax_enable_x64", True)

def main():
    # ---------------------------------------------------------
    # 1. Hyperparameters & Configuration
    # ---------------------------------------------------------
    SEED = 42
    N_SAMPLES = 256
    TEST_RATIO = 0.25
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-3
    EPOCHS = 40
    LAMBDA_PHYS = 2.5
    QUBITS = 6
    LAYERS = 5
    MC_SAMPLES = 100  # For Uncertainty Quantification
    
    KEY = jr.PRNGKey(SEED)
    k_data, k_model, k_train, k_eval = jr.split(KEY, 4)

    # Create directories for results
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/artifacts", exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data Pipeline
    # ---------------------------------------------------------
    print(f"--- Synthesizing Dataset (N={N_SAMPLES}) ---")
    X_raw, Y_raw, Br_range, r0_range, cs_range = synthesize_dataset_np(seed=SEED, N=N_SAMPLES)
    
    # Range configuration for the Log-Space Adaptor
    physical_ranges = [Br_range, r0_range, cs_range]
    
    # Scale features (Target weights based on variance for balancing)
    scaler = StandardScalerJAX.fit(X_raw)
    X_scaled = scaler.transform(X_raw)
    
    # Balancing weights for the 3 target parameters
    ranges_vec = jnp.array([r[1]-r[0] for r in physical_ranges])
    target_weights = 1.0 / (ranges_vec**2)

    X_train, Y_train, X_test, Y_test = train_test_split_jax(
        X_scaled, Y_raw, test_ratio=TEST_RATIO, key=k_data
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ---------------------------------------------------------
    # 3. Model & Optimizer Setup
    # ---------------------------------------------------------
    adaptor = OutputLogRangeAdaptor(physical_ranges)
    model = PIQNN(
        in_dim=X_train.shape[-1], 
        out_dim=3, 
        n_layers=LAYERS, 
        adaptor=adaptor, 
        key=k_model
    )

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ---------------------------------------------------------
    # 4. Physics-Informed Training Functions
    # ---------------------------------------------------------
    def loss_fn(model, x_batch, y_batch, key):
        # Forward pass (batch-enabled via PIQNN internal vmap)
        y_pred = model(x_batch)
        
        # 1. Supervised Loss (MSE weighted by physical scale)
        error = y_pred - y_batch
        sup_loss = jnp.mean(jnp.sum((error**2) * target_weights, axis=1))
        
        # 2. Physics Loss (Residuals evaluated on unscaled features)
        # Reconstruct physical units for Lambert W evaluation
        x_unscaled = scaler.inverse_transform(x_batch)
        phys_loss = physics_residuals(x_unscaled, y_pred)
        
        total_loss = sup_loss + LAMBDA_PHYS * phys_loss
        return total_loss, (sup_loss, phys_loss)

    @eqx.filter_jit
    def train_step(model, opt_state, x_batch, y_batch, key):
        (loss, (sup, phys)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, x_batch, y_batch, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, sup, phys

    # ---------------------------------------------------------
    # 5. Execution Loop
    # ---------------------------------------------------------
    print(f"--- Starting PIQNN Training ({EPOCHS} Epochs) ---")
    history = []
    
    for epoch in range(EPOCHS):
        # Shuffle training data
        epoch_key = jr.fold_in(k_train, epoch)
        perm = jr.permutation(epoch_key, jnp.arange(len(X_train)))
        X_train, Y_train = X_train[perm], Y_train[perm]
        
        n_batches = len(X_train) // BATCH_SIZE
        epoch_sup, epoch_phys = 0.0, 0.0
        
        for i in range(n_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            xb, yb = X_train[start:end], Y_train[start:end]
            
            model, opt_state, _, sup, phys = train_step(
                model, opt_state, xb, yb, jr.fold_in(epoch_key, i)
            )
            epoch_sup += sup
            epoch_phys += phys
        
        # Logging
        avg_sup = epoch_sup / n_batches
        avg_phys = epoch_phys / n_batches
        print(f"Epoch {epoch+1:02d} | Sup Loss: {avg_sup:.4e} | Phys Loss: {avg_phys:.4e}")
        history.append([epoch+1, float(avg_sup), float(avg_phys)])

    # ---------------------------------------------------------
    # 6. Evaluation & Uncertainty Quantification (UQ)
    # ---------------------------------------------------------
    print("--- Running Posterior Inversion & UQ Analysis ---")
    
    # MC Dropout logic vectorized using JAX vmap
    def predict_uq(x):
        # Process multiple forward passes for a single point
        keys = jr.split(k_eval, MC_SAMPLES)
        # Note: In a real dropout scenario, you'd add noise here
        # For VQC, we can treat parameter jitter or stochastic gates as noise
        return jax.vmap(model)(jnp.tile(x, (MC_SAMPLES, 1)))

    Y_pred = model(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(Y_test, Y_pred)
    labels = ["Br_r0 (T)", "r0 (m)", "cs (m/s)"]
    
    print("\n--- FINAL TEST METRICS ---")
    for i, lab in enumerate(labels):
        print(f"{lab:10s} | RMSE: {metrics['rmse'][i]:.4e} | RelMAE: {metrics['rel_mae'][i]*100:6.2f}%")

    # ---------------------------------------------------------
    # 7. Save Artifacts & Visualization
    # ---------------------------------------------------------
    # Save training log to CSV
    log_df = pd.DataFrame(history, columns=['epoch', 'sup_loss', 'phys_loss'])
    log_df.to_csv("results/logs/training_history.csv", index=False)
    
    # Save diagnostic plots
    plot_diagnostic_results(
        log_df, 
        np.array(Y_test), 
        np.array(Y_pred), 
        labels
    )
    
    # Save model weights (Equinox/JAX style)
    eqx.tree_serialise_seeds("results/artifacts/piqnn_model.eqx", model)
    
    print("\n[SUCCESS] Pipeline completed. All results saved in results/ directory.")

if __name__ == "__main__":
    main()
