#!/usr/bin/env python3
"""
Neural network training with enhanced numerical features + title embeddings
(but NO domain/user embeddings) to find the optimal configuration
"""
import os
import pickle
import shutil
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import backend.config as cfg
from backend.data_processing import create_data_loader_fixed as create_data_loader, prepare_features_fixed as prepare_features


class NumericalPlusTitleNN(nn.Module):
    """Neural network that processes numerical and title features separately then combines them."""
    
    def __init__(self, numerical_dim, title_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        # Process numerical features
        self.numerical_net = nn.Sequential(
            nn.Linear(numerical_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # Process title embeddings  
        self.title_net = nn.Sequential(
            nn.Linear(title_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Combined processing
        combined_dim = (hidden_dim // 2) + hidden_dim  # numerical + title
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, numerical, title):
        # Process each feature type separately
        num_features = self.numerical_net(numerical)
        title_features = self.title_net(title)
        
        # Combine and process
        combined = torch.cat([num_features, title_features], dim=1)
        return self.combined_net(combined).squeeze()


def test_configuration(config, X_num_train, X_num_val, X_num_test, 
                      X_title_train, X_title_val, X_title_test,
                      y_train, y_val, y_test, verbose=True):
    """Test a specific neural network configuration."""
    
    # Convert to tensors
    X_num_train_tensor = torch.FloatTensor(X_num_train)
    X_num_val_tensor = torch.FloatTensor(X_num_val)
    X_num_test_tensor = torch.FloatTensor(X_num_test)

    X_title_train_tensor = torch.FloatTensor(X_title_train)
    X_title_val_tensor = torch.FloatTensor(X_title_val)
    X_title_test_tensor = torch.FloatTensor(X_title_test)

    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    y_test_tensor = torch.FloatTensor(y_test)
    
    model = NumericalPlusTitleNN(
        numerical_dim=X_num_train.shape[1], 
        title_dim=X_title_train.shape[1],
        hidden_dim=config.get('hidden_dim', 128),
        dropout=config.get('dropout', 0.1)
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = config.get('patience', 300)
    best_model_state = None
    
    if verbose:
        print(f"📊 Testing: {config['name']} (LR={config['lr']:.0e}, Epochs={config['epochs']})")
    
    # Log configuration to wandb if this is the main run
    if hasattr(config, 'is_main_run') and config.get('is_main_run', False):
        wandb.log({
            f"config/{k}": v for k, v in config.items() 
            if k not in ['name', 'is_main_run']
        })
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        optimizer.zero_grad()
        pred = model(X_num_train_tensor, X_title_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Validation every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_num_val_tensor, X_title_val_tensor)
                val_r2 = r2_score(y_val, val_pred.numpy())
                
                # Log metrics to wandb if this is the main run
                if hasattr(config, 'is_main_run') and config.get('is_main_run', False):
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "val_r2": val_r2,
                        "best_val_r2": max(best_val_r2, val_r2),
                        "patience_counter": patience_counter
                    })
                
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 50
                
                if verbose and epoch % 200 == 0:
                    print(f"    Epoch {epoch:4d}: Val R² = {val_r2:.6f} (best: {best_val_r2:.6f})")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_num_test_tensor, X_title_test_tensor)
        test_r2 = r2_score(y_test, test_pred.numpy())
        test_loss = criterion(test_pred, y_test_tensor).item()
    
    if verbose:
        print(f"    Final: Val R² = {best_val_r2:.6f}, Test R² = {test_r2:.6f}")
    
    # Log final results to wandb if this is the main run
    if hasattr(config, 'is_main_run') and config.get('is_main_run', False):
        wandb.log({
            "final_val_r2": best_val_r2,
            "final_test_r2": test_r2,
            "final_test_loss": test_loss,
            "epochs_trained": min(epoch + 1, config['epochs'])
        })
    
    return {
        'val_r2': best_val_r2,
        'test_r2': test_r2,
        'test_loss': test_loss,
        'model_state': best_model_state,
        'epochs_trained': min(epoch + 1, config['epochs'])
    }


def create_run_directory():
    """Creates a unique directory for this training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("artifacts/training_runs", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir, config, model_params=None):
    """Saves the configuration and model parameters for this run."""
    config_dict = dict(config)
    
    if model_params:
        config_dict.update(model_params)
    
    config_dict['timestamp'] = datetime.now().isoformat()
    config_dict['run_directory'] = run_dir
    
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    return config_path


def update_runs_summary(run_dir, config, final_metrics):
    """Updates the master runs summary file."""
    summary_file = "artifacts/training_runs/runs_summary.csv"
    os.makedirs("artifacts/training_runs", exist_ok=True)
    
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'run_dir': os.path.basename(run_dir),
        'test_r2': final_metrics.get('test_r2', -999),
        'test_loss': final_metrics.get('test_loss', -999),
        'best_val_r2': final_metrics.get('best_val_r2', -999),
        'learning_rate': config['lr'],
        'hidden_dim': config.get('hidden_dim', 128),
        'dropout_rate': config.get('dropout', 0.1),
        'weight_decay': config.get('weight_decay', 1e-5),
        'epochs_trained': final_metrics.get('epochs_trained', -1),
        'architecture': 'numerical_plus_title'
    }
    
    write_header = not os.path.exists(summary_file)
    
    with open(summary_file, 'a') as f:
        if write_header:
            f.write(','.join(summary_data.keys()) + '\n')
        f.write(','.join(str(v) for v in summary_data.values()) + '\n')
    
    print(f"📊 Run summary updated: {summary_file}")


def train():
    """Main function to run the training pipeline."""
    print("🔍 NEURAL NETWORK: NUMERICAL + TITLE EMBEDDINGS (NO DOMAIN/USER)")
    print("=" * 65)
    
    # --- Initialize Weights & Biases ---
    wandb.init(
        project="hackernews-regression", 
        name=f"numerical_plus_title_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "architecture": "numerical_plus_title",
            "features": "numerical_enhanced_36d_plus_title_200d", 
            "approach": "separate_processing_then_combine",
            "target": "log_score_prediction"
        }
    )
    
    # --- Create Run Directory ---
    run_dir = create_run_directory()
    print(f"🗂️  Created run directory: {run_dir}")
    
    # --- Data Preparation with Enhanced Features ---
    print("🚀 Loading and preparing data...")
    config = cfg  # Use config directly instead of wandb
    data = prepare_features(config)
    
    # --- Train/Val/Test Split FIRST (before computing stats!) ---
    indices = np.arange(len(data["y"]))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    
    # CRITICAL: Compute domain/user stats ONLY on training data (no data leakage!)
    print("Adding domain/user statistical features (from TRAINING data only)...")
    
    train_domain_ids = data['X_domain_ids'][train_idx]
    train_user_ids = data['X_user_ids'][train_idx]
    train_scores = data['y'][train_idx]
    
    domain_stats = {}
    for domain_id in np.unique(train_domain_ids):
        mask = train_domain_ids == domain_id
        domain_stats[domain_id] = train_scores[mask].mean()
    
    user_stats = {}
    for user_id in np.unique(train_user_ids):
        mask = train_user_ids == user_id
        user_stats[user_id] = train_scores[mask].mean()
    
    # Apply stats to ALL samples (using training-derived stats)
    global_mean = train_scores.mean()
    domain_means = np.array([domain_stats.get(did, global_mean) for did in data['X_domain_ids']])
    user_means = np.array([user_stats.get(uid, global_mean) for uid in data['X_user_ids']])
    
    # Enhanced numerical features (36D)
    X_numerical_enhanced = np.column_stack([
        data['X_numerical'],  # Original 34 features
        domain_means,         # Feature 35: Domain mean score (from training only)
        user_means           # Feature 36: User mean score (from training only)
    ])
    
    # Title embeddings (200D)
    X_title_emb = data['X_title_embeddings']
    
    print(f"📊 Data: {len(data['y']):,} samples")
    print(f"  Numerical features: {X_numerical_enhanced.shape[1]}D (enhanced)")
    print(f"  Title embeddings: {X_title_emb.shape[1]}D (GloVe)")
    print(f"  Total input: {X_numerical_enhanced.shape[1] + X_title_emb.shape[1]}D")
    print(f"Split: Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,}")
    
    # Log data info to wandb
    wandb.log({
        "data/total_samples": len(data['y']),
        "data/train_samples": len(train_idx),
        "data/val_samples": len(val_idx), 
        "data/test_samples": len(test_idx),
        "data/numerical_features": X_numerical_enhanced.shape[1],
        "data/title_embedding_dim": X_title_emb.shape[1],
        "data/total_features": X_numerical_enhanced.shape[1] + X_title_emb.shape[1]
    })
    
    # Split the features
    X_num_train = X_numerical_enhanced[train_idx]
    X_num_val = X_numerical_enhanced[val_idx] 
    X_num_test = X_numerical_enhanced[test_idx]

    X_title_train = X_title_emb[train_idx]
    X_title_val = X_title_emb[val_idx]
    X_title_test = X_title_emb[test_idx]

    y_train = data['y'][train_idx]
    y_val = data['y'][val_idx]
    y_test = data['y'][test_idx]

    # Scale numerical features (fit on training only)
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_num_test_scaled = scaler.transform(X_num_test)

    # --- Baseline Comparisons ---
    print(f"\n🔍 BASELINE COMPARISONS:")
    
    # Combine features for linear regression
    X_combined_train = np.column_stack([X_num_train_scaled, X_title_train])
    X_combined_val = np.column_stack([X_num_val_scaled, X_title_val])
    X_combined_test = np.column_stack([X_num_test_scaled, X_title_test])

    # LinearRegression with numerical only
    print("  Testing LinearRegression (Numerical Only)...")
    lr_num_only = LinearRegression()
    lr_num_only.fit(X_num_train_scaled, y_train)
    lr_num_test_pred = lr_num_only.predict(X_num_test_scaled)
    lr_num_test_r2 = r2_score(y_test, lr_num_test_pred)

    # LinearRegression with numerical + title
    print("  Testing LinearRegression (Numerical + Title)...")
    lr_combined = LinearRegression()
    lr_combined.fit(X_combined_train, y_train)
    lr_combined_val_pred = lr_combined.predict(X_combined_val)
    lr_combined_test_pred = lr_combined.predict(X_combined_test)
    lr_combined_val_r2 = r2_score(y_val, lr_combined_val_pred)
    lr_combined_test_r2 = r2_score(y_test, lr_combined_test_pred)

    print(f"LinearRegression (Numerical Only) Test R²:   {lr_num_test_r2:.6f}")
    print(f"LinearRegression (Numerical + Title) Test R²: {lr_combined_test_r2:.6f}")
    print(f"Title embeddings add: {lr_combined_test_r2 - lr_num_test_r2:+.6f} R² points")

    # Log baseline results to wandb
    wandb.log({
        "baseline/lr_numerical_only_r2": lr_num_test_r2,
        "baseline/lr_combined_r2": lr_combined_test_r2,
        "baseline/title_embeddings_improvement": lr_combined_test_r2 - lr_num_test_r2
    })

    # --- Neural Network Configuration Testing ---
    print(f"\n🔍 NEURAL NETWORK CONFIGURATION TESTING:")
    
    configs = [
        {"lr": 2e-3, "epochs": 500, "hidden_dim": 128, "dropout": 0.1, "patience": 10000, "name": "Lower LR, 500 Epochs, No Early Stopping"},
    ]

    best_nn_r2 = -float('inf')
    best_config = None
    best_result = None
    config_results = []

    for i, config in enumerate(configs):
        # Mark the best config as main run for detailed logging
        if i == 0:  # For now, log the first config in detail
            config['is_main_run'] = True
            
        result = test_configuration(
            config, X_num_train_scaled, X_num_val_scaled, X_num_test_scaled,
            X_title_train, X_title_val, X_title_test,
            y_train, y_val, y_test
        )
        
        config_results.append({
            'config_name': config['name'],
            'test_r2': result['test_r2'],
            'val_r2': result['val_r2'],
            'lr': config['lr'],
            'hidden_dim': config.get('hidden_dim', 128),
            'dropout': config.get('dropout', 0.1)
        })
        
        if result['test_r2'] > best_nn_r2:
            best_nn_r2 = result['test_r2']
            best_config = config
            best_result = result

    # Log all configuration results
    for i, result in enumerate(config_results):
        wandb.log({
            f"configs/config_{i}_test_r2": result['test_r2'],
            f"configs/config_{i}_val_r2": result['val_r2'],
            f"configs/config_{i}_lr": result['lr'],
            f"configs/config_{i}_hidden_dim": result['hidden_dim'],
        })

    # --- Results Summary ---
    print(f"\n📈 FINAL COMPARISON:")
    print(f"  LinearRegression (Num only):      {lr_num_test_r2:.6f}")
    print(f"  LinearRegression (Num + Title):   {lr_combined_test_r2:.6f}")
    print(f"  Neural Network (Num + Title):     {best_nn_r2:.6f}")
    print(f"  Best NN Config: {best_config['name']}")

    print(f"\n🎯 IMPROVEMENTS:")
    print(f"  Title embeddings vs Numerical only: {lr_combined_test_r2 - lr_num_test_r2:+.6f}")
    print(f"  Neural Network vs LinearRegression:  {best_nn_r2 - lr_combined_test_r2:+.6f}")
    print(f"  Total NN improvement over Num only:  {best_nn_r2 - lr_num_test_r2:+.6f}")

    # Log final comparison metrics
    wandb.log({
        "final/lr_numerical_only": lr_num_test_r2,
        "final/lr_combined": lr_combined_test_r2,
        "final/best_nn": best_nn_r2,
        "final/title_improvement": lr_combined_test_r2 - lr_num_test_r2,
        "final/nn_vs_lr_improvement": best_nn_r2 - lr_combined_test_r2,
        "final/total_nn_improvement": best_nn_r2 - lr_num_test_r2
    })

    if best_nn_r2 > lr_combined_test_r2:
        print("✅ SUCCESS! Neural Network beats LinearRegression!")
        print("💡 Numerical + Title embeddings work well together")
        wandb.log({"status": "success_nn_beats_lr"})
    elif best_nn_r2 > lr_combined_test_r2 - 0.003:
        print("🔶 CLOSE! Neural Network nearly matches LinearRegression")
        print("💡 This combination is promising - might need more tuning")
        wandb.log({"status": "close_performance"})
    else:
        print("🚨 Neural Network still underperforms LinearRegression")
        print("💡 Even this simpler combination struggles")
        wandb.log({"status": "underperforming"})

    # Show progress toward target
    target_r2 = 0.2
    if best_nn_r2 > 0.1:
        print(f"\n🎉 Great progress! R² = {best_nn_r2:.4f} is halfway to target R² = {target_r2}")
    elif best_nn_r2 > 0.05:
        print(f"\n🔥 Good progress! R² = {best_nn_r2:.4f} is getting closer to meaningful performance")
    else:
        print(f"\n📊 Current R² = {best_nn_r2:.4f} - still far from target R² = {target_r2}")
        print("💭 The fundamental signal in this dataset might be limited")

    wandb.log({
        "target/target_r2": target_r2,
        "target/progress_ratio": best_nn_r2 / target_r2
    })

    # --- Save Results and Artifacts ---
    print(f"\n💾 Saving results...")
    
    # Save the best model
    best_model = NumericalPlusTitleNN(
        numerical_dim=X_num_train_scaled.shape[1], 
        title_dim=X_title_train.shape[1],
        hidden_dim=best_config.get('hidden_dim', 128),
        dropout=best_config.get('dropout', 0.1)
    )
    best_model.load_state_dict(best_result['model_state'])
    
    os.makedirs(cfg.ARTIFACTS_DIR, exist_ok=True)
    torch.save(best_model.state_dict(), cfg.MODEL_PATH)
    torch.save(best_model.state_dict(), os.path.join(run_dir, "best_model.pth"))
    
    # Save configuration
    model_params = {
        'architecture': 'numerical_plus_title',
        'numerical_dim': X_num_train_scaled.shape[1],
        'title_dim': X_title_train.shape[1],
        'total_samples': len(data["y"]),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx)
    }
    config_path = save_run_config(run_dir, best_config, model_params)
    print(f"Configuration saved: {config_path}")
    
    # Save encoders and scaler
    with open(cfg.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(run_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create prediction scatter plot
    best_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_num_test_scaled)
        X_title_test_tensor = torch.FloatTensor(X_title_test)
        test_preds = best_model(X_test_tensor, X_title_test_tensor).numpy()
    
    y_pred_orig = np.expm1(test_preds)
    y_true_orig = np.expm1(y_test)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_orig, y_pred_orig, alpha=0.1, s=5)
    plt.plot([y_true_orig.min(), y_true_orig.max()],
             [y_true_orig.min(), y_true_orig.max()],
             'r--', lw=2, label='Ideal Fit')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Actual Score'); plt.ylabel('Predicted Score')
    plt.title(f'Neural Network: Predicted vs. Actual (R²={best_nn_r2:.4f})')
    plt.grid(True); plt.legend()
    
    plot_path = os.path.join(run_dir, "predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Plot saved: {plot_path}")
    
    # Log the scatter plot to wandb
    wandb.log({"predictions/scatter_plot": wandb.Image(plt)})
    plt.close()
    
    # Create and log additional plots
    # Residuals plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - test_preds
    plt.scatter(test_preds, residuals, alpha=0.3, s=5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (log scale)')
    plt.ylabel('Residuals (log scale)')
    plt.title('Residuals Plot')
    plt.grid(True)
    wandb.log({"predictions/residuals_plot": wandb.Image(plt)})
    plt.close()
    
    # Log model architecture summary
    wandb.log({
        "model/total_parameters": sum(p.numel() for p in best_model.parameters()),
        "model/trainable_parameters": sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    })
    
    # Update runs summary
    final_metrics = {
        'test_r2': best_result['test_r2'],
        'test_loss': best_result['test_loss'],
        'best_val_r2': best_result['val_r2'],
        'epochs_trained': best_result['epochs_trained']
    }
    update_runs_summary(run_dir, best_config, final_metrics)
    
    print(f"✅ Artifacts saved to '{cfg.ARTIFACTS_DIR}' and '{run_dir}'")
    
    # --- Cleanup ---
    print("🧹 Cleaning up temporary embedding files...")
    temp_dir = "data/temp_embeddings"
    if hasattr(data["X_title_embeddings"], 'filename') and os.path.dirname(data["X_title_embeddings"].filename) == temp_dir:
        del data
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("✅ Cleanup complete.")
    else:
        print("⏩ No temporary files to clean up or path mismatch.")
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    train() 