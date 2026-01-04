import os
import json
import shutil
import joblib
from datetime import datetime
import numpy as np

def save_training_results(model, feature_scaler, target_scaler, ticker, metrics, config, is_best=False):
    """
    Saves model/scaler to archive, updates manifest, and optionally promotes to production.
    
    Args:
        model: Trained Keras model
        scaler: Fitted scaler
        ticker: Stock symbol (e.g., "NVDA")
        metrics: Dictionary of results (e.g., {"loss": 0.05, "val_loss": 0.04})
        config: Dictionary of training configs used
        is_best: Boolean, if True, overwrites the 'production' model
    """
    # 1. Setup Directories 
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", ticker)
    archive_dir = os.path.join(base_dir, "archive")
    production_dir = os.path.join(base_dir, "production")
    
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(production_dir, exist_ok=True)
    
    # 2. Create Filenames (Timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_loss = metrics.get("loss", 0.0)
    filename_base = f"{ticker}_{timestamp}_loss-{final_loss:.4f}"
    
    model_path = os.path.join(archive_dir, f"{filename_base}.h5")
    feature_scaler_path = os.path.join(archive_dir, f"{filename_base}_feature_scaler.gz")
    target_scaler_path = os.path.join(archive_dir, f"{filename_base}_target_scaler.gz")
    
    # 3. Save to Archive
    print(f"Archiving model: {filename_base} ....")
    model.save(model_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    
    # 4. Update Manifest
    manifest_path = os.path.join(base_dir, "manifest.json")
    
    # Create a log entry
    entry = {
        "timestamp": timestamp,
        "model_filename": f"{filename_base}.h5",
        "feature_scaler_filename": f"{filename_base}_feature_scaler.gz",
        "target_scaler_filename": f"{filename_base}_target_scaler.gz",
        "metrics": metrics,
        "config": config,
        "promoted_to_prod": is_best
    }
    
    # Load existing manifest or start new
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                manifest = []
    else:
        manifest = []
        
    manifest.append(entry)
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    # Promote to prod
    if is_best:
        print(f".... Promoting {ticker} model to Production ....")
        # We copy the files to generic names 'models.h5', 'feature_scaler.gz' and 'target_scaler.gz'
        shutil.copy(model_path, os.path.join(production_dir, "model.h5"))
        shutil.copy(feature_scaler_path, os.path.join(production_dir, "feature_scaler.gz"))
        shutil.copy(target_scaler_path, os.path.join(production_dir, "target_scaler.gz"))
    
    