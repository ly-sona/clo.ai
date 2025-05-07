#!/usr/bin/env python
# utils/io.py
"""Helpers for saving and loading XGBoost models."""

import os, time, joblib, xgboost as xgb
from pathlib import Path

def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def save_xgb(model: xgb.Booster, save_path: Path) -> Path:
    """
    Save an XGBoost Booster to the specified path.
    The model will be saved in both JSON and pickle formats.
    Returns the path to the saved model.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in both JSON and pickle formats
    json_path = save_path.with_suffix('.json')
    pkl_path = save_path.with_suffix('.pkl')
    
    model.save_model(str(json_path))
    joblib.dump(model, pkl_path)
    
    return json_path

def load_xgb(model_path: Path) -> xgb.Booster:
    """
    Load an XGBoost Booster from the specified path.
    Tries to load from JSON first, then falls back to pickle.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Try JSON first
    json_path = model_path.with_suffix('.json')
    if json_path.exists():
        model = xgb.Booster()
        model.load_model(str(json_path))
        return model
        
    # Fall back to pickle
    pkl_path = model_path.with_suffix('.pkl')
    if pkl_path.exists():
        return joblib.load(pkl_path)
        
    raise FileNotFoundError(f"No model file found at {model_path} (tried .json and .pkl)")