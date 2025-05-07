#!/usr/bin/env python
# power_model.py
"""Train an XGBoost regressor on CircuitNet power maps, optimise one weight
   via DEAP, save the booster on Drive, and display a sample heat‑map."""

from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

from utils.io         import save_xgb
from utils.visual     import plot_heatmap
from utils.deap_helpers import deap_minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths for CircuitNet dataset (matching Colab notebook)
CIRCUITNET_DIR = Path("/content/circuitnet_data")
POWER_DIR = CIRCUITNET_DIR / "power_all"
SAVE_NAME = "power_xgb"
SAVE_DIR = Path("/content/drive/MyDrive/circuitnet_artifacts")  # Save to Drive for persistence

# ──────────────────────────────────────────────────────────────#
def load_power() -> dict[str, np.ndarray]:
    """Load power maps from the CircuitNet dataset."""
    files = list(POWER_DIR.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy files found in {POWER_DIR}")
    logger.info(f"Loaded {len(files)} power maps from {POWER_DIR}")
    return {p.name: np.load(p) for p in files}

def preprocess(pw: dict[str, np.ndarray]):
    """Preprocess power maps into features and labels."""
    X, y = [], []
    for arr in pw.values():
        X.append(arr.reshape(-1, 1).astype(np.float32))
        y.append(arr.flatten())
    return np.vstack(X), np.hstack(y)

def train_xgb(Xtr, ytr, Xval, yval) -> xgb.Booster:
    """Train an XGBoost model on the power data."""
    dtr, dval = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xval, label=yval)
    params = {"objective": "reg:squarederror",
              "eval_metric": ["rmse"],
              "max_depth": 3, "learning_rate": 0.1, "seed": 42}
    return xgb.train(params, dtr, 1000,
                     [(dtr, "train"), (dval, "val")],
                     early_stopping_rounds=50, verbose_eval=50)

# ──────────────────────────────────────────────────────────────#
def main():
    # Load and visualize a sample power map
    power = load_power()
    first_name, first_arr = next(iter(power.items()))
    plot_heatmap(first_arr, title=f"Sample map – {first_name}")

    # Preprocess and split data
    X, y = preprocess(power)
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr, Xval,  ytr, yval  = train_test_split(Xtr, ytr, test_size=0.25, random_state=42)

    # Train model
    model = train_xgb(Xtr, ytr, Xval, yval)
    rmse  = np.sqrt(mean_squared_error(ytest, model.predict(xgb.DMatrix(Xtest))))
    r2    = r2_score(ytest, model.predict(xgb.DMatrix(Xtest)))
    logger.info(f"Test RMSE={rmse:.4f}  R²={r2:.4f}")

    # Optimize weights using DEAP
    def fitness(indiv):  # minimise RMSE
        pred = model.predict(xgb.DMatrix(Xtest * indiv[0]))
        return (np.sqrt(mean_squared_error(ytest, pred)),)

    w_best = deap_minimize(fitness, n_weights=1, ngen=10)
    logger.info(f"Best weight: {w_best}")

    # Save model
    save_path = SAVE_DIR / SAVE_NAME
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_xgb(model, save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()