#!/usr/bin/env python
# circuit_optimize.py
"""
Parse an ISCAS‑85 netlist, train an XGBoost delay regressor from the CircuitNet dataset,
optimise a single global gate‑size weight via DEAP, save the booster + schematic to a specified location.
"""

from pathlib import Path
import re, argparse, pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import json
from typing import Tuple, Dict, List, Optional, Union
import os

from utils.io           import save_xgb, load_xgb
from utils.visual       import draw_gate_graph
from utils.deap_helpers import deap_minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths for CircuitNet dataset (matching Colab notebook)
CIRCUITNET_DIR = Path("/content/drive/MyDrive/circuitnet_data")
DELAY_DATASET = CIRCUITNET_DIR / "delay_dataset.csv"
NETLIST = CIRCUITNET_DIR / "c17.bench"
SAVE_NAME = "delay_xgb"
SAVE_DIR = Path("/content/drive/MyDrive/circuitnet_artifacts")

class CircuitOptimizer:
    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the circuit optimizer with optional model parameters.
        
        Args:
            model_params: Dictionary of XGBoost parameters. If None, uses defaults.
        """
        self.model_params = model_params or {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "learning_rate": 0.1
        }
        self.model = None
        self.best_weight = None

    def load_dataset(self, data_path: Union[str, Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from the CircuitNet delay dataset.
        
        Args:
            data_path: Optional path to dataset file. If None, uses default CircuitNet path.
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        if data_path is None:
            data_path = DELAY_DATASET
            
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
            
        # Load the delay dataset CSV
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        
        logger.info(f"Loaded {len(X)} samples from CircuitNet delay dataset")
        return X, y

    def parse_bench(self, bench_path: Union[str, Path] = None) -> Tuple[List[str], List[str], List[str], Dict, Dict]:
        """Parse ISCAS-85 bench format netlist."""
        if bench_path is None:
            bench_path = NETLIST
            
        bench_path = Path(bench_path)
        if not bench_path.exists():
            raise FileNotFoundError(f"Bench file not found: {bench_path}")
            
        bench_text = bench_path.read_text()
        ins, outs, gates, conns, types = [], [], [], {}, {}
        
        for ln in bench_text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if ln.startswith("INPUT"):
                ins.append(ln.split("(")[1].rstrip(")"))
            elif ln.startswith("OUTPUT"):
                outs.append(ln.split("(")[1].rstrip(")"))
            else:
                m = re.match(r"(\w+)\s*=\s*(\w+)\(([^)]+)\)", ln)
                if m:
                    g, typ, ins_str = m.groups()
                    gates.append(g)
                    types[g] = typ.lower()
                    conns[g] = [p.strip() for p in ins_str.split(",")]
        return ins, outs, gates, conns, types

    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, val_size: float = 0.25,
              n_estimators: int = 500, early_stopping_rounds: int = 50) -> float:
        """Train the XGBoost model and return test RMSE."""
        Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=test_size, random_state=0)
        Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=val_size, random_state=0)

        dtr, dval = xgb.DMatrix(Xtr, label=ytr), xgb.DMatrix(Xval, label=yval)
        self.model = xgb.train(
            self.model_params,
            dtr, n_estimators,
            [(dtr, "tr"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        rmse = np.sqrt(mean_squared_error(ytest, self.model.predict(xgb.DMatrix(Xtest))))
        logger.info(f"Baseline RMSE={rmse:.4f}")
        return rmse

    def optimize_weights(self, X: np.ndarray, y: np.ndarray, n_weights: int = 1, ngen: int = 12) -> float:
        """Optimize gate weights using DEAP."""
        if self.model is None:
            raise ValueError("Model must be trained before optimizing weights")
            
        Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
        
        def fitness(indiv):
            pred = self.model.predict(xgb.DMatrix(Xtest * indiv[0]))
            return (np.sqrt(mean_squared_error(ytest, pred)),)
            
        self.best_weight = deap_minimize(fitness, n_weights=n_weights, ngen=ngen)
        logger.info(f"Best weight: {self.best_weight}")
        return self.best_weight[0]

    def save_model(self, save_path: Path = None, include_weights: bool = True):
        """Save the trained model and optionally the optimized weights."""
        if self.model is None:
            raise ValueError("No model to save")
            
        if save_path is None:
            save_path = SAVE_DIR / SAVE_NAME
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_xgb(self.model, save_path)
        
        if include_weights and self.best_weight is not None:
            weights_path = save_path.with_suffix('.weights.json')
            with open(weights_path, 'w') as f:
                json.dump({'best_weight': self.best_weight[0]}, f)
                
        logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: Path) -> None:
        """Load a saved model and its weights if available."""
        self.model = load_xgb(model_path)
        
        weights_path = model_path.with_suffix('.weights.json')
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                weights = json.load(f)
                self.best_weight = [weights['best_weight']]
                
        logger.info(f"Model loaded from {model_path}")

    def generate_schematic(self, bench_path: Path, output_path: Path = None) -> None:
        """Generate a schematic with the optimized weights applied."""
        if self.best_weight is None:
            raise ValueError("No optimized weights available")
            
        if output_path is None:
            output_path = SAVE_DIR / "gate_graph.png"
            
        bench_text = bench_path.read_text()
        _, _, gates, conns, types = self.parse_bench(bench_text)
        sizes = {g: self.best_weight[0] for g in gates}
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        draw_gate_graph(conns, types, sizes, out=output_path)
        logger.info(f"Schematic saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and optimize circuit models")
    parser.add_argument("--data", type=Path, default=DELAY_DATASET, help="Path to CircuitNet delay dataset")
    parser.add_argument("--bench", type=Path, default=NETLIST, help="Path to .bench file")
    parser.add_argument("--output", type=Path, default=SAVE_DIR / SAVE_NAME, help="Path to save model")
    parser.add_argument("--schematic", type=Path, default=SAVE_DIR / "gate_graph.png", help="Path to save schematic")
    args = parser.parse_args()

    optimizer = CircuitOptimizer()
    
    try:
        # Load and train
        X, y = optimizer.load_dataset(args.data)
        optimizer.train(X, y)
        
        # Optimize weights
        optimizer.optimize_weights(X, y)
        
        # Save model
        optimizer.save_model(args.output)
        
        # Generate schematic
        optimizer.generate_schematic(args.bench, args.schematic)
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main()