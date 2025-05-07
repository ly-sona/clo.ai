# power_model.py
"""Train an XGBoost regressor on CircuitNet power maps, optimise one weight
   via DEAP, save the booster on Drive, and display a sample heat‑map."""

from pathlib import Path
import numpy as np, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.io         import save_xgb
from utils.visual     import plot_heatmap
from utils.deap_helpers import deap_minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths for CircuitNet dataset
CIRCUITNET_DIR = Path("/content/drive/MyDrive/1Xp2y29Le6Doo3meKhTZClVwxG_7z2QuF")
POWER_DIR = CIRCUITNET_DIR / "IR_drop_features"  # Updated to match CircuitNet structure
SAVE_NAME = "power_xgb"
SAVE_DIR = Path("/content/drive/MyDrive/circuitnet_artifacts")

class PowerModel:
    def __init__(self, data_dir='circuitnet_data', sample_limit=1_000_000):
        """Initialize the PowerModel with data directory and sample limit."""
        self.data_dir = data_dir
        self.sample_limit = sample_limit
        self.model = None
        self.evals_result = None

    def load_power_data(self, directory):
        """Loads power data from .npy files in a directory."""
        data = {}
        print(f"[Checkpoint] Loading power data files from {directory}...")
        for filename in tqdm(os.listdir(directory), desc="Loading files"):
            if filename.endswith(".npy"):
                filepath = os.path.join(directory, filename)
                data[filename] = np.load(filepath)
        return data

    def preprocess_data(self, power_data):
        """Preprocesses the power data for machine learning."""
        if not power_data:
            print("Error: No power data available for preprocessing.")
            return None, None

        print("[Checkpoint] Starting data preprocessing...")
        X_list = []
        y_list = []
        
        # First pass: count total samples and determine sampling rate
        print("[Checkpoint] Counting total samples...")
        total_samples = sum(power_array.size for power_array in power_data.values())
        
        # Calculate sampling rate if needed
        if total_samples > self.sample_limit:
            sampling_rate = self.sample_limit / total_samples
            print(f"[Checkpoint] Will sample {sampling_rate:.2%} of data")
        else:
            sampling_rate = 1.0
            print("[Checkpoint] No downsampling needed")
        
        # Process data
        print("[Checkpoint] Processing and sampling data...")
        for power_array in tqdm(power_data.values(), desc="Processing files"):
            # Flatten and sample if needed
            if sampling_rate < 1.0:
                # Sample indices
                n_samples = int(power_array.size * sampling_rate)
                indices = np.random.choice(power_array.size, n_samples, replace=False)
                # Get values at sampled indices
                values = power_array.flatten()[indices]
            else:
                values = power_array.flatten()
            
            # Reshape for features (matching provided code)
            X = values.reshape(-1, 1).astype(np.float32)
            X_list.append(X)
            y_list.append(values)
            
            # Print progress every 1000 files
            if len(X_list) % 1000 == 0:
                current_samples = sum(x.shape[0] for x in X_list)
                print(f"[Progress] Processed {len(X_list)} files, {current_samples:,} samples so far")

        print("[Checkpoint] Concatenating sampled data...")
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        
        print(f"[Checkpoint] Final dataset shape: X={X_all.shape}, y={y_all.shape}")
        return X_all, y_all

    def train_model(self, X_train, y_train, X_val, y_val, callbacks=None, max_depth=3, learning_rate=0.1):
        """Trains an XGBoost model on the given data."""
        print("Training XGBoost model...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'seed': 42,
            'verbosity': 0,  # Keep verbosity low during training
            'max_depth': max_depth,
            'learning_rate': learning_rate,
        }
        evals_result = {}  # Store results for early stopping
        model = xgb.train(params, dtrain,
                         num_boost_round=1000,
                         evals=[(dtrain, 'train'), (dval, 'val')],
                         early_stopping_rounds=50,
                         evals_result=evals_result,
                         verbose_eval=50)
        print(f"Best iteration: {model.best_iteration}")
        print(f"Best RMSE: {evals_result['val']['rmse'][model.best_iteration]:.4f}")
        print(f"Best MAE: {evals_result['val']['mae'][model.best_iteration]:.4f}")
        self.model = model
        self.evals_result = evals_result
        return model, evals_result

    def evaluate_model(self, X_test, y_test):
        """Evaluates the trained XGBoost model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        print("Evaluating model...")
        dtest = xgb.DMatrix(X_test, label=y_test)
        y_pred = self.model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R-squared: {r2:.4f}")
        return {'rmse': rmse, 'r2': r2}

    def visualize_power_data(self, data, title="Power Data Visualization"):
        """Visualizes power data using a heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
        plt.title(title)
        plt.show()

    def predict_power(self, X):
        """Predicts power consumption for given input data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

    def train_and_evaluate(self, power_data_dir, test_size=0.2, val_size=0.25):
        """Complete pipeline for training and evaluating the power model."""
        # 1. Load and preprocess data
        power_data = self.load_power_data(power_data_dir)
        X, y = self.preprocess_data(power_data)
        
        if X is None or y is None:
            raise ValueError("Failed to preprocess power data")

        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42)

        # 3. Train model
        self.train_model(X_train, y_train, X_val, y_val)

        # 4. Evaluate model
        metrics = self.evaluate_model(X_test, y_test)

        return metrics

    def deap_fitness_function(self, individual, X, y):
        """Fitness function for DEAP."""
        weighted_X = X * individual[0]  # Apply the single weight
        dtrain = xgb.DMatrix(weighted_X, label=y)
        y_pred = self.model.predict(dtrain)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return (rmse,)

    def optimize_feature_weights(self, X, y, num_generations=10, pop_size=50):
        """Optimizes feature weights using DEAP."""
        print("Optimizing feature weights with DEAP...")
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attribute", np.random.rand)  # Initialize with random weight
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=1)  # Only one weight
        toolbox.register("population", tools.initRepeat, list,
                        toolbox.individual)

        toolbox.register("evaluate", self.deap_fitness_function, X=X, y=y)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=0.2, indpb=1.0) #mean=1, std=0.2
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        population, logbook = algorithms.eaMuPlusLambda(population, toolbox,
                                                    mu=pop_size,
                                                    lambda_=pop_size,
                                                    cxpb=0.5,
                                                    mutpb=0.2,
                                                    ngen=num_generations,
                                                    stats=None,
                                                    halloffame=hof,
                                                    verbose=True)

        print("DEAP optimization complete.")
        return hof[0]

    def apply_optimized_weights(self, X, y, optimized_weights):
        """Applies the optimized feature weights to the input data and evaluates the model."""
        print("Applying optimized weights and evaluating...")
        weighted_X = X * optimized_weights[0]
        dtest = xgb.DMatrix(weighted_X, label=y)
        y_pred = self.model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        print(f"RMSE with optimized weights: {rmse:.4f}")
        print(f"R-squared with optimized weights: {r2:.4f}")
        return {'rmse': rmse, 'r2': r2}

# ──────────────────────────────────────────────────────────────#
def main():
    # Load and visualize a sample power map
    power = PowerModel().load_power_data(POWER_DIR)
    first_name, first_arr = next(iter(power.items()))
    PowerModel().visualize_power_data(first_arr, title=f"Sample map – {first_name}")

    # Preprocess and split data
    X, y = PowerModel().preprocess_data(power)
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr, Xval,  ytr, yval  = train_test_split(Xtr, ytr, test_size=0.25, random_state=42)

    # Train model
    model, evals_result = PowerModel().train_model(Xtr, ytr, Xval, yval)
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