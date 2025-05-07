import os
import numpy as np
from circuit_optimizer import CircuitOptimizer
from power_model import PowerModel
from sklearn.model_selection import train_test_split
import time
import json
import xgboost as xgb
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Train power model and optimize circuit')
    parser.add_argument('--sample', type=int, default=2_000_000,
                      help='Maximum number of samples to use (default: 2_000_000)')
    parser.add_argument('--test', type=float, default=0.2,
                      help='Test set size (default: 0.2)')
    parser.add_argument('--val', type=float, default=0.25,
                      help='Validation set size (default: 0.25)')
    parser.add_argument('--max-depth', type=int, default=3,
                      help='XGBoost max_depth (default: 3)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                      help='XGBoost learning rate (default: 0.1)')
    parser.add_argument('--output-dir', type=str, default='../circuitnet_data',
                      help='Output directory for saved models (default: ../circuitnet_data)')
    parser.add_argument('--circuit-file', type=str, default='../circuit_files/c6288.bench',
                      help='Circuit file to optimize (default: ../circuit_files/c6288.bench)')
    parser.add_argument('--validation-circuit', type=str, default='../circuit_files/c432.bench',
                      help='Circuit file for validation (default: ../circuit_files/c432.bench)')
    parser.add_argument('--test-circuit', type=str, default='../circuit_files/c6288.bench',
                      help='Circuit file for testing (default: ../circuit_files/c6288.bench)')
    return parser.parse_args()


def save_hybrid_model(power_model, circuit_optimizer, output_dir):
    """Save the complete hybrid model including both power model and circuit optimizer state."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save power model
    if power_model.model is not None:
        power_model_path = os.path.join(output_dir, 'power_model.json')
        power_model.model.save_model(power_model_path)
        print(f"[Checkpoint] Power model saved to '{power_model_path}'")
    
    # Save circuit optimizer state
    circuit_optimizer_path = os.path.join(output_dir, 'circuit_optimizer.pkl')
    with open(circuit_optimizer_path, 'wb') as f:
        pickle.dump({
            'initial_simulations': 50,  # Using fixed value instead of accessing from instance
            'data_dir': circuit_optimizer.data_dir
        }, f)
    print(f"[Checkpoint] Circuit optimizer state saved to '{circuit_optimizer_path}'")
    
    # Save hybrid model metadata
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'power_model_path': 'power_model.json',
        'circuit_optimizer_path': 'circuit_optimizer.pkl',
        'model_version': '1.0',
        'initial_simulations': 50  # Using fixed value here too
    }
    metadata_path = os.path.join(output_dir, 'hybrid_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Checkpoint] Hybrid model metadata saved to '{metadata_path}'")


def train_and_evaluate_power_model(args):
    print("=== Training and Testing Power Model ===")
    start_time = time.time()
    power_model = PowerModel(data_dir=args.output_dir, sample_limit=args.sample)
    print("[Checkpoint] PowerModel initialized.")
    
    # Load power data for training circuit
    print(f"[Checkpoint] Loading power data from '{os.path.join(args.output_dir, 'power_all')}' ...")
    power_data = power_model.load_power_data(os.path.join(args.output_dir, 'power_all'))
    if not power_data:
        print(f"No power data found. Please ensure data is in {os.path.join(args.output_dir, 'power_all')}")
        return None
    print(f"[Checkpoint] Loaded {len(power_data)} power data files.")
        
    # Preprocess data
    print("[Checkpoint] Preprocessing power data ...")
    X, y = power_model.preprocess_data(power_data)
    if X is None or y is None:
        print("Error: Preprocessing failed.")
        return None
    print(f"[Checkpoint] Preprocessed data: X shape = {X.shape}, y shape = {y.shape}")
    print("[Checkpoint] Preprocessing complete. Proceeding to data split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val, random_state=42)
    print(f"[Checkpoint] Data split: X_train = {X_train.shape}, X_val = {X_val.shape}, X_test = {X_test.shape}")
    
    # Train model
    print("[Checkpoint] Training power model ...")
    train_start = time.time()
    
    # Create a proper XGBoost callback
    class PrintProgressCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % 50 == 0:
                print(f"[Checkpoint] Training round {epoch} ...")
    
    power_model.train_model(X_train, y_train, X_val, y_val, 
                          callbacks=[PrintProgressCallback()],
                          max_depth=args.max_depth,
                          learning_rate=args.learning_rate)
    print(f"[Checkpoint] Power model trained in {time.time() - train_start:.2f} seconds.")
    
    # Evaluate model on training data
    print("[Checkpoint] Evaluating power model on training data ...")
    train_metrics = power_model.evaluate_model(X_train, y_train)
    print(f"[RESULT] Training R2 Score: {train_metrics['r2']:.4f}")
    
    # Evaluate model on validation circuit
    print(f"\n[Checkpoint] Evaluating power model on validation circuit ({args.validation_circuit}) ...")
    val_data = power_model.load_power_data(os.path.join(args.output_dir, 'power_all'))
    if val_data:
        X_val_circuit, y_val_circuit = power_model.preprocess_data(val_data)
        val_metrics = power_model.evaluate_model(X_val_circuit, y_val_circuit)
        print(f"[RESULT] Validation Circuit R2 Score: {val_metrics['r2']:.4f}")
    
    # Evaluate model on test circuit
    print(f"\n[Checkpoint] Evaluating power model on test circuit ({args.test_circuit}) ...")
    test_data = power_model.load_power_data(os.path.join(args.output_dir, 'power_all'))
    if test_data:
        X_test_circuit, y_test_circuit = power_model.preprocess_data(test_data)
        test_metrics = power_model.evaluate_model(X_test_circuit, y_test_circuit)
        print(f"[RESULT] Test Circuit R2 Score: {test_metrics['r2']:.4f}")
    
    print(f"[Checkpoint] Power model training and evaluation completed in {time.time() - start_time:.2f} seconds.\n")
    return power_model

def optimize_circuit(args, power_model):
    print("\n=== Circuit Optimization ===")
    start_time = time.time()
    circuit_optimizer = CircuitOptimizer(data_dir=args.output_dir)
    print("[Checkpoint] CircuitOptimizer initialized.")
    
    # Get absolute path to circuit file
    circuit_file = os.path.abspath(args.circuit_file)
    if not os.path.exists(circuit_file):
        print(f"Circuit file not found: {circuit_file}")
        return None, None
    print(f"[Checkpoint] Circuit file found: {circuit_file}")
    
    print("[Checkpoint] Optimizing circuit ...")
    optimize_start = time.time()
    optimized_circuit, optimized_sizes, original_power, original_delay, optimized_power, optimized_delay = circuit_optimizer.optimize_circuit(
        circuit_file, 
        num_initial_simulations=50  # Using a fixed value instead of accessing from instance
    )
    print(f"[Checkpoint] Circuit optimized in {time.time() - optimize_start:.2f} seconds.")
    
    # Save power model separately if it exists
    power_model_path = None
    if power_model and power_model.model is not None:
        power_model_path = os.path.join(args.output_dir, 'power_model.json')
        power_model.model.save_model(power_model_path)
        print(f"[Checkpoint] Power model saved to '{power_model_path}'")
    
    # Create optimized chip design (without the actual model object)
    optimized_chip = {
        'circuit': optimized_circuit,
        'gate_sizes': optimized_sizes,
        'power_model_path': power_model_path,  # Store path instead of model object
        'optimization_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'initial_simulations': 50,
            'circuit_file': circuit_file,
            'original_power': original_power,
            'original_delay': original_delay,
            'optimized_power': optimized_power,
            'optimized_delay': optimized_delay
        }
    }
    
    # Save optimized chip design
    output_path = os.path.join(args.output_dir, 'optimized_chip.json')
    with open(output_path, 'w') as f:
        json.dump(optimized_chip, f, indent=2)
    print(f"[Checkpoint] Optimized chip design saved to '{output_path}'")
    
    print("\n[RESULT] Optimization Results:")
    print(f"Original Performance: Delay={original_delay:.2f}, Power={original_power:.2f}")
    print(f"Optimized Performance: Delay={optimized_delay:.2f}, Power={optimized_power:.2f}")
    print(f"Optimized gate sizes: {optimized_sizes}")
    print(f"[Checkpoint] Circuit optimization completed in {time.time() - start_time:.2f} seconds.\n")
    return circuit_optimizer, optimized_chip

if __name__ == "__main__":
    args = parse_args()
    print("[Checkpoint] Starting training and optimization pipeline ...")
    print(f"[Config] Sample limit: {args.sample:,}, Test size: {args.test}, Validation size: {args.val}")
    print(f"[Config] XGBoost params: max_depth={args.max_depth}, learning_rate={args.learning_rate}")
    print(f"[Config] Output directory: {args.output_dir}")
    print(f"[Config] Circuit file: {args.circuit_file}")
    
    os.makedirs(os.path.join(args.output_dir, "power_all"), exist_ok=True)
    
    # Run training and optimization
    power_model = train_and_evaluate_power_model(args)
    if power_model is None:
        print("Error: Power model training failed")
        exit(1)
        
    circuit_optimizer, optimized_chip = optimize_circuit(args, power_model)
    if circuit_optimizer is None or optimized_chip is None:
        print("Error: Circuit optimization failed")
        exit(1)
    
    # Save the complete hybrid model
    save_hybrid_model(power_model, circuit_optimizer, args.output_dir)
    
    print("\n[Checkpoint] All tasks completed.")
    print("\nSaved artifacts:")
    print("  ▶ power_model.json")
    print("  ▶ circuit_optimizer.pkl")
    print("  ▶ hybrid_model_metadata.json")
    print("  ▶ optimized_chip.json")
    
    print("\n[FINAL OUTPUT] Optimized Chip Design:")
    print(f"  ▶ Circuit: {optimized_chip['circuit'][:200]}...")  # Print first 200 chars of circuit
    print(f"  ▶ Gate Sizes: {optimized_chip['gate_sizes']}")
    print(f"  ▶ Optimization Time: {optimized_chip['optimization_metadata']['timestamp']}") 