import os
from circuit_optimizer import CircuitOptimizer
from power_model import PowerModel

def main():
    # Initialize models
    circuit_optimizer = CircuitOptimizer(data_dir='circuitnet_data')
    power_model = PowerModel(data_dir='circuitnet_data')

    # Example 1: Optimize a circuit
    print("\n=== Circuit Optimization Example ===")
    circuit_file = 'circuitnet_data/c17.bench'  # Replace with your circuit file
    if os.path.exists(circuit_file):
        optimized_circuit, optimized_gate_sizes = circuit_optimizer.optimize_circuit(
            circuit_file, num_initial_simulations=5)
        print("\nOptimized Circuit:")
        print(optimized_circuit)
        print("\nOptimized Gate Sizes:")
        print(optimized_gate_sizes)

    # Example 2: Train and evaluate power model
    print("\n=== Power Model Example ===")
    power_data_dir = 'circuitnet_data/power_all'  # Replace with your power data directory
    if os.path.exists(power_data_dir):
        metrics = power_model.train_and_evaluate(power_data_dir)
        print("\nPower Model Metrics:")
        print(metrics)

        # Example of predicting power for new data
        if power_model.model is not None:
            # Load a sample power map for visualization
            power_data = power_model.load_power_data(power_data_dir)
            if power_data:
                first_name, first_arr = next(iter(power_data.items()))
                power_model.visualize_power_data(first_arr, title=f"Sample Power Map - {first_name}")

if __name__ == "__main__":
    main() 