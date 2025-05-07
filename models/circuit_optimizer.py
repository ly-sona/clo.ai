import os
import numpy as np
import re
from deap import base, creator, tools, algorithms
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict, deque

class CircuitOptimizer:
    def __init__(self, data_dir='circuitnet_data'):
        """Initialize the CircuitOptimizer with data directory."""
        self.data_dir = data_dir
        self.model = None
        self.evals_result = None
        
    def load_iscas85(self, file_path):
        """Loads an ISCAS85 bench file."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading ISCAS85 file: {e}")
            return None

    def parse_iscas85(self, iscas85_content):
        """Parses an ISCAS85 bench file."""
        inputs = []
        outputs = []
        gates = []
        connections = {}
        gate_types = {}

        lines = iscas85_content.strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('INPUT'):
                inputs.append(line.split('(')[1].rstrip(')'))
            elif line.startswith('OUTPUT'):
                outputs.append(line.split('(')[1].rstrip(')'))
            else:
                match = re.match(r"(\w+)\s*=\s*(\w+)\(([^)]+)\)", line)
                if match:
                    gate_name, gate_type, input_str = match.groups()
                    gate_type = gate_type.lower()
                    inputs_to_gate = [p.strip() for p in input_str.split(',')]

                    gates.append(gate_name)
                    gate_types[gate_name] = gate_type
                    connections[gate_name] = inputs_to_gate

        return inputs, outputs, gates, connections, gate_types

    def get_gate_sizes(self, gate_types, gates):
        """Gets the gate sizes."""
        gate_sizes = {}
        for gate in gates:
            if gate_types[gate] in ['nand', 'nor', 'and', 'or', 'inv', 'buf']:
                gate_sizes[gate] = 1.0
            else:
                gate_sizes[gate] = 1.0
        return gate_sizes

    def extract_features(self, inputs, outputs, gates, connections, gate_types, gate_sizes):
        """Extracts features from the ISCAS85 circuit description."""
        num_gates = len(gates)
        total_fanin = sum(len(conn) for conn in connections.values())
        avg_fanin = total_fanin / num_gates if num_gates else 0
        max_fanin = max(len(conn) for conn in connections.values()) if connections else 0

        X = np.array([num_gates, avg_fanin, max_fanin]).reshape(1, -1).astype(np.float32)
        y = np.array([0.0])
        return X, y

    def simulate_circuit(self, iscas85_content, gate_sizes):
        """Simulates the circuit to get performance metrics."""
        delay = 0
        lines = iscas85_content.strip().splitlines()
        for line in lines:
            if line.startswith(('INPUT', 'OUTPUT', '#')):
                continue
            parts = line.split()
            if len(parts) > 2:
                gate_name = parts[0]
                if gate_name in gate_sizes:
                    delay += gate_sizes[gate_name]
                else:
                    delay += 1
        power = 0.1 * len(lines)
        return delay + np.random.rand(), power + np.random.rand()

    def train_xgboost_model(self, X_train, y_train, X_val, y_val):
        """Trains an XGBoost model."""
        print("Training XGBoost model...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'seed': 42,
            'verbosity': 0,
            'max_depth': 3,
            'learning_rate': 0.1,
        }
        evals_result = {}
        model = xgb.train(params, dtrain,
                         num_boost_round=1000,
                         evals=[(dtrain, 'train'), (dval, 'val')],
                         early_stopping_rounds=50,
                         evals_result=evals_result,
                         verbose_eval=50)
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

    def deap_fitness_function(self, individual, X, y):
        """Fitness function for DEAP."""
        weighted_X = X * individual[0]
        dtrain = xgb.DMatrix(weighted_X, label=y)
        y_pred = self.model.predict(dtrain)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return (rmse,)

    def optimize_circuit_layout(self, X, y, num_generations=10, pop_size=50):
        """Optimizes circuit layout parameters using DEAP."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        print("Optimizing circuit layout with DEAP...")
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attribute", np.random.rand)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=1)
        toolbox.register("population", tools.initRepeat, list,
                        toolbox.individual)

        toolbox.register("evaluate", self.deap_fitness_function, X=X, y=y)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=1, sigma=0.2, indpb=1.0)
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

    def generate_optimized_iscas85(self, original_iscas85, optimized_gate_sizes):
        """Generates the optimized ISCAS85 netlist based on the parameters."""
        optimized_iscas85 = original_iscas85
        lines = optimized_iscas85.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith(('INPUT', 'OUTPUT', '#', ' ')):
                gate_name_match = re.match(r"(\w+)\s*=", line)
                if gate_name_match:
                    gate_name = gate_name_match.group(1)
                    if gate_name in optimized_gate_sizes:
                        lines[i] = f"{line}  # size={optimized_gate_sizes[gate_name]:.2f}"
        optimized_iscas85 = "\n".join(lines)
        return optimized_iscas85

    def optimize_circuit(self, file_path, num_initial_simulations=5):
        """Main function to optimize a circuit."""
        # 1. Load and parse ISCAS85 file
        iscas85_content = self.load_iscas85(file_path)
        if iscas85_content is None:
            raise ValueError("Failed to load ISCAS85 file")

        inputs, outputs, gates, connections, gate_types = self.parse_iscas85(iscas85_content)
        all_nodes = inputs + gates + outputs
        coords_dict = self.generate_gate_coordinates(connections, all_nodes, inputs)
        layout_xy = np.array([coords_dict.get(n, (0, 0)) for n in all_nodes])
        initial_gate_sizes = self.get_gate_sizes(gate_types, gates)

        # 2. Initial simulation
        original_delay, original_power = self.simulate_circuit(iscas85_content, initial_gate_sizes)
        print(f"Initial Circuit Performance: Delay={original_delay:.2f}, Power={original_power:.2f}")

        # 3. Prepare data for XGBoost
        simulation_results_delay = []
        X_list = []
        for _ in range(num_initial_simulations):
            gate_sizes = {gate: np.random.uniform(0.5, 2.0) for gate in gates}
            delay, _ = self.simulate_circuit(iscas85_content, gate_sizes)
            simulation_results_delay.append(delay)
            X, _ = self.extract_features(inputs, outputs, gates, connections, gate_types, gate_sizes)
            X_list.append(X)

        y_train_delay = np.array(simulation_results_delay).astype(np.float32)
        X = np.concatenate(X_list, axis=0)

        # 4. Split data and train model
        X_train, X_test, y_train_delay, y_test_delay = train_test_split(
            X, y_train_delay, test_size=0.2, random_state=42)
        X_train, X_val, y_train_delay, y_val_delay = train_test_split(
            X_train, y_train_delay, test_size=0.25, random_state=42)

        self.train_xgboost_model(X_train, y_train_delay, X_val, y_val_delay)
        evaluation_metrics = self.evaluate_model(X_test, y_test_delay)
        print("Evaluation Metrics:", evaluation_metrics)

        # 5. Optimize with DEAP
        optimized_weights = self.optimize_circuit_layout(X_test, y_test_delay)
        optimized_gate_sizes = {gate: optimized_weights[0] for gate in gates}

        # 6. Generate and simulate optimized circuit
        optimized_iscas85_content = self.generate_optimized_iscas85(
            iscas85_content, optimized_gate_sizes)
        optimized_delay, optimized_power = self.simulate_circuit(
            optimized_iscas85_content, optimized_gate_sizes)
        print(f"Optimized Circuit Performance: Delay={optimized_delay:.2f}, Power={optimized_power:.2f}")

        return optimized_iscas85_content, optimized_gate_sizes, original_power, original_delay, optimized_power, optimized_delay 

    def generate_gate_coordinates(self, connections, gates, inputs):
        """Assigns (x, y) coordinates to each gate based on logic level (distance from inputs)."""
        # Build a graph: key = gate, value = list of gates it drives
        graph = defaultdict(list)
        indegree = {g: 0 for g in gates}
        for gate, fanins in connections.items():
            for src in fanins:
                if src in gates:
                    graph[src].append(gate)
                    indegree[gate] += 1
        # Gates with no fanins (primary inputs)
        level = {}
        queue = deque()
        for gate in gates:
            if indegree[gate] == 0:
                level[gate] = 0
                queue.append(gate)
        # BFS to assign levels
        while queue:
            g = queue.popleft()
            for succ in graph[g]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    level[succ] = level[g] + 1
                    queue.append(succ)
        # Group gates by level
        level_to_gates = defaultdict(list)
        for gate, lvl in level.items():
            level_to_gates[lvl].append(gate)
        # Assign coordinates
        coords = {}
        for lvl, gates_in_lvl in level_to_gates.items():
            for i, gate in enumerate(gates_in_lvl):
                coords[gate] = (i, lvl)
        return coords 