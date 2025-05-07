from deap import base, creator, tools, algorithms
import numpy as np
import os
import re
import networkx as nx

# Guard against repeat-run errors
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

class CircuitOptimizer:
    def __init__(self, data_dir='circuitnet_data'):
        self.data_dir = data_dir
        self.initial_simulations = 50  # Increased from 10 to 50 for better surrogate model

    def load_iscas85(self, file_path):
        with open(file_path, 'r') as f:
            return f.read()

    def parse_iscas85(self, iscas85_content):
        gates = {}
        for line in iscas85_content.split('\n'):
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                gate_name = parts[0]
                gate_type = parts[1]
                inputs = parts[2:-1]
                output = parts[-1]
                gates[gate_name] = {'type': gate_type, 'inputs': inputs, 'output': output}
        return gates

    def optimize_circuit(self, circuit_file, num_initial_simulations=10):
        iscas85_content = self.load_iscas85(circuit_file)
        gates = self.parse_iscas85(iscas85_content)
        if not gates:
            return None, None

        # Create a simple fitness function (example: minimize number of gates)
        def fitness(individual):
            return (len(gates) * individual[0],)

        # Create the toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, 0.1, 2.0)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create the population
        pop = toolbox.population(n=num_initial_simulations)
        hof = tools.HallOfFame(1)

        # Run the algorithm
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, halloffame=hof, verbose=True)

        # Get the best individual
        best_individual = hof[0]
        optimized_weights = best_individual

        # Assign the optimized weight to all gates
        optimized_gate_sizes = {gate: optimized_weights[0] for gate in gates}

        return iscas85_content, optimized_gate_sizes 