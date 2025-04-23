# deap_model.py
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import xgboost as xgb
from features import extract_features

# Layout configuration
N_MACROS = 10
GRID_WIDTH, GRID_HEIGHT = 10, 10

# Load trained XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgb_layout_model.json')  # Ensure this matches --model_output from training

def repair(layout):
    """
    Ensures that macros don't overlap on the grid.
    """
    seen = set()
    for i in range(len(layout)):
        pos = tuple(layout[i])
        while pos in seen:
            layout[i] = [random.randint(0, GRID_WIDTH-1),
                         random.randint(0, GRID_HEIGHT-1)]
            pos = tuple(layout[i])
        seen.add(pos)
    return layout

def evaluate(individual):
    """
    Evaluate a layout (flattened individual) by predicting power.
    """
    layout = np.array(individual).reshape(-1, 2)
    layout = repair(layout)
    features = extract_features(layout)
    predicted_power = xgb_model.predict(features.reshape(1, -1))[0]
    return predicted_power,

def mutate_layout(individual, indpb=0.2):
    """
    Slightly shift macro positions with probability indpb.
    """
    for i in range(0, len(individual), 2):
        if random.random() < indpb:
            individual[i] = min(GRID_WIDTH-1, max(0, individual[i] + random.choice([-1, 1])))
        if random.random() < indpb:
            individual[i+1] = min(GRID_HEIGHT-1, max(0, individual[i+1] + random.choice([-1, 1])))
    return individual,

def crossover_layout(ind1, ind2):
    """
    Two-point crossover at macro boundaries.
    """
    size = len(ind1)
    cxpoint = random.randrange(2, size, 2)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize power
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, GRID_WIDTH - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2 * N_MACROS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover_layout)
toolbox.register("mutate", mutate_layout)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(threshold=0.5, generations=50, pop_size=30, min_generations=5):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    fitness_over_time = []

    for gen in range(generations):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)

        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)

        min_fit = min(f.fitness.values[0] for f in pop)
        fitness_over_time.append(min_fit)
        logbook.record(gen=gen, **stats.compile(pop))

        if min_fit < threshold and gen >= min_generations:
            print(f"Threshold reached! Predicted power: {min_fit}")
            break

    return pop, logbook, hof, fitness_over_time

if __name__ == "__main__":
    pop, logbook, hof, fitness = run_ga(threshold=0.3)
    if fitness:
        plt.plot(range(len(fitness)), fitness, marker='o', linestyle='-')
        plt.xlabel("Generation")
        plt.ylabel("Predicted Power (mW)")
        plt.title("Fitness Over Generations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No fitness data to plot!")
    print("Best layout (flattened):", hof[0])