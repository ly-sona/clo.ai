# Import required libraries
import random                          # For randomness in mutation and crossover
import numpy as np                    # For numerical operations
import matplotlib.pyplot as plt       # For plotting fitness over generations
from deap import base, creator, tools, algorithms  # Main DEAP components
import xgboost as xgb                 # XGBoost for surrogate power prediction

# ----- Layout Configurations -----
N_MACROS = 10                         # Number of components to place in the layout
GRID_WIDTH, GRID_HEIGHT = 10, 10      # Dimensions of the layout grid (10x10)

# ----- Load Pre-trained XGBoost Model -----
### TODO: Important change the name of this to the xgb_layout_model and keep it in the same directory 
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgb_layout_model.json')  # Load trained model from file

# ----- Feature Extraction from Layout -----
def extract_features(layout):
    """
    Convert the layout into a feature vector suitable for XGBoost input.
    """
    layout = np.array(layout).reshape(-1, 2)   # Convert flat list to Nx2 shape
    com = np.mean(layout, axis=0)              # Center of mass of the layout
    dists = [np.linalg.norm(layout[i] - layout[j]) 
             for i in range(len(layout)) 
             for j in range(i+1, len(layout))] # Pairwise distances between macros
    avg_dist = np.mean(dists)                  # Average pairwise distance
    return np.array([*com, avg_dist])          # Return as feature vector

# ----- Repair Function to Remove Overlaps -----
def repair(layout):
    """
    Ensures that no two macros occupy the same grid cell.
    """
    seen = set()  # Keep track of used positions
    for i in range(len(layout)):
        while tuple(layout[i]) in seen:
            layout[i] = [random.randint(0, GRID_WIDTH-1), 
                         random.randint(0, GRID_HEIGHT-1)]  # Reassign if overlap
        seen.add(tuple(layout[i]))
    return layout

# # ----- Evaluation Function Using XGBoost -----
def evaluate(individual):
    """
    Evaluate the predicted power using the XGBoost model.
    """
    layout = np.array(individual).reshape(N_MACROS, 2)  # Reshape layout
    layout = repair(layout)                             # Fix overlaps
    features = extract_features(layout)                 # Extract features
    predicted_power = xgb_model.predict(np.array([features]))[0]  # Predict power
    return predicted_power,                             # Return as a tuple


# Mock data for testing 
#  def evaluate(individual):
#     """
#     Fake evaluation function just for testing structure.
#     """
#     return random.uniform(0.3, 1.0),  # The comma makes it a tuple!

# ----- Custom Mutation Operator -----
def mutate_layout(individual, indpb=0.2):
    """
    Slightly shift macro positions with some probability (indpb).
    """
    for i in range(0, len(individual), 2):  # Every (x, y) pair
        if random.random() < indpb:
            individual[i] = min(GRID_WIDTH-1, max(0, individual[i] + random.choice([-1, 1])))
        if random.random() < indpb:
            individual[i+1] = min(GRID_HEIGHT-1, max(0, individual[i+1] + random.choice([-1, 1])))
    return individual,

# ----- Custom Crossover Operator -----
def crossover_layout(ind1, ind2):
    """
    Combine parts of two parents to create two new children.
    """
    size = len(ind1)
    cxpoint = random.randrange(2, size, 2)  # Select crossover point between macros
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]  # Swap parts
    return ind1, ind2

# ----- DEAP Setup -----
# Define fitness and individual types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
creator.create("Individual", list, fitness=creator.FitnessMin)

# Register functions to DEAP's toolbox
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, GRID_WIDTH - 1)  # Generate random int
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2 * N_MACROS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover_layout)
toolbox.register("mutate", mutate_layout)
toolbox.register("select", tools.selTournament, tournsize=3)

# ----- Genetic Algorithm Loop -----
def run_ga(threshold=0.5, generations=50, pop_size=30, min_generations=5):
    """
    Run the genetic algorithm until power is below threshold or max generations reached.
    """
    pop = toolbox.population(n=pop_size)       # Create initial population
    hof = tools.HallOfFame(1)                  # Save the best solution
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Track fitness
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()                  # Record history
    fitness_over_time = []

    for gen in range(generations):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)  # Apply crossover & mutation

        # Evaluate each new offspring
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))  # Select survivors
        hof.update(pop)                              # Update best found

        # Track stats
        min_fit = min(f.fitness.values[0] for f in pop)
        fitness_over_time.append(min_fit)
        logbook.record(gen=gen, **stats.compile(pop))

        if min_fit < threshold and gen >= min_generations:  # Stop early if threshold reached
            print(f"Threshold reached! Predicted power: {min_fit}")
            break

    return pop, logbook, hof, fitness_over_time

# ----- Run GA and Plot Results -----
pop, logbook, hof, fitness = run_ga(threshold=0.3)

# Plot predicted power over generations
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

# Display best layout found
print("Best layout (flattened):", hof[0])
