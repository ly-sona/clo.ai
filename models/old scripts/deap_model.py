import random, numpy as np, matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import xgboost as xgb
from .features import extract_features        # keep for layout stats
from .def_writer import write_def

# ─── Layout config ─────────────────────────────────────────────────────
N_MACROS = 10
GRID_WIDTH, GRID_HEIGHT = 10, 10

# ─── Load trained *congestion* model ───────────────────────────────────
xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_congestion_model.json")   # ← make sure this file exists

# ─── GA helpers ────────────────────────────────────────────────────────
def repair(layout):
    seen = set()
    for i in range(len(layout)):
        pos = tuple(layout[i])
        while pos in seen:
            layout[i] = [random.randint(0, GRID_WIDTH - 1),
                         random.randint(0, GRID_HEIGHT - 1)]
            pos = tuple(layout[i])
        seen.add(pos)
    return layout

def evaluate(individual):
    """
    Return **predicted congestion** for a flattened layout.
    Lower is better.
    """
    layout = np.array(individual).reshape(-1, 2)
    layout = repair(layout)
    feat = extract_features(layout)
    cong = xgb_model.predict(feat.reshape(1, -1))[0]
    return (cong,)

def mutate_layout(ind, indpb=0.2):
    for i in range(0, len(ind), 2):
        if random.random() < indpb:
            ind[i] = min(GRID_WIDTH - 1, max(0, ind[i] + random.choice([-1, 1])))
        if random.random() < indpb:
            ind[i + 1] = min(GRID_HEIGHT - 1, max(0, ind[i + 1] + random.choice([-1, 1])))
    return (ind,)

def crossover_layout(ind1, ind2):
    cx = random.randrange(2, len(ind1), 2)
    ind1[cx:], ind2[cx:] = ind2[cx:], ind1[cx:]
    return ind1, ind2

# ─── DEAP setup ────────────────────────────────────────────────────────
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

tb = base.Toolbox()
tb.register("attr_int", random.randint, 0, GRID_WIDTH - 1)
tb.register("individual", tools.initRepeat, creator.Individual,
            tb.attr_int, n=2 * N_MACROS)
tb.register("population", tools.initRepeat, list, tb.individual)

tb.register("evaluate", evaluate)
tb.register("mate", crossover_layout)
tb.register("mutate", mutate_layout)
tb.register("select", tools.selTournament, tournsize=3)

def run_ga(threshold=5.0, generations=50, pop_size=30, min_generations=5):
    pop, hof = tb.population(n=pop_size), tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min); stats.register("avg", np.mean)

    fitness_curve = []
    for gen in range(generations):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(pop, tb, cxpb=0.5, mutpb=0.3)
        for ind, fit in zip(offspring, map(tb.evaluate, offspring)):
            ind.fitness.values = fit

        pop = tb.select(offspring, k=len(pop))
        hof.update(pop)

        gen_min = min(ind.fitness.values[0] for ind in pop)
        fitness_curve.append(gen_min)
        if gen_min < threshold and gen >= min_generations:
            print(f"Threshold reached! Predicted congestion: {gen_min:.3f}")
            break

    return pop, hof[0], fitness_curve

# ─── CLI run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    _, best, curve = run_ga()
    best_xy = np.array(best).reshape(-1, 2)
    write_def(best_xy, GRID_WIDTH, GRID_HEIGHT,
              design_name="GA_LAYOUT", outfile="best_layout.def")

    plt.plot(curve, marker="o")
    plt.xlabel("Generation"); plt.ylabel("Predicted Congestion")
    plt.title("Fitness over Generations"); plt.grid(True); plt.tight_layout()
    plt.show()
    print("Best layout:", best)