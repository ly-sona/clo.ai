#!/usr/bin/env python
# utils/deap_helpers.py
"""One‑liner DEAP optimiser for ‘minimise single weight’ use‑case."""

import numpy as np
from deap import base, creator, tools, algorithms

def deap_minimize(fitness_fn,
                  n_weights: int = 1,
                  ngen: int = 20,
                  pop: int = 60,
                  sigma: float = 0.2,
                  mu: float = 1.0,
                  cxpb: float = 0.5,
                  mutpb: float = 0.2,
                  tourn: int = 3):
    """Return best individual discovered by DEAP (single‑objective minimisation)."""
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    tb = base.Toolbox()
    tb.register("attr", np.random.rand)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr, n=n_weights)
    tb.register("population",  tools.initRepeat, list, tb.individual)

    tb.register("evaluate", fitness_fn)
    tb.register("mate",     tools.cxBlend, alpha=0.5)
    tb.register("mutate",   tools.mutGaussian, mu=mu, sigma=sigma, indpb=1.0)
    tb.register("select",   tools.selTournament, tournsize=tourn)

    popu = tb.population(pop)
    hof, _ = algorithms.eaMuPlusLambda(
        popu, tb, mu=pop, lambda_=pop, cxpb=cxpb, mutpb=mutpb,
        ngen=ngen, halloffame=tools.HallOfFame(1), verbose=False
    )
    return hof[0]