# %%
from deap import base
from deap import creator
from deap import tools

import random
import numpy
import math
from numba import jit


import cart_pole
import elitism
import time

# 粒子群算法
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt


# PSO Algorithm constants:
pop = 4800
max_iter = 100
w = 0.95
c1 = 0.5
c2 = 0.5

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# create the cart pole task class:
cartPole = cart_pole.CartPole(RANDOM_SEED)
NUM_OF_PARAMS = len(cartPole)
# boundaries for layer size parameters:

# weight and bias values are bound between -1 and 1:
BOUNDS_LOW, BOUNDS_HIGH = -1.0, 1.0  # boundaries for all dimensions



# %%
# fitness calculation using the CrtPole class:
@jit
def score(individual):
    return cartPole.getScore(individual),



# %%
# Genetic Algorithm flow:
mode = 'multiprocessing'
set_run_mode(score, mode)
pso = PSO(func=score, n_dim=NUM_OF_PARAMS, pop=pop, max_iter=max_iter, lb=-1.0, ub=1.0, w=w, c1=c1, c2=c2)

tt = time.time()
pso.run()
tt2 = time.time()


# %%
# print best solution found:
best = pso.gbest_x
print()
print("Best Solution = ", best)
print("Best Fitness = ", pso.gbest_y)
print('Time used: {} sec'.format(tt2-tt))




# %%
# save best solution for a replay:
cartPole.saveParams(best)
cartPole.replay(best)


# %%
# find average score of 100 episodes using the best solution found:
print("Running 100 episodes using the best solution...")
scores = []
for test in range(100):
    scores.append(cart_pole.CartPole().getScore(best))
print("scores = ", scores)
print("Avg. score = ", sum(scores) / len(scores))



