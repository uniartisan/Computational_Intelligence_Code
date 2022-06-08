# %% [markdown]
# 神经网络控制倒立摆
# 使用PSO算法控制训练过程

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

# draw opt image
import matplotlib.pyplot as plt



# PSO Algorithm constants:
pop = 5200
max_iter = 30
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
# @jit
def score(individual):
    # RANDOM_SEED = random.randint(1,10000)
    # cartPole1 = cart_pole.CartPole(RANDOM_SEED)
    cartPole1 = cart_pole.CartPole()
    # train 使用距离信息
    return -1 * cartPole1.getTrainScore(individual),



# %%
# Genetic Algorithm flow:
mode = 'multiprocessing'
set_run_mode(score, mode)
pso = PSO(func=score, n_dim=NUM_OF_PARAMS, pop=pop, max_iter=max_iter,
          lb=BOUNDS_LOW, ub=BOUNDS_HIGH, w=w, c1=c1, c2=c2)

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
cartPole = cart_pole.CartPole(RANDOM_SEED)
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


# %%
plt.plot(pso.gbest_y_hist)
plt.show()


