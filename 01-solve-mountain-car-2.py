# %% [markdown]
# 找到一组使得小车可以上山的解，使用PSO算法寻找

# %%
from deap import base
from deap import creator
from deap import tools

import random
import numpy
import math
from numba import jit

import mountain_car
import elitism
import time

# 粒子群算法
from sko.PSO import PSO
from sko.tools import set_run_mode
import matplotlib.pyplot as plt

# PSO Algorithm constants:
pop = 9600
max_iter = 40
w = 0.95
c1 = 0.5
c2 = 0.5

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the Zoo test class:
car = mountain_car.MountainCar(RANDOM_SEED)

def result2op(x):
    op = []
    for i in x:
        if i<0.5:
            op.append(0)
        elif i<1.5:
            op.append(1)
        else:
            op.append(2)
        
    return op


# fitness calculation
# @jit
def getCarScore(individual):
    individuals = result2op(individual)
    q_loss = 0
    # for i in range(len(individual)):
    #     q_loss+=math.pow(individual[i]-individuals[i],2)

    return car.getScore(individuals)+q_loss/200,  # return a tuple


# %%
# 多线程优化
mode = 'multiprocessing'
set_run_mode(getCarScore, mode)
pso = PSO(func=getCarScore, n_dim=200, pop=pop,
          max_iter=max_iter, lb=0, ub=2.49, w=w, c1=c1, c2=c2)

tt = time.time()
pso.run()
tt2 = time.time()


# %%
pso.gbest_x

# %%
# print best solution:
best =result2op(pso.gbest_x)
print()
print("Best Solution = ", best)
print("Best Fitness = ", pso.gbest_y)
print('Time used: {} sec'.format(tt2-tt))


# %%
# save best solution for a replay:
car.saveActions(best)
car.replay(best)

# %%
plt.plot(pso.gbest_y_hist)
plt.show()


