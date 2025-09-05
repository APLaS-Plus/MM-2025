from sko.PSO import PSO
from sko.tools import set_run_mode
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.cal_mask_time import Q2_cal_mask_time_optimized, Q2_constraint_ueq
from utils.base import *
from utils.geo import *

# M1射中之前
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"]**2)) / MISSILE_SPEED
print(f"max_time: {max_time}")

func = Q2_cal_mask_time_optimized

lb = [-140, 0, 0]
ub = [140, max_time, max_time]

ueq = (Q2_constraint_ueq,)
print(type(ueq))

set_run_mode(func, "multiprocessing")
pso = PSO(func=func, n_dim=3, pop=40, max_iter=800, lb=lb, ub=ub, constraint_ueq=ueq)
pso.run()
print(pso.gbest_x)
print(pso.gbest_y)

plt.plot(pso.gbest_y_hist)
plt.show()
