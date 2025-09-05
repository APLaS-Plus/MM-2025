from sko.PSO import PSO
from sko.tools import set_run_mode
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.Q3cal_mask_time import Q3_constraint_ueq, Q3_cal_mask_time_optimized
from utils.base import *
from utils.geo import *

# M1射中之前
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"]**2)) / MISSILE_SPEED
print(f"max_time: {max_time}")

def func(x):
    return -Q3_cal_mask_time_optimized(x)

lb = [-140] + [0] * 6
ub = [140] + [max_time] * 6

ueq = (Q3_constraint_ueq,)
# print(type(ueq))

set_run_mode(func, "multiprocessing")
pso = PSO(func=func, n_dim=7, pop=90, max_iter=800, lb=lb, ub=ub, constraint_ueq=ueq, c1=2, c2=2, w=0.3)
pso.run()
print(f" Vx, drop1, bomb1, drop2, bomb2, drop3, bomb3: {pso.gbest_x}")
print(-pso.gbest_y.item())

plt.plot(pso.gbest_y_hist)
plt.savefig("run/Q3_loss.png")