from sko.PSO import PSO
from sko.tools import set_run_mode
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.Q1Q2cal_mask_time import Q2_cal_mask_time, Q2_constraint_ueq, Q2_cal_mask_time_optimized
from utils.base import *
from utils.geo import *

# M1射中之前
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"]**2)) / MISSILE_SPEED
print(f"max_time: {max_time}")

def func(x):
    return -Q2_cal_mask_time_optimized(x)

lb = [-140, 0, 0]
ub = [140, max_time, max_time]

ueq = (Q2_constraint_ueq,)
# print(type(ueq))

set_run_mode(func, "multiprocessing")
pso = PSO(func=func, n_dim=3, pop=96, max_iter=200, lb=lb, ub=ub, constraint_ueq=ueq, c1=1, c2=1, w=0.9)
pso.run()
print(f" Vx, drop, bomb: {pso.gbest_x}")
print(-pso.gbest_y.item())

plt.plot(pso.gbest_y_hist)
plt.savefig("run/Q2_loss.png")

#  Vx, drop, bomb: [-70.09965038   0.           2.46359669]
# 3.131565782891446