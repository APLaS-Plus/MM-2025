from sko.PSO import PSO
from sko.tools import set_run_mode
import numpy as np
import math
import matplotlib.pyplot as plt
from utils.Q3cal_mask_time import Q3_constraint_ueq, Q3_cal_mask_time, Q3_cal_mask_time_optimized
from utils.base import *
from utils.geo import *

# M1射中之前
max_time = math.sqrt(sum(MISSILES_INITIAL["M1"]**2)) / MISSILE_SPEED
print(f"max_time: {max_time}")

def func(x):
    return -Q3_cal_mask_time_optimized(x)

lb = [-140, 0] + [0] * 6
ub = [140, 30] + [max_time] * 6

ueq = (Q3_constraint_ueq,)
# print(type(ueq))

w = 0.9
c = (1 - w) / 2

set_run_mode(func, "multiprocessing")
pso = PSO(func=func, n_dim=8, pop=90, max_iter=200, lb=lb, ub=ub, constraint_ueq=ueq, c1=c, c2=c, w=w, verbose=True)

pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# print(type(pso.gbest_y_hist))
# print(pso.gbest_y_hist)

try:
    plot_data = plot_data = [item[0] if isinstance(item, np.ndarray) else item for item in pso.gbest_y_hist]
except:
    plot_data = pso.gbest_y_hist
plt.plot(plot_data)

plt.savefig("run/Q3_loss.png")

# best_x is  [1.04982409e+02 1.39647206e+01 5.82782933e-03 0.00000000e+00
#  1.67813094e+01 2.14645528e+01 0.00000000e+00 8.58353653e-01] best_y is [-7.40370185]