import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

firing_rates = np.arange(0.1, 200, 0.1)
cell_t_rate = 20
t_scaling_params = [1.5, 1, 0.66, 0.66]


def func(values, a, c):
    return a * np.exp(-c * (values-cell_t_rate)) + t_scaling_params[3]
x = np.array([0, cell_t_rate, cell_t_rate*10, cell_t_rate*20])
y = np.array([t_scaling_params[0], t_scaling_params[1], t_scaling_params[2], t_scaling_params[3]])
popt, pcov = curve_fit(func, x, y, [t_scaling_params[0], .1])

yy = func(firing_rates, *popt)

plt.plot(firing_rates, yy) #  Excitatory curve
plt.plot(firing_rates, 1/yy) #  Inhibitory curve

plt.show()

print popt


