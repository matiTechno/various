import matplotlib.pyplot as plt
import random

vals = [random.random() for _ in range(20)]
args = [x / (len(vals)-1) for x in range(len(vals))]
h = 1 / (len(args)-1)
vals[0] = 0
vals[-1] = 1

vals_init = vals.copy()
valsp = vals.copy()
func = [0.5*x**3 + 2*x**2 - 1.5*x for x in args]

# Gauss Seidel method
# Laplace equation
for _ in range(100):
    for i in range(1, len(vals) -1):
        vals[i] = 0.5 * (vals[i+1] + vals[i-1])

# Poisson equation
for _ in range(100):
    for i in range(1, len(valsp) -1):
        valsp[i] = 0.5 * (valsp[i+1] + valsp[i-1] - (3*args[i] + 4)*h**2)
        # we want the second derivative to be 3x + 4


fig, axs = plt.subplots(2)
axs[0].plot(args, vals_init, label='init')
axs[0].scatter([args[0], args[-1]], [vals[0], vals[-1]], label='target')
axs[0].plot(args, vals, label='result')
axs[0].set_title('Laplace eq; Gauss Seidel 100 iterations')

axs[1].plot(args, vals_init, label='init')
axs[1].plot(args, func, label='target')
axs[1].plot(args, valsp, label='result')
axs[1].set_title('Poisson eq')

axs[0].legend()
axs[1].legend()
plt.show()
