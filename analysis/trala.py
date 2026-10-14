import numpy as np

x = [0.1, 0.15, 0.2, 0.05, 0.2, 0.05, 0.2, 0.05]

print(1 - 10*np.var(x))

x1 = []
for i in range(len(x)):
    x1.append(1-x[i])

print(1 - 10*np.var(x1))
