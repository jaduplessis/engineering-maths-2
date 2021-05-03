import numpy as np

ndatapoints = 10000
x = np.random.uniform(-1, 1, ndatapoints)
y = np.random.uniform(-1, 1, ndatapoints)

incircle = np.zeros(ndatapoints)
for i in range(ndatapoints):
    incircle[i] = (x[i]**2 + y[i]**2) <= 1

print(sum(incircle) / float(ndatapoints))
print(np.pi / 4)
