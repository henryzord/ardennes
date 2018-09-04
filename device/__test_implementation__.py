from cpu_device import choice
import numpy as np

a = np.array(['a', 'b', 'c', 'd', 'e'])
p = np.array([0.1, 0.4, 0.2, 0.2, 0.1])
replace = True
size = (3, 3)

print(choice(a, size, replace, p))
# print(choice(a=a, size=size, replace=replace, p=p))
