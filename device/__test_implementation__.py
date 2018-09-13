from cpu_device import choice
import numpy as np
from collections import Counter

a = np.array(['a', 'b', 'c', 'd', 'e'])
p = np.array([0.1, 0.4, 0.2, 0.2, 0.1])
replace = True
size = 1000000

counts = Counter(choice(a, size, replace, p))
_sum = sum(list(counts.values()))

for i in range(len(p)):
    print('%s: real: %f observed: %f' % (a[i], p[i], counts[a[i]] / float(_sum)))
