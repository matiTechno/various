"""
algorithm: insertion sort (in-place)
input: (a_0, ..., a_n)
output: sorted sequence

for k <- 1 to n
    i <- k
    while i > 0 and a_i < a_(i-1)
        do  swap(a_i, a_(i-1))
            i <- i - 1
"""

import random

values = [random.randint(0,100) for _ in range(20)]
print(values)

for k in range(1, len(values)):
    i = k
    while i and values[i] < values[i-1]:
        values[i], values[i-1] = values[i-1], values[i]

print(values)
