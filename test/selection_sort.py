import random

"""
algorithm: selection sort (in-place)
input: (a_1, ..., a_n)
output: sorted list

for k <- 0 to n
    do m <- min(a_k, ..., a_n)
    swap(a_k, m)
"""

values = []

for i in range(20):
    values.append(random.randint(0,100))

print(values)

for idx in range(len(values) - 1):
    min = values[idx]
    idx_min = idx

    for i in range(idx, len(values)):
        if values[i] < min:
            min = values[i]
            idx_min = i
    #swap
    values[idx], values[idx_min] = values[idx_min], values[idx]

print(values)
