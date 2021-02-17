"""
algorithm: merge sort
input: (a_1, ..., a_n)
output: sorted sequence

step <- 2
half <- 1
while half < n
    do  i <- 0
        while i + half <= n
            do  l <- ( a_i, ..., a_(i + half - 1) )
                end <- min(i + step - 1, n)
                r <- ( a_(i + half), ..., a_(end) )
                (a_i, ..., a_(end)) <- Merge(l, r)
                i <- i + step
        step <- 2 step
        half <- 2 half

Merge
input: l, r - sorted sequences
output: o - merged (sorted) sequence

o <- empty
while |l| and |r|
    do  if first(l) < first(r)
            then o <- (o_1, ..., o_n, first(l))
            pop_first(l)

            else o <- (o_1, ..., o_n, first(r))
            pop_first(r)

o <- (o_1, ..., o_n, l_1, ..., l_n, r_1, ..., r_n)
return o
"""

def merge_sort(values):
    step = 2
    half = 1

    while half < len(values):
        i = 0
        while i + half < len(values):
            l = values[i : i + half]
            end = int( min(i + step, len(values)) )
            r = values[i + half : end]
            m = merge(l, r)

            for k in range(len(m)):
                values[i + k] = m[k]
            i += step
        step *= 2
        half *= 2

def merge(l, r):
    o = []

    while len(l) and len(r):
        if l[0] < r[0]:
            o.append(l[0])
            l.pop(0)
        else:
            o.append(r[0])
            r.pop(0)
    return o + l + r

import random

for i in range(1, 100):
    values = [random.randint(1,100) for _ in range(i)]
    merge_sort(values)

    for k in range(0, len(values) - 1):
        assert values[k] <= values[k+1]

print("success")
