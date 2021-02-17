import random
import math

def quicksort1(list):
    copy = list.copy()
    impl1(copy, 0, len(list)-1)
    return copy

def impl1(list, lo, hi):

    if lo >= hi:
        return
    piv_id = partition1(list, lo, hi)
    impl1(list, lo, piv_id - 1)
    impl1(list, piv_id + 1, hi)

# Lomuto partition scheme

def partition1(list, lo, hi):
    piv = list[hi]
    i = lo
    for k in range(lo, hi + 1):
        if list[k] < piv:
            list[k], list[i] = list[i], list[k]
            i += 1
    # in case of input like: 6 6 6 6 6 3
    # in the outter function we are discarding the i-th element,
    # so it must be the smallest one of the {a_i, ..., a_hi} subset
    list[i], list[k] = list[k], list[i]
    return i

def quicksort2(list):
    copy = list.copy()
    impl2(copy, 0, len(list)-1)
    return copy

def impl2(list, lo, hi):
    if lo >= hi:
        return
    piv_id = partition2(list, lo, hi)
    # we don't know if the piv_id element is the biggest of the left subset
    # so it must not be discarded
    # note we have parititioned the set into two subsets
    # {a_1, ..., a_(piv_id)} <= pivot
    # {a_(piv_id + 1), ..., a_n} >= pivot
    impl2(list, lo, piv_id)
    impl2(list, piv_id + 1, hi)


# this is a Hoare partition

def partition2(list, lo, hi):
    p = list[math.floor((lo + hi)/2)]
    i = lo
    j = hi

    while True:
        while list[i] < p:
            i += 1
        while list[j] > p:
            j -= 1

        if j <= i:
            return j
        list[i], list[j] = list[j], list[i]
        i += 1
        j -= 1

#test

for i in range(1, 100):
    list = [random.randint(0,1000) for _ in range(i)]
    assert sorted(list) == quicksort1(list) == quicksort2(list)

print('success')
