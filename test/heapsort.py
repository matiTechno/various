"""
some comments on heaps

I will use implicit data structure for a heap (an array).
Nodes are laid out in the breadth-first order.
If the index of a node in the array is 'i', then:

left(i) = 2i + 1
right(i) = 2i + 2
parent(i) = floor( (i-1)/2 )

This is possible because heap is a complete binary tree and then
the equations are easy to derive.

There are two major operations on a heap, bubble-up and bubble-down.
These move a target element down or up the tree by swapping with neighbours.

API functions are: extract() and insert()

There will be two versions of a heap, max-heap and min-heap and they differ
in the heap property - how the elements are ordered.

Main uses of a heap:
    implementation of priority queue
    heapsort

Here, I will implement only the min-heap.
"""

import math
import random

def left_idx(i):
    return 2 * i + 1

def right_idx(i):
    return 2 * i + 2

def parent_idx(i):
    return math.floor((i-1)/2)

class Heap:
    def __init__(self, values = None):

        if values is None:
            self.list = []
            return

        self.list = values.copy()

        i = parent_idx(len(self.list) - 1)

        while i >= 0:
            self.bubble_down(i)
            i -= 1

    def insert(self, value):
        self.list.append(value)
        # bubbling up
        i = len(self.list) - 1

        while True:
            pid = parent_idx(i)
            if pid < 0:
                break
            if self.list[i] < self.list[pid]:
                self.list[i], self.list[pid] = self.list[pid], self.list[i]
                i = pid
            else:
                break

    def extract(self):
        assert len(self.list)
        if len(self.list) < 2:
            return self.list.pop(0)

        top = self.list[0]
        self.list[0] = self.list.pop()
        self.bubble_down(0)
        return top

    def bubble_down(self, i):
        while True:
            lid = left_idx(i)

            if lid >= len(self.list):
                break;

            rid = right_idx(i)
            sid = lid # id of a smaller node

            if rid < len(self.list) and self.list[rid] < self.list[lid]:
                sid = rid

            if self.list[sid] < self.list[i]:
                self.list[sid], self.list[i] = self.list[i], self.list[sid]
                i = sid
            else:
                break

    def size(self):
        return len(self.list)

    def sorted_list(self):
        l = []
        while self.size():
            l.append(self.extract())
        assert self.size() == 0
        return l

values = [random.randint(0,100) for _ in range(20)]
print(values)

heap1 = Heap(values)
heap2 = Heap()

for v in values: # test another method of construction
    heap2.insert(v)

l1 = heap1.sorted_list()
l2 = heap2.sorted_list()
l3 = sorted(values)
print(l1)
print(l2)
print(l3)
assert l1 == l2 == l3
