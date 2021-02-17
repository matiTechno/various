import random
import functools
import matplotlib.pyplot as plt

def get_column(mat, idx):
    return [row[idx] for row in mat]

def point_cmp(lhs, rhs):
    if lhs[0] == rhs[0]:
        return -(lhs[1] - rhs[1])
    return lhs[0] - rhs[0]

def signed_area(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]

def tr_vec(p1, p2):
    return (p2[0] - p1[0], p2[1] - p1[1])

points = [(random.random(), random.random()) for _ in range(100)]
points.sort(key = functools.cmp_to_key(point_cmp))

# test if the algorithm handles this case correctly
points.append( (points[-1][0], points[-1][1] + 0.1) )

assert len(points) >= 2

upper = []
upper.append(points[0])
upper.append(points[1])

for i in range(2, len(points)):
    upper.append(points[i])

    while len(upper) > 2:
        p0 = upper[-3]
        p1 = upper[-2]
        p2 = upper[-1]
        v1 = tr_vec(p1, p0)
        v2 = tr_vec(p1, p2)
        sa = signed_area(v1, v2)

        if sa >= 0:
            upper.pop(-2)
        else:
            break

lower = []
lower.append(points[-1])
lower.append(points[-2])

for i in reversed(range(len(points) - 2)):
    lower.append(points[i])

    while len(lower) > 2:
        p0 = lower[-3]
        p1 = lower[-2]
        p2 = lower[-1]
        v1 = tr_vec(p1, p0)
        v2 = tr_vec(p1, p2)
        sa = signed_area(v1, v2)

        if sa >= 0:
            lower.pop(-2)
        else:
            break

lower.pop(0)
lower.pop()
convex_hull = upper + lower
poly = upper + lower + [upper[0]]

plt.scatter(get_column(points, 0), get_column(points, 1))
plt.plot(get_column(poly, 0), get_column(poly, 1))
plt.show()
