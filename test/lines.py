import random
import matplotlib.pyplot as plt

class vec2:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

class Segment:
    def __init__(self, s = vec2(), e = vec2()):
        self.s = s
        self.e = e

def vadd(lhs, rhs):
    return vec2(lhs.x + rhs.x, lhs.y + rhs.y)

def vsub(lhs, rhs):
    return vec2(lhs.x - rhs.x, lhs.y - rhs.y)

def vmul(scal, vec):
    return vec2(scal * vec.x, scal * vec.y)

def dot(lhs, rhs):
    return lhs.x * rhs.x + lhs.y * rhs.y

def rot90(vec):
    return vec2(-vec.y, vec.x)

def intersect(seg1, seg2):
    d1 = vsub(seg1.e, seg1.s)
    d2 = vsub(seg2.e, seg2.s)
    norm_d2 = rot90(d2)
    denom = dot(norm_d2, d1)

    if denom == 0:
        return None

    t1 = dot( norm_d2, vsub(seg2.s, seg1.s) ) / denom
    t2 = dot( rot90(d1), vsub(seg1.s, seg2.s) ) / dot(rot90(d1), d2)

    if (t1 > 1 or t1 < 0) or (t2 > 1 or t2 < 0):
        return None
    return vadd( seg1.s, vmul(t1, d1) )

segments = []

for _ in range(20):
    s = vec2(random.random(), random.random())
    e = vec2(random.random(), random.random())
    segments.append(Segment(s, e))

segments.append( Segment(vec2(0.5,0), vec2(0.5, 1)) )

points = []

for s1 in segments:
    for s2 in segments:
        p = intersect(s1, s2)
        if p is not None:
            points.append(p)

for seg in segments:
    plt.plot( [seg.s.x, seg.e.x], [seg.s.y, seg.e.y] )

plt.scatter( [p.x for p in points], [p.y for p in points], color='r' )
plt.show()
