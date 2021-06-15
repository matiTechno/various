import matplotlib.pyplot as plt
import cmath
import math

def dft_impl(x):
    N = len(x)
    coeffs = []
    for k in range(N):
        coeff = 0
        for n, xn in enumerate(x):
            coeff += xn * cmath.exp(-1j * 2 * cmath.pi * k * n / N)
        coeffs.append(coeff)
    return coeffs

def dft(x):
    coeffs = dft_impl(x)
    return [c / len(x) for c in coeffs]

def idft(coeffs):
    x = dft_impl([c.conjugate() for c in coeffs])
    return [c.conjugate() for c in x]

def reconstruct(coeffs, period, t):
    y = 0
    for k, coeff in enumerate(coeffs):
        if k > N/2:
            k -= N
        y += coeff * cmath.exp(1j * 2 * cmath.pi * k * t / period)
    return y

def gen_spline(points, t):
    N = len(points)
    pid = int(t * N)
    assert pid != N
    p0 = points[pid-1] if pid != 0 else points[N-1]
    p1 = points[pid]
    p2 = points[(pid + 1) % N]
    p3 = points[(pid + 2) % N]
    # convert to segment-local time
    t = (t * N) - pid
    tt = t*t
    ttt = tt*t
    out = []
    for i in range(len(p0)):
        x = ( (-ttt + 2*tt - t) * p0[i] + (ttt - 2*tt + 1) * p1[i] + (-ttt + tt + t) * p2[i] +
            (ttt - tt) * p3[i] )
        out.append(x)
    return out

control_points = [(1,1),(0,4),(-1,1),(-4,0),(-1,-1),(0,-4),(1,-1),(4,0)]
N = 10
samples = [gen_spline(control_points, t) for t in [1/N * i for i in range(N)]]
samples = [complex(s[0], s[1]) for s in samples]

coeffs = dft(samples)
inverse = idft(coeffs)

for s, i in zip(samples, inverse):
    assert abs(s - i) < 0.00001

fineN = 20 * N
# reconstruction
re_samples = [ reconstruct(coeffs, 1, t) for t in [1 / fineN * i for i in range(fineN)] ]
re_samples.append(re_samples[0])
# perfect reconstruction
perf_samples = [gen_spline(control_points, t) for t in [1 / fineN * i for i in range(fineN)]]
perf_samples.append(perf_samples[0])

fig, ax = plt.subplots()
ax.plot([p[0] for p in perf_samples], [p[1] for p in perf_samples], color='orange',
        label='function')
ax.plot([p.real for p in re_samples], [p.imag for p in re_samples], color='b',
        label='reconstruction based on samples')
ax.scatter([p.real for p in samples], [p.imag for p in samples], color='r',
        label='samples of the function')
#ax.scatter([p[0] for p in control_points], [p[1] for p in control_points], color='g')
ax.legend()
plt.show()

