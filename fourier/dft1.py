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

# harmonics - list of (index, amplitude) pairs

def harmonic_combi(period, harmonics):
    def fn(x):
        y = 0
        for h in harmonics:
            y += h[1] * math.cos(2 * math.pi * h[0] * x / period + math.pi/6)
        return y
    return fn

"""
For a perfect reconstruction either uncomment second fn or increase number of samples to 11.
I will try to showcase it on google colab.
N/2 ceofficient (only when N is odd) is not accurate when phase shift occurs in N/2 term of an original function.
"""

interv= [0, 10]
period = interv[1] - interv[0]
N = 10
T = period / N # sampling period
fn = harmonic_combi(period, [(1,1),(2,0.5),(3,0.1),(4,0.1),(5,0.1)])
#fn = harmonic_combi(period, [(1,1),(2,0.5),(3,0.1),(4,0.1)])
#fn = lambda x: x**2

spoints = [interv[0] + i * T for i in range(N)]
samples = [fn(x) for x in spoints]
coeffs = dft(samples)
inverse = idft(coeffs)
real_inv = [c.real for c in inverse] 

fig, ax = plt.subplots()

# render the original function
fineN = 100
fine_points = [interv[0] - period/2 + i * ((period + period/2) / fineN) for i in range(fineN)]
ax.plot(fine_points, [fn(x) for x in fine_points])

ax.scatter(spoints, samples)
ax.scatter(spoints, real_inv, s=10)

# we will try to reconstruct a continuous signal from the DFT coefficients

fac = 10
spoints2 = [interv[0] + i * T / 10 for i in range(N * fac)]
samples2 = [reconstruct(coeffs, period, t).real for t in spoints2]

ax.plot(spoints2, samples2)
plt.show()

