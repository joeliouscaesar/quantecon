"""
Scipy exercises
"""

# 13.1
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci_stats

# option parameters
mu = 4
sigma = 0.25
beta = 0.99
n = 10
K = 40

# f is pdf for asset returns
f = lambda x:sci_stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
# # sanity I did that right
# draws = sci_stats.lognorm.rvs(size=1000, s=sigma, scale=np.exp(mu))
# ldraws = np.log(draws)
# ldraws.mean()
# ldraws.std()

def g(x):
    option_return = np.where(x - K < 0, 0, x - K)
    return beta**n * option_return * f(x)

# domain/values for plot
x = np.linspace(0, 400, 1000)
gs = g(x)

# plot!
fig, ax = plt.subplots()
ax.plot(x, gs)
plt.show()


# 13.2 
import scipy.integrate as sci_int
import time
start = time.time()
ev, err = sci_int.quad(g, 0, 400)
end = time.time()
print(f"time: {end-start}")

# # sanity check
# x = np.linspace(0, 400, 4000)
# start = time.time()
# measure = f(x).sum()
# g(x).sum() / measure
# end = time.time()
# print(f"time: {end-start}")

# 13.3
M = 10_000_000
# rvs from the lognorm directly, could've also gone the normal route
draws = sci_stats.lognorm.rvs(size=M, s=sigma,scale=np.exp(mu))
ev2 = np.where(draws < K, 0, draws - K).mean()
# final answer needs discounting
ev2 * beta ** n

# 13.4
def bisect(f, a, b, tol=10e-5):
    middle = (a+b)/2
    if (b - a) <= tol:
        return middle
    # in this case recursive call
    if f(middle) > 0:
        return bisect(f, a, middle, tol)
    else:
        return bisect(f, middle, b, tol)

def f(x):
    return np.sin(4*(x - 0.25)) + x + x**20 - 1

bisect(f, 0, 1)