"""
11. Numpy
"""
import numpy as np
import quantecon as qe
#11.1

def p(x,coef):
    """
    Evaulates a polynomial defined by coef at x
    """
    degree = coef.size
    xs = np.repeat(x, degree)
    xterms = xs.cumprod() / x
    return (xterms * coef).sum()
    
def p_naive(x, coef):
    val = 0.0
    for (k, c) in enumerate(coef):
        val += x**k * c
    return val

def my_time(f, *args):
    start = time.time()
    f(*args)
    end = time.time()
    return end - start

import random
import time

coefs = [random.uniform(-1,1) for _ in range(3000)]
coefs_np = np.array(coefs)

naive_times = [my_time(p_naive, 0.1, coefs) for _ in range(1000)]
np_times = [my_time(p, 0.1, coefs_np) for _ in range(1000)]

avg_np_time = np.array(np_times).mean()
avg_naive_time = np.array(naive_times).mean()
avg_np_time/avg_naive_time

#11.2
class DiscreteRv:
    def __init__(self, q:np.ndarray):
        self.q = q
    def draw(self, size):
        uniform_draws = np.random.uniform(0,1,size)
        cdf = self.q.cumsum()
        return cdf.searchsorted(uniform_draws)

coin = DiscreteRv(np.array([0.75, 0.25]))
flips = coin.draw(10000)
flips.mean()
flips.var()

# 11.3
import seaborn as sns
class ECDF:
    def __init__(self, sample:np.ndarray):
        self.sample = sample
    def __call__(self, x):
        return (self.sample <= x).mean()
    def plot(self, a, b):
        points = np.linspace(a,b,100)
        # reshape to broadcast
        points.shape = (100, 1)
        cdf_evals = (points >= self.sample).mean(axis=1)
        # reshape back so we can plot it
        points.shape = (100,)
        sns.lineplot(x=points, y=cdf_evals)
        plt.show()

rev_eng = ECDF(flips)
rev_eng(0.1)
rev_eng.plot(-1, 2)

# normal dist
normies = np.random.normal(0, 1, 10000)
norm = ECDF(normies)
norm.plot(-3, 3)

# 11.4
# x = np.random.randn(4,4)
# y = np.random.randn(4)
# x / y

def slower(rows, cols):
    denoms = [random.normalvariate(0, 1) for _ in range(rows)]
    A = []
    for _ in range(rows):
        vals = [random.normalvariate(0, 1)/denoms[d] for d in range(cols)]
        A.append(vals)
    return A
slower(4,4)

def slow_more_dims(d1,d2,d3):
    denoms = [random.normalvariate(0,1) for _ in range(d3)]
    A = []
    for _ in range(d1):
        B = []
        for _ in range(d2):
            vals = [random.normalvariate(0, 1)/denoms[d] for d in range(d3)]
            B.append(vals)
        A.append(B)
    return A

with qe.Timer():
    B = slow_more_dims(1000,100,100)

with qe.Timer():
    x = np.random.randn(1000, 100, 100)
    y = np.random.randn(100)
    B2 = x / y

