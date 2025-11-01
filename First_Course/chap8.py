"""
8. OOP II: Building Classes
"""
# 8.1
import numpy as np

class ECDF:
    """
    Class representing an empirical cdf

    Attributes:
        observations:np.array - given sample
    """
    def __init__(self, obs:np.ndarray):
        self.observations = obs
    def __call__(self, x):
        """
        Evaluates the ecdf for a given value x
        """
        obs_le_x = (self.observations <= x)
        return obs_le_x.mean()

# generator instance
rng = np.random.default_rng()
unif_sample = rng.uniform(0, 1, 10000)

F_unif = ECDF(unif_sample)
F_unif(0.5)
F_unif(0.9)


# 8.2
class Polynomial:
    """
    Class representing a polynomial. Stores coefficients. Assumes low order terms are first
    e.g. self.coefficients = [1, 0, 2] <--> 2x^2 + 1  
    """
    def __init__(self, coefficients):
        self.coefficients = coefficients
    def evaluate(self, x):
        """
        Evaluates polynomial at x
        """
        value = 0.0
        for (k, coef) in enumerate(self.coefficients):
            value += coef * x**k
        return value
    def differentiate(self):
        """
        Differentiates polynomial (inplace)
        """
        new_coefficients = []
        for (k, coef) in enumerate(self.coefficients):
            if k == 0:
                # skip order 0 term
                continue
            new_coef = coef * k
            new_coefficients.append(new_coef)
        self.coefficients = new_coefficients

# imports just for plots!
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plotPoly(poly:Polynomial):
    data = {
        "x":[i/2 for i in range(-100,100)],
        "y":[poly.evaluate(i/2) for i in range(-100,100)]
    }
    data = pd.DataFrame(data)
    sns.scatterplot(data, x="x", y="y")
    plt.show()

poly0 = Polynomial([-1, 0, 1])
plotPoly(poly0)

poly1 = Polynomial([-1, 0, 1, -2])
plotPoly(poly1)

# edge case :)
poly2 = Polynomial([])
plotPoly(poly2)

# alright now let's check diff
poly0.differentiate()
plotPoly(poly0)
poly0.differentiate()
plotPoly(poly0)

plotPoly(poly1)
poly1.differentiate()
plotPoly(poly1)
poly1.differentiate()
plotPoly(poly1)
poly1.coefficients