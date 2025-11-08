# 12.1 
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# prepping data to plot
thetas = np.linspace(0, 2, 10)
xs = np.linspace(0, 5, 100)
for theta in thetas:
    fs = np.cos(np.pi*theta*xs)*np.exp(-xs)
    ax.plot(xs, fs, label=rf'$\theta={round(theta,3)}$')
ax.set_title("Pretty Plot")
ax.legend()

plt.show()


