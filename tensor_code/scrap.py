import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering in Matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Create a test plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y, label=r"$\sin(x)$")  # Use LaTeX formatting
ax.set_title(r"Test Plot: $\sin(x)$")  # Use LaTeX formatting
ax.set_xlabel(r"$x$")  # Use LaTeX formatting
ax.set_ylabel(r"$\sin(x)$")  # Use LaTeX formatting
ax.legend()

# Save and show the plot
fig.savefig('latex_test_plot.png')
plt.show()
