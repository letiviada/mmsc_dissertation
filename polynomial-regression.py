import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

def volume(x):
    """Throughput function will be something smooth that 
    is monotonically decreasing as a function of alpha
    for a given beta"""
    a= 1; b=-0.5; c=-0.1; d=-0.01
    return a*x**0 + b*x**2 + c*x**2 + d*x**3


# Range of alpha
x_plot = np.linspace(0, 1, 101)

# In the very worst case, train-test split removes the two ends of alpha from the train set - see fig
x_train = np.linspace(0.1, 0.9, 81)
x_train = np.sort(np.random.choice(a=x_train, size=11, replace=False, p=None))
y_train = volume(x_train)

# Need 2D arrays for ml model
X_train = x_train[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]


# Prepare plot
fig, ax = plt.subplots()
linewidth = 2
ax.set_prop_cycle(
    color=["black", "tab:blue", "tab:orange", "tab:green", "tab:red"]
)

# Plot physical model
ax.plot(x_plot, volume(x_plot), linewidth=linewidth, label="Physical model")

# Plot training data
ax.scatter(x_train, y_train, label="Training data")

# Train and plot a polynomial regressor for various degrees
for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-3))
    model.fit(X_train, y_train)
    y_plot = model.predict(X_plot)
    ax.plot(x_plot, y_plot, label=f"Polynomial of degree {degree}")
# See that only problems are at end points

# Or can use spline rather than polynomial (but have to deal with endpoints - see fig)
model = make_pipeline(SplineTransformer(n_knots=4, degree=3), Ridge(alpha=1e-3))
model.fit(X_train, y_train)

y_plot = model.predict(X_plot)
ax.plot(x_plot, y_plot, label="Spline")

ax.legend(loc="lower center")
# ax.set_ylim(0,1.1)
plt.show()