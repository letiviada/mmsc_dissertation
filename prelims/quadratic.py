import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from multiscale.plotting import create_fig, style_and_colormap

# We have imported the necessary libraries to run the code.
#  Now we will load the data and split it into training and testing sets.
#  We will also train the model and evaluate it.


# In this example we are going to data defining a straight line.
X = np.linspace(-5,5,100).reshape(-1,1)
a,b,c = 2,-3,-1
y = a * X.flatten()**2 + b * X.flatten() + c + np.random.normal(0,1, X.shape[0])
sol = lambda x: a*x**2 + b*x +c

# Split the data into training and testing sets
# I am going to test in 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest = RandomForestRegressor(n_estimators=100, random_state=42)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean squared error: {mse}')
print(f'R2 score: {r2}')

# Plot the results
# ----------------

_, colors = style_and_colormap(num_positions = 3)
fig, axes = create_fig(1, 1, title='Random Forest Regressor', figsize=(10, 6),dpi = 100)

axes[0].scatter(X_test, y_test, color=colors[0])
axes[0].scatter(X_train, y_train, color=colors[1])
axes[0].scatter(X_test, y_pred, color=colors[2])
axes[0].plot(X.flatten(),sol(X.flatten()), color = 'k', linestyle = '--')



plt.show()
