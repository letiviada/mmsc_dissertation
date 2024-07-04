from sklearn.model_selection import train_test_split, GridSearchCV
from utils import obtain_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import numpy as np
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
sys.path.append('/home/viadacampos/Documents/mmsc_dissertation')
from multiscale.plotting import create_fig, style_and_colormap, save_figure

file = "regression/models/termination_time_model.pkl"
best_model = joblib.load(file)
data = obtain_data('termination_time')
inputs, outputs = data[['alpha', 'beta']], data['termination_time']#
inputs, outputs = joblib.load("regression/models/train_data.pkl")

y_pred = best_model.predict(inputs)

mse = mean_squared_error(outputs, y_pred)
r2 = r2_score(outputs, y_pred)
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")

sns.set_theme()
fig, ax = create_fig(nrows = 1, ncols = 2 ,dpi = 100)
_, colors = style_and_colormap(num_positions = 7, colormap = 'tab20b')
colors = colors.tolist()
sns.scatterplot(x = outputs, y = y_pred, ax = ax[0])
# plot the results of the model
ax[0].plot(outputs, outputs, color = 'black')
# plot the predicted values against the actual values
sns.scatterplot(x = inputs['alpha'], y = y_pred, marker = 'x', label = 'Predicted Test', ax = ax[1])
sns.scatterplot(x = inputs['alpha'], y = outputs, marker = 'o', label = 'Actual Test', ax = ax[1])
ax[1].set_xlabel('Adhesivity',fontsize='20')
ax[1].set_ylabel('Termination Time', fontsize = '20')  
save_figure(fig, 'regression/figures/data_larger/solution_regression')
plt.show()
