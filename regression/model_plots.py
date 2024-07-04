from sklearn.model_selection import train_test_split, GridSearchCV
from treat_data import get_data_from_json, get_input_output
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import numpy as np
import sys
sys.path.append('/Users/letiviada/dissertation_mmsc')
from multiscale.plotting import create_fig, style_and_colormap, save_figure

y_pred = best_model.predict(X_test)
X_test_n = X_test.reset_index(drop=True)
y_test_n = y_test.reset_index(drop=True)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
end = time.time()
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")
print(f"Time taken: {end-start} seconds")

sns.set_theme()
fig, ax = create_fig(nrows = 1, ncols = 2 ,dpi = 100)
_, colors = style_and_colormap(num_positions = 7, colormap = 'tab20b')
colors = colors.tolist()
sns.scatterplot(x = y_test, y = y_pred, ax = ax[0])
# plot the results of the model
ax[0].plot(y_test, y_test, color = 'black')
# plot the predicted values against the actual values
sns.scatterplot(x = X_test['alpha'], y = y_pred, marker = 'x', label = 'Predicted', ax = ax[1])
sns.scatterplot(x = X_test['alpha'], y = y_test, marker = 'o', label = 'Actual', ax = ax[1])
save_figure(fig, 'regression/figures/data/solutioin_regression')
plt.show()
