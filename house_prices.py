from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['AvgHouseValue'] = housing.target
print(data.head())

# Split the data into training and testing sets
X = data.drop('AvgHouseValue', axis=1)
y = data['AvgHouseValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest = RandomForestRegressor(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'ccp_alpha': np.arange(0, 0.1, 0.01)
}
grid_search = GridSearchCV(forest, param_grid, cv = 5)

grid_search.fit(X_train, y_train)
print("Best parameters:" , grid_search.best_params_)


