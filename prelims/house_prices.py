from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
start = time.time()
# Load the data
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['AvgHouseValue'] = housing.target
#print(data.head())

# Split the data into training and testing sets
X = data.drop('AvgHouseValue', axis=1)
y = data['AvgHouseValue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest = RandomForestRegressor(random_state=42)

forest.fit(X_train,y_train)
feature = forest.feature_importances_
feauture_names = X.columns
importance_df = pd.DataFrame({'Feature':feauture_names, 'Importance': feature})
importance_df = importance_df.sort_values(by = 'Importance', ascending = False)
print(importance_df)

top_features = importance_df['Feature'].iloc[:2].tolist()
#print(f'Top 3 features: {top_features}')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100],
    'max_depth': [10, 20, 30],
    'ccp_alpha': np.arange(0, 0.1, 0.01)
}
grid_search = GridSearchCV(forest, param_grid, cv = 5, n_jobs= 5)
X_top_train = X_train[top_features]
X_top_test = X_test[top_features]
grid_search.fit(X_top_train, y_train)
print("Best parameters:" , grid_search.best_params_)
#mse = mean_squared_error(y_test,y_pred)
#r2 = r2_score(y_test, y_pred)


#print(f'Mean squared error:{mse}')
#print(f'R2 score:{r2}')
end = time.time()
print(f'Time taken: {end-start} seconds')
