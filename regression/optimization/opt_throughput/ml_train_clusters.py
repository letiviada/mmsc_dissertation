import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data = pd.read_csv('regression/optimization/opt_throughput/data/data_clusters.csv')
data['ratio'] = data['ratio'] * 1000
for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster'] == cluster]

    min_ratio_index = cluster_data['ratio'].idxmin()
    max_ratio_index = cluster_data['ratio'].idxmax()
    min_ratio_row = cluster_data.loc[min_ratio_index]
    max_ratio_row = cluster_data.loc[max_ratio_index]
    edge_rows = pd.concat([min_ratio_row.to_frame().T, max_ratio_row.to_frame().T], ignore_index=True)
    cluster_data = cluster_data.drop([min_ratio_index, max_ratio_index])

    X_train1, X_test, y_train1, y_test = train_test_split(cluster_data[['ratio']], cluster_data['adhesivity'],
                                                        test_size=0.2, random_state=42)
    
    X_train, y_train = pd.concat([X_train1, edge_rows[['ratio']]]), pd.concat([y_train1, edge_rows['adhesivity']])
    model_base = make_pipeline(PolynomialFeatures(), Ridge())
    param_grid = {
            'polynomialfeatures__degree': np.arange(3,15),
            'ridge__alpha': np.logspace(-10, 5, 16)
        }
    grid_search = GridSearchCV(model_base, param_grid,
                    cv = 8, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_estimator_} with score {grid_search.best_score_}")
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)
    data.loc[data['cluster'] == cluster, 'predicted_adhesivity'] = model.predict(data[data['cluster'] == cluster][['ratio']])

data.to_csv('/Users/letiviada/dissertation_mmsc/regression/optimization/opt_throughput/data/data_with_predictions.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 8))
sns.scatterplot(data=data, x='ratio', y='adhesivity', hue='cluster', ax = axes[0],palette='tab10')
sns.scatterplot(data=data, x='ratio', y='predicted_adhesivity', hue = 'cluster',  ax = axes[1],palette='tab10', marker='x')
plt.show()
