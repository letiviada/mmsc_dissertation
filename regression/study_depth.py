
from sklearn.model_selection import cross_val_score, KFold,train_test_split, GridSearchCV, RandomizedSearchCV
from utils import save_model
from utils import obtain_data, clean_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import time
import numpy as np
def main(depth):
    data = clean_data(
filename='/Users/letiviada/dissertation_mmsc/multiscale/results/mono-dispersed/performance_indicators/performance_indicators_phi_1.0.json')
    inputs, outputs = data[['Adhesivity', 'Particle Size']], data['Termination time']
   
    # Split the data into training and testing sets
    # ---------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=0)

    # Create the model
    # ----------------
    rand_forest_model = RandomForestRegressor(n_estimators = 300,
                                              max_depth=depth,random_state=42)

    # Train the model
    # ---------------
    kf = KFold(n_splits=8, shuffle=True, random_state=42) # 8-fold cross-validation the shuffle 
    # parameter is set to True to shuffle the data before splitting it (default is False)
    # Perform cross-validation
    r2_scores = cross_val_score(rand_forest_model, inputs, outputs, cv=8, scoring='r2')

    # Calculate the average scores

    average_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)

    print(f"Average R2: {average_r2}")
    print(f"Standard deviation of R2: {std_r2}")


    # Predict the values
    # -------------------
    #y_pred = rand_forest_model.predict(X_test)
    #mse = mean_squared_error(y_test, y_pred)
    #r2 = r2_score(y_test, y_pred)
    #print(f"Mean squared error: {mse}")
    #print(f"R2 score for test: {r2}")

if __name__ == '__main__':
    scores_dict = {}
    depths = []
    main(50)

