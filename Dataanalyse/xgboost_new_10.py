import dask.dataframe as dd
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from joblib import dump
import numpy as np
import time
import torch
import pandas as pd
import json

# Load the data in chunks

path='../THESIS/Dataanalyse/all_data_axis2.csv'
df = dd.read_csv(path)

#open feature_important_axis1.json
with open('../Thesis/syntheic_data/feature_important_axis2.json') as f:
    feature_important_axis1 = json.load(f)

features=feature_important_axis1

# Define the XGBoost classifier
classifier = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='mlogloss', 
    num_class=3,

    tree_method='hist',  # Use histogram-based algorithm
    device='cuda'  # Enable GPU acceleration   
    
    )

# Define the parameter grid
param_grid = {
    "n_estimators": np.arange(500, 1600, 100),
    "max_depth": np.arange(10, 150),
    "learning_rate": np.arange(0.2, 0.6, 0.01),
    "subsample": np.arange(0.5, 1.01, 0.05),
    "colsample_bytree": np.arange(0.5, 1.01, 0.05),
    "gamma": np.arange(0, 5),
}

# Initialize RandomizedSearchCV with Dask
model = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_grid,
    scoring="accuracy",
    n_jobs=1,
    cv=5,
    n_iter=100
)
start_time = time.time()

# Train the model using Dask
X = df[features]
y = df['Schadensklasse']
model.fit(X, y)

end_time = time.time()
# Output the results
training_time = end_time - start_time
print(f"Estimated training time: {training_time:.2f} seconds")

# Output the results
print("Best parameters set:")
print(f"Best score: {model.best_score_}")

best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print(f"\t{param_name}: {best_parameters[param_name]}")
# save the best parameters in a file
with open('best_10_parameters_axis2.txt', 'w') as f:
    f.write(f"Best score: {model.best_score_}\n")
    for param_name in sorted(param_grid.keys()):
        f.write(f"\t{param_name}: {best_parameters[param_name]}\n")
# Get the top 10 important features
importances = model.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [(features[i], importances[i]) for i in indices]


# Save the best model
dump(model.best_estimator_, 'best_xgboost_model_10_axis2.joblib')
# Load the best model and plot the important features
