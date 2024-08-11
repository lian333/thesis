import dask.dataframe as dd
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from joblib import dump
import numpy as np
import time
import pandas as pd
# Load the data in chunks
path=r'D:\studydata\Masterarbeit\lian333\Thesis\Dataanalyse\all_data_axis1.csv'
df = dd.read_csv(path)
df=pd.DataFrame(df.compute())
#open feature_important_axis1.json
import json
with open(r'D:\studydata\Masterarbeit\lian333\Thesis\syntheic_data\feature_important_axis1.json') as f:
    feature_important_axis1 = json.load(f)

features=feature_important_axis1

#features = [f for f in df.columns if f not in ['Schadensklasse', 'Timestamp', 'ID','Date']]

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
    "n_estimators": np.arange(100, 1000, 100),
    "max_depth": np.arange(10, 100),
    "learning_rate": np.arange(0.1, 0.5, 0.01),
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
with open('best_10parameters_axis1.txt', 'w') as f:
    f.write(f"Best score: {model.best_score_}\n")
    for param_name in sorted(param_grid.keys()):
        f.write(f"\t{param_name}: {best_parameters[param_name]}\n")
# Get the top 10 important features
importances = model.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [(features[i], importances[i]) for i in indices]


# Save the best model
dump(model.best_estimator_, '10_best_xgboost_modeltestaxis1.joblib')
# Load the best model and plot the important features
