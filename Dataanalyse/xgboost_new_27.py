import dask.dataframe as dd
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score
from joblib import dump
import numpy as np
import time

# Load the data in chunks
path='../THESIS/Dataanalyse/all_data_axis1.csv'
df = dd.read_csv(path)


cols = ['Mittelwert_x', 'Mittelwert_y', 'Mittelwert_z',
        'Variance_x', 'Variance_y', 'Variance_z', 'Effektivwert_x',
        'Effektivwert_y', 'Effektivwert_z', 'Standardabweichung_x',
        'Standardabweichung_y', 'Standardabweichung_z', 'Woehlbung_x',
        'Woehlbung_y', 'Woehlbung_z', 'Schiefe_x', 'Schiefe_y', 'Schiefe_z',
        'Mittlere_Absolute_Abweichung_x', 'Mittlere_Absolute_Abweichung_y',
        'Mittlere_Absolute_Abweichung_z', 'Zentrales_Moment_x',
        'Zentrales_Moment_y', 'Zentrales_Moment_z', 'Median_x', 'Median_y',
        'Median_z']
features = [f for f in df.columns if f not in ['Schadensklasse', 'Timestamp', 'ID','Date']]

# Define the XGBoost classifier
classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', num_class=3)

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
    n_iter=10
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
with open('best_parameters_axis1.txt', 'w') as f:
    f.write(f"Best score: {model.best_score_}\n")
    for param_name in sorted(param_grid.keys()):
        f.write(f"\t{param_name}: {best_parameters[param_name]}\n")
# Get the top 10 important features
importances = model.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [(features[i], importances[i]) for i in indices]

with open('best_parameters_axis1.txt', 'a') as f:
    f.write("Top 10 important features:\n")
    for feature, importance in top_features:
        f.write(f"{feature}: {importance:.4f}\n")

# Save the best model
dump(model.best_estimator_, 'best_xgboost_modeltestaxis1.joblib')
# Load the best model and plot the important features
