import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV
folderpath=Path(r'Dataanalyse\model')

def apply_scaler(X_tr, Y_tr, X_te, Y_te):
    # Scalers for X and Y, fit on training data
    X_sc = StandardScaler().fit(X_tr)
    Y_sc = StandardScaler().fit(Y_tr)

    # Transform training data
    X_tr_scaled = X_sc.transform(X_tr)
    Y_tr_scaled = Y_sc.transform(Y_tr)

    # Transform test data
    X_te_scaled = X_sc.transform(X_te)
    Y_te_scaled = Y_sc.transform(Y_te)
    return X_tr_scaled, Y_tr_scaled, X_te_scaled, Y_te_scaled, X_sc, Y_sc

def select( X_tr, X_te, Y_tr, Y_te ):
    X_tr1= X_tr.head(800)
    X_te1= X_te.head(200)
    Y_tr1 = Y_tr.head(800)
    Y_te1 = Y_te.head(200)
    return X_tr1, X_te1,Y_tr1,Y_te1
def test_model_matrix_random_search(X, Y, random_state=None, n_iter=100):
    """
    Perform random search on degree and alpha of Kernel Ridge.

    Args:
        X (np.ndarray): Training features.
        Y (np.ndarray): Target features.
        random_state (int, optional): Random seed. Defaults to None.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 100.

    Returns:
        pd.DataFrame: Random search results.
    """

    # K-Folds cross-validator
    k = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Train test split
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=random_state
    )
    X_tr, X_te, Y_tr, Y_te  =  select( X_tr, X_te, Y_tr, Y_te )
    # scale features
    x, y, x_test, y_test, X_sc, Y_sc = apply_scaler(X_tr, Y_tr, X_te, Y_te)
    #print(f"Saving scalers to {folderpath}...")
    pickle.dump(X_sc, open(folderpath.joinpath(f"X_sc.p"), "wb"))
    pickle.dump(Y_sc, open(folderpath.joinpath(f"Y_sc.p"), "wb"))

    # Define the model
    model = KernelRidge(kernel="poly")

    # Define the parameter distributions
    param_distributions = {
        'degree': [1,2,3, 4, 5],  # Assuming you want to test degrees 1 to 5
        'alpha': np.random.uniform(low=0.05, high=1, size=100),  # Continuous distribution from 0.05 to 1

    }

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model, 
        param_distributions, 
        n_iter=n_iter, 
        cv=k, 
        random_state=random_state, 
        n_jobs=-1,
        refit=True,
        scoring='r2'

    )

    # Fit RandomizedSearchCV
    random_search.fit(x, y.ravel())

    # Getting the best model
    best_model = random_search.best_estimator_

    # Test best model on the test set
    test_score = best_model.score(x_test, y_test.ravel())

    # get the best parameter of the model
    best_params = random_search.best_params_

    # return the results in the form of [random_state, alpha, degree, test_score, best_model]

    results=[random_state, best_params['alpha'], best_params['degree'], test_score, best_model]
    print(results[0:4])

    return results



if __name__ == "__main__":

    data = pd.read_csv(r'Dataanalyse\axis2_demo_tablepart1_0.csv')

    #data = data.head(1000)
    cols = ['Mittelwert_x', 'Mittelwert_y', 'Mittelwert_z',
            'Variance_x', 'Variance_y', 'Variance_z', 'Effektivwert_x',
            'Effektivwert_y', 'Effektivwert_z', 'Standardabweichung_x',
            'Standardabweichung_y', 'Standardabweichung_z', 'Woehlbung_x',
            'Woehlbung_y', 'Woehlbung_z', 'Schiefe_x', 'Schiefe_y', 'Schiefe_z',
            'Mittlere_Absolute_Abweichung_x', 'Mittlere_Absolute_Abweichung_y',
            'Mittlere_Absolute_Abweichung_z', 'Zentrales_Moment_x',
            'Zentrales_Moment_y', 'Zentrales_Moment_z', 'Median_x', 'Median_y',
            'Median_z',]
    X=data[cols]
    Y=pd.DataFrame(data['Schadensklasse'])

    # loop over multiple random states
    states = np.random.randint(1000, size=100)
    results = []
    newresults=pd.DataFrame

    for random_state in states:
        # for each random state perform grid search on a kernel ridge model
        # and return the result
        result = test_model_matrix_random_search(X, Y, random_state=random_state)
        results.append(result)

    # get all the results and save it as Dataframe
    newresults=pd.DataFrame(results)
    print("Random states used:", states)

    # get the max value in 'test_score' column and find the row which contain the best test_score
    max_value=newresults.iloc[:,-2].max()
    best_result=newresults[newresults.iloc[:,-2]== max_value]
    best_result_list = best_result.values.tolist()[0]
    best_result_model = best_result_list[-1]

    name = (
                f"{best_result_list[-2]:.4f}_d{best_result_list[2]}-a{best_result_list[1]:.2f}-rs{best_result_list[0]}"
                )
    # save the model with the best result
    folderpath=Path(r'Dataanalyse\model')
    print(f"Saving model to {folderpath}...")
    pickle.dump(
        best_result_model, open(folderpath.joinpath(f"{name}.p"), "wb")
    )
    print("best_para:", best_result_list)




    fig, ax = plt.subplots(figsize=[8, 12])
    ax=plt.scatter(x=newresults.iloc[:,2], y= newresults.iloc[:,3], alpha=1)    
    plt.xlabel("degree")
    plt.ylabel("score")
    plt.savefig(folderpath.parent.joinpath("performance.png"), dpi=150)

