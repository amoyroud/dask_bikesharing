import dask.dataframe as dd
import dask.array as da
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import RFE

from utils import SEED, model_name, get_train, get_holdout


def tree_selection(data):
    """
    Uses the feature importances of a decision tree to select the most relevant features (importance > 0.1%)

    Conservative to ensure not too much information is lost.

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with the "irrelevant" columns removed
    """
    print("\tPerforming Tree selection, removing features with importance less than 0.1%")
    X = data.drop("cnt", axis=1)
    y = data["cnt"]

    rf = RandomForestRegressor(random_state=SEED)

    rf.fit(X, y)

    importances = rf.feature_importances_

    for i, u in enumerate(importances):
        if u < 0.001:
            print("\t\tDropping {0}, importance of only {1:.5f}%".format(X.columns[i], u * 100))

    to_select = X.loc[:, importances > 0.001]
    to_select["cnt"] = y

    return to_select


def r2_selection(data):
    """
    Perform forward stepwise feature selection to try and maximize R^2 with the least features possible

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with only those columns that increased the R^2 at each forward step
    """
    model = RandomForestRegressor(random_state=SEED)
    tscv = TimeSeriesSplit(n_splits=5)
    X = data.drop("cnt", axis=1)
    y = data["cnt"]

    print("\tPerforming forward stepwise feature selection based on a {0} and R2".format(model_name(model)))

    best_r2 = 0
    included = []
    excluded = list(X.columns)

    while True:
        to_check = []
        for feat in excluded:
            to_fit = included + [feat]
            scores = cross_val_score(model, X=X[to_fit], y=y, scoring="r2", cv=tscv)
            to_check.append(da.mean(scores))
        best_index = da.argmax(to_check)

        if to_check[best_index] > best_r2:
            included.append(excluded[best_index])
            excluded.pop(best_index)
            print("\t\tAdding feature {}, improved R2 from {:.5f} to {:.5f}"
                  .format(included[-1], best_r2, to_check[best_index]))
            best_r2 = to_check[best_index]
        else:
            print("Final features selected: {}".format(included))
            retval = X[included]
            retval["cnt"] = y
            return retval


def rfe(data):
    """
    Perform recursive feature elimination to retrieve the 50 most relevant features based on feature importances

    Differs from 'tree_selection' only in the recursive nature (tree_selection only fits a model once, RFE fits a model
    50 times)

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe with only the 50 most relevant features based on RFE
    """
    data = data.copy()

    model = RandomForestRegressor(random_state=SEED)
    fe = RFE(estimator=model, n_features_to_select=50)
    print("\tPerforming recursive feature elimination based on a Random Forest...")

    final_data = data.loc[:, fe.fit(data.drop("cnt", axis=1), data["cnt"]).support_]
    print("\t\tRemaining Features: {}".format(list(final_data.columns)))

    final_data["cnt"] = data["cnt"]

    return final_data
