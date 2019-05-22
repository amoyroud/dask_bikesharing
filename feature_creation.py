import featuretools as ft
import dask.dataframe as dd
import dask.array as da
import dask_ml.cluster
from dask_ml.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from utils import get_train, get_holdout, PATH, SEED
from dask_ml.model_selection import GridSearchCV


def categorize_time(r):
    """
    Helper function for 'commute_hours'

    :param r: a row of the data frame
    :return: Boolean, True if hour is during commute hours else flase
    """
    return ((r["hr"] >= 5 and r["hr"] < 10) or
           (r["hr"] >= 16 and r["hr"] < 20)) and \
            r["workingday"]


def commute_hours(data):
    """
    Creates a column that declares whether the hour is during commute hours

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe containing the new column
    """
    print("\tAdding variable for Commute Hours, 1 for yes and 0 for false")
    data = data.copy()
    data["commute_hours"] = data.apply(categorize_time, axis=1)
    return data


def cluster_variable(data):
    """
    Creates a column that gives a cluster id based on KMeans clustering of all features

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe containing the new column
    """
    print("\tAdding cluster variable...")
    data = data.copy()
    to_cluster = dd.get_dummies(data)
    train = get_train(to_cluster)
    holdout = get_holdout(to_cluster)

    kmeans = KMeans(n_clusters=5, random_state=SEED).fit(train.drop("cnt", axis=1)) # magic numbers, blech

    data["cluster"] = da.append(kmeans.labels_, kmeans.predict(holdout.drop("cnt", axis=1)))

    data["cluster"] = data["cluster"].astype("category")

    return data


def weather_cluster(data):
    """
    Creates a column that gives a cluster id based on KMeans clustering of only weather-related features

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe containing the new column
    """
    print("\tAdding clustering variable based on weather-related features...")
    df = data.copy()[["weathersit", "temp", "atemp", "hum", "windspeed"]]
    to_cluster = dd.get_dummies(df)
    train = get_train(to_cluster)
    holdout = get_holdout(to_cluster)

    kmeans = KMeans(n_clusters=5, random_state=SEED).fit(train)  # magic numbers, blech

    data["weather_cluster"] = da.append(kmeans.labels_, kmeans.predict(holdout))

    data["weather_cluster"] = data["weather_cluster"].astype("category")

    return data


def deep_features(data):
    """
    Performs deep feature synthesis on the dataframe to generate new columns based on different groups

    Currently, the only group is by season.

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe containing the new column
    """
    print("\tPerforming Deep Feature Synthesis...")
    df = data.copy().reset_index()
    count = df["cnt"]
    df = df.drop("cnt", axis=1)

    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id="bikeshare_hourly",
                                  index="index",
                                 dataframe=df,
                                 already_sorted=True)

    es = es.normalize_entity(base_entity_id="bikeshare_hourly",
                             new_entity_id="seasons",
                             index="season")

    f_mtx, f_defs = ft.dfs(entityset=es,
                           target_entity="bikeshare_hourly",
                           agg_primitives=["std", "max", "min", "mean"]) # ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "n_unique", "mode"]

    extra_features = f_mtx.iloc[:, len(df.columns):len(f_mtx)]
    for f in extra_features.columns:
        print("\t\tCreated feature {}".format(f))

    f_mtx["cnt"] = count

    return f_mtx


def subcount_forecast(data, feature):
    """
    Creates a new a column that is the predicted value of the input feature

    Essentially an abstraction for 'prediction_forecasts'

    :param data: a pandas dataframe where each row is an hour
    :param feature: a String containing the feature that should be forecasted (one of: casual, registered)
    :return: a pandas dataframe containing the new column
    """
    var_name = feature + "_forecast"
    print("\tAdding {} variable...".format(var_name))
    df = dd.get_dummies(data.copy().drop("cnt", axis=1))
    to_predict = dd.read_csv(PATH)[feature]
    df[feature] = to_predict

    train = get_train(df)

    model = RandomForestRegressor(random_state=SEED)
    model_params = {"n_estimators": list(range(10, 110, 10))}
    #tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(estimator=model, param_grid=model_params, scoring="r2", cv=None, refit=True)
    grid_search.fit(train.drop(feature, axis=1), train[feature])
    print("\t\tPredictions for GridSearchCV on {}: {:.5f} +/- {:.5f}"
          .format(feature,
                  grid_search.best_score_,
                  grid_search.cv_results_["std_test_score"][da.argmax(grid_search.cv_results_["mean_test_score"])]))

    data[var_name] = grid_search.best_estimator_.predict(dd.get_dummies(data.drop("cnt", axis=1)))

    return data


def prediction_forecasts(data):
    """
    Creates two new columns, one that is predictions for the 'casual' column and the other for the 'registered' column

    We wouldn't actually have the registered or casual variables at the time of predicting the "cnt" variable, but we
    do have all the other information and we have past information for registered and count. So the theory here is that
    we can use predictions for these values that would be accurate enough to be extremely helpful as features for a
    model

    :param data: a pandas dataframe where each row is an hour
    :return: a pandas dataframe containing the new columns
    """

    data = data.copy()
    casual = subcount_forecast(data, "casual")["casual_forecast"]
    registered = subcount_forecast(data, "registered")["registered_forecast"]

    data["casual_forecast"] = casual
    data["registered_forecast"] = registered

    return data
