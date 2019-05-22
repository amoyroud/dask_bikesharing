
import dask.dataframe as dd
from sklearn.decomposition import PCA
from dask_ml.decomposition import PCA as daskpca
from utils import get_train, get_holdout


def pca(data):
    train = get_train(data).drop("cnt", axis=1)
    test = get_holdout(data).drop("cnt", axis=1)

    pca = daskpca(n_components=0.95, svd_solver="full").fit(train)

    print("\tPerforming PCA dimensionality reduction...")

    pca_train = dd.DataFrame(data=pca.transform(train))
    pca_test = dd.DataFrame(data=pca.transform(test))

    new_df = pca_train.append(pca_test)
    new_df["cnt"] = data["cnt"]

    return new_df
