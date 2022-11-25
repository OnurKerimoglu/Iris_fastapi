from sklearn import datasets
from sklearn.model_selection import train_test_split


def get_data(datasetname):
    if datasetname == 'iris':
        iris = datasets.load_iris()
        X = iris.data[:, :]  # we take all features
        y = iris.target
    else:
        raise(ValueError(f'Unknown dataset name: {datasetname}'))

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42)

    return X_train, X_test, y_train, y_test
