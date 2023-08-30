import pathlib

from sklearn.preprocessing import LabelEncoder

from lasts.utils import get_project_root
import numpy as np


class Dataset(object):
    def __init__(self, X_train, y_train, X_test, y_test, random_state=0, name=""):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train, self.y_test, self.label_encoder = self._encode_labels(
            y_train, y_test
        )
        self.name = name
        self.random_state = random_state

    def __call__(self, *args, **kwargs):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def _encode_labels(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        return y_train, y_test, le

    def __str__(self):
        return (
            "Dataset Name: %s\nX_train: %s\nX_test: %s\ny_train: %s\ny_test: %s\nLabel Encoding: %s"
            % (
                self.name,
                self.X_train.shape,
                self.X_test.shape,
                self.y_train.shape,
                self.y_test.shape,
                self.label_encoder.classes_,
            )
        )


def dataset_list():
    root = get_project_root()
    path = (root / "datasets" / "cached").glob("**/*")
    return [x.name for x in path if x.is_dir()]


def load_raw_cached_dataset(path):
    X_train = np.load(path / "X_train.npy")
    X_test = np.load(path / "X_test.npy")
    y_train = np.load(path / "y_train.npy")
    y_test = np.load(path / "y_test.npy")
    return X_train, y_train, X_test, y_test


def dataset_loader_raw(name):
    root = get_project_root()
    path = root / "datasets" / "cached"
    X_train, y_train, X_test, y_test = load_raw_cached_dataset(path=path / name)
    return X_train, y_train, X_test, y_test


def dataset_loader(name, verbose=True):
    X_train, y_train, X_test, y_test = dataset_loader_raw(name)
    data = Dataset(X_train, y_train, X_test, y_test, name=name)
    if verbose:
        print(str(data))
    return data


if __name__ == "__main__":
    cbf = dataset_loader("ERing")
