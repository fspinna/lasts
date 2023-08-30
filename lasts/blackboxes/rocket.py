import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket
from tensorflow.keras.utils import to_categorical
from lasts.utils import convert_numpy_to_sktime


class RocketWrapper(object):
    def __init__(self, random_state=None, **kwargs):
        self.clf = make_pipeline(
            Rocket(n_jobs=-1, random_state=random_state),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
            **kwargs
        )
        self._is_fitted = False

    def fit(self, X, y):
        X = convert_numpy_to_sktime(X)
        y = y.astype(str)
        self.clf.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        X = convert_numpy_to_sktime(X.astype(float))
        y_pred = self.clf.predict(X)
        return y_pred.astype(int)

    def predict_proba(self, X):
        return to_categorical(self.predict(X), num_classes=len(self.clf.classes_))

    def score(self, X, y):
        X = convert_numpy_to_sktime(X)
        y = y.astype(str)
        return self.clf.score(X, y)


if __name__ == "__main__":
    from lasts.datasets.datasets import load_uea_dataset

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    ) = load_uea_dataset("Libras")

    clf = RocketWrapper()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
