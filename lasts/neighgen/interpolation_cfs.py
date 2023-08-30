from lasts.base import NeighborhoodGenerator
import numpy as np
from lasts.neighgen.utils import linear_interpolation


class InterpolationCfs(NeighborhoodGenerator):
    def __init__(self, blackbox, labels=None, unpicklable_variables=None):
        self.blackbox = blackbox
        self.labels = labels
        self.unpicklable_variables = unpicklable_variables


def interpolation_cfs(x, x_label, X, blackbox, interpolation_f, n_interpolations=1000):
    counterexemplars = list()
    for i in range(len(X)):
        x_prime = X[i : i + 1].ravel()
        interpolations = interpolation_f(x.ravel(), x_prime, n_interpolations)
        y = blackbox.predict(interpolations)
        counterexemplars.append(interpolations[y != x_label][-1].ravel())
    return np.array(counterexemplars)[:, :, np.newaxis]


def linear_interpolations(a, b, n=100):
    interpolations = list()
    for t in range(1, n):
        interpolations.append(linear_interpolation(a, b, t / n))
    return np.array(interpolations)[:, :, np.newaxis]


if __name__ == "__main__":
    from lasts.blackboxes.loader import cached_blackbox_loader
    from lasts.datasets.datasets import build_cbf
    import matplotlib.pyplot as plt

    random_state = 0
    np.random.seed(random_state)
    dataset_name = "cbf"

    (
        _,
        _,
        _,
        _,
        _,
        _,
        X_exp_train,
        y_exp_train,
        X_exp_val,
        y_exp_val,
        X_exp_test,
        y_exp_test,
    ) = build_cbf(n_samples=600, random_state=random_state)

    blackbox = cached_blackbox_loader("cbf_knn.joblib")
    x = X_exp_train[0:1]
    X = X_exp_train[1:2]
    cfs = interpolation_cfs(
        x, blackbox.predict(x)[0], X_exp_train[1:3], blackbox, linear_interpolations
    )

    plt.plot(x.ravel())
    plt.show()
    for cf in cfs:
        plt.plot(cf.ravel())
    plt.show()
