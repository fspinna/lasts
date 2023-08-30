import glob
import math
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, norm
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pathlib
from collections import defaultdict


def coverage_score_scikit_tree(dt: DecisionTreeClassifier, leaf_id):
    """
    Number of records classified in a leaf w.r.t. total number of records
    Parameters
    ----------
    dt
    leaf_id

    Returns
    -------

    """
    n = dt.tree_.value[0].sum()
    n_lhs = dt.tree_.value[leaf_id].sum()
    return n_lhs / n


def precision_score_scikit_tree(dt: DecisionTreeClassifier, X, y, leaf_id):
    """
    Number of records that are correctly classified in a leaf w.r.t number of records in that leaf
    Parameters
    ----------
    dt
    X
    y
    leaf_id

    Returns
    -------

    """
    y_surrogate = dt.predict(X)
    X_leave_ids = dt.apply(X)
    idxs = np.argwhere(
        X_leave_ids == leaf_id
    )  # indexes of records of X in the leaf leaf_id
    n_y = (
        y[idxs] == y_surrogate[idxs]
    ).sum()  # n of records of the leaf leaf_id that are correctly classified
    n_lhs = len(idxs)
    return n_y / n_lhs


def make_path(folder):
    path = pathlib.Path(folder)
    pathlib.Path(path).mkdir(exist_ok=True, parents=True)
    return path


def get_project_root():
    return pathlib.Path(__file__).parent


def vector_to_dict(x, feature_names):
    return {k: v for k, v in zip(feature_names, x)}


def compute_medoid(X):
    distance_matrix = pairwise_distances(X, n_jobs=-1)
    medoid_idx = np.argmin(distance_matrix.sum(axis=0))
    return X[medoid_idx]


def convert_list_to_sktime(X):
    df_dict = defaultdict(list)
    for i, dim in enumerate(X):
        for ts in dim:
            df_dict[i].append(pd.Series(ts.ravel()))
    return pd.DataFrame(df_dict)


def convert_numpy_to_sktime(X):
    df_dict = dict()
    for ts in X:
        for dimension in range(X.shape[2]):
            if dimension in df_dict.keys():
                df_dict[dimension].append(pd.Series(ts[:, dimension]))
            else:
                df_dict[dimension] = [pd.Series(ts[:, dimension])]
    df = pd.DataFrame(df_dict)
    return df


def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))


def explanation_error(true_importance, pred_importance):
    return np.abs(true_importance - pred_importance).sum() / len(true_importance)


def baseline_error(true_importance):
    ones = np.ones_like(true_importance)
    zeros = np.zeros_like(true_importance)
    baseline = min(
        explanation_error(true_importance, ones),
        explanation_error(true_importance, zeros),
    )
    # if baseline == 0:
    #     baseline = 1
    return baseline


def convert_sktime_to_numpy(X):
    np_tss = []
    for ts in X.iloc:
        np_ts = []
        for dimension in range(len(X.columns)):
            np_ts.append(np.array(ts[dimension]).reshape(1, -1, 1))
        np_ts = np.concatenate(np_ts, axis=2)
        np_tss.append(np_ts)
    np_tss = np.concatenate(np_tss)
    return np_tss


def sliding_window_distance(ts, s):
    distances = []
    for i in range(len(ts) - len(s) + 1):
        ts_s = ts[i : i + len(s)]
        dist = np.linalg.norm(s - ts_s)
        distances.append(dist)
    return np.argmin(distances)


def sliding_window_euclidean(ts, s):
    distances = []
    for i in range(len(ts) - len(s) + 1):
        ts_s = ts[i : i + len(s)]
        dist = np.linalg.norm(s - ts_s)
        distances.append(dist)
    return np.min(distances)


def choose_z(
    x,
    encoder,
    decoder,
    n=1000,
    x_label=None,
    blackbox=None,
    check_label=False,
    verbose=False,
    mse=False,
):
    X = np.repeat(x, n, axis=0)
    Z = encoder.predict(X)
    Z_tilde = decoder.predict(Z)
    if check_label:
        y_tilde = blackbox.predict(Z_tilde)
        y_correct = np.nonzero(y_tilde == x_label)
        if len(Z_tilde[y_correct]) == 0:
            if verbose:
                warnings.warn("No instances with the same label of x found.")
        else:
            Z_tilde = Z_tilde[y_correct]
            Z = Z[y_correct]
    if mse:
        distances = []
        for z_tilde in Z_tilde:
            distances.append(((x - z_tilde) ** 2).sum())
        distances = np.array(distances)
    else:
        # distances = cdist(x[:, :, 0], Z_tilde[:, :, 0]) # does not work for multi ts
        distances = cdist(
            x.reshape(-1, x.shape[1] * x.shape[2]),
            Z_tilde.reshape(-1, Z_tilde.shape[1] * Z_tilde.shape[2]),
        )
    best_z = Z[np.argmin(distances)]
    return best_z.reshape(1, -1)


def plot_choose_z(x, encoder, decoder, n=100, K=None):
    Z = []
    for i in range(n):
        Z.append(encoder.predict(x).ravel())
    Z = np.array(Z)
    Z_tilde = decoder.predict(Z)
    distances = cdist(x[:, :, 0], Z_tilde[:, :, 0]).ravel()
    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=distances,
        cmap="Greens_r",
        norm=matplotlib.colors.PowerNorm(gamma=0.1),
    )
    if K is not None:
        plt.scatter(K[:, 0], K[:, 1], c="lightgray")
    plt.show()


def euclidean_norm(Z):
    Z_norm = list()
    for z in Z:
        Z_norm.append(np.linalg.norm(z))
    return np.array(Z_norm)


def norm_distance(Z, distance=wasserstein_distance):
    """Compute the distance between the euclidean norms of the instances in Z
    and instances extracted from a gaussian distribution

    Parameters
    ----------
    Z: array of shape (n_samples, n_features)
        datasets
    distance: string, optional (default=wasserstein_distance)
        type of distance
    Returns
    -------
    distance: int
        distance between the euclidean norms
    """

    Z_norm = euclidean_norm(Z)
    rnd_norm = euclidean_norm(np.random.normal(size=Z.shape))
    return distance(Z_norm, rnd_norm)


def plot_norm_distributions(norm_array_list, labels=None):
    norm_array_df = pd.DataFrame(norm_array_list).T
    norm_array_df.columns = labels
    for column in norm_array_df.columns:
        sns.kdeplot(norm_array_df[column])
    plt.show()


def reconstruction_accuracy(X, encoder, decoder, blackbox, repeat=1, verbose=True):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        y_tilde = blackbox.predict(decoder.predict(encoder.predict(X)))
        accuracy = accuracy_score(y, y_tilde)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


def reconstruction_accuracy_vae2(
    X, encoder, decoder, blackbox, repeat=1, n=100, check_label=True, verbose=True
):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        Xs = np.repeat(X, n, axis=2)
        # Z = list()
        Z_tilde = list()
        for n_ in range(Xs.shape[0]):
            X_ = Xs[n_, :, :].T[:, :, np.newaxis]
            Z_ = encoder.predict(X_)
            Z_tilde_ = decoder.predict(Z_)
            if check_label:
                y_tilde_ = blackbox.predict(Z_tilde_)
                y_correct = np.nonzero(y_tilde_ == y[n_])
                if len(Z_tilde_[y_correct]) == 0:
                    if verbose:
                        warnings.warn("No instances with the same label of x found.")
                else:
                    Z_tilde_ = Z_tilde_[y_correct]
                    # Z_ = Z_[y_correct]
            distances = cdist(X_[:, :, 0], Z_tilde_[:, :, 0])
            # best_z = Z_[np.argmin(distances)].ravel()
            best_z_tilde = Z_tilde_[np.argmin(distances)].ravel()
            # Z.append(best_z)
            Z_tilde.append(best_z_tilde)
        # Z = np.array(Z)
        Z_tilde = np.array(Z_tilde)[:, :, np.newaxis]
        accuracy = accuracy_score(y, blackbox.predict(Z_tilde))
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


def reconstruction_accuracy_vae(
    X, encoder, decoder, blackbox, repeat=1, n=100, check_label=True, verbose=True
):
    y = blackbox.predict(X)
    accuracies = []
    for i in range(repeat):
        Z = list()
        for x in X:
            if check_label:
                x_label = blackbox.predict(x[np.newaxis, :, :])
            else:
                x_label = None
            z = choose_z(
                x=x[np.newaxis, :, :],
                encoder=encoder,
                decoder=decoder,
                n=n,
                x_label=x_label,
                blackbox=blackbox,
                check_label=check_label,
                verbose=verbose,
            )
            Z.append(z.ravel())
        Z = np.array(Z)
        y_tilde = blackbox.predict(decoder.predict(Z))
        accuracy = accuracy_score(y, y_tilde)
        accuracies.append(accuracy)
    accuracies = np.array(accuracies)
    accuracies_mean = accuracies.ravel().mean()
    accuracies_std = np.std(accuracies.ravel())
    if verbose:
        print("Accuracy:", accuracies_mean, "±", accuracies_std)
    return accuracies_mean


def exemplars_and_counterexemplars_similarities(x, x_label, X, y):
    """Compute similarities between an instance and exemplars and counterexemplars
    Parameters
    ----------
    x : {array-like}
        Instance. Shape [n_samples, n_features].
    x_label : {integer}
        Instance label.
    X : {array-like}
        Data. Shape [n_samples, n_features].
    y : {array-like}
        Data labels of shape [n_samples]
    Returns
    -------
    s_exemplars : similarities from the exemplars
    s_counterexemplars : similarities from the counterexemplars
    """
    x = x.ravel().reshape(1, -1)
    exemplar_idxs = np.argwhere(y == x_label).ravel()
    exemplars = X[exemplar_idxs]
    counterexemplar_idxs = np.argwhere(y != x_label).ravel()
    counterexemplars = X[counterexemplar_idxs]
    s_exemplars = 1 / (1 + cdist(exemplars, x))
    s_counterexemplars = 1 / (1 + cdist(counterexemplars, x))

    return s_exemplars.ravel(), s_counterexemplars.ravel()


def triangle_distribution(size, **kwargs):
    coordinates = np.array([[1, 0], [-1, 0], [0, math.sqrt(3)]])
    idxs = np.random.randint(0, 3, size=(size[0],))
    return coordinates[idxs]


def plot_reconstruction(X, encoder, decoder, figsize=(20, 15), n=0):
    X_tilde = decoder.predict(encoder.predict(X))
    g = 1
    plt.figure(figsize=figsize)
    for i in range(n, n + 5):
        # display original
        ax = plt.subplot(5, 1, g)
        g += 1
        plt.plot(X[i], label="real")
        plt.plot(X_tilde[i], label="reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()


def plot_reconstruction_vae(X, encoder, decoder, figsize=(20, 15), n=0):
    Z = list()
    for x in X:
        z = choose_z(x[np.newaxis, :, :], encoder, decoder)
        Z.append(z.ravel())
    Z = np.array(Z)
    X_tilde = decoder.predict(Z)
    g = 1
    plt.figure(figsize=figsize)
    for i in range(n, n + 5):
        # display original
        ax = plt.subplot(5, 1, g)
        g += 1
        plt.plot(X[i], label="real")
        plt.plot(X_tilde[i], label="reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.legend()
    plt.show()


def plot_choose_z_latent_space(X, encoder, decoder, blackbox, n=1000, figsize=(20, 15)):
    Z = list()
    for x in X:
        x_label = blackbox.predict(x[np.newaxis, :, :])[0]
        z = choose_z(x[np.newaxis, :, :], encoder, decoder, n, x_label, blackbox)
        Z.append(z.ravel())
    Z = np.array(Z)
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.show()


def plot_labeled_latent_space_matrix(Z, y, **kwargs):
    Z = list(Z)
    Z = pd.DataFrame(Z)
    pd.plotting.scatter_matrix(
        Z,
        c=y,
        cmap="viridis",
        diagonal="kde",
        alpha=1,
        s=100,
        figsize=kwargs.get("figsize", (8, 8)),
    )


def probability_density_mean(Z):
    return norm.pdf(Z).mean()


def means_from_df_list(df_list, labels=None):
    df_means = pd.DataFrame()
    for df in df_list:
        df_means = df_means.append(df.mean(), ignore_index=True)
    if labels is not None:
        df_means.index = labels
    return df_means


def read_metrics_csv(folder, **kwargs):
    names = sorted(
        [
            os.path.basename(filename.replace(".csv", ""))
            for filename in glob.glob(folder + "*.csv")
        ]
    )
    df_list = list()
    for name in names:
        df = pd.read_csv(folder + name + ".csv", sep=";")
        df_list.append(df)
    df_means = means_from_df_list(df_list, labels=names)
    return df_list, df_means, names


def usefulness_scores(X, y, x, x_label, n=[1, 2, 4, 8, 16]):
    """Compute the knn prediction for x using n exemplars and counterexemplars
    Parameters
    ----------
    x_label
    X : array of shape (n_samples, n_features)
        datasets
    y : array of shape (n_samples,)
        datasets labels
    x : array of shape (n_features,)
        instance to benchmark
    n : list, optional (default = [1,2,4,8,16])
        number of instances to extract from every class
    Returns
    -------
    accuracy_by_n : array of shape (len(n),)
        prediction accuracy for x, for every n
    """
    x = x.ravel()
    dfs = dict()
    for key in n:
        dfs[key] = {"X": [], "y": []}
    for unique_label in np.unique(y):
        same_label_idxs = np.argwhere(y == unique_label).ravel()
        same_label_records = X[same_label_idxs]
        for n_record in n:
            random_idxs = np.random.choice(
                same_label_records.shape[0],
                min(n_record, same_label_records.shape[0]),
                replace=False,
            ).ravel()
            dfs[n_record]["X"].extend(same_label_records[random_idxs])
            dfs[n_record]["y"].extend(np.repeat(unique_label, len(random_idxs)))
    for key in dfs.keys():
        dfs[key]["X"] = np.array(dfs[key]["X"])
        dfs[key]["y"] = np.array(dfs[key]["y"])
    accuracy_by_n = dict()
    for key in dfs.keys():
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(dfs[key]["X"], dfs[key]["y"])
        # y_by_n.append(knn.predict(x.ravel().reshape(1, -1))[0])
        accuracy_by_n[key] = knn.score(x.ravel().reshape(1, -1), [x_label])
    return accuracy_by_n


def numpy_to_list_multi(array):
    df_multi = list()
    for dim in range(array.shape[2]):
        df_multi.append(array[:, :, dim : dim + 1])
    return df_multi


def is_np_array(array):
    return isinstance(array, np.ndarray)


def format_multivariate_input(array):
    if is_np_array(array):
        return numpy_to_list_multi(array)
    else:
        return array


def find_linear_trend(ts):
    p = np.polyfit(x=np.arange(len(ts)), y=ts, deg=1)
    trend = (np.arange(len(ts)) * p[0]) + p[1]
    return trend, p[0], p[1]


def minmax_scale(X, minimum=None, maximum=None):
    if X.min() == X.max():
        return np.zeros_like(X)
    if minimum is None:
        return (X - X.min()) / (X.max() - X.min())
    else:
        return (X - minimum) / (maximum - minimum)


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out
