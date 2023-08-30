import numpy as np
import itertools
from tqdm import tqdm


def place_ones(
    size, count
):  # can be improved by generating the permutations of only certain features
    for positions in itertools.combinations(range(size), count):
        p = [0] * size
        for i in positions:
            p[i] = 1
        yield p


def constant_segmentation(ts_len, segments_size):
    change_points = list(np.arange(0, ts_len + 1)[segments_size::segments_size])
    if change_points[-1] < (ts_len):
        change_points.append(ts_len)
    return get_change_points_indexes(change_points)


def get_change_points_indexes(change_points):
    """From list of ending segment idxs to list of tuple with starting and ending idxs

    Parameters
    ----------
    segmentation

    Returns
    -------

    Examples
    --------
    >>> print(get_change_points_indexes([5,9,12]))
    [(0, 5), (5, 9), (9, 12)]
    """
    change_points_indexes = []
    if len(change_points) == 1:
        change_points_indexes.append((0, change_points[0]))
    for i in range(len(change_points) - 1):
        if i == 0:
            change_points_indexes.append((0, change_points[i]))
        change_points_indexes.append((change_points[i], change_points[i + 1]))
    return change_points_indexes


def constant_mapping(Z, x, change_points, constant_val=0):
    assert Z.shape[1] == len(change_points)
    X = list()
    for i, z in enumerate(Z):
        x_copy = x.copy().ravel()
        for val, change_point in zip(z, change_points):
            if not val:  # if the value is 0 the feature needs to be replaced
                x_copy[change_point[0] : change_point[1]] = constant_val
        X.append(x_copy)
    return np.array(X)[:, :, np.newaxis]


def sedc(x, blackbox, mapping_f, n_features):
    x_label = blackbox.predict(x)[0]
    cfs = list()
    columns_to_check = np.array([])
    used_features = None
    for i in tqdm(range(n_features)):
        permutations = 1 - np.array(
            list(place_ones(n_features, i + 1))
        )  # permutations without repetition
        if len(columns_to_check) > 0:  # if there are some features to ignore
            if used_features.sum() == 0:
                break
            permutations = permutations[
                (permutations[:, columns_to_check] != 0).prod(axis=1) == 1
            ]  # remove
            # permutations having 0 in the feature to ignore
        if len(permutations) == 0:  # if there are not permutations to check
            break
        Z = mapping_f(permutations)  # map permutations to data
        y = blackbox.predict(Z)  # predict labels of the perturbed dataset
        cf_idxs = np.argwhere(y != x_label).ravel()
        if len(cf_idxs) > 0:  # if there are counterfactuals
            cfs.append(Z[cf_idxs])
            used_features = permutations[cf_idxs].min(
                axis=0
            )  # mask for features to ignore in the future
            columns_to_check = np.argwhere(
                used_features == 0
            ).ravel()  # features to ignore in the future
    if len(cfs) > 0:
        return np.concatenate(cfs)
    else:
        return np.array(cfs)


def sedc_proba(x, predict_proba, mapping_f, n_features):
    x_label = np.argmax(predict_proba(x), axis=1)[0]
    x_proba = predict_proba(x)[0]
    counterfactuals = list()
    columns_to_ignore = np.array([])
    pbar = tqdm(range(n_features))
    for i in pbar:
        permutations = 1 - np.array(
            list(place_ones(n_features, i + 1))
        )  # permutations of i+1 0's without repetition
        if len(columns_to_ignore) > 0:  # if there are some features to ignore
            if (
                len(columns_to_ignore) == n_features
            ):  # if all columns need to be ignored
                break
            permutations = permutations[
                (permutations[:, columns_to_ignore] != 0).prod(axis=1) == 1
            ]  # remove
            # permutations having 0's in the features to ignore
        if len(permutations) == 0:  # if there are not permutations to check
            break
        pbar.set_description("{} permutations".format(len(permutations)))
        Z = mapping_f(permutations)  # map permutations to data
        y_proba = predict_proba(Z)  # predict labels of the perturbed dataset
        cf_idxs = np.argwhere(np.argmax(y_proba, axis=1) != x_label).ravel()
        pcf_idxs = np.argwhere(
            x_proba[x_label] > y_proba[:, x_label]
        ).ravel()  # idxs of records that are more
        # counterfactual than before
        pcf_idxs = np.array(
            [idx for idx in pcf_idxs if idx not in cf_idxs]
        )  # remove idxs that are already in cf_idxs
        if (
            len(pcf_idxs) > 0
        ):  # if there are some records that are becoming counterfactuals
            # in future cycles only perturbations involving these features are checked
            used_features = 1 - (permutations[pcf_idxs]).min(
                axis=0
            )  # find the features that i will check in
            # the future
            columns_to_ignore_pcf = np.argwhere(
                used_features == 0
            ).ravel()  # features to ignore in the future
            columns_to_ignore = columns_to_ignore_pcf
        if len(cf_idxs) > 0:  # if there are counterfactuals
            counterfactuals.append(Z[cf_idxs])
            used_features = permutations[cf_idxs].min(
                axis=0
            )  # mask for features to ignore in the future
            columns_to_ignore_cf = np.argwhere(
                used_features == 0
            ).ravel()  # features to ignore in the future
            columns_to_ignore = columns_to_ignore_cf
        if (len(pcf_idxs) > 0) and (len(cf_idxs) > 0):  # maybe not needed
            columns_to_ignore = np.array(
                list(set(np.concatenate([columns_to_ignore_pcf, columns_to_ignore_cf])))
            )
    if len(counterfactuals) > 0:
        return np.concatenate(counterfactuals)
    else:
        return np.array(counterfactuals)


def sedc_ts(x, blackbox, change_points, mapping_f, mapping_kwargs=dict(), **kwargs):
    def f(Z):
        X = mapping_f(Z, x, change_points, **mapping_kwargs)
        return X

    cfs = sedc_proba(x, blackbox.predict_proba, f, len(change_points))
    return cfs


if __name__ == "__main__":
    from lasts.blackboxes.loader import cached_blackbox_loader
    from lasts.datasets.datasets import build_cbf, load_ucr_dataset
    import matplotlib.pyplot as plt

    random_state = 0
    np.random.seed(random_state)
    dataset_name = "cbf"

    X_train, *_ = build_cbf(n_samples=600, random_state=random_state)

    blackbox = cached_blackbox_loader("cbf_cnn.h5")

    x = X_train[0:1]

    change_points = constant_segmentation(len(x.ravel()), 4)
    cfs = sedc_ts(x, blackbox, change_points=change_points, mapping_f=constant_mapping)
    plt.plot(x.ravel())
    plt.show()
    for cf in cfs:
        plt.plot(cf.ravel())
        plt.show()
