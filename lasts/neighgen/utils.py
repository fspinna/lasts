import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from lasts.utils import euclidean_norm


def check_neighborhood_norm(
    size,
    kind,
    n=1000,
    n_single_neighborhood=1,
    threshold=1,
    distribution=np.random.normal,
    **kwargs
):
    Zs = list()
    for i in range(n):
        z = np.random.normal(size=size)
        Z = vicinity_sampling(
            z=z,
            n=n_single_neighborhood,
            threshold=threshold,
            kind=kind,
            distribution=distribution,
            verbose=False,
            **kwargs
        )
        Zs.append(Z)
    Zs = np.concatenate(Zs)
    return Zs, euclidean_norm(Zs)


def filter_neighborhood(
    Z, y, ratio=0.5, ignore_instance_after_first_match=False, inverse=False
):
    labels = np.unique(y)
    if len(labels) > 2:
        raise Exception("Dataset labels must be binarized.")
    idxs_a = np.argwhere(y == labels[0]).ravel()
    Z_a = Z[idxs_a]
    idxs_b = np.argwhere(y == labels[1]).ravel()
    Z_b = Z[idxs_b]
    distance_matrix = cdist(Z_a, Z_b)
    distance_dict = {"a": list(), "b": list(), "dist": list()}
    for row in range(distance_matrix.shape[0]):
        for column in range(distance_matrix.shape[1]):
            distance_dict["a"].append(idxs_a[row])
            distance_dict["b"].append(idxs_b[column])
            distance_dict["dist"].append(distance_matrix[row, column])
    df = pd.DataFrame(distance_dict)
    df_sorted = df.sort_values(["dist"], axis=0)
    idxs_to_filter = set()
    df_idx = 0
    while len(idxs_to_filter) / len(Z) < ratio:
        if ignore_instance_after_first_match:
            if (
                df_sorted.iloc[df_idx]["a"] in idxs_to_filter
                or df_sorted.iloc[df_idx]["b"] in idxs_to_filter
            ):
                df_idx += 1
                continue
        idxs_to_filter.add(df_sorted.iloc[df_idx]["a"])
        idxs_to_filter.add(df_sorted.iloc[df_idx]["b"])
        df_idx += 1
    idxs = set(range(Z.shape[0]))
    if inverse:
        idxs_to_keep = idxs_to_filter
    else:
        idxs_to_keep = idxs.difference(idxs_to_filter)
    idxs_to_keep = np.array(list(idxs_to_keep), dtype=np.int)
    Z_filtered = Z[idxs_to_keep]
    return Z_filtered


def test_generators():
    z = np.array([[1, 1]])
    for kind in [
        "gaussian_matched",
        "gaussian",
        "gaussian_global",
        "uniform_sphere",
        "by_rejection",
    ]:
        Z = vicinity_sampling(
            z, kind=kind, epsilon=1, r=1, distribution=np.random.normal, n=500
        )
        plt.scatter(Z[:, 0], Z[:, 1])
        plt.scatter(z[:, 0], z[:, 1], c="red")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()


def vicinity_sampling(
    z,
    n=1000,
    threshold=None,
    kind="gaussian_matched",
    distribution=None,
    distribution_kwargs=dict(),
    verbose=True,
    **kwargs
):
    if verbose:
        print("\nSampling -->", kind)
    if kind == "gaussian":
        Z = gaussian_vicinity_sampling(z, threshold, n)
    elif kind == "gaussian_matched":
        Z = gaussian_matched_vicinity_sampling(z, threshold, n)
    elif kind == "gaussian_global":
        Z = gaussian_global_sampling(z, n)
    elif kind == "uniform_sphere":
        Z = uniform_sphere_vicinity_sampling(z, n, threshold)
    elif kind == "uniform_sphere_scaled":
        Z = uniform_sphere_scaled_vicinity_sampling(z, n, threshold)
    elif kind == "by_rejection":
        Z = sample_by_rejection(
            distribution=distribution,
            center=z,
            r=threshold,
            distribution_kwargs=distribution_kwargs,
            n=n,
            verbose=verbose,
        )
    else:
        raise Exception("Vicinity sampling kind not valid")
    return Z


def gaussian_matched_vicinity_sampling(z, epsilon, n=1):
    return gaussian_vicinity_sampling(z, epsilon, n) / np.sqrt(1 + (epsilon**2))


def gaussian_vicinity_sampling(z, epsilon, n=1):
    return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)


def gaussian_global_sampling(z, n=1):
    return np.random.normal(size=(n, z.shape[1]))


def sample_by_rejection(
    distribution, center, r, distribution_kwargs=dict(), n=1000, verbose=True
):
    Z = []
    count = 0
    while len(Z) < n:
        Z_sample = distribution(size=(n, center.shape[1]), **distribution_kwargs)
        distances = cdist(center, Z_sample).ravel()
        Z.extend(Z_sample[np.nonzero((distances <= r).ravel())])
        count += 1
        if verbose:
            print(
                "   iteration", str(count) + ":", "found", len(Z), "samples", end="\r"
            )
    if verbose:
        print()
    Z = np.array(Z)
    Z = Z[np.random.choice(Z.shape[0], n, replace=False), :]
    return Z


def uniform_sphere_origin(n, d, r=1):
    """Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled
    by "radius" (length of points are in range [0, "radius"]).

    Parameters
    ----------
    n : int
        number of points to generate
    d : int
        dimensionality of each point
    r : float
        radius of the sphere

    Returns
    -------
    array of shape (n, d)
        sampled points
    """
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(d, n))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(n) ** (1 / d)
    # Return the list of random (direction & length) points.
    return r * (random_directions * random_radii).T


def uniform_sphere_vicinity_sampling(z, n=1, r=1):
    Z = uniform_sphere_origin(n, z.shape[1], r)
    translate(Z, z)
    return Z


def uniform_sphere_scaled_vicinity_sampling(z, n=1, threshold=1):
    Z = uniform_sphere_origin(n, z.shape[1], r=1)
    Z *= threshold
    translate(Z, z)
    return Z


def translate(X, center):
    """Translates a origin centered array to a new center

    Parameters
    ----------
    X : array
        data to translate centered in the axis origin
    center : array
        new center point

    Returns
    -------
    None
    """
    for axis in range(center.shape[-1]):
        X[..., axis] += center[..., axis]


def spherical_interpolation(a, b, t):
    if t <= 0:
        return a
    elif t >= 1:
        return b
    elif np.allclose(a, b):
        return a
    omega = np.arccos(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * a + np.sin(t * omega) / so * b


def linear_interpolation(a, b, t):
    return (t * a) + ((1 - t) * b)


def gaussian_matched_interpolation(a, b, t):
    return linear_interpolation(a, b, t) / np.sqrt((t**2) + (1 - t) ** 2)


def interpolate(a, b, kind="linear", n=10):
    a = a.ravel()
    b = b.ravel()
    interpolation_matrix = list()
    for t in np.arange(1 / n, 1, 1 / n):
        if kind == "linear":
            interpolation_vector = linear_interpolation(a, b, t)
        elif kind == "gaussian_matched":
            interpolation_vector = gaussian_matched_interpolation(a, b, t)
        elif kind == "slerp":
            interpolation_vector = spherical_interpolation(a, b, t)
        else:
            raise ValueError("Invalid interpolation kind")
        interpolation_matrix.append(interpolation_vector)
    return np.array(interpolation_matrix)


def find_endpoint(a, midpoint, kind="linear"):
    a = a.ravel()
    midpoint = midpoint.ravel()
    if kind == "linear":
        b = (2 * midpoint) - a
    elif kind == "gaussian_matched":
        b = (midpoint * (2 ** (1 / 2))) - a
    return b


def binary_sampling_search(
    z,
    z_label,
    blackbox,
    lower_threshold=0,
    upper_threshold=4,
    n=10000,
    n_batch=1000,
    stopping_ratio=0.01,
    kind="gaussian_matched",
    vicinity_sampler_kwargs=dict(),
    verbose=True,
    check_upper_threshold=True,
    final_counterfactual_search=True,
    downward_only=True,
    **kwargs
):
    if verbose:
        print("Binary sampling search:", kind)

    # sanity check for the upper threshold
    if check_upper_threshold:
        for i in range(int(n / n_batch)):
            Z = vicinity_sampling(
                z=z,
                n=n_batch,
                threshold=upper_threshold,
                kind=kind,
                verbose=False,
                **vicinity_sampler_kwargs
            )
            y = blackbox.predict(Z)
            if not np.all(y == z_label):
                break
        if i == list(range(int(n / n_batch)))[-1]:
            raise Exception(
                "No counterfactual found, increase upper_threshold or n_search."
            )

    change_lower = False
    latest_working_threshold = upper_threshold
    Z_counterfactuals = list()
    while lower_threshold / upper_threshold < stopping_ratio:
        if change_lower:
            if downward_only:
                break
            lower_threshold = threshold
        threshold = (lower_threshold + upper_threshold) / 2
        if threshold <= 1e-8:  # if the threshold is getting insanely low
            warnings.warn("Threshold is too low, loop exited")
            break
        change_lower = True
        if verbose:
            print("   Testing threshold value:", threshold)
        for i in range(int(n / n_batch)):
            Z = vicinity_sampling(
                z=z,
                n=n_batch,
                threshold=threshold,
                kind=kind,
                verbose=False,
                **vicinity_sampler_kwargs
            )
            y = blackbox.predict(Z)
            if not np.all(y == z_label):  # if we found already some counterfactuals
                counterfactuals_idxs = np.argwhere(y != z_label).ravel()
                Z_counterfactuals.append(Z[counterfactuals_idxs])
                latest_working_threshold = threshold
                upper_threshold = threshold
                change_lower = False
                break
    if verbose:
        print("   Best threshold found:", latest_working_threshold)
    if final_counterfactual_search:
        if verbose:
            print(
                "   Final counterfactual search... (this could take a while)", end=" "
            )
        Z = vicinity_sampling(
            z=z,
            n=n,
            threshold=latest_working_threshold,
            kind=kind,
            verbose=False,
            **vicinity_sampler_kwargs
        )
        y = blackbox.predict(Z)
        counterfactuals_idxs = np.argwhere(y != z_label).ravel()
        Z_counterfactuals.append(Z[counterfactuals_idxs])
        if verbose:
            print("Done!")
    Z_counterfactuals = np.concatenate(Z_counterfactuals)
    closest_counterfactual = min(
        Z_counterfactuals, key=lambda p: sum((p - z.ravel()) ** 2)
    )
    return closest_counterfactual.reshape(1, -1), latest_working_threshold
