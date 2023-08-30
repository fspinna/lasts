import warnings

import matplotlib
from matplotlib import pyplot as plt, patheffects
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import scipy
from scipy.stats import norm

from sklearn.tree import DecisionTreeClassifier, plot_tree
from lasts.neighgen.utils import interpolate
from lasts.utils import compute_medoid, means_from_df_list, cartesian


def plot_subsequences(
    shapelets, ts_length, ts_max, ts_min, figsize=(10, 3), color="mediumblue", dpi=60
):
    for i, shapelet in enumerate(shapelets):
        plt.figure(figsize=figsize)
        plt.gca().set_ylim((ts_min, ts_max))
        plt.gca().set_xlim((0, ts_length))
        plt.plot(shapelet.ravel(), lw=3, color=color)
        plt.axis("off")
        plt.text(
            len(shapelet.ravel()),
            (shapelet.ravel().max()) + (ts_max - ts_min) / 10,
            str(i),
            c=color,
        )
        plt.show()


def plot_subsequences_grid(
    subsequence_list,
    n,
    m,
    starting_idx=0,
    random=False,
    color="mediumblue",
    fontsize=12,
    text_height=0,
    dpi=72,
    figsize=(10, 5),
    linewidth=3,
    **kwargs
):
    fig, axs = plt.subplots(n, m, figsize=figsize, dpi=dpi)
    fig.patch.set_visible(False)
    for i in range(n):
        for j in range(m):
            if random:
                starting_idx = np.random.randint(0, len(subsequence_list))
            axs[i, j].plot(
                subsequence_list[starting_idx].ravel()
                - subsequence_list[starting_idx].mean(),
                lw=linewidth,
                color=color,
            )
            axs[i, j].set_aspect("equal", adjustable="datalim")
            y_lim = axs[i, j].get_ylim()
            x_lim = axs[i, j].get_xlim()
            # axs[i, j].set_xlim((0, l))
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            axs[i, j].axis("off")
            axs[i, j].text(
                np.min(x_lim),
                0 + text_height,
                str(starting_idx),
                fontsize=fontsize,
                color=color,
                horizontalalignment="center",
                verticalalignment="center",
                weight="bold",
                path_effects=[
                    patheffects.Stroke(
                        linewidth=linewidth, foreground="white", alpha=0.6
                    ),
                    patheffects.Normal(),
                ],
            )
            starting_idx += 1
    plt.tight_layout()
    plt.show()


def plot_binary_heatmap(
    x_label,
    y,
    X_binary,
    figsize=(8, 8),
    dpi=60,
    fontsize=20,
    labelsize=20,
    step=1,
    aspect=0.5,
    show=True,
    **kwargs
):
    """Plots a heatmap of the contained and not contained shapelet
    Parameters
    ----------
    explainer : Sbgdt object
    exemplars_labels : int
        instance to explain label

    Returns
    -------
    self
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = y.copy()
    X = X_binary.copy()
    # 0: no shapelet, 1: shapelet counterfactual, 2: shapelet factual
    X[y == x_label] *= 2
    sorted_by_class_idxs = y.argsort()
    sorted_dataset = X[sorted_by_class_idxs]
    cmap = matplotlib.colors.ListedColormap(["white", "#d62728", "#2ca02c"])
    plt.ylabel("subsequences", fontsize=fontsize)
    plt.xlabel("time series", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.matshow(sorted_dataset.T, cmap=cmap, aspect=aspect)
    ax.set_yticks(np.arange(0, sorted_dataset.shape[1], step=step))
    # ax.set_aspect(aspect * sorted_dataset.shape[0] / sorted_dataset.shape[1])
    # ax.set_aspect(aspect)
    plt.tight_layout()
    if show:
        plt.show()


def plot_shapelet_heatmap(
    X,
    y,
    figsize=(8, 8),
    dpi=60,
    fontsize=20,
    labelsize=20,
    step=1,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y = y.copy()
    X = X.copy()
    sorted_by_class_idxs = y.argsort()
    X_sorted = X[sorted_by_class_idxs]
    plt.ylabel("subsequences", fontsize=fontsize)
    plt.xlabel("time series", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.matshow(X_sorted.T)
    ax.set_yticks(np.arange(0, X_sorted.shape[1], step=step))
    ax.set_aspect(0.5 * X_sorted.shape[0] / X_sorted.shape[1])
    plt.show()


def plot_sklearn_decision_tree(dt: DecisionTreeClassifier, dpi=300):
    plt.figure(dpi=dpi)
    plot_tree(dt)
    plt.show()


def plot_shapelet_rule(
    x,
    shapelets_idxs,
    shapelets,
    starting_idxs,
    condition_operators,
    shapelets_names=None,
    title="",
    legend_label="",
    figsize=(20, 5),
    dpi=72,
    fontsize=20,
    text_height=0,
    labelfontsize=15,
    loc="best",
    frameon=True,
    forced_y_lim=None,
    return_y_lim=False,
    formatter="%.1f",
    **kwargs
):
    if shapelets_names is None:
        shapelets_names = ["" for _ in shapelets_idxs]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title, fontsize=fontsize)
    ax.plot(x.ravel(), c="royalblue", alpha=0.2, lw=3, label=legend_label)
    for i, (shp_idx, shp, st_idx, op) in enumerate(
        zip(shapelets_idxs, shapelets, starting_idxs, condition_operators)
    ):
        ax.plot(
            np.arange(st_idx, st_idx + len(shp)),
            shp,
            linestyle="-" if op == ">" else "--",
            alpha=0.5 if op == ">" else 0.5,
            label="contained" if op == ">" else "not-contained",
            c="#2ca02c" if op == ">" else "#d62728",
            lw=5,
        )
        plt.text(
            (st_idx + st_idx + len(shp) - 2) / 2
            if i != 0
            else 1 + (st_idx + st_idx + len(shp)) / 2,
            shp[int(len(shp) / 2)] + text_height,
            "%s %s" % (str(shp_idx), str(shapelets_names[i])),
            fontsize=fontsize - 2,
            c="#2ca02c" if op == ">" else "#d62728",
            horizontalalignment="center",
            verticalalignment="center",
            weight="bold",
            path_effects=[
                patheffects.Stroke(linewidth=3, foreground="white", alpha=0.6),
                patheffects.Normal(),
            ],
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if formatter is not None:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(formatter))
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.xlabel("time-steps", fontsize=fontsize)
    plt.ylabel("value", fontsize=fontsize)
    plt.legend(
        by_label.values(),
        by_label.keys(),
        frameon=frameon,
        fontsize=labelfontsize,
        loc=loc,
    )
    if forced_y_lim is not None:
        plt.gca().set_ylim(forced_y_lim)
    if return_y_lim:
        y_lim = plt.gca().get_ylim()
    plt.show()
    if return_y_lim:
        return y_lim


def plot_latent_space_z(
    Z,
    y,
    z,
    z_label,
    K=None,
    closest_counterfactual=None,
    K_alpha=0,
    neigh_alpha=0.5,
    locator_base=1.0,
    show=True,
    gaussian_reference=False,
    s=20,
    set_major_locator=True,
    legend_markersize=None,
    legend_fontsize=12,
    plot_line=False,
    **kwargs
):
    """Plot a 2d scatter representation of the latent space
    Parameters
    ----------
    Z : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    z : array of shape (n_features,)
        latent instance to explain
    z_label : int
        latent instance to explain label

    Returns
    -------
    None
    """
    z = z.ravel()

    fig, ax = plt.subplots(
        figsize=kwargs.get("figsize", (6, 6)), dpi=kwargs.get("dpi", 72)
    )
    ax.set_title(r"Latent Neighborhood: $Z$", fontsize=kwargs.get("fontsize", 12))

    # plots generated neighborhood points
    exemplars = np.argwhere(y == z_label)
    counterexemplars = np.argwhere(y != z_label)

    if K is not None:
        ax.scatter(
            K[:, 0],
            K[:, 1],
            c="gray",
            alpha=K_alpha,  # alpha=0.2,
            label=r"$\mathcal{N}(0, 1)$",
        )

    if gaussian_reference:
        n = 100
        grid_x = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, n))
        grid_y = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, n))
        K = cartesian([grid_x, grid_y])
        # K = (K - K.mean()) / K.std()
        sns.kdeplot(
            x=K[:, 0],
            y=K[:, 1],
            color="gray",
            alpha=0.4,
            cut=3,
            fill=True,
            linestyles="--",
            ax=ax,
            cmap="Greys",
            cbar=True,
            cbar_kws={
                "label": "pdf of a standard bivariate normal distribution",
                "format": "%.2f",
                "location": "right",
            }
            # label=r"$\mathcal{N}(0, 1)$"
        )
        sns.kdeplot(
            x=K[:, 0],
            y=K[:, 1],
            color="gray",
            alpha=0.4,
            cut=3,
            fill=False,
            linestyles="--",
            ax=ax,
            cmap="Greys",
            zorder=0
            # label=r"$\mathcal{N}(0, 1)$"
        )

    ax.scatter(
        Z[:, 0][exemplars],
        Z[:, 1][exemplars],
        c="#2ca02c",
        alpha=neigh_alpha,
        label=r"$Z_=$",
        s=s,
    )
    ax.scatter(
        Z[:, 0][counterexemplars],
        Z[:, 1][counterexemplars],
        c="#d62728",
        alpha=neigh_alpha,
        label=r"$Z_\neq$",
        s=s,
    )

    if closest_counterfactual is not None:
        if plot_line:
            plt.plot(
                (z.ravel()[0], closest_counterfactual.ravel()[0]),
                (z.ravel()[1], closest_counterfactual.ravel()[1]),
                lw=3,
                c="gray",
                ls="--",
                alpha=0.8,
            )
        ax.scatter(
            closest_counterfactual[:, 0],
            closest_counterfactual[:, 1],
            c="#d62728",
            alpha=0.9,
            label=r"$\mathbf{z}_\neq$",
            marker="X",
            edgecolors="white",
            s=200,
        )
        cb = plt.colorbar(matplotlib.cm.ScalarMappable())
        cb.remove()

    # marks the instance to explain with an X
    ax.scatter(
        z[0],
        z[1],
        label=r"$\mathbf{z}$",
        c="royalblue",
        marker="X",
        edgecolors="white",
        s=200,
    )

    if kwargs.get("legend"):
        legend = ax.legend(fontsize=legend_fontsize, frameon=True, loc="lower left")
        if legend_markersize is not None:
            legend.legendHandles[0]._sizes = [legend_markersize]
            legend.legendHandles[1]._sizes = [legend_markersize]
            # legend.legendHandles[0]._legmarker.set_markersize(legend_markersize)
            # legend.legendHandles[1]._legmarker.set_markersize(legend_markersize)
    loc = matplotlib.ticker.MultipleLocator(
        base=locator_base
    )  # this locator puts ticks at regular intervals
    if set_major_locator:
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.tick_params(axis="both", which="major", labelsize=kwargs.get("fontsize", 12))
    if show:
        plt.show()


def plot_latent_space_matrix_z(Z, y, z, z_label, K=None, **kwargs):
    """Plots a scatter matrix of the latent space
    Parameters
    ----------
    Z : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    z : array of shape (n_features,)
        latent instance to explain
    z_label : int
        latent instance to explain label

    Returns
    -------
    None
    """

    y = 1 * (y == z_label)

    Z = np.concatenate([Z, z])
    y = np.concatenate([y, np.repeat(2, z.shape[0])])

    if K is not None:
        Z = np.concatenate([Z, K])
        y = np.concatenate([y, np.repeat(3, K.shape[0])])
        markers = [".", ".", "X", "."]
    else:
        markers = [".", ".", "X"]

    Z = list(Z)
    Z = pd.DataFrame(Z)
    Z["y"] = y

    g = sns.pairplot(
        Z,
        hue="y",
        markers=markers,
        palette={0: "#d62728", 1: "#2ca02c", 2: "royalblue", 3: "gray"},
        aspect=1,
        corner=True,
        height=4,
        plot_kws=dict(s=200),
    )
    g._legend.set_title("")
    if K is not None:
        new_labels = [r"$Z_\neq$", r"$Z_=$", r"$z$", r"$K$"]
    else:
        new_labels = [r"$Z_\neq$", r"$Z_=$", r"$z$"]
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)
    plt.show()


def morphing_matrix(blackbox, decoder, x_label, labels=None, n=7, **kwargs):
    """Plots a 2d matrix of instances sampled from a normal distribution
    only meaningful with a 2d normal latent space (es. with VAE, AAE)
    Parameters
    ----------
    blackbox : BlackboxWrapper object
        a wrapped blackbox
    decoder : object
        a trained decoder
    x_label : int
        instance to explain label
    labels : list of shape (n_classes,), optional (default = None)
        list of classes labels
    n : int, optional (default = 7)
        number of instances per latent dimension
    Returns
    -------
    self
    """

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))[::-1]
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    fig, axs = plt.subplots(
        n, n, figsize=kwargs.get("figsize", (10, 5)), dpi=kwargs.get("dpi", 72)
    )
    fig.suptitle("Classes Morphing", fontsize=kwargs.get("fontsize", 12))
    fig.patch.set_visible(False)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sampled = np.array([[xi, yi]])
            z_sampled_tilde = decoder.predict(z_sampled).ravel()
            z_sampled_label = blackbox.predict(z_sampled_tilde.reshape(1, -1, 1))[0]
            color = "#2ca02c" if z_sampled_label == x_label else "#d62728"
            if z_sampled_label == x_label:
                label = (
                    r"$b(\tilde{z}) = $" + labels[z_sampled_label]
                    if labels
                    else r"$b(\tilde{z}) = $" + str(z_sampled_label)
                )
            else:
                label = (
                    r"$b(\tilde{z}) \neq $" + labels[x_label]
                    if labels
                    else r"$b(\tilde{z}) \neq $" + str(x_label)
                )
            axs[i, j].plot(z_sampled_tilde, color=color, label=label)
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            axs[i, j].axis("off")

    d = dict()
    for a in fig.get_axes():
        if a.get_legend_handles_labels()[1][0] not in d:
            d[a.get_legend_handles_labels()[1][0]] = a.get_legend_handles_labels()[0][0]

    labels, handles = zip(*sorted(zip(d.keys(), d.values()), key=lambda t: t[0]))
    plt.legend(handles, labels, fontsize=kwargs.get("fontsize", 12))
    plt.show()


def plot_interpolation(
    z,
    z_prime,
    x_label,
    decoder,
    blackbox,
    interpolation="linear",
    n=100,
    hide_interpolation=False,
    title="Interpolation",
    **kwargs
):
    interpolation_matrix = interpolate(z, z_prime, interpolation, n)
    decoded_interpolation_matrix = decoder.predict(interpolation_matrix)
    z_tilde = decoder.predict(z)
    z_prime_tilde = decoder.predict(z_prime)
    y = blackbox.predict(decoded_interpolation_matrix)

    exemplars_idxs = np.argwhere(y == x_label).ravel()
    counterexemplars_idxs = np.argwhere(y != x_label).ravel()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    plt.title(title, fontsize=kwargs.get("fontsize", 12))
    plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
    plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis="both", which="major", labelsize=kwargs.get("fontsize", 12))
    if not hide_interpolation:
        plt.plot(
            decoded_interpolation_matrix[:, :, 0][exemplars_idxs].T,
            c="#2ca02c",
            alpha=kwargs.get("alpha", 0.1),
        )
        if counterexemplars_idxs.shape[0] != 0:
            plt.plot(
                decoded_interpolation_matrix[:, :, 0][counterexemplars_idxs].T,
                c="#d62728",
                alpha=kwargs.get("alpha", 0.1),
            )
    plt.plot(
        z_tilde.ravel(),
        c="royalblue",
        linestyle="-",
        lw=3,
        alpha=0.9,
        label=r"$\hat{X}$",
    )
    plt.plot(
        z_prime_tilde.ravel(),
        c="#d62728",
        linestyle="-",
        lw=3,
        alpha=0.9,
        label=r"$\hat{X}_\neq$",
    )
    plt.legend()
    plt.show()


def plot_exemplars_and_counterexemplars(
    Z_tilde,
    y,
    x,
    z_tilde,
    x_label,
    labels=None,
    plot_x=True,
    plot_z_tilde=True,
    legend=False,
    no_axes_labels=False,
    **kwargs
):
    """Plots x, z_tilde; exemplars; counterexemplars
    Parameters
    ----------
    Z_tilde : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    x : array of shape (n_features,)
        instance to explain
    z_tilde : array of shape (n_features,)
        autoencoded instance to explain
    x_label : int
        instance to explain label
    labels : list of shape (n_classes,), optional (default = None)
        list of classes labels

    Returns
    -------
    self
    """
    exemplars_idxs = np.argwhere(y == x_label).ravel()
    counterexemplars_idxs = np.argwhere(y != x_label).ravel()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Instance to explain: " + r"$b(x)$" + " = " + labels[x_label] if labels
    #           else "Instance to explain: " + r"$b(x)$" + " = " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))
    plt.title(
        r"$b(X)$" + " = " + labels[x_label]
        if labels
        else r"$b(X)$" + " = " + str(x_label),
        fontsize=kwargs.get("fontsize", 12),
    )
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis="both", which="major", labelsize=kwargs.get("fontsize", 12))
    if plot_x:
        plt.plot(x.ravel(), c="royalblue", linestyle="-", lw=3, alpha=0.9, label=r"$X$")
    if plot_z_tilde:
        plt.plot(
            z_tilde.ravel(),
            c="orange",
            linestyle="-",
            lw=3,
            alpha=0.9,
            label=r"$\hat{X}$",
        )
    if legend:
        plt.legend(fontsize=kwargs.get("fontsize", 12), frameon=False)
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Exemplars: " + r"$b(\tilde{Z}_{=})$" + " = " + labels[x_label] if labels
    #           else "Exemplars: " + r"$b(\tilde{Z}_{=})$" + " = " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))

    plt.title(
        r"$b(\hat{\mathcal{X}}_{=})$" + " = " + labels[x_label]
        if labels
        else r"$b(\hat{\mathcal{X}}_{=})$" + " = " + str(x_label),
        fontsize=kwargs.get("fontsize", 12),
    )
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis="both", which="major", labelsize=kwargs.get("fontsize", 12))
    plt.plot(
        Z_tilde[:, :, 0][exemplars_idxs].T, c="#2ca02c", alpha=kwargs.get("alpha", 0.1)
    )
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    # plt.title("Counterexemplars: " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label] if labels
    #           else "Counterexemplars: " + r"$b(\tilde{Z}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
    #           fontsize=kwargs.get("fontsize", 12))
    plt.title(
        r"$b(\hat{\mathcal{X}}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label]
        if labels
        else r"$b(\hat{\mathcal{X}}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
        fontsize=kwargs.get("fontsize", 12),
    )
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis="both", which="major", labelsize=kwargs.get("fontsize", 12))
    plt.plot(
        Z_tilde[:, :, 0][counterexemplars_idxs].T,
        c="#d62728",
        alpha=kwargs.get("alpha", 0.1),
    )
    plt.show()

    plt.figure(figsize=kwargs.get("figsize", (10, 3)), dpi=kwargs.get("dpi", 72))
    plt.title(
        "Neighborhood: " + r"$\hat{\mathcal{X}}$", fontsize=kwargs.get("fontsize", 12)
    )
    if not no_axes_labels:
        plt.ylabel("value", fontsize=kwargs.get("fontsize", 12))
        plt.xlabel("time-steps", fontsize=kwargs.get("fontsize", 12))
    plt.tick_params(axis="both", which="major", labelsize=kwargs.get("fontsize", 12))
    plt.plot(
        Z_tilde[:, :, 0][exemplars_idxs].T, c="#2ca02c", alpha=kwargs.get("alpha", 0.1)
    )
    plt.plot(
        Z_tilde[:, :, 0][counterexemplars_idxs].T,
        c="#d62728",
        alpha=kwargs.get("alpha", 0.1),
    )
    plt.plot(z_tilde.ravel(), c="royalblue", linestyle="-", lw=3, alpha=0.9)
    plt.show()


def plot_exemplars_and_counterexemplars_multi(
    Z_tilde,
    y,
    x,
    z_tilde,
    x_label,
    labels=None,
    plot_x=True,
    plot_z_tilde=True,
    legend=False,
    no_axes_labels=False,
    figsize=(10, 10),
    dpi=72,
    fontsize=12,
    alpha=0.1,
    formatter="%.1f",
    **kwargs
):
    """Plots x, z_tilde; exemplars; counterexemplars
    Parameters
    ----------
    Z_tilde : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    x : array of shape (n_features,)
        instance to explain
    z_tilde : array of shape (n_features,)
        autoencoded instance to explain
    x_label : int
        instance to explain label
    labels : list of shape (n_classes,), optional (default = None)
        list of classes labels

    Returns
    -------
    self
    """
    exemplars_idxs = np.argwhere(y == x_label).ravel()
    counterexemplars_idxs = np.argwhere(y != x_label).ravel()

    fig, axs = plt.subplots(
        nrows=x.shape[2], ncols=1, figsize=figsize, dpi=dpi, sharex=True
    )
    axs[0].set_title(
        r"$b(X)$" + " = " + labels[x_label]
        if labels
        else r"$b(X)$" + " = " + str(x_label),
        fontsize=fontsize,
    )
    if not no_axes_labels:
        plt.xlabel("time-steps", fontsize=fontsize)
    for i, ax in enumerate(axs):
        ax.set_ylabel("dim_" + str(i), fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if formatter is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter(formatter)
            )
    if plot_x:
        plot_mts_array(x, c="royalblue", linestyle="-", lw=3, alpha=0.9, label=r"$x$")
    if plot_z_tilde:
        if plot_x:
            plot_mts_array(
                z_tilde,
                c="orange",
                linestyle="-",
                lw=3,
                alpha=0.9,
                label=r"$\tilde{z}$",
            )
        else:
            plot_mts_array(
                z_tilde,
                c="royalblue",
                linestyle="-",
                lw=3,
                alpha=0.9,
                label=r"$\tilde{z}$",
            )
    if legend:
        plt.legend()
    plt.show()

    fig, axs = plt.subplots(
        nrows=x.shape[2], ncols=1, figsize=figsize, dpi=dpi, sharex=True
    )
    axs[0].set_title(
        r"$b(\hat{\mathcal{X}}_{=})$" + " = " + labels[x_label]
        if labels
        else r"$b(\hat{\mathcal{X}}_{=})$" + " = " + str(x_label),
        fontsize=fontsize,
    )
    if not no_axes_labels:
        plt.ylabel("value", fontsize=fontsize)
        plt.xlabel("time-steps", fontsize=fontsize)
    for i, ax in enumerate(axs):
        ax.set_ylabel("dim_" + str(i), fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if formatter is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter(formatter)
            )
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plot_mts_array(Z_tilde[exemplars_idxs], c="#2ca02c", alpha=alpha)
    plt.show()

    fig, axs = plt.subplots(
        nrows=x.shape[2], ncols=1, figsize=figsize, dpi=dpi, sharex=True
    )
    axs[0].set_title(
        r"$b(\hat{\mathcal{X}}_\neq)$" + " " + r"$\neq$" + " " + labels[x_label]
        if labels
        else r"$b(\hat{\mathcal{X}}_\neq)$" + " " + r"$\neq$" + " " + str(x_label),
        fontsize=fontsize,
    )
    if not no_axes_labels:
        plt.ylabel("value", fontsize=fontsize)
        plt.xlabel("time-steps", fontsize=fontsize)
    for i, ax in enumerate(axs):
        ax.set_ylabel("dim_" + str(i), fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if formatter is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter(formatter)
            )
    plot_mts_array(Z_tilde[counterexemplars_idxs], c="#d62728", alpha=alpha)
    plt.show()

    fig, axs = plt.subplots(
        nrows=x.shape[2], ncols=1, figsize=figsize, dpi=dpi, sharex=True
    )
    axs[0].set_title("Neighborhood: " + r"$\hat{\mathcal{X}}$", fontsize=fontsize)
    if not no_axes_labels:
        plt.ylabel("value", fontsize=fontsize)
        plt.xlabel("time-steps", fontsize=fontsize)
    for i, ax in enumerate(axs):
        ax.set_ylabel("dim_" + str(i), fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if formatter is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter(formatter)
            )
    plot_mts_array(Z_tilde[exemplars_idxs], c="#2ca02c", alpha=alpha)
    plot_mts_array(Z_tilde[counterexemplars_idxs], c="#d62728", alpha=alpha)
    plot_mts_array(z_tilde, c="royalblue", linestyle="-", lw=3, alpha=0.9)
    plt.show()


def plot_subsequence_mapping(
    subsequence_dictionary, name_dictionary, feature_idx, **kwargs
):
    plt.title(str(feature_idx) + " : " + name_dictionary[feature_idx].decode("utf-8"))
    plt.plot(subsequence_dictionary[feature_idx][:, :, 0].T, c="gray", alpha=0.1)
    # plt.plot(subsequence_dictionary[feature_idx][:,:,0].mean(axis=0).ravel(), c="red")
    plt.plot(compute_medoid(subsequence_dictionary[feature_idx][:, :, 0]), c="red")
    plt.show()


def plot_scattered_feature_importance(
    ts, feature_importance, figsize=(20, 3), labels=None, dpi=60, fontsize=20
):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(list(range(len(ts))), ts, c=feature_importance, cmap="Reds")
    plt.show()


def plot_feature_importance(
    ts,
    feature_importance,
    figsize=(20, 5),
    labels=None,
    dpi=72,
    fontsize=20,
    cmap="Reds",
    title="",
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(ts.ravel(), c="royalblue", alpha=1, lw=3)
    ax.pcolorfast(
        (0, len(feature_importance) - 1),
        ax.get_ylim(),
        feature_importance[np.newaxis],
        cmap=cmap,
        alpha=1,
    )
    plt.title(title, fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.xlabel("time-steps", fontsize=fontsize)
    plt.ylabel("value", fontsize=fontsize)
    # fig.show()
    plt.show()


def plot_feature_importance_on_ts(
    ts,
    feature_importance,
    figsize=(10, 3),
    labels=None,
    dpi=72,
    fontsize=12,
    cmap="Blues",
    norm=None,
    title="",
    linewidth=3,
    colorbar=False,
    labelfontsize=12,
    label=r"$X$",
    formatter="%.1f",
    legend=False,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.title(title, fontsize=fontsize)
    points = np.array([range(len(ts)), ts]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=0.99999
    )
    lc.set_array(feature_importance)
    ax.plot(ts, color="gray", alpha=0.5, label=label)
    ax.add_collection(lc)
    ax.autoscale_view()
    if formatter is not None:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(formatter))
    plt.xlabel("time-steps", fontsize=fontsize)
    plt.ylabel("value", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    if legend:
        plt.legend()
    plt.show()


def plot_feature_importance_on_ts_multi(
    mts,
    feature_importances,
    figsize=(10, 3),
    labels=None,
    dpi=72,
    fontsize=20,
    cmap="Blues",
    norm=None,
    title="",
    formatter="%.1f",
    **kwargs
):
    fig, axs = plt.subplots(
        nrows=mts.shape[2], ncols=1, figsize=figsize, dpi=dpi, sharex=True
    )
    axs[0].set_title(title, fontsize=fontsize)
    for i, ax in enumerate(axs):
        points = np.array(
            [range(len(mts[:, :, i].ravel())), mts[:, :, i].ravel()]
        ).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=5, alpha=0.99999)
        lc.set_array(feature_importances[:, :, i].ravel())
        ax.plot(mts[:, :, i].ravel(), color="gray", alpha=0.5)
        ax.add_collection(lc)
        ax.autoscale_view()
        ax.set_ylabel("dim_" + str(i), fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if formatter is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter(formatter)
            )

    plt.xlabel("time-steps", fontsize=fontsize)
    plt.show()


def plot_changing_shape(
    ts1,
    ts2,
    feature_importance,
    figsize=(20, 5),
    labels=None,
    dpi=72,
    fontsize=20,
    cmap_ts1="Greens",
    cmap_ts2="Reds",
    title="",
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.title(title, fontsize=fontsize)

    points = np.array([range(len(ts1)), ts1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap_ts1, linewidth=5, alpha=0.9)
    lc.set_array(feature_importance)
    ax.add_collection(lc)
    ax.autoscale_view()

    points = np.array([range(len(ts2)), ts2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap_ts2, linewidth=5, alpha=0.9)
    lc.set_array(feature_importance)
    ax.add_collection(lc)
    ax.autoscale_view()

    plt.xlabel("time-steps", fontsize=fontsize)
    plt.ylabel("value", fontsize=fontsize)
    plt.show()


def plot_usefulness(
    lasts_df,
    real_df,
    dataset_label="",
    figsize=(7, 3),
    fontsize=20,
    dpi=72,
    alpha=1,
    **kwargs
):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(dataset_label, fontsize=fontsize)
    plt.plot(
        np.array(lasts_df).mean(axis=0),
        label="lasts",
        lw=3,
        marker="o",
        c="#01665e",
        alpha=alpha,
    )
    plt.plot(
        np.array(real_df).mean(axis=0),
        label="real",
        lw=3,
        marker="o",
        c="darkgoldenrod",
        alpha=alpha,
    )
    plt.xticks(ticks=list(range(len(lasts_df.columns))), labels=lasts_df.columns)
    plt.gca().set_ylim((0.3, 1.05))
    plt.yticks(ticks=np.arange(3, 11) / 10)
    plt.ylabel("accuracy", fontsize=fontsize)
    plt.xlabel("nbr (counter)exemplars", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    loc = matplotlib.ticker.MultipleLocator(
        base=0.1
    )  # this locator puts ticks at regular intervals
    axes = plt.gca()
    # y_lim = plt.gca().get_ylim()
    axes.yaxis.set_major_locator(loc)
    plt.legend(
        frameon=False,
        fontsize=fontsize,
        loc="lower right",
        ncol=2,
        columnspacing=2,
        handletextpad=0.5,
    )
    plt.show()


def boxplot_from_df(
    df, figsize=(8.0, 6.0), dpi=72, labels=None, fontsize=18, ylabel="", title=""
):
    medianprops = dict(linestyle="-", linewidth=3, color="#8c510a")
    meanprops = dict(marker="D", markeredgecolor="#003c30", markerfacecolor="#003c30")
    boxprops = dict(linestyle="-", linewidth=3, color="#01665e")
    whiskerprops = dict(linestyle="-", linewidth=3, color="#01665e")
    capprops = dict(linestyle="-", linewidth=3, color="#01665e")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    if isinstance(df, list):
        plt.boxplot(
            df,
            medianprops=medianprops,
            boxprops=boxprops,
            capprops=capprops,
            whiskerprops=whiskerprops,
            meanprops=meanprops,
            showmeans=False,
            showfliers=False,
        )
    else:
        plt.boxplot(
            df.values,
            medianprops=medianprops,
            boxprops=boxprops,
            capprops=capprops,
            whiskerprops=whiskerprops,
            meanprops=meanprops,
            showmeans=False,
            showfliers=False,
        )
    # plt.axhline(y=1, color='r', linestyle='--')
    plt.xticks(
        np.array(list(range(len(df.columns)))) + 1,
        list(df.columns) if labels is None else labels,
    )
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.show()


def boxplots_from_df_list(df_list, labels=None, print_means=True, figsize=(8.0, 6.0)):
    medianprops = dict(linestyle="-", linewidth=3, color="#8c510a")
    meanprops = dict(marker="D", markeredgecolor="#003c30", markerfacecolor="#003c30")
    boxprops = dict(linestyle="-", linewidth=3, color="#01665e")
    whiskerprops = dict(linestyle="-", linewidth=3, color="#01665e")
    capprops = dict(linestyle="-", linewidth=3, color="#01665e")
    if labels is None:
        labels = np.array(range(len(df_list))) + 1
    for column in df_list[0].columns:
        array = df_list[0][column].values[np.newaxis, :]
        if len(df_list) > 1:
            for df in df_list[1:]:
                array = np.concatenate(
                    [array, df[column].values[np.newaxis, :]], axis=0
                )
        # array = array[0]
        plt.figure(figsize=figsize)
        plt.title(column)
        plt.boxplot(
            array.T,
            medianprops=medianprops,
            boxprops=boxprops,
            capprops=capprops,
            whiskerprops=whiskerprops,
            meanprops=meanprops,
            showmeans=False,
            showfliers=False,
        )
        plt.xticks(list(np.array(range(len(df_list))) + 1), labels)
        plt.show()
    if print_means:
        df_means = means_from_df_list(df_list, labels)
        print(df_means)
    return


def plot_boxplot_groups(
    df_dict,
    figsize=(20, 5),
    dpi=72,
    title="",
    ylabel="",
    fontsize=10,
    sharey=True,
    fontsize_keys=10,
    fontsize_columns=10,
    x_ticks_rotation="vertical",
    boxplot_widths=0.5,
    y_scale="linear",
    ax_labels=None,
    xticklabels=None,
    y_label_rotation="vertical",
    showmeans=False,
    **kwargs
):
    medianprops = dict(linestyle="-", linewidth=3, color="#8c510a")
    meanprops = dict(marker="D", markeredgecolor="red", markerfacecolor="red")
    boxprops = dict(linestyle="-", linewidth=3, color="#01665e")
    whiskerprops = dict(linestyle="-", linewidth=3, color="#01665e")
    capprops = dict(linestyle="-", linewidth=3, color="#01665e")
    n_groups = len(df_dict.keys())
    fig, axs = plt.subplots(1, n_groups, figsize=figsize, sharey=sharey, dpi=dpi)
    plt.suptitle(title, fontsize=fontsize)
    axs[0].set_yscale(y_scale)
    axs[0].tick_params(axis="y", which="major", labelsize=fontsize_columns)
    axs[0].tick_params(axis="y", which="minor", labelsize=fontsize_columns)
    axs[0].set_ylabel(ylabel, fontsize=fontsize_columns, rotation=y_label_rotation)
    for i, name in enumerate(sorted(list(df_dict.keys()))):
        df = df_dict[name]
        ax = axs[i]
        if xticklabels is None:
            ax.xaxis.set_ticklabels(
                list(df.columns), fontsize=fontsize_columns, rotation=x_ticks_rotation
            )
        else:
            ax.xaxis.set_ticklabels(
                xticklabels, fontsize=fontsize_columns, rotation=x_ticks_rotation
            )
        df_list = list()
        for column in df.columns:
            df_list.append(list(df[column].dropna()))
        df = df_list
        # ax.xaxis.set_ticks(np.array(list(range(len(df.columns)))) + 1)
        ax.boxplot(
            df,
            medianprops=medianprops,
            boxprops=boxprops,
            capprops=capprops,
            whiskerprops=whiskerprops,
            meanprops=meanprops,
            showmeans=showmeans,
            showfliers=False,
            widths=boxplot_widths,
        )
        ax.set_title(
            name if ax_labels is None else ax_labels[i], fontsize=fontsize_keys
        )
    plt.tight_layout()
    plt.show()


def plot_mts2(
    mts, sharex=False, sharey=False, dpi=72, figsize=(20, 10), dim_labels=None, **kwargs
):
    fig, axs = plt.subplots(
        nrows=len(mts), ncols=1, sharex=sharex, sharey=sharey, dpi=dpi, figsize=figsize
    )
    for i, (dim, ax) in enumerate(zip(mts, axs)):
        ax.plot(np.squeeze(dim).T)
        ax.set_ylabel("dim_" + str(i) if dim_labels is None else dim_labels[i])
    return fig


def plot_mts(mts, **kwargs):
    fig = plt.gcf()
    axs = fig.axes
    for i, (dim, ax) in enumerate(zip(mts, axs)):
        ax.plot(np.squeeze(dim).T, **kwargs)
    return fig


def plot_mts_array(mts, **kwargs):
    fig = plt.gcf()
    axs = fig.axes
    for i, ax in enumerate(axs):
        ax.plot(np.squeeze(mts[:, :, i]).T, **kwargs)
    return fig


def plot_multi_subsequence_rule(
    x,
    shapelets_idxs,
    shapelets,
    starting_idxs,
    condition_operators,
    dimension_idxs,
    shapelets_names=None,
    dimension_names=None,
    title="",
    legend_label="",
    figsize=(20, 5),
    dpi=72,
    fontsize=20,
    text_height=None,
    labelfontsize=15,
    loc="lower right",
    frameon=True,
    sharex=False,
    formatter="%.1f",
    **kwargs
):
    if shapelets_names is None:
        shapelets_names = ["" for _ in shapelets_idxs]
    lengths = np.array([dim.shape[1] for dim in x])
    if lengths.min() == lengths.max():  # if all dims have equal n. of timesteps
        sharex = True
    fig, axs = plt.subplots(
        nrows=len(x), ncols=1, figsize=figsize, dpi=dpi, sharex=sharex
    )
    if text_height is None:
        text_height = [0 for _ in axs]
    axs[0].set_title(title, fontsize=fontsize)
    plot_mts(x, c="royalblue", alpha=0.2, lw=3, label=legend_label)
    for i, (shp_idx, shp, st_idx, op, dim_idx) in enumerate(
        zip(
            shapelets_idxs,
            shapelets,
            starting_idxs,
            condition_operators,
            dimension_idxs,
        )
    ):
        axs[dim_idx].plot(
            np.arange(st_idx, st_idx + len(shp)),
            shp,
            linestyle="-" if op == ">" else "--",
            alpha=0.5 if op == ">" else 0.5,
            label="contained" if op == ">" else "not-contained",
            c="#2ca02c" if op == ">" else "#d62728",
            lw=5,
        )
        axs[dim_idx].text(
            (st_idx + st_idx + len(shp) - 2) / 2
            if i != 0
            else 1 + (st_idx + st_idx + len(shp)) / 2,
            shp[int(len(shp) / 2)] + text_height[dim_idx],
            "%s %s" % (str(shp_idx), str(shapelets_names[i])),
            fontsize=fontsize - 2,
            c="#2ca02c" if op == ">" else "#d62728",
            horizontalalignment="center",
            verticalalignment="center",
            weight="bold",
            path_effects=[
                patheffects.Stroke(linewidth=3, foreground="white", alpha=0.6),
                patheffects.Normal(),
            ],
        )
    handles = list()
    labels = list()
    for i, ax in enumerate(axs):
        ax.set_ylabel(
            "dim_" + str(i) if dimension_names is None else dimension_names[i],
            fontsize=fontsize,
        )
        handle, label = ax.get_legend_handles_labels()
        handles.extend(handle)
        labels.extend(label)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if formatter is not None:
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter(formatter)
            )
    by_label = dict(zip(labels, handles))
    plt.xlabel("time-steps", fontsize=fontsize)
    plt.legend(
        by_label.values(),
        by_label.keys(),
        frameon=frameon,
        fontsize=labelfontsize,
        loc=loc,
    )
    plt.show()


def plot_grouped_history(history):
    if isinstance(history, dict):
        history = pd.DataFrame(history)
    val_metrics = list()
    for column in history.columns:
        if "val_" in column:
            val_metrics.append(column.replace("val_", ""))
            val_metrics.append(column)
    for i in range(0, len(val_metrics), 2):
        plt.title(val_metrics[i])
        plt.plot(history[val_metrics[i]], label="train")
        plt.plot(history[val_metrics[i + 1]], label="val")
        plt.legend()
        plt.show()
    for column in history.columns:
        if column not in val_metrics:
            plt.title(column)
            plt.plot(history[column], label="train")
            plt.legend()
            plt.show()


def plot_latent_space(
    Z,
    y=None,
    title="",
    labels_names=None,
    figsize=(5, 5),
    dpi=96,
    legend=True,
    fontsize=20,
    labelsize=20,
    legendfontsize=15,
):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    plt.figure(figsize=figsize, dpi=dpi)
    # plt.suptitle(title)
    plt.title(title, fontsize=fontsize)
    if Z.shape[1] != 2:
        warnings.warn(
            "Latent space is not bidimentional, only the first 2 dimensions will be plotted."
        )
    if y is None:
        plt.scatter(Z[:, 0], Z[:, 1])
    else:
        for label in np.unique(y):
            idxs = np.nonzero(y == label)
            plt.scatter(
                Z[:, 0][idxs],
                Z[:, 1][idxs],
                c=colors[label % len(colors)],
                label=label if labels_names is None else labels_names[label],
            )
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    plt.tick_params(axis="both", which="major", labelsize=labelsize)
    plt.gca().set_aspect("equal", adjustable="box")
    if legend:
        plt.legend(loc="lower right", fontsize=legendfontsize, framealpha=0.95)
    """if labels is not None:
        for i, label in enumerate(labels):
            legend.get_texts()[i].set_text(label)"""
    plt.show()


def plot_latent_space_matrix(Z, y=None, **kwargs):
    """Plots a scatter matrix of the latent space
    Parameters
    ----------
    Z : array of shape (n_samples, n_features)
        latent instances
    y : array of shape (n_samples,)
        latent instances labels
    z : array of shape (n_features,)
        latent instance to explain
    z_label : int
        latent instance to explain label

    Returns
    -------
    None
    """
    Z = list(Z)
    Z = pd.DataFrame(Z)
    if y is not None:
        Z["y"] = y

        g = sns.pairplot(Z, hue="y", aspect=1, corner=True, height=4, diag_kind="kde")
        g._legend.set_title("")
    else:
        sns.pairplot(Z, aspect=1, corner=True, height=4, diag_kind="kde")
    plt.show()
    return


if __name__ == "__main__":
    cmap = shap_cmap()
