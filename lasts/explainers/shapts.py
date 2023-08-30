#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:35:09 2020

@author: francesco
"""
import shap
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib
import sys

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from lasts.utils import explanation_error, baseline_error, find_linear_trend
from lasts.base import (
    ModularExplainer,
    SaliencyBasedExplanation,
    Pickler,
    Plotter,
    Evaluator,
)
from lasts.plots import plot_feature_importance_on_ts
from datetime import datetime


class ShapTS(ModularExplainer):
    def __init__(self, blackbox, verbose=True, unpicklable_variables=None, **kwargs):
        self.blackbox = blackbox
        self.unpicklable_variables = unpicklable_variables
        self.verbose = verbose
        self.kwargs = kwargs

        self.x_ = None
        self.x_label_ = None
        self.explanation_ = None
        self.plotter = ShapTSPlotter(self)
        self.evaluator = ShapTSEvaluator(self)
        self.pickler = Pickler(self)
        self.shap_values_ = None
        self.runtime_ = None

    def fit(self, x):
        start_time = datetime.now()
        self.x_ = x
        self.x_label_ = self.blackbox.predict(x)[0]
        shap_values, change_points = shap_ts(
            ts=x.ravel(), classifier=self.blackbox, **self.kwargs
        )
        shap_values_reshaped = reshape_shap_output_pointwise(
            shap_values, change_points
        )[:, :, np.newaxis]
        self.shap_values_ = shap_values_reshaped
        self.runtime_ = (datetime.now() - start_time).total_seconds()
        return self

    def explain(self):
        self.explanation_ = SaliencyBasedExplanation(
            self.shap_values_[self.x_label_ : self.x_label_ + 1]
        )
        return self.explanation_

    def fit_explain(self, x):
        self.fit(x)
        return self.explain()

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind=kind, **kwargs)

    def evaluate(self, metric, **kwargs):
        return self.evaluator.evaluate(metric=metric, **kwargs)

    def save(self, folder, name=""):
        # self.blackbox = None  # not really needed after fitting
        super().save(folder, name + "_shapts")

    @classmethod
    def load(cls, folder, name="", custom_load=None):
        return super().load(folder, name + "_shapts", custom_load)


class ShapTSPlotter(Plotter):
    def __init__(self, shap_explainer: ShapTS):
        self.shap_explainer = shap_explainer

    def plot_shap_values(
        self, figsize=(10, 3), dpi=72, cmap="coolwarm_r", norm=None, **kwargs
    ):
        if norm is None:
            norm = matplotlib.colors.CenteredNorm()
        norm(
            self.shap_explainer.shap_values_
        )  # this way the norm checks all the shap values
        plot_feature_importance_on_ts(
            self.shap_explainer.x_.ravel(),
            self.shap_explainer.explanation_.saliency.ravel(),
            cmap=cmap,  # FIXME: probably wrong (remove _r)
            norm=norm,
            figsize=figsize,
            dpi=dpi,
            **kwargs
        )

    def plot(self, kind, **kwargs):
        pass


class ShapTSEvaluator(Evaluator):
    METRICS = ["runtime"]

    def __init__(self, shap_explainer: ShapTS):
        self.shap_explainer = shap_explainer

    def runtime_score(self):
        return self.shap_explainer.runtime_

    def evaluate(self, metric, **kwargs):
        if metric == "runtime":
            return self.runtime_score()
        else:
            raise Exception("Metric not valid.")


def preprocess_shap_values(shap_values, kind="absolute"):
    shap_values_ = shap_values.copy()
    minmax = MinMaxScaler()
    if kind == "discard_negative":  # treat negative shapley values as 0
        shap_values_[shap_values < 0] = 0
    else:
        pass
    if kind == "no_change":
        shap_values_preprocessed = minmax.fit_transform(
            shap_values_.reshape(-1, 1)
        ).ravel()
    else:
        shap_values_preprocessed = minmax.fit_transform(
            np.abs(shap_values_).reshape(-1, 1)
        ).ravel()
    return shap_values_preprocessed


def get_latent_shap_explainer(Z, blackbox, decoder, **kwargs):
    def f(Z):
        return blackbox.predict_proba(decoder.predict(Z))

    explainer = shap.KernelExplainer(f, Z)
    return explainer


def get_latent_shap_explainer_binary(Z, x_label, blackbox, decoder, **kwargs):
    def f(Z):
        return 1 * (blackbox.predict(decoder.predict(Z)) == x_label)

    explainer = shap.KernelExplainer(f, Z)
    return explainer


def get_change_points(
    ts, model="rbf", jump=5, pen=1, segments_size=None, figsize=(20, 3), plot=True
):
    if model == "constant":
        change_points = list(np.arange(0, len(ts) + 1)[segments_size::segments_size])
        if change_points[-1] < (len(ts)):
            change_points.append(len(ts))
    else:
        algo = rpt.Pelt(model=model, jump=jump).fit(ts)
        change_points = algo.predict(pen=pen)
    if plot:
        rpt.display(
            ts,
            true_chg_pts=change_points,
            computed_chg_pts=change_points,
            figsize=figsize,
        )
        plt.show()
    return change_points


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
    [(0,5),(5,9),(9,12)]
    """
    change_points_indexes = []
    if len(change_points) == 1:
        change_points_indexes.append((0, change_points[0]))
    for i in range(len(change_points) - 1):
        if i == 0:
            change_points_indexes.append((0, change_points[i]))
        change_points_indexes.append((change_points[i], change_points[i + 1]))
    return change_points_indexes


def linear_consecutive_segmentation(z, change_points):
    """Different type of segmentation: if there are consecutive ones in z they count as only one 1

    Parameters
    ----------
    z
    segmentation

    Returns
    -------

    Examples
    --------
    >>> linear_consecutive_segmentation(z = [0,1,1,0], change_points=[10, 15, 70, 100])
    (array([0, 1, 0]), [10, 70, 100])
    """
    new_change_points = []
    i = 0
    while i < len(change_points):
        idx = change_points[i]
        if z[i] == 1:
            if (i + 1 == len(change_points)) or (z[i + 1] == 0):
                new_change_points.append(idx)
            else:
                i += 1
                continue
        else:
            new_change_points.append(idx)
        i += 1
    new_z = z[np.insert(np.diff(z).astype(np.bool), 0, True)]
    return new_z, new_change_points


def two_points_linear_interpolation(change_points_couple, ts):
    # linear interpolation between two points
    n_points = np.abs(np.diff(change_points_couple))[0]
    if change_points_couple[1] == len(ts):
        change_amplitude = ts[change_points_couple[0]] - ts[change_points_couple[1] - 1]
    else:
        change_amplitude = ts[change_points_couple[0]] - ts[change_points_couple[1]]
    steps = abs(change_amplitude / n_points)
    new_vals = []
    for i in range(0, n_points):
        if change_amplitude > 0:
            new_vals.append(ts[change_points_couple[0]] - (i * steps))
        else:
            new_vals.append(ts[change_points_couple[0]] + (i * steps))
    return np.array(new_vals)


def mask_ts(zs, change_points, ts, background):
    zs = 1 - zs  # invert 0 and 1 for np.argwhere
    ts = ts.ravel().copy()
    if background in ["linear_trend"]:
        trend, _, _ = find_linear_trend(ts.ravel())

    change_points_indexes = get_change_points_indexes(change_points)

    masked_tss = []
    for z in zs:
        if background == "linear_consecutive":
            z, new_segmentation = linear_consecutive_segmentation(z, change_points)
            change_points_indexes = get_change_points_indexes(new_segmentation)
        seg_to_change = np.argwhere(z).ravel()
        masked_ts = ts.copy()
        for seg_index in seg_to_change:
            if background in ["linear", "linear_consecutive"]:
                masked_ts[
                    change_points_indexes[seg_index][0] : change_points_indexes[
                        seg_index
                    ][1]
                ] = two_points_linear_interpolation(
                    change_points_indexes[seg_index], ts
                )
            elif background in ["linear_trend"]:
                masked_ts[
                    change_points_indexes[seg_index][0] : change_points_indexes[
                        seg_index
                    ][1]
                ] = trend[
                    change_points_indexes[seg_index][0] : change_points_indexes[
                        seg_index
                    ][1]
                ]
            elif type(background) == int:
                masked_ts[
                    change_points_indexes[seg_index][0] : change_points_indexes[
                        seg_index
                    ][1]
                ] = background
            else:
                raise Exception("background not valid.")
        masked_tss.append(masked_ts)
    masked_tss = np.array(masked_tss)
    return masked_tss


def shap_ts(
    ts,
    classifier,
    nsamples=1000,
    background="linear",
    pen=1,
    model="rbf",
    jump=5,
    plot=False,
    figsize=(20, 3),
    segments_size=None,
    **kwargs
):
    change_points = get_change_points(
        ts,
        model=model,
        jump=jump,
        pen=pen,
        segments_size=segments_size,
        figsize=figsize,
        plot=plot,
    )

    def f(z):
        tss = mask_ts(z, change_points, ts, background)[:, :, np.newaxis]
        return classifier.predict_proba(tss)

    explainer = shap.KernelExplainer(f, data=np.zeros((1, len(change_points))))

    shap_values = explainer.shap_values(
        np.ones((1, len(change_points))), nsamples=nsamples, silent=True
    )
    return shap_values, change_points


def reshape_shap_output_pointwise(shap_values, change_points):
    change_points_indexes = get_change_points_indexes(change_points)
    shap_matrix = []  # (classes, shap_values point by point)
    for shap_array in shap_values:
        shap_array = shap_array.flatten()
        reshaped_shap_array = []
        for i, shap_value in enumerate(shap_array):
            for _ in range(change_points_indexes[i][1] - change_points_indexes[i][0]):
                reshaped_shap_array.append(shap_value)
        shap_matrix.append(reshaped_shap_array)
    return np.array(shap_matrix)


def plot_shap_explanation(
    ts,
    shap_values,
    figsize=(20, 3),
    labels=None,
    dpi=60,
    fontsize=20,
    no_axes_labels=False,
):
    colors_pointwise = []
    minima = np.array(shap_values).min()
    maxima = np.array(shap_values).max()

    # these are here to avoid error in case there aren't values under or over 0 (for DivergingNorm)
    if minima == 0:
        minima -= sys.float_info.epsilon
    if maxima == 0:
        maxima += sys.float_info.epsilon

    norm = matplotlib.colors.DivergingNorm(vmin=minima, vcenter=0, vmax=maxima)

    div = [[norm(minima), "#d62728"], [norm(0), "white"], [norm(maxima), "#2ca02c"]]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", div)

    for shap_array in shap_values:
        normalized_shap_array = norm(shap_array).flatten()
        colors_pointwise.append(normalized_shap_array)
    colors_pointwise = np.array(colors_pointwise)

    for i in range(colors_pointwise.shape[0]):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_title(
            "SHAP - " + r"$b(x)$" + " = " + labels[i] if labels else i,
            fontsize=fontsize,
        )
        ax.plot(ts.reshape(1, -1).T, c="royalblue", alpha=1, lw=3)
        ax.pcolorfast(
            (0, len(colors_pointwise[i, :]) - 1),
            ax.get_ylim(),
            colors_pointwise[i, :][np.newaxis],
            cmap=cmap,
            alpha=1,
            vmin=0,
            vmax=1,
        )
        if not no_axes_labels:
            plt.ylabel("value", fontsize=fontsize)
            plt.xlabel("time-steps", fontsize=fontsize)
        plt.tick_params(axis="both", which="major", labelsize=fontsize)
        fig.show()
        plt.show()


def shap_cmap():
    blues = matplotlib.cm.get_cmap(name="Blues")
    reds = matplotlib.cm.get_cmap(name="Reds")
    cmap = ListedColormap([reds(10), blues(10)])
    return cmap


def plot_color_gradients(cmap_list):
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)
    axs[0].set_title("colormaps", fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=plt.get_cmap(name))
        ax.text(
            -0.01,
            0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    from lasts.blackboxes.loader import cached_blackbox_loader
    from joblib import load
    from lasts.datasets.datasets import build_cbf

    random_state = 0
    # np.random.seed(random_state)
    dataset_name = "cbf"

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
    ) = build_cbf(n_samples=600, random_state=random_state)

    blackbox = cached_blackbox_loader("cbf_rocket.joblib")

    i = 3
    x = X_exp_test[i : i + 1]

    # shapts = ShapTimeSeries()
    kwargs = {
        "nsamples": 1000,
        "background": "linear_consecutive",
        "pen": 1,
        "model": "rbf",
        "jump": 5,
        "plot": False,
        "segments_size": None,
    }
    shapts = ShapTS(blackbox, **kwargs)
    shapts.fit_explain(x)
    shapts.plotter.plot_shap_values()
