import matplotlib
import matplotlib.pyplot as plt

from lasts.explainers.shapts import ShapTS
from lasts.explainers.shapts import shap_ts, reshape_shap_output_pointwise
from lasts.base import BlackBox, Plotter
from lasts.plots import plot_feature_importance_on_ts_multi
from datetime import datetime


def from3dto2d(ts_3d):
    return ts_3d.transpose(0, 2, 1).reshape(-1, ts_3d.shape[1] * ts_3d.shape[2])


def from2dto3d(ts_2d, shape_3d):
    return ts_2d.reshape(-1, shape_3d[2], shape_3d[1]).transpose(0, 2, 1)


class Wrapper3D2D(BlackBox):
    def __init__(self, blackbox, shape_3d):
        self.blackbox = blackbox
        self.shape_3d = shape_3d

    def predict(self, X):
        return self.blackbox.predict(from2dto3d(X, shape_3d=self.shape_3d))

    def predict_proba(self, X):
        return self.blackbox.predict_proba(from2dto3d(X, shape_3d=self.shape_3d))


class ShapTSMulti(ShapTS):
    def __init__(self, blackbox, shape_3d, **kwargs):
        super().__init__(blackbox, **kwargs)
        self.blackbox = Wrapper3D2D(blackbox=blackbox, shape_3d=shape_3d)
        self.plotter = ShapTSMultiPlotter(self)

    def fit(self, x):
        start_time = datetime.now()
        self.x_ = x
        self.x_label_ = self.blackbox.predict(x)[0]
        shap_values, change_points = shap_ts(
            ts=from3dto2d(x).ravel(), classifier=self.blackbox, **self.kwargs
        )
        shap_values_reshaped = reshape_shap_output_pointwise(shap_values, change_points)
        self.shap_values_ = from2dto3d(shap_values_reshaped.reshape(1, -1), x.shape)
        self.runtime_ = (datetime.now() - start_time).total_seconds()
        return self


class ShapTSMultiPlotter(Plotter):
    def __init__(self, shap_explainer: ShapTSMulti):
        self.shap_explainer = shap_explainer

    def plot_shap_values(self, figsize=(20, 5), dpi=72, **kwargs):
        norm = matplotlib.colors.CenteredNorm()
        norm(
            self.shap_explainer.shap_values_
        )  # this way the norm checks all the shap values
        plot_feature_importance_on_ts_multi(
            self.shap_explainer.x_,
            self.shap_explainer.explanation_.saliency,
            cmap="coolwarm_r",
            norm=norm,
            figsize=figsize,
            dpi=dpi,
            **kwargs
        )

    def plot(self, kind, **kwargs):
        pass


if __name__ == "__main__":
    import numpy as np
    from lasts.datasets.datasets import build_multivariate_cbf
    from lasts.plots import plot_mts_array

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
    ) = build_multivariate_cbf()

    blackbox = cached_blackbox_loader("cbfmulti_rocket.joblib")

    i = 3
    x = X_exp_test[i : i + 1]

    kwargs = {
        "nsamples": 1000,
        "background": "linear_consecutive",
        "pen": 1,
        "model": "rbf",
        "jump": 5,
        "plot": False,
        "segments_size": None,
    }
    shapts = ShapTSMulti(blackbox, shape_3d=x.shape, **kwargs)
    shapts.fit_explain(x)
    shapts.plotter.plot_shap_values()
