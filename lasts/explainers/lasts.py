from lasts.base import ModularExplainer, Plotter, Evaluator, Explanation, Pickler
import warnings
from datetime import datetime
import numpy as np


class Lasts(ModularExplainer):
    def __init__(
        self,
        blackbox,
        encoder,
        decoder,
        neighgen,
        surrogate=None,
        labels=None,
        verbose=False,
        binarize_surrogate_labels=True,
        unpicklable_variables=None,
    ):
        self.blackbox = blackbox
        self.encoder = encoder
        self.decoder = decoder
        self.neighgen = neighgen
        self.surrogate = surrogate
        self.labels = labels
        self.verbose = verbose
        self.binarize_surrogate_labels = binarize_surrogate_labels
        self.plotter = LastsPlotter(self)
        self.evaluator = LastsEvaluator(self)
        self.pickler = Pickler(self)
        if unpicklable_variables is not None:
            self.pickler.save(unpicklable_variables)

        self.x_ = None
        self.z_ = None
        self.Z_ = None
        self.z_tilde_ = None
        self.x_label_ = None
        self.y_ = None
        self.z_tilde_label_ = None
        self.explanation_ = None
        self.runtime_neighgen_ = None
        self.runtime_surrogate_ = None

    def _fit_prepare(self, x, z_fixed=None):
        self.x_ = x
        self.x_label_ = self.blackbox.predict(self.x_)[0]
        self.z_ = self.encoder.predict(x) if z_fixed is None else z_fixed
        self.z_tilde_ = self.decoder.predict(self.z_)
        self.z_label_ = self.blackbox.predict(self.z_tilde_)[0]
        self.z_tilde_label_ = self.blackbox.predict(self.z_tilde_)[0]
        if self.x_label_ != self.z_tilde_label_:
            warnings.warn(
                "The x label before the autoencoding is %s but the label after the autoencoding is %s."
                % (str(self.z_tilde_label_), str(self.x_label_))
            )
        return self

    def _fit_neighgen(self):
        start_time = datetime.now()
        self.neighgen.set_labels(self.labels)
        self.neighgen.fit(self.z_, self.x_label_)
        self.runtime_neighgen_ = (datetime.now() - start_time).total_seconds()
        Z = self.neighgen.predict()
        self.Z_ = Z
        self.Z_tilde_ = self.decoder.predict(Z)
        self.y_ = self.blackbox.predict(self.Z_tilde_)
        return self

    def _fit_surrogate(self):
        start_time = datetime.now()
        # print(self.Z_tilde_.shape, np.unique(self.y_), np.unique(1 * (self.y_ == self.x_label_)))
        if not self.binarize_surrogate_labels:
            self.surrogate.set_labels(self.labels)
            self.surrogate.fit(self.Z_tilde_, self.y_)
        else:
            self.surrogate.set_labels(
                [
                    "not " + str(self.x_label_)
                    if self.labels is None
                    else "not " + self.labels[self.x_label_],
                    str(self.x_label_)
                    if self.labels is None
                    else self.labels[self.x_label_],
                ]
            )
            self.surrogate.fit(self.Z_tilde_, 1 * (self.y_ == self.x_label_))
        self.runtime_surrogate_ = (datetime.now() - start_time).total_seconds()
        return self.surrogate

    def fit_partial(self, x, z_fixed=None):
        self._fit_prepare(x, z_fixed)
        self._fit_neighgen()
        return self

    def fit(self, x, z_fixed=None):
        self.fit_partial(x, z_fixed)
        self._fit_surrogate()
        return self

    def fit_explain(self, x, z_fixed=None):
        self.fit(x, z_fixed)
        return self.explain()

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind, **kwargs)

    def evaluate(self, metric, **kwargs):
        return self.evaluator.evaluate(metric, **kwargs)

    def explain(self):
        neighgen_explanation = self.neighgen.explain()
        surrogate_explanation = self.surrogate.explain(self.z_tilde_)
        explanation = LastsExplanation(neighgen_explanation, surrogate_explanation)
        self.explanation_ = explanation
        return explanation

    def save(self, folder, name=""):
        self.encoder = None  # not really needed
        super().save(folder, name + "_lasts")

    @classmethod
    def load(cls, folder, name="", custom_load=None):
        return super().load(folder, name + "_lasts", custom_load)


class LastsPlotter(Plotter):
    def __init__(self, lasts_explainer):
        self.lasts_explainer = lasts_explainer
        # assert isinstance(self.lasts_explainer.surrogate, ShapeletTree)

    def plot(self, kind, **kwargs):
        if kind == "subsequences":
            self.lasts_explainer.surrogate.plot(kind=kind, **kwargs)

        elif kind == "subsequences_grid":
            self.lasts_explainer.surrogate.plot(
                kind=kind, n=kwargs.pop("n"), m=kwargs.pop("m"), **kwargs
            )

        elif kind == "subsequences_heatmap":
            self.lasts_explainer.surrogate.plot(kind=kind, **kwargs)

        elif kind == "subsequences_binary_heatmap":
            self.lasts_explainer.surrogate.plot(
                kind=kind,
                x_label=self.lasts_explainer.x_label_
                if not self.lasts_explainer.binarize_surrogate_labels
                else 1,
                **kwargs
            )

        elif kind == "tree":
            self.lasts_explainer.surrogate.plot(kind=kind, **kwargs)

        elif kind == "factual_rule":
            self.lasts_explainer.surrogate.plot(
                kind=kind,
                x=self.lasts_explainer.z_tilde_,
                draw_on=self.lasts_explainer.x_,
                **kwargs
            )

        elif kind == "counterfactual_rule":
            self.lasts_explainer.surrogate.plot(
                kind=kind,
                x=self.lasts_explainer.z_tilde_,
                draw_on=self.lasts_explainer.x_,
                **kwargs
            )

        elif kind == "rules":
            self.lasts_explainer.surrogate.plot(
                kind=kind,
                x=self.lasts_explainer.z_tilde_,
                draw_on=self.lasts_explainer.x_,
                **kwargs
            )

        elif kind == "latent_space":
            closest_counterfactual = (
                self.lasts_explainer.neighgen.closest_counterexemplar_
            )
            self.lasts_explainer.neighgen.plot(
                kind=kind,
                closest_counterfactual=closest_counterfactual
                if kwargs.get("closest_counterexemplar") is not None
                else None,
                **kwargs
            )

        elif kind == "latent_space_matrix":
            self.lasts_explainer.neighgen.plot(kind=kind, **kwargs)

        elif kind == "morphing_matrix":
            self.lasts_explainer.neighgen.plot(
                kind=kind, n=kwargs.pop("n", 7), **kwargs
            )

        elif kind == "counterexemplar_interpolation":
            self.lasts_explainer.neighgen.plot(
                kind=kind,
                interpolation=kwargs.pop("interpolation", "linear"),
                n=kwargs.pop("n", 100),
                **kwargs
            )

        elif kind == "manifest_space":
            self.lasts_explainer.neighgen.plot(
                kind=kind, x=self.lasts_explainer.x_, **kwargs
            )

        elif kind == "saliency_map":
            self.lasts_explainer.neighgen.plot(
                "variation_delta", draw_on=self.lasts_explainer.x_.ravel(), **kwargs
            )

        elif kind == "counterexemplar_shape_change":
            self.lasts_explainer.neighgen.plot(kind, **kwargs)

        else:
            raise Exception("Plot kind not valid.")

        return self


class LastsEvaluator(Evaluator):
    METRICS = [
        "surrogate_tree_factual_coverage",
        "surrogate_tree_factual_precision",
        "surrogate_fidelity",
        "surrogate_fidelity_x",
        "surrogate_tree_counterfactual_coverage",
        "surrogate_tree_counterfactual_precision",
        "silhouette_latent",
        "silhouette_manifest",
        "silhouette_latent_binary",
        "lof_latent",
        "lof_manifest",
        "lof_latent_z",
        "lof_manifest_z",
        "factual_rule_length",
        "counterfactual_rule_length",
        "surrogate_runtime",
        "neighgen_runtime",
        "total_runtime",
    ]

    BENCHMARKS = ["usefulness"]

    def __init__(self, lasts_explainer: Lasts):
        self.lasts_explainer = lasts_explainer

    def usefulness_benchmark(self, **kwargs):
        return self.lasts_explainer.neighgen.evaluate(metric="usefulness", **kwargs)

    def factual_rule_length(self, **kwargs):
        return len(self.lasts_explainer.explanation_.surrogate_explanation.factual_rule)

    def counterfactual_rule_length(self, **kwargs):
        return len(
            self.lasts_explainer.explanation_.surrogate_explanation.counterfactual_rule
        )

    def evaluate(self, metric, **kwargs):
        if metric == "surrogate_tree_factual_coverage":
            return self.lasts_explainer.surrogate.evaluate(
                metric="tree_coverage", x=self.lasts_explainer.z_tilde_
            )
        elif metric == "surrogate_tree_factual_precision":
            return self.lasts_explainer.surrogate.evaluate(
                metric="tree_precision",
                X=self.lasts_explainer.surrogate.X_,
                x=self.lasts_explainer.z_tilde_,
                y=self.lasts_explainer.surrogate.y_,
            )
        elif metric == "surrogate_tree_counterfactual_precision":
            counterfactual_ts = self.lasts_explainer.surrogate.find_counterfactual_tss(
                self.lasts_explainer.z_tilde_
            )[0:1]
            return self.lasts_explainer.surrogate.evaluate(
                metric="tree_precision",
                X=self.lasts_explainer.surrogate.X_,
                x=counterfactual_ts,
                y=self.lasts_explainer.surrogate.y_,
            )

        elif metric == "surrogate_tree_counterfactual_coverage":
            counterfactual_ts = self.lasts_explainer.surrogate.find_counterfactual_tss(
                self.lasts_explainer.z_tilde_
            )[0:1]
            return self.lasts_explainer.surrogate.evaluate(
                metric="tree_coverage", x=counterfactual_ts
            )

        elif metric == "surrogate_fidelity":
            return self.lasts_explainer.surrogate.evaluate(
                metric="accuracy_score",
                X=self.lasts_explainer.surrogate.X_,
                y=self.lasts_explainer.surrogate.y_  # no need to binarize because we take the dataset inside the
                # surrogate model that is already transformed in the right way
            )

        elif metric == "surrogate_fidelity_x":
            return self.lasts_explainer.surrogate.evaluate(
                metric="accuracy_score",
                X=self.lasts_explainer.z_tilde_,
                y=np.array([self.lasts_explainer.x_label_])
                if not self.lasts_explainer.binarize_surrogate_labels
                else np.array([1]),
            )

        elif metric == "silhouette_latent":
            return self.lasts_explainer.neighgen.evaluate(metric=metric, **kwargs)

        elif metric == "silhouette_manifest":
            return self.lasts_explainer.neighgen.evaluate(
                metric=metric, distance=kwargs.pop("distance", "euclidean"), **kwargs
            )

        elif metric == "silhouette_latent_binary":
            return self.lasts_explainer.neighgen.evaluate(metric=metric, **kwargs)

        elif metric == "silhouette_manifest_binary":
            return self.lasts_explainer.neighgen.evaluate(
                metric=metric, distance=kwargs.pop("distance", "euclidean"), **kwargs
            )
        elif metric == "lof_latent":
            return self.lasts_explainer.neighgen.evaluate(metric=metric, **kwargs)

        elif metric == "lof_manifest":
            return self.lasts_explainer.neighgen.evaluate(
                metric=metric, distance=kwargs.pop("distance", "euclidean"), **kwargs
            )

        elif metric == "lof_latent_z":
            return self.lasts_explainer.neighgen.evaluate(metric=metric, **kwargs)

        elif metric == "lof_manifest_z":
            return self.lasts_explainer.neighgen.evaluate(
                metric=metric, distance=kwargs.pop("distance", "euclidean"), **kwargs
            )
        elif metric == "factual_rule_length":
            return self.factual_rule_length(**kwargs)

        elif metric == "counterfactual_rule_length":
            return self.counterfactual_rule_length(**kwargs)

        elif metric == "surrogate_runtime":
            return self.lasts_explainer.runtime_surrogate_

        elif metric == "neighgen_runtime":
            return self.lasts_explainer.runtime_neighgen_

        elif metric == "total_runtime":
            return (
                self.lasts_explainer.runtime_surrogate_
                + self.lasts_explainer.runtime_neighgen_
            )

        elif metric == "usefulness_benchmark":
            return self.usefulness_benchmark(**kwargs)

        else:
            raise Exception("Metric not valid.")


class LastsExplanation(Explanation):
    def __init__(self, neighgen_explanation, surrogate_explanation):
        self.neighgen_explanation = neighgen_explanation
        self.surrogate_explanation = surrogate_explanation

    def __str__(self):
        return "%s\n%s" % (
            str(self.neighgen_explanation),
            str(self.surrogate_explanation),
        )

    def explain(self):
        return self.neighgen_explanation.explain(), self.surrogate_explanation.explain()


if __name__ == "__main__":
    from lasts.blackboxes.loader import cached_blackbox_loader
    from lasts.datasets.datasets import build_cbf
    from lasts.autoencoders.variational_autoencoder_v2 import load_model
    from lasts.autoencoders.variational_autoencoder import load_model
    from lasts.utils import get_project_root, choose_z
    from lasts.surrogates.sax_tree import SaxTree
    from lasts.surrogates.shapelet_tree import ShapeletTree
    from lasts.neighgen.counter_generator import CounterGenerator
    from lasts.wrappers import DecoderWrapper
    from lasts.surrogates.utils import generate_n_shapelets_per_size

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

    # encoder, decoder, autoencoder = load_model(get_project_root() / "autoencoders" / "cached" / "vae_v2" /
    #                                            "cbf__ldim2" / "cbf__ldim2_vae")
    encoder, decoder, autoencoder = load_model(
        get_project_root() / "autoencoders" / "cached" / "vae" / "cbf" / "cbf_vae"
    )

    i = 0
    x = X_exp_test[i].ravel().reshape(1, -1, 1)
    z_fixed = choose_z(
        x,
        encoder,
        decoder,
        n=1000,
        x_label=blackbox.predict(x)[0],
        blackbox=blackbox,
        check_label=True,
        mse=False,
    )

    neighgen = CounterGenerator(
        blackbox,
        DecoderWrapper(decoder),
        n_search=10000,
        n_batch=1000,
        lower_threshold=0,
        upper_threshold=4,
        kind="gaussian_matched",
        sampling_kind="uniform_sphere",
        vicinity_sampler_kwargs=dict(),
        stopping_ratio=0.01,
        check_upper_threshold=True,
        final_counterfactual_search=True,
        verbose=True,
        custom_sampling_threshold=None,
        custom_closest_counterfactual=None,
        n=500,
        balance=False,
        forced_balance_ratio=0.5,
        redo_search=True,
        cut_radius=True,
    )

    n_shapelets_per_size = generate_n_shapelets_per_size(X_exp_train.shape[1])
    shapelet_model_kwargs = {
        "l": 0.1,
        "r": 2,
        "optimizer": "sgd",
        "n_shapelets_per_size": n_shapelets_per_size,
        "weight_regularizer": 0.01,
        "max_iter": 100,
    }

    # surrogate = ShapeletTree(random_state=random_state, shapelet_model_kwargs=shapelet_model_kwargs)
    surrogate = SaxTree(random_state=random_state)

    lasts_ = Lasts(
        blackbox,
        encoder,
        DecoderWrapper(decoder),
        neighgen,
        surrogate,
        verbose=True,
        binarize_surrogate_labels=True,
        labels=["cylinder", "bell", "funnel"],
    )

    lasts_.fit(x, z_fixed)

    exp = lasts_.explain()

    lasts_.plot("latent_space")
    lasts_.plot("morphing_matrix")
    lasts_.plot("counterexemplar_interpolation")
    lasts_.plot("manifest_space")
    lasts_.plot("saliency_map")

    lasts_.plot("subsequences_heatmap")
    lasts_.plot("rules")
    lasts_.neighgen.plotter.plot_counterexemplar_shape_change()
