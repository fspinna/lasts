import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import LocalOutlierFactor

from lasts.metrics import dtw_variation_delta
from lasts.neighgen.utils import vicinity_sampling, binary_sampling_search
from sklearn.model_selection import train_test_split
import warnings

from lasts.wrappers import DecoderBlackboxWrapper, DecoderWrapper
from lasts.base import (
    Plotter,
    Evaluator,
    NeighborhoodGenerator,
    ExampleBasedExplanation,
    Pickler,
    Explanation,
    SaliencyBasedExplanation,
)
from lasts.plots import (
    plot_latent_space_z,
    plot_latent_space_matrix_z,
    morphing_matrix,
    plot_interpolation,
    plot_exemplars_and_counterexemplars,
    plot_scattered_feature_importance,
    plot_feature_importance,
    plot_feature_importance_on_ts,
    plot_changing_shape,
)
from lasts.utils import usefulness_scores

from sklearn.metrics import silhouette_score
from tslearn.metrics import dtw


class NormalGenerator(object):
    def __init__(self, **kwargs):
        pass

    def generate_neighborhood(self, z, n=1000, **kwargs):
        Z = np.random.normal(size=(n, z.shape[1]))
        return Z


class CounterGenerator(NeighborhoodGenerator):
    def __init__(
        self,
        blackbox,
        decoder=None,
        n_search=10000,
        n_batch=1000,
        lower_threshold=0,
        upper_threshold=4,
        kind="gaussian_matched",
        sampling_kind=None,
        vicinity_sampler_kwargs=dict(),
        stopping_ratio=0.01,
        check_upper_threshold=True,
        final_counterfactual_search=True,
        verbose=True,
        custom_sampling_threshold=None,
        custom_closest_counterfactual=None,
        n=500,
        balance=False,
        forced_balance_ratio=None,
        redo_search=True,
        cut_radius=False,
        labels=None,
        unpicklable_variables=None,
    ):
        """
        Search and generates a neighborhood around z
        Parameters
        ----------
        n_search: int, optional (default = 10000)
            total n. of instances generated at any step of the search algorithm
        n_batch: int, optional (default = 1000)
            batch n. of instances generated at any step of the search algorithm
        lower_threshold: int, optional (default = 0)
            threshold to refine the search, only used if downward_only=False
        upper_threshold: int, optional (default = 4)
            starting threshold
        kind: string, optional (default = "gaussian_matched")
            counterfactual search kind
        sampling_kind: string, optional (default = None)
            sampling_kind, if None the sampling kind is the same as kind
        vicinity_sampler_kwargs: dictionary, optional (default = dict())
        stopping_ratio: float, optional (default = 0.01)
            ratio at which to stop the counterfactual search algorithm i.e. stop if
            lower_threshold/upper_threshold < stopping_ratio. Only used if downward_only=True
        check_upper_threshold: bool, optional (default = True)
            check if with the starting upper threshold the search finds some counterexemplars
        final_counterfactual_search: bool, optional (default = True)
            after the search algorithm stops, search a last time for counterexemplars
        verbose: bool, optional (default = True)
        custom_sampling_threshold: float, optional (default = None)
            pass a threshold directly without searching it
        custom_closest_counterfactual: array of size (1, n_features), optional (default = None)
            pass a counterexemplar directly
        n: int, optional (default = 500)
            n. of instance of the neighborhood
        balance: bool, optional (default = False)
            balance the neighborhood labels after generating it (reduces n)
        forced_balance_ratio: float, optional (default = None)
            balance the neighborhood labels while generating it.
            A value of 0.5 means we want the same n. of instances per label
        redo_search: bool, optional (default = True)
            redo the search if even if it has been run before
        cut_radius:
            after the search of the counterexemplar and best_threshold, replace the threshold with the distance between
            the counterexemplar and z (useful only if the threshold of sampling_kind is a distance)
        kwargs
        """

        self.blackbox = blackbox
        self.decoder = decoder
        if decoder is not None:
            # FIXME: horrible, should find another solution
            if unpicklable_variables is not None:
                unpicklable_variables_ = list()
                for var in unpicklable_variables:
                    if var in ["blackbox", "decoder"]:
                        unpicklable_variables_.append(var)
            else:
                unpicklable_variables_ = None
            self.predictor = DecoderBlackboxWrapper(
                DecoderWrapper(
                    decoder.decoder
                ),  # FIXME: don't like it but to pickle the object it's mandatory
                blackbox,
                unpicklable_variables=unpicklable_variables_,
            )
        else:
            self.predictor = blackbox
        self.n_search = n_search
        self.n_batch = n_batch
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.kind = kind
        self.sampling_kind = sampling_kind
        self.vicinity_sampler_kwargs = vicinity_sampler_kwargs
        self.stopping_ratio = stopping_ratio
        self.check_upper_threshold = check_upper_threshold
        self.final_counterfactual_search = final_counterfactual_search
        self.verbose = verbose
        self.custom_sampling_threshold = custom_sampling_threshold
        self.custom_closest_counterfactual = custom_closest_counterfactual
        self.n = n
        self.balance = balance
        self.forced_balance_ratio = forced_balance_ratio
        self.redo_search = redo_search
        self.cut_radius = cut_radius
        self.labels = labels

        self.plotter = CounterGeneratorPlotter(self)
        self.evaluator = CounterGeneratorEvaluator(self)
        self.pickler = Pickler(self)
        if unpicklable_variables is not None:
            self.pickler.save(unpicklable_variables)

        self.closest_counterexemplar_ = None
        self.best_threshold_ = None
        self.z_ = None
        self.z_tilde_ = None
        self.z_label_ = None
        self.Z_ = None
        self.y_ = None
        self.Z_tilde_ = None

    def fit(self, z, z_label=None, **kwargs):
        """Search and generates a neighborhood around z
        Parameters
        ----------
        z: array of shape (1, n_features)
            instance to explain
        Returns
        -------
        Z : array of size (n, n_features)
            generated neighborhood
        """
        self.z_ = z
        self.z_tilde_ = self.decoder.predict(self.z_)
        if z_label is None:
            z_label = self.predictor.predict(z)[
                0
            ]  # FIXME: was without [0] before, seem to work anyway
        else:
            #  TODO: maybe should check if z_label==self.predictor.predict(z) and raise a warning
            pass
        self.z_label_ = z_label
        if (self.closest_counterexemplar_ is None or self.redo_search) and (
            self.custom_closest_counterfactual is None
        ):
            self._counterfactual_search(
                z=z,
                z_label=z_label,
                n_search=self.n_search,
                n_batch=self.n_batch,
                lower_threshold=self.lower_threshold,
                upper_threshold=self.upper_threshold,
                kind=self.kind,
                vicinity_sampler_kwargs=self.vicinity_sampler_kwargs,
                stopping_ratio=self.stopping_ratio,
                check_upper_threshold=self.check_upper_threshold,
                final_counterfactual_search=self.final_counterfactual_search,
                verbose=self.verbose,
                **kwargs,
            )

        kind = self.sampling_kind if self.sampling_kind is not None else self.kind

        Z = self._neighborhood_sampling(
            z=z,
            z_label=z_label,
            n_batch=self.n_batch,
            kind=kind,
            vicinity_sampler_kwargs=self.vicinity_sampler_kwargs,
            verbose=self.verbose,
            custom_sampling_threshold=self.custom_sampling_threshold,
            custom_closest_counterfactual=self.custom_closest_counterfactual,
            n=self.n,
            balance=self.balance,
            forced_balance_ratio=self.forced_balance_ratio,
            cut_radius=self.cut_radius,
            **kwargs,
        )
        self.Z_ = Z
        if self.decoder is not None:
            self.Z_tilde_ = self.decoder.predict(self.Z_)
        self.y_ = self.predictor.predict(self.Z_)
        return self

    def predict(self):
        return self.Z_

    def _counterfactual_search(
        self,
        z,
        z_label=None,
        n_search=10000,
        n_batch=1000,
        lower_threshold=0,
        upper_threshold=4,
        kind="gaussian_matched",
        vicinity_sampler_kwargs=dict(),
        stopping_ratio=0.01,
        check_upper_threshold=True,
        final_counterfactual_search=True,
        verbose=True,
        **kwargs,
    ):
        if z_label is None:
            z_label = self.predictor.predict(z)
        self.closest_counterexemplar_, self.best_threshold_ = binary_sampling_search(
            z=z,
            z_label=z_label,
            blackbox=self.predictor,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            n=n_search,
            n_batch=n_batch,
            kind=kind,
            vicinity_sampler_kwargs=vicinity_sampler_kwargs,
            stopping_ratio=stopping_ratio,
            check_upper_threshold=check_upper_threshold,
            final_counterfactual_search=final_counterfactual_search,
            verbose=verbose,
            **kwargs,
        )

        return self.closest_counterexemplar_, self.best_threshold_

    def _neighborhood_sampling(
        self,
        z,
        z_label=None,
        n_batch=1000,
        kind="gaussian_matched",
        vicinity_sampler_kwargs=dict(),
        verbose=True,
        custom_sampling_threshold=None,
        custom_closest_counterfactual=None,
        n=500,
        balance=False,
        forced_balance_ratio=None,
        cut_radius=False,
        **kwargs,
    ):
        if z_label is None:
            z_label = self.predictor.predict(z)
        if custom_closest_counterfactual is not None:
            self.closest_counterexemplar_ = custom_closest_counterfactual
        if cut_radius:
            self.best_threshold_ = np.linalg.norm(z - self.closest_counterexemplar_)
            if verbose:
                print("Setting new threshold at radius:", self.best_threshold_)
            if kind not in ["uniform_sphere"]:
                warnings.warn(
                    "cut_radius=True, but for the method "
                    + kind
                    + " the threshold is not a radius."
                )
        if custom_sampling_threshold is not None:
            self.best_threshold_ = custom_sampling_threshold
            if verbose:
                print("Setting custom threshold:", self.best_threshold_)

        Z = vicinity_sampling(
            z=self.closest_counterexemplar_,
            n=n,
            threshold=self.best_threshold_,
            kind=kind,
            verbose=verbose,
            **vicinity_sampler_kwargs,
        )

        if forced_balance_ratio is not None:
            y = self.predictor.predict(Z)
            y = 1 * (y == z_label)
            n_minority_instances = np.unique(y, return_counts=True)[1].min()
            if (n_minority_instances / n) < forced_balance_ratio:
                if verbose:
                    print("Forced balancing neighborhood...", end=" ")
                n_desired_minority_instances = int(forced_balance_ratio * n)
                n_desired_majority_instances = n - n_desired_minority_instances
                minority_class = np.argmin(np.unique(y, return_counts=True)[1])
                sampling_strategy = (
                    n_desired_minority_instances / n_desired_majority_instances
                )
                while n_minority_instances < n_desired_minority_instances:
                    Z_ = vicinity_sampling(
                        z=self.closest_counterexemplar_,
                        n=n_batch,
                        threshold=self.best_threshold_
                        if custom_sampling_threshold is None
                        else custom_sampling_threshold,
                        kind=kind,
                        verbose=False,
                        **vicinity_sampler_kwargs,
                    )
                    y_ = self.predictor.predict(Z_)
                    y_ = 1 * (y_ == z_label)
                    n_minority_instances += np.unique(y_, return_counts=True)[1][
                        minority_class
                    ]
                    Z = np.concatenate([Z, Z_])
                    y = np.concatenate([y, y_])
                rus = RandomUnderSampler(
                    random_state=0, sampling_strategy=sampling_strategy
                )
                Z, y = rus.fit_resample(Z, y)
                if len(Z) > n:
                    Z, _ = train_test_split(Z, train_size=n, stratify=y)
                if verbose:
                    print("Done!")

        if balance:
            if verbose:
                print("Balancing neighborhood...", end=" ")
            rus = RandomUnderSampler(random_state=0)
            y = self.predictor.predict(Z)
            y = 1 * (y == self.predictor.predict(z))
            Z, _ = rus.fit_resample(Z, y)
            if verbose:
                print("Done!")
        return Z

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind, **kwargs)

    def evaluate(self, metric, **kwargs):
        return self.evaluator.evaluate(metric, **kwargs)

    def explain(self):
        exemplars = self.Z_tilde_[np.nonzero(self.y_ == self.z_label_)]
        counterexemplars = self.Z_tilde_[np.nonzero(self.y_ != self.z_label_)]
        example_explanation = ExampleBasedExplanation(exemplars, counterexemplars)
        saliency = np.abs(
            self.z_tilde_ - self.decoder.predict(self.closest_counterexemplar_)
        )  # .ravel())
        # saliency = dtw_variation_delta(self.z_tilde_.ravel(), self.decoder.predict(
        #     self.closest_counterexemplar_).ravel())
        saliency_explanation = SaliencyBasedExplanation(saliency)
        explanation = CounterGeneratorExplanation(
            example_explanation, saliency_explanation
        )
        return explanation

    def save(self, folder, name=""):
        super().save(folder, name + "_neighgen")

    @classmethod
    def load(cls, folder, name="", custom_load=None):
        return super().load(folder, name + "_neighgen", custom_load)


class CounterGeneratorPlotter(Plotter):
    PLOTS = []

    def __init__(self, counter_generator: CounterGenerator):
        self.counter_generator = counter_generator

    def plot_bidimensional_latent_space(self, **kwargs):
        plot_latent_space_z(
            Z=self.counter_generator.Z_,
            y=self.counter_generator.y_,
            z=self.counter_generator.z_,
            z_label=self.counter_generator.z_label_,
            legend=True,
            **kwargs,
        )

    def plot_multidimensional_latent_space(self, **kwargs):
        plot_latent_space_matrix_z(
            Z=self.counter_generator.Z_,
            y=self.counter_generator.y_,
            z=self.counter_generator.z_,
            z_label=self.counter_generator.z_label_,
            **kwargs,
        )

    def plot_morphing_matrix(self, n, **kwargs):
        morphing_matrix(
            blackbox=self.counter_generator.blackbox,
            decoder=self.counter_generator.decoder,
            x_label=self.counter_generator.z_label_,
            labels=self.counter_generator.labels,
            n=n,
            **kwargs,
        )

    def plot_counterexemplar_interpolation(self, interpolation, n, **kwargs):
        plot_interpolation(
            z=self.counter_generator.z_,
            z_prime=self.counter_generator.closest_counterexemplar_,
            x_label=self.counter_generator.z_label_,
            decoder=self.counter_generator.decoder,
            blackbox=self.counter_generator.blackbox,
            interpolation=interpolation,
            n=n,
            **kwargs,
        )

    def plot_manifest_space(self, x=None, **kwargs):
        plot_exemplars_and_counterexemplars(
            Z_tilde=self.counter_generator.Z_tilde_,
            y=self.counter_generator.y_,
            x=x
            if x is not None
            else self.counter_generator.z_tilde_,  # if x is not given
            z_tilde=self.counter_generator.z_tilde_,
            x_label=self.counter_generator.z_label_,
            labels=self.counter_generator.labels,
            plot_x=False if x is None else True,
            **kwargs,
        )
        return self

    def plot_variation_delta_scatter(self, **kwargs):
        delta = np.abs(
            self.counter_generator.z_tilde_.ravel()
            - self.counter_generator.decoder.predict(
                self.counter_generator.closest_counterexemplar_
            ).ravel()
        )
        plot_scattered_feature_importance(
            ts=self.counter_generator.z_tilde_.ravel(), feature_importance=delta
        )

    def plot_variation_delta(self, **kwargs):
        delta = np.abs(
            self.counter_generator.z_tilde_.ravel()
            - self.counter_generator.decoder.predict(
                self.counter_generator.closest_counterexemplar_
            ).ravel()
        )
        plot_feature_importance(
            ts=self.counter_generator.z_tilde_.ravel(), feature_importance=delta
        )

    def plot_variation_delta_on_ts(self, **kwargs):
        delta = np.abs(
            self.counter_generator.z_tilde_.ravel()
            - self.counter_generator.decoder.predict(
                self.counter_generator.closest_counterexemplar_
            ).ravel()
        )
        draw_on = kwargs.pop("draw_on")
        plot_feature_importance_on_ts(
            ts=self.counter_generator.z_tilde_.ravel() if draw_on is None else draw_on,
            feature_importance=delta,
            **kwargs,
        )

    def plot_counterexemplar_shape_change(self, **kwargs):
        plot_changing_shape(
            self.counter_generator.z_tilde_.ravel(),
            self.counter_generator.decoder.predict(
                self.counter_generator.closest_counterexemplar_
            ).ravel(),
            np.abs(
                self.counter_generator.z_tilde_.ravel()
                - self.counter_generator.decoder.predict(
                    self.counter_generator.closest_counterexemplar_
                ).ravel()
            ),
            **kwargs,
        )

    def plot(self, kind, **kwargs):
        if kind == "latent_space":
            self.plot_bidimensional_latent_space(**kwargs)

        elif kind == "latent_space_matrix":
            self.plot_multidimensional_latent_space(**kwargs)

        elif kind == "morphing_matrix":
            self.plot_morphing_matrix(n=kwargs.pop("n", 7), **kwargs)

        elif kind == "counterexemplar_interpolation":
            self.plot_counterexemplar_interpolation(
                interpolation=kwargs.pop("interpolation", "linear"),
                n=kwargs.pop("n", 100),
                **kwargs,
            )

        elif kind == "manifest_space":
            self.plot_manifest_space(x=kwargs.pop("x", None), **kwargs)

        elif kind == "variation_delta":
            self.plot_variation_delta_on_ts(**kwargs)

        elif kind == "counterexemplar_shape_change":
            self.plot_counterexemplar_shape_change(**kwargs)

        else:
            raise Exception("Plot kind not valid.")


class CounterGeneratorEvaluator(Evaluator):
    METRICS = []
    BENCHMARKS = []

    def __init__(self, counter_generator: CounterGenerator):
        self.counter_generator = counter_generator

    def usefulness_benchmark(self, n=[1, 2, 4, 8, 16], binarize=False, **kwargs):
        Z_tilde_shape = self.counter_generator.Z_tilde_.shape
        if binarize:
            accuracy_by_n = usefulness_scores(
                self.counter_generator.Z_tilde_.reshape(
                    -1, Z_tilde_shape[1] * Z_tilde_shape[2]
                ),
                1 * (self.counter_generator.y_ == self.counter_generator.z_label_),
                self.counter_generator.z_tilde_.reshape(
                    -1, Z_tilde_shape[1] * Z_tilde_shape[2]
                ),  # FIXME: or x?
                1,
                n=n,
            )
        else:
            accuracy_by_n = usefulness_scores(
                self.counter_generator.Z_tilde_.reshape(
                    -1, Z_tilde_shape[1] * Z_tilde_shape[2]
                ),
                self.counter_generator.y_,
                self.counter_generator.z_tilde_.reshape(
                    -1, Z_tilde_shape[1] * Z_tilde_shape[2]
                ),  # FIXME: or x?
                self.counter_generator.z_label_,
                n=n,
            )
        return accuracy_by_n

    def latent_silhouette_score(self, **kwargs):
        return silhouette_score(self.counter_generator.Z_, self.counter_generator.y_)

    def latent_binary_silhouette_score(self, **kwargs):
        y = 1 * (self.counter_generator.y_ == self.counter_generator.z_label_)
        return silhouette_score(self.counter_generator.Z_, y)

    def manifest_silhouette_score(self, distance="euclidean", **kwargs):
        if distance == "dtw":
            metric = dtw
        else:
            metric = distance
        return silhouette_score(
            self.counter_generator.Z_tilde_[:, :, 0],
            self.counter_generator.y_,
            metric=metric,
        )

    def manifest_binary_silhouette_score(self, distance="euclidean", **kwargs):
        y = 1 * (self.counter_generator.y_ == self.counter_generator.z_label_)
        if distance == "dtw":
            metric = dtw
        else:
            metric = distance
        return silhouette_score(
            self.counter_generator.Z_tilde_[:, :, 0], y, metric=metric
        )

    def latent_lof_score(self, **kwargs):
        lof = LocalOutlierFactor(metric="euclidean", novelty=True)
        lof.fit(self.counter_generator.Z_)
        lof_scores = lof.predict(self.counter_generator.Z_)
        return lof_scores.mean()

    def latent_lof_score_z(self, **kwargs):
        lof = LocalOutlierFactor(metric="euclidean", novelty=True)
        lof.fit(self.counter_generator.Z_)
        lof_score = lof.predict(self.counter_generator.z_)[0]
        return lof_score

    def manifest_lof_score(self, distance="euclidean", **kwargs):
        if distance == "dtw":
            metric = dtw
        else:
            metric = distance
        lof = LocalOutlierFactor(metric=metric, novelty=True)
        lof.fit(self.counter_generator.Z_tilde_[:, :, 0])
        lof_scores = lof.predict(self.counter_generator.Z_tilde_[:, :, 0])
        return lof_scores.mean()

    def manifest_lof_score_z(self, distance="euclidean", **kwargs):
        if distance == "dtw":
            metric = dtw
        else:
            metric = distance
        lof = LocalOutlierFactor(metric=metric, novelty=True)
        lof.fit(self.counter_generator.Z_tilde_[:, :, 0])
        lof_score = lof.predict(self.counter_generator.z_tilde_[:, :, 0])[0]
        return lof_score

    def evaluate(self, metric, **kwargs):
        if metric == "silhouette_latent":
            return self.latent_silhouette_score(**kwargs)
        elif metric == "silhouette_manifest":
            return self.manifest_silhouette_score(
                metric=kwargs.pop("distance", "euclidean"), **kwargs
            )
        elif metric == "silhouette_latent_binary":
            return self.latent_binary_silhouette_score(**kwargs)
        elif metric == "silhouette_manifest_binary":
            return self.manifest_binary_silhouette_score(
                metric=kwargs.pop("distance", "euclidean"), **kwargs
            )
        elif metric == "lof_latent":
            return self.latent_lof_score(**kwargs)
        elif metric == "lof_manifest":
            return self.manifest_lof_score(
                metric=kwargs.pop("distance", "euclidean"), **kwargs
            )
        elif metric == "lof_latent_z":
            return self.latent_lof_score_z(**kwargs)
        elif metric == "lof_manifest_z":
            return self.manifest_lof_score_z(
                metric=kwargs.pop("distance", "euclidean"), **kwargs
            )
        elif metric == "usefulness":
            return self.usefulness_benchmark(**kwargs)
        else:
            raise Exception("Metric not valid.")


class CounterGeneratorExplanation(Explanation):
    def __init__(
        self,
        example_explanation: ExampleBasedExplanation,
        saliency_explanation: SaliencyBasedExplanation,
    ):
        self.example_explanation = example_explanation
        self.saliency_explanation = saliency_explanation

    def explain(self):
        return self.example_explanation.explain(), self.saliency_explanation.explain()


if __name__ == "__main__":
    from lasts.deprecated.blackboxes import blackbox_loader
    from lasts.datasets.datasets import build_cbf
    from lasts.autoencoders.variational_autoencoder_v2 import load_model
    from lasts.utils import get_project_root, choose_z, usefulness_scores
    from lasts.benchmarks.usefulness import usefulness_scores

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

    blackbox = blackbox_loader("cbf_knn.joblib")

    encoder, decoder, autoencoder = load_model(
        get_project_root()
        / "autoencoders"
        / "cached"
        / "vae_v2"
        / "cbf__ldim2"
        / "cbf__ldim2_vae"
    )

    i = 2
    x = X_exp_test[i].ravel().reshape(1, -1, 1)
    x_label = blackbox.predict(x)[0]
    z = choose_z(
        x,
        encoder,
        decoder,
        n=1000,
        x_label=x_label,
        blackbox=blackbox,
        check_label=True,
        mse=False,
    )

    neigh = CounterGenerator(
        blackbox,
        decoder,
        n_search=10000,
        n_batch=1000,
        lower_threshold=0,
        upper_threshold=4,
        kind="uniform_sphere",
        sampling_kind=None,
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
        labels=["cyl", "bell", "fun"],
    )

    neigh.fit(z)

    Z = neigh.predict()
    neigh.plot("latent_space")
    neigh.plot("latent_space_matrix")
    neigh.plot("morphing_matrix")
    neigh.plot("counterexemplar_interpolation")
    neigh.plot("variation_delta")

    print(neigh.evaluate("silhouette_manifest", distance="dtw"))
    print(neigh.evaluate("silhouette_latent"))
    print(neigh.evaluate("silhouette_manifest_binary", distance="dtw"))
    print(neigh.evaluate("silhouette_latent_binary"))
    print(neigh.evaluate("lof_latent", distance="dtw"))
    print(neigh.evaluate("lof_manifest"))
    print(neigh.evaluate("lof_latent_z", distance="dtw"))
    print(neigh.evaluate("lof_manifest_z"))
