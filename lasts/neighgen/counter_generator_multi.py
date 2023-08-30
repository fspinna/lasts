from lasts.base import SaliencyBasedExplanation, ExampleBasedExplanation
from lasts.neighgen.counter_generator import (
    CounterGenerator,
    CounterGeneratorPlotter,
    CounterGeneratorEvaluator,
    CounterGeneratorExplanation,
)
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

from lasts.plots import (
    plot_exemplars_and_counterexemplars_multi,
    plot_feature_importance_on_ts_multi,
)


class CounterGeneratorMulti(CounterGenerator):
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
        super().__init__(
            blackbox,
            decoder,
            n_search,
            n_batch,
            lower_threshold,
            upper_threshold,
            kind,
            sampling_kind,
            vicinity_sampler_kwargs,
            stopping_ratio,
            check_upper_threshold,
            final_counterfactual_search,
            verbose,
            custom_sampling_threshold,
            custom_closest_counterfactual,
            n,
            balance,
            forced_balance_ratio,
            redo_search,
            cut_radius,
            labels,
            unpicklable_variables,
        )
        self.plotter = CounterGeneratorMultiPlotter(self)
        self.evaluator = CounterGeneratorMultiEvaluator(self)

    def explain(self):
        exemplars = self.Z_tilde_[np.nonzero(self.y_ == self.z_label_)]
        counterexemplars = self.Z_tilde_[np.nonzero(self.y_ != self.z_label_)]
        example_explanation = ExampleBasedExplanation(exemplars, counterexemplars)
        saliency = np.abs(
            self.z_tilde_ - self.decoder.predict(self.closest_counterexemplar_)
        )
        saliency_explanation = SaliencyBasedExplanation(saliency)
        explanation = CounterGeneratorExplanation(
            example_explanation, saliency_explanation
        )
        return explanation


class CounterGeneratorMultiPlotter(CounterGeneratorPlotter):
    def __init__(self, counter_generator):
        super().__init__(counter_generator=counter_generator)

    def plot_manifest_space(self, x=None, **kwargs):
        plot_exemplars_and_counterexemplars_multi(
            Z_tilde=self.counter_generator.Z_tilde_,
            y=self.counter_generator.y_,
            x=x
            if x is not None
            else self.counter_generator.z_tilde_,  # if x is not given
            z_tilde=self.counter_generator.z_tilde_,
            x_label=self.counter_generator.z_label_,
            labels=self.counter_generator.labels,
            plot_x=False if x is None else True,
            **kwargs
        )
        return self

    def plot_variation_delta_on_ts(self, **kwargs):
        delta = np.abs(
            self.counter_generator.z_tilde_
            - self.counter_generator.decoder.predict(
                self.counter_generator.closest_counterexemplar_
            )
        )
        plot_feature_importance_on_ts_multi(
            mts=self.counter_generator.z_tilde_, feature_importances=delta, **kwargs
        )


class CounterGeneratorMultiEvaluator(CounterGeneratorEvaluator):
    def __init__(self, counter_generator):
        super().__init__(counter_generator=counter_generator)

    def manifest_silhouette_score(self, distance="euclidean", **kwargs):
        return silhouette_score(
            self.counter_generator.Z_tilde_.reshape(
                -1,
                self.counter_generator.Z_tilde_.shape[1]
                * self.counter_generator.Z_tilde_.shape[2],
            ),
            self.counter_generator.y_,
            metric=distance,
        )

    def manifest_binary_silhouette_score(self, distance="euclidean", **kwargs):
        y = 1 * (self.counter_generator.y_ == self.counter_generator.z_label_)
        return silhouette_score(
            self.counter_generator.Z_tilde_.reshape(
                -1,
                self.counter_generator.Z_tilde_.shape[1]
                * self.counter_generator.Z_tilde_.shape[2],
            ),
            y,
            metric=distance,
        )

    def manifest_lof_score(self, distance="euclidean", **kwargs):
        lof = LocalOutlierFactor(metric=distance, novelty=True)
        lof.fit(
            self.counter_generator.Z_tilde_.reshape(
                -1,
                self.counter_generator.Z_tilde_.shape[1]
                * self.counter_generator.Z_tilde_.shape[2],
            )
        )
        lof_scores = lof.predict(
            self.counter_generator.Z_tilde_.reshape(
                -1,
                self.counter_generator.Z_tilde_.shape[1]
                * self.counter_generator.Z_tilde_.shape[2],
            )
        )
        return lof_scores.mean()

    def manifest_lof_score_z(self, distance="euclidean", **kwargs):
        lof = LocalOutlierFactor(metric=distance, novelty=True)
        lof.fit(
            self.counter_generator.Z_tilde_.reshape(
                -1,
                self.counter_generator.Z_tilde_.shape[1]
                * self.counter_generator.Z_tilde_.shape[2],
            )
        )
        lof_score = lof.predict(
            self.counter_generator.z_tilde_.reshape(
                -1,
                self.counter_generator.z_tilde_.shape[1]
                * self.counter_generator.z_tilde_.shape[2],
            )
        )[0]
        return lof_score
