from lasts.explainers.lasts import Lasts, LastsPlotter, LastsEvaluator, LastsExplanation
import numpy as np


class LastsMulti(Lasts):
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
        super().__init__(
            blackbox,
            encoder,
            decoder,
            neighgen,
            surrogate,
            labels,
            verbose,
            binarize_surrogate_labels,
            unpicklable_variables,
        )
        self.plotter = LastsMultiPlotter(self)
        self.evaluator = LastsMultiEvaluator(self)


class LastsMultiPlotter(LastsPlotter):
    def __init__(self, lasts_explainer: LastsMulti):
        super().__init__(lasts_explainer=lasts_explainer)


class LastsMultiEvaluator(LastsEvaluator):
    def __init__(self, lasts_explainer: LastsMulti):
        super().__init__(lasts_explainer=lasts_explainer)


if __name__ == "__main__":
    from lasts.blackboxes.loader import cached_blackbox_loader
    from lasts.datasets.datasets import build_multivariate_cbf
    from lasts.autoencoders.loader import vae_loader
    from lasts.utils import get_project_root, choose_z, reconstruction_accuracy_vae
    from lasts.surrogates.sax_tree_multi import SaxTreeMultivariate
    from lasts.surrogates.shapelet_tree_multi import ShapeletTreeMultivariate
    from lasts.neighgen.counter_generator_multi import CounterGeneratorMulti
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
    ) = build_multivariate_cbf()

    # blackbox = cached_blackbox_loader("cbfmulti_resnet.h5")
    blackbox = cached_blackbox_loader("cbfmulti_rocket.joblib")

    encoder, decoder, autoencoder = vae_loader("cbfmulti__ldim2")

    i = 29
    x = X_exp_test[i : i + 1]

    print(reconstruction_accuracy_vae(X_exp_test, encoder, decoder, blackbox))

    z_fixed = choose_z(
        x,
        encoder,
        decoder,
        n=1000,
        x_label=blackbox.predict(x)[0],
        blackbox=blackbox,
        check_label=True,
    )

    neighgen = CounterGeneratorMulti(
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
    surrogate = SaxTreeMultivariate(random_state=random_state)

    lasts_ = LastsMulti(
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

    # lasts_.plot("morphing_matrix")
    # lasts_.plot("counterexemplar_interpolation")
    lasts_.plot("manifest_space")
    # lasts_.plot("saliency_map")
    #
    # lasts_.plot("subsequences_heatmap")
    lasts_.plot("rules")
    # lasts_.neighgen.plotter.plot_counterexemplar_shape_change()
