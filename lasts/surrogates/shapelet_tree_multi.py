import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings
from joblib import dump, load
import pathlib

from lasts.surrogates.shapelet_transformer import ShapeletTransformer
from lasts.surrogates.shapelet_tree import (
    ShapeletTree,
    ShapeletTreeEvaluator,
    ShapeletTreePlotter,
)
from lasts.utils import (
    precision_score_scikit_tree,
    make_path,
    format_multivariate_input,
)
from lasts.plots import (
    plot_subsequences,
    plot_subsequences_grid,
    plot_binary_heatmap,
    plot_tree,
    plot_shapelet_rule,
    plot_multi_subsequence_rule,
)
from lasts.explanations.tree import SklearnDecisionTreeConverter, prune


class ShapeletTreeMultivariate(ShapeletTree):
    def __init__(
        self,
        labels=None,
        random_state=None,
        tau=None,
        tau_quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        shapelet_model_kwargs={
            "l": 0.1,
            "r": 2,
            "optimizer": "sgd",
            "weight_regularizer": 0.01,
            "max_iter": 100,
        },
        decision_tree_grid_kwargs={
            "min_samples_split": [0.002, 0.01, 0.05, 0.1, 0.2],
            "min_samples_leaf": [0.001, 0.01, 0.05, 0.1, 0.2],
            "max_depth": [None, 2, 4, 6, 8, 10, 12, 16],
        },
        prune_duplicate_tree_leaves=True,
        n_shapelets_per_sizes="heuristic",
    ):
        self.labels = labels
        self.random_state = random_state
        if tau is not None:
            self.tau = tau
            self.tau_ = tau
        else:
            self.tau = None
            self.tau_ = None
        self.tau_quantiles = tau_quantiles
        self.shapelet_model_kwargs = shapelet_model_kwargs
        self.decision_tree_grid_kwargs = decision_tree_grid_kwargs
        self.prune_duplicate_tree_leaves = prune_duplicate_tree_leaves
        self.n_shapelets_per_sizes = n_shapelets_per_sizes
        self.plotter = ShapeletTreeMultivariatePlotter(self)
        self.evaluator = ShapeletTreeMultivariateEvaluator(self)

        self.decision_tree_ = None
        self.decision_tree_queryable_ = None
        self.shapelet_transformers_ = None
        self.X_ = None
        self.y_ = None
        self.X_thresholded_ = None
        self.X_transformed_ = None
        self.tree_graph_ = None
        self.feature_mapping_dict_ = None

    def fit(self, X, y):
        X = format_multivariate_input(X)
        self.X_ = X
        self.y_ = y
        self.shapelet_transformers_ = list()
        prev_size = 0
        self.feature_mapping_dict_ = dict()
        for i, X_dim in enumerate(X):
            shapelet_model_kwargs = self.shapelet_model_kwargs.copy()
            if isinstance(self.n_shapelets_per_sizes, list):
                shapelet_model_kwargs[
                    "n_shapelets_per_sizes"
                ] = self.n_shapelets_per_sizes[i]
            else:
                shapelet_model_kwargs[
                    "n_shapelets_per_sizes"
                ] = self.n_shapelets_per_sizes
            shapelet_transformer = ShapeletTransformer(
                labels=self.labels,
                random_state=self.random_state,
                shapelet_model_kwargs=self.shapelet_model_kwargs,
            )
            shapelet_transformer.fit(X_dim, y)
            for feature_index in range(shapelet_transformer.X_transformed_.shape[1]):
                self.feature_mapping_dict_[feature_index + prev_size] = [
                    i,
                    feature_index,
                ]
            prev_size += shapelet_transformer.X_transformed_.shape[1]
            self.shapelet_transformers_.append(shapelet_transformer)
        self.X_transformed_ = list()
        for X_dim, transformer in zip(X, self.shapelet_transformers_):
            self.X_transformed_.append(transformer.transform(X_dim))
        self.X_transformed_ = np.concatenate(self.X_transformed_, axis=1)

        if self.tau is not None:
            self.X_thresholded_ = 1 * (self.X_transformed_ < self.tau_)
            clf = DecisionTreeClassifier()
            param_grid = self.decision_tree_grid_kwargs
            grid = GridSearchCV(
                clf, param_grid=param_grid, scoring="accuracy", n_jobs=1, verbose=0
            )
            grid.fit(self.X_thresholded_, y)
        else:
            grids = []
            grids_scores = []
            for quantile in self.tau_quantiles:
                X_thresholded = 1 * (
                    self.X_transformed_ < (np.quantile(self.X_transformed_, quantile))
                )
                clf = DecisionTreeClassifier()
                param_grid = self.decision_tree_grid_kwargs
                grid = GridSearchCV(
                    clf, param_grid=param_grid, scoring="accuracy", n_jobs=1, verbose=0
                )
                grid.fit(X_thresholded, y)
                grids.append(grid)
                grids_scores.append(grid.best_score_)
            grid = grids[np.argmax(np.array(grids_scores))]
            best_quantile = self.tau_quantiles[np.argmax(np.array(grids_scores))]
            self.tau_ = np.quantile(self.X_transformed_, best_quantile)
            self.X_thresholded_ = 1 * (self.X_transformed_ < self.tau_)

        clf = DecisionTreeClassifier(
            **grid.best_params_, random_state=self.random_state
        )
        clf.fit(self.X_thresholded_, y)
        # if self.prune_duplicate_tree_leaves:
        #     prune_duplicate_leaves(clf)  # FIXME: does it influence the .tree properties?
        if (
            self.prune_duplicate_tree_leaves
        ):  # TODO: should check which of the two pruning techniques is the best
            clf = prune(clf)

        self.decision_tree_ = clf
        self.decision_tree_queryable_ = SklearnDecisionTreeConverter(clf)

    def transform(self, X):
        X = format_multivariate_input(X)
        X_transformed = list()
        for X_dim, transformer in zip(X, self.shapelet_transformers_):
            X_transformed.append(transformer.transform(X_dim))
        X_transformed = np.concatenate(X_transformed, axis=1)
        X_thresholded = 1 * (X_transformed < self.tau_)
        return X_thresholded

    def predict(self, X):
        X = format_multivariate_input(X)
        return self.decision_tree_.predict(self.transform(X))

    def locate(self, X):
        X = format_multivariate_input(X)
        locations = list()
        for X_dim, transformer in zip(X, self.shapelet_transformers_):
            locations.append(transformer.locate(X_dim))
        return locations

    def get_shapelet_by_idx(self, idx):
        dim_idx, shp_idx = self.feature_mapping_dict_[idx]
        return (
            self.shapelet_transformers_[dim_idx]
            .shapelet_model_.shapelets_[shp_idx]
            .ravel()
        )

    def save(self, folder, name=""):
        path = make_path(folder)
        for i, shapelet_transformer in enumerate(self.shapelet_transformers_):
            shapelet_transformer.save(
                folder, name="_" + str(i) + "shapeletmultitransformer"
            )
        dump(self, path / (name + "_shapeletmultitree.joblib"))
        return self

    @classmethod
    def load(cls, folder, name=""):
        #  TODO: test if it works (the problem could be the list of transformer which is not a class implementing a
        #   load method as usual
        path = pathlib.Path(folder)
        shapelet_tree = load(path / (name + "_shapeletmultitree.joblib"))
        shapelet_tree.shapelet_transformers_ = list()
        shapelet_transformers = list()
        for i in range(len(shapelet_tree.X_)):
            shapelet_transformer = ShapeletTransformer.load(
                folder, name="_" + str(i) + "shapeletmultitransformer"
            )
            shapelet_transformers.append(shapelet_transformer)
        shapelet_tree.shapelet_transformers_ = shapelet_transformers
        return shapelet_tree


class ShapeletTreeMultivariatePlotterOld(ShapeletTreePlotter):
    def __init__(self, shapelet_tree_multivariate: ShapeletTreeMultivariate):
        super().__init__(shapelet_tree=shapelet_tree_multivariate)

    def _map_to_dimension(self, x, rule):
        shapelets_idxs = [
            premise.attribute for premise in rule.premises
        ]  # tree idxs of shapelets
        operators_list = [
            premise.operator for premise in rule.premises
        ]  # operators in the rule
        shapelets_list = [
            self.shapelet_tree.get_shapelet_by_idx(idx) for idx in shapelets_idxs
        ]  #

        # map all the things for plotting by dimension ex. [[shp_dim1][shp_dim2, shp_dim2]...]
        predicted_locations_by_dim = self.shapelet_tree.locate(
            x
        )  # list of predicted locations, one list per dim
        shapelets_idxs_by_dim = [[] for _ in range(len(x))]
        operators_by_dim = [[] for _ in range(len(x))]
        shapelets_by_dim = [[] for _ in range(len(x))]
        orig_shapelets_idxs_by_dim = [[] for _ in range(len(x))]
        for shp_idx, operator, shapelet in zip(
            shapelets_idxs, operators_list, shapelets_list
        ):
            dim_idx, shp_idx_dim = self.shapelet_tree.feature_mapping_dict_[shp_idx]
            shapelets_idxs_by_dim[dim_idx].append(shp_idx_dim)
            operators_by_dim[dim_idx].append(operator)
            shapelets_by_dim[dim_idx].append(shapelet)
            orig_shapelets_idxs_by_dim[dim_idx].append(shp_idx)
        return (
            predicted_locations_by_dim,
            shapelets_idxs_by_dim,
            orig_shapelets_idxs_by_dim,
            operators_by_dim,
            shapelets_by_dim,
        )

    def plot_factual_rule(self, x, return_y_lim=False, **kwargs):
        # TODO: the problem is that the idx of the shapelet in the tree is not the same as that inside the different
        #  models. Learning shapelets can take also a multivariate input (but not with different shaped dimensions)
        factual_id = self.shapelet_tree.find_leaf_id(x)
        factual_rule = (
            self.shapelet_tree.decision_tree_queryable_.get_factual_rule_by_idx(
                factual_id, as_contained=True, labels=self.shapelet_tree.labels
            )
        )

        (
            pred_locs_by_dim,
            shp_idxs_by_dim,
            orig_shp_idxs_by_dim,
            ops_by_dim,
            shp_by_dim,
        ) = self._map_to_dimension(x, factual_rule)

        y_lims = list()
        for dim_idx, (
            shp_idxs,
            orig_shp_idxs,
            operators,
            shapelets,
            predicted_locations,
        ) in enumerate(
            zip(
                shp_idxs_by_dim,
                orig_shp_idxs_by_dim,
                ops_by_dim,
                shp_by_dim,
                pred_locs_by_dim,
            )
        ):
            if len(shp_idxs) == 0:
                continue
            starting_idxs = [predicted_locations[0, idx] for idx in shp_idxs]
            y_lim = plot_shapelet_rule(
                x=kwargs.get("draw_on", x[dim_idx]),
                shapelets_idxs=orig_shp_idxs,
                shapelets=shapelets,
                starting_idxs=starting_idxs,
                condition_operators=operators,
                title="Factual Rule\n(%s)" % ("dim_" + str(dim_idx)),
                legend_label=r"$X$",
                return_y_lim=return_y_lim,
                **kwargs
            )
            y_lims.append(y_lim)
        if return_y_lim:
            return y_lims
        else:
            return None

    def plot_counterfactual_rule(self, x, y_lims=None, **kwargs):
        factual_id = self.shapelet_tree.find_leaf_id(x)
        counterfactual_rule = (
            self.shapelet_tree.decision_tree_queryable_.get_counterfactual_rule_by_idx(
                factual_id, as_contained=True, labels=self.shapelet_tree.labels
            )
        )
        (
            pred_locs_by_dim,
            shp_idxs_by_dim,
            orig_shp_idxs_by_dim,
            ops_by_dim,
            shp_by_dim,
        ) = self._map_to_dimension(x, counterfactual_rule)

        for dim_idx, (
            shp_idxs,
            orig_shp_idxs,
            operators,
            shapelets,
            predicted_locations,
        ) in enumerate(
            zip(
                shp_idxs_by_dim,
                orig_shp_idxs_by_dim,
                ops_by_dim,
                shp_by_dim,
                pred_locs_by_dim,
            )
        ):
            if len(shp_idxs) == 0:
                continue
            starting_idxs = [predicted_locations[0, idx] for idx in shp_idxs]
            plot_shapelet_rule(
                x=kwargs.get("draw_on", x[dim_idx]),
                shapelets_idxs=orig_shp_idxs,
                shapelets=shapelets,
                starting_idxs=starting_idxs,
                condition_operators=operators,
                title="Counterfactual Rule\n(%s)" % ("dim_" + str(dim_idx)),
                legend_label=r"$X$",
                return_y_lim=False,
                **kwargs
            )

        # counterfactual rule plotted on a counterfactual z_tilde
        # get all the leave ids
        (
            _,
            counterfactual_leaf,
        ) = self.shapelet_tree.decision_tree_queryable_._minimum_distance(
            self.shapelet_tree.decision_tree_queryable_._get_node_by_idx(factual_id)
        )  # FIXME: ugly
        leave_idxs = self.shapelet_tree.decision_tree_.apply(
            self.shapelet_tree.X_thresholded_
        )
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.argwhere(leave_idxs == counterfactual_leaf).ravel()
        # choose one counterfactual among those in the leaf
        counterfactual_idx = counterfactuals_idxs[kwargs.get("z_tilde_idx", 0)]
        counterfactual_ts = [
            self.shapelet_tree.X_[i][counterfactual_idx : counterfactual_idx + 1]
            for i in range(len(x))
        ]
        counterfactual_y = self.shapelet_tree.y_[counterfactual_idx]
        if counterfactual_y != counterfactual_rule.consequence:
            warnings.warn(
                "The real class is different from the one predicted by the tree.\ny_real=%s "
                "y_pred=%s" % (counterfactual_y, counterfactual_rule.consequence)
            )
        (
            pred_locs_by_dim,
            shp_idxs_by_dim,
            orig_shp_idxs_by_dim,
            ops_by_dim,
            shp_by_dim,
        ) = self._map_to_dimension(counterfactual_ts, counterfactual_rule)

        for dim_idx, (
            shp_idxs,
            orig_shp_idxs,
            operators,
            shapelets,
            predicted_locations,
        ) in enumerate(
            zip(
                shp_idxs_by_dim,
                orig_shp_idxs_by_dim,
                ops_by_dim,
                shp_by_dim,
                pred_locs_by_dim,
            )
        ):
            if len(shp_idxs) == 0:
                continue
            starting_idxs = [predicted_locations[0, idx] for idx in shp_idxs]
            plot_shapelet_rule(
                x=counterfactual_ts[dim_idx],
                shapelets_idxs=orig_shp_idxs,
                shapelets=shapelets,
                starting_idxs=starting_idxs,
                condition_operators=operators,
                title="Counterfactual Rule (on a counterexemplar)\n(%s)"
                % ("dim_" + str(dim_idx)),
                legend_label=r"$\hat{X}'_\neq$",
                forced_y_lim=y_lims if y_lims is None else y_lims[dim_idx],
                **kwargs
            )


class ShapeletTreeMultivariatePlotter(ShapeletTreePlotter):
    def __init__(self, shapelet_tree_multivariate: ShapeletTreeMultivariate):
        super().__init__(shapelet_tree=shapelet_tree_multivariate)

    def _map_to_dimension(self, x, rule):
        shapelets_idxs = [
            premise.attribute for premise in rule.premises
        ]  # tree idxs of shapelets
        operators_list = [
            premise.operator for premise in rule.premises
        ]  # operators in the rule
        shapelets_list = [
            self.shapelet_tree.get_shapelet_by_idx(idx) for idx in shapelets_idxs
        ]

        predicted_locations_by_dim = self.shapelet_tree.locate(
            x
        )  # list of predicted locations, one list per dim
        starting_idxs = list()
        shapelets_dims = list()
        for shp_idx in shapelets_idxs:
            dim_idx, shp_idx_dim = self.shapelet_tree.feature_mapping_dict_[shp_idx]
            starting_idxs.append(predicted_locations_by_dim[dim_idx][0, shp_idx_dim])
            shapelets_dims.append(dim_idx)
        return (
            shapelets_idxs,
            operators_list,
            shapelets_list,
            starting_idxs,
            shapelets_dims,
        )

    def plot_factual_rule(self, x, return_y_lim=False, **kwargs):
        x = format_multivariate_input(x)
        factual_id = self.shapelet_tree.find_leaf_id(x)
        factual_rule = (
            self.shapelet_tree.decision_tree_queryable_.get_factual_rule_by_idx(
                factual_id, as_contained=True, labels=self.shapelet_tree.labels
            )
        )

        (
            shapelets_idxs,
            operators_list,
            shapelets_list,
            starting_idxs,
            shapelets_dims,
        ) = self._map_to_dimension(x, factual_rule)

        plot_multi_subsequence_rule(
            x=kwargs.get("draw_on", x),
            shapelets_idxs=shapelets_idxs,
            shapelets=shapelets_list,
            starting_idxs=starting_idxs,
            condition_operators=operators_list,
            dimension_idxs=shapelets_dims,
            title="Factual Rule",
            legend_label=r"$x$",
            **kwargs
        )

    def plot_counterfactual_rule(self, x, y_lims=None, **kwargs):
        x = format_multivariate_input(x)
        factual_id = self.shapelet_tree.find_leaf_id(x)
        counterfactual_rule = (
            self.shapelet_tree.decision_tree_queryable_.get_counterfactual_rule_by_idx(
                factual_id, as_contained=True, labels=self.shapelet_tree.labels
            )
        )
        (
            shapelets_idxs,
            operators_list,
            shapelets_list,
            starting_idxs,
            shapelets_dims,
        ) = self._map_to_dimension(x, counterfactual_rule)

        plot_multi_subsequence_rule(
            x=kwargs.get("draw_on", x),
            shapelets_idxs=shapelets_idxs,
            shapelets=shapelets_list,
            starting_idxs=starting_idxs,
            condition_operators=operators_list,
            dimension_idxs=shapelets_dims,
            title="Counterfactual Rule",
            legend_label=r"$X$",
            **kwargs
        )

        # counterfactual rule plotted on a counterfactual z_tilde
        # get all the leave ids
        (
            _,
            counterfactual_leaf,
        ) = self.shapelet_tree.decision_tree_queryable_._minimum_distance(
            self.shapelet_tree.decision_tree_queryable_._get_node_by_idx(factual_id)
        )  # FIXME: ugly
        leave_idxs = self.shapelet_tree.decision_tree_.apply(
            self.shapelet_tree.X_thresholded_
        )
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.argwhere(leave_idxs == counterfactual_leaf).ravel()
        # choose one counterfactual among those in the leaf
        counterfactual_idx = counterfactuals_idxs[kwargs.get("z_tilde_idx", 0)]
        counterfactual_ts = [
            self.shapelet_tree.X_[i][counterfactual_idx : counterfactual_idx + 1]
            for i in range(len(x))
        ]
        counterfactual_y = self.shapelet_tree.y_[counterfactual_idx]
        if counterfactual_y != counterfactual_rule.consequence:
            warnings.warn(
                "The real class is different from the one predicted by the tree.\ny_real=%s "
                "y_pred=%s" % (counterfactual_y, counterfactual_rule.consequence)
            )
        (
            shapelets_idxs,
            operators_list,
            shapelets_list,
            starting_idxs,
            shapelets_dims,
        ) = self._map_to_dimension(counterfactual_ts, counterfactual_rule)

        plot_multi_subsequence_rule(
            x=counterfactual_ts,
            shapelets_idxs=shapelets_idxs,
            shapelets=shapelets_list,
            starting_idxs=starting_idxs,
            condition_operators=operators_list,
            dimension_idxs=shapelets_dims,
            title="Counterfactual Rule (on a counterexemplar)",
            legend_label=r"$\hat{X}'_\neq$",
            **kwargs
        )

    def plot_subsequences(self, **kwargs):
        for shapelet_transformer in self.shapelet_tree.shapelet_transformers_:
            shapelet_transformer.plot(kind="subsequences", **kwargs)

    def plot_subsequences_grid(self, n, m, **kwargs):
        for shapelet_transformer in self.shapelet_tree.shapelet_transformers_:
            shapelet_transformer.plot(kind="subsequences_grid", **kwargs)

    def plot(self, kind, **kwargs):
        if kind == "subsequences":
            self.plot_subsequences(**kwargs)
        elif kind == "subsequences_grid":
            self.plot_subsequences_grid(n=kwargs.pop("n"), m=kwargs.pop("m"), **kwargs)
        elif kind == "subsequences_heatmap":
            super().plot_shapelet_heatmap(**kwargs)
            # self.plot_binary_heatmap(
            #     x_label=kwargs.pop("x_label"),
            #     **kwargs
            # )

        elif kind == "subsequences_binary_heatmap":
            super().plot_binary_heatmap(x_label=kwargs.pop("x_label"), **kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        elif kind == "factual_rule":
            self.plot_factual_rule(x=kwargs.pop("x"), **kwargs)

        elif kind == "counterfactual_rule":
            self.plot_counterfactual_rule(x=kwargs.pop("x"), **kwargs)

        elif kind == "rules":
            x = kwargs.pop("x")
            y_lims = self.plot_factual_rule(
                x=x,
                return_y_lims=kwargs.pop(
                    "return_y_lims", True
                ),  # ensure that the plots have the same y scale
                **kwargs
            )
            self.plot_counterfactual_rule(x=x, y_lims=y_lims, **kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        else:
            raise Exception("Plot kind not valid.")
        return self


class ShapeletTreeMultivariateEvaluator(ShapeletTreeEvaluator):
    def __init__(self, shapelet_tree_multi: ShapeletTreeMultivariate):
        super().__init__(shapelet_tree=shapelet_tree_multi)


def main1():
    from lasts.datasets.datasets import load_generali_subsampled, load_generali
    from lasts.surrogates.utils import generate_n_shapelets_per_size

    X_train = [np.random.random(size=(10, 235, 1)), np.random.random(size=(10, 20, 1))]
    y_train = np.random.randint(0, 2, size=10)

    clf = ShapeletTreeMultivariate(
        random_state=0,
        prune_duplicate_tree_leaves=True,
    )

    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    from lasts.datasets.datasets import build_multivariate_cbf

    X_train, y_train, _, _, X_test, y_test, *_ = build_multivariate_cbf()

    clf = ShapeletTreeMultivariate(
        random_state=0,
        prune_duplicate_tree_leaves=True,
    )

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

    i = 1
    x = X_test[i : i + 1]

    clf.plot(kind="rules", x=x)
