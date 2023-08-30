import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict, LearningShapelets
import warnings
from joblib import dump, load
import pathlib

from lasts.base import Plotter, Evaluator, Surrogate, RuleBasedExplanation
from lasts.utils import (
    coverage_score_scikit_tree,
    precision_score_scikit_tree,
    make_path,
)
from lasts.plots import (
    plot_subsequences,
    plot_subsequences_grid,
    plot_binary_heatmap,
    plot_tree,
    plot_shapelet_rule,
    plot_shapelet_heatmap,
    plot_sklearn_decision_tree,
)
from lasts.explanations.tree import SklearnDecisionTreeConverter, prune


class ShapeletTree(Surrogate):
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
            "n_shapelets_per_size": "heuristic",
            "weight_regularizer": 0.01,
            "max_iter": 100,
        },
        decision_tree_grid_kwargs={
            "min_samples_split": [0.002, 0.01, 0.05, 0.1, 0.2],
            "min_samples_leaf": [0.001, 0.01, 0.05, 0.1, 0.2],
            "max_depth": [None, 2, 4, 6, 8, 10, 12, 16],
        },
        prune_duplicate_tree_leaves=True,
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
        self.plotter = ShapeletTreePlotter(self)
        self.evaluator = ShapeletTreeEvaluator(self)

        self.decision_tree_ = None
        self.decision_tree_queryable_ = None
        self.shapelet_model_ = None
        self.X_ = None
        self.y_ = None
        self.X_transformed_ = None
        self.X_thresholded_ = None
        self.tree_graph_ = None

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        n_shapelets_per_size = self.shapelet_model_kwargs.get(
            "n_shapelets_per_size", "heuristic"
        )
        if n_shapelets_per_size == "heuristic":
            n_ts, ts_sz = X.shape[:2]
            n_classes = len(set(y))
            n_shapelets_per_size = grabocka_params_to_shapelet_size_dict(
                n_ts=n_ts,
                ts_sz=ts_sz,
                n_classes=n_classes,
                l=self.shapelet_model_kwargs.get("l", 0.1),
                r=self.shapelet_model_kwargs.get("r", 2),
            )

        shp_clf = LearningShapelets(
            n_shapelets_per_size=n_shapelets_per_size,
            optimizer=self.shapelet_model_kwargs.get("optimizer", "sgd"),
            weight_regularizer=self.shapelet_model_kwargs.get(
                "weight_regularizer", 0.01
            ),
            max_iter=self.shapelet_model_kwargs.get("max_iter", 100),
            random_state=self.random_state,
            verbose=self.shapelet_model_kwargs.get("verbose", 0),
        )

        shp_clf.fit(X, y)
        X_transformed = shp_clf.transform(X)
        self.X_transformed_ = X_transformed

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

        clf = DecisionTreeClassifier(**grid.best_params_)
        clf.fit(self.X_thresholded_, y)
        # if self.prune_duplicate_tree_leaves:
        #     prune_duplicate_leaves(clf)  # FIXME: does it influence the .tree properties?
        if (
            self.prune_duplicate_tree_leaves
        ):  # TODO: should check which of the two pruning techniques is the best
            clf = prune(clf)

        self.decision_tree_ = clf
        self.decision_tree_queryable_ = SklearnDecisionTreeConverter(clf)
        self.shapelet_model_ = shp_clf

        return self

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def predict(self, X):
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : {array-like}
            Test data. Shape [n_samples, n_features].
        Returns
        -------
        y : array of shape [n_samples]
        """

        X_transformed = self.shapelet_model_.transform(X)
        X_thresholded = 1 * (X_transformed < self.tau_)
        y = self.decision_tree_.predict(X_thresholded)
        return y

    def transform(self, X):
        X_transformed = self.shapelet_model_.transform(X)
        X_thresholded = 1 * (X_transformed < self.tau_)
        return X_thresholded

    def locate(self, x):
        return self.shapelet_model_.locate(x)

    def get_shapelet_by_idx(self, idx):
        return self.shapelet_model_.shapelets_[idx].ravel()

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind, **kwargs)
        return self

    def evaluate(self, metric, **kwargs):
        return self.evaluator.evaluate(metric, **kwargs)

    def find_leaf_id(self, ts):
        ts_thresholded = self.transform(ts)
        leaf_id = self.decision_tree_.apply(ts_thresholded)[0]
        return leaf_id

    def find_counterfactual_leaf_id(self, ts):
        leaf_id = self.find_leaf_id(ts)
        _, counterfactual_leaf = self.decision_tree_queryable_._minimum_distance(
            self.decision_tree_queryable_._get_node_by_idx(leaf_id)
        )  # FIXME: ugly
        return counterfactual_leaf

    def find_counterfactual_tss(self, ts):
        leave_idxs = self.decision_tree_.apply(self.X_thresholded_)
        counterfactual_leaf = self.find_counterfactual_leaf_id(ts)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.nonzero(leave_idxs == counterfactual_leaf)
        counterfactual_tss = self.X_[counterfactuals_idxs]
        return counterfactual_tss

    def explain(self, x):
        factual_id = self.find_leaf_id(x)
        factual_rule = self.decision_tree_queryable_.get_factual_rule_by_idx(
            factual_id, as_contained=True, labels=self.labels
        )
        counterfactual_rule = (
            self.decision_tree_queryable_.get_counterfactual_rule_by_idx(
                factual_id, as_contained=True, labels=self.labels
            )
        )
        explanation = RuleBasedExplanation(
            factual_rule=factual_rule, counterfactual_rule=counterfactual_rule
        )
        return explanation

    def save(self, folder, name=""):
        path = make_path(folder)
        self.shapelet_model_.to_pickle(path=path / (name + "_shapeletmodel.joblib"))
        self.shapelet_model_ = None
        dump(self, path / (name + "_shapelettree.joblib"))
        return self

    @classmethod
    def load(cls, folder, name=""):
        path = pathlib.Path(folder)
        shapelet_tree = load(path / (name + "_shapelettree.joblib"))
        shapelet_model = LearningShapelets().from_pickle(
            path / (name + "_shapeletmodel.joblib")
        )
        shapelet_tree.shapelet_model_ = shapelet_model
        return shapelet_tree


class ShapeletTreePlotter(Plotter):
    PLOTS = [
        "subsequences",
        "subsequences_grid",
        "subsequences_heatmap",
        "factual_rule",
        "counterfactual_rule",
        "rules",
        "tree",
    ]

    def __init__(self, shapelet_tree: ShapeletTree):
        self.shapelet_tree = shapelet_tree

    def plot_subsequences(self, **kwargs):
        plot_subsequences(
            shapelets=self.shapelet_tree.shapelet_model_.shapelets_.copy(),
            ts_length=self.shapelet_tree.X_.shape[1],
            ts_max=self.shapelet_tree.X_.max(),
            ts_min=self.shapelet_tree.X_.min(),
            **kwargs
        )

    def plot_subsequences_grid(self, n, m, **kwargs):
        plot_subsequences_grid(
            subsequence_list=self.shapelet_tree.shapelet_model_.shapelets_.copy(),
            n=n,
            m=m,
            **kwargs
        )

    def plot_binary_heatmap(self, x_label, **kwargs):
        plot_binary_heatmap(
            x_label=x_label,
            y=self.shapelet_tree.y_,
            X_binary=self.shapelet_tree.X_thresholded_,
            step=kwargs.pop("step", 5),
            **kwargs
        )

    def plot_shapelet_heatmap(self, **kwargs):
        plot_shapelet_heatmap(
            X=self.shapelet_tree.X_transformed_,
            y=self.shapelet_tree.y_,
            step=kwargs.pop("step", 5),
            **kwargs
        )

    def plot_tree(self, **kwargs):
        plot_sklearn_decision_tree(dt=self.shapelet_tree.decision_tree_, **kwargs)

    def plot_factual_rule(self, x, return_y_lim=False, **kwargs):
        predicted_locations = self.shapelet_tree.locate(x)
        factual_id = self.shapelet_tree.find_leaf_id(x)
        factual_rule = (
            self.shapelet_tree.decision_tree_queryable_.get_factual_rule_by_idx(
                factual_id, as_contained=True, labels=self.shapelet_tree.labels
            )
        )
        shapelets_idxs = [premise.attribute for premise in factual_rule.premises]
        starting_idxs = [predicted_locations[0, idx] for idx in shapelets_idxs]
        operators = [premise.operator for premise in factual_rule.premises]
        shapelets = [
            self.shapelet_tree.get_shapelet_by_idx(idx) for idx in shapelets_idxs
        ]
        y_lim = plot_shapelet_rule(
            x=kwargs.get("draw_on", x),
            shapelets_idxs=shapelets_idxs,
            shapelets=shapelets,
            starting_idxs=starting_idxs,
            condition_operators=operators,
            title="Factual Rule\n" + str(factual_rule),
            legend_label=r"$x$",
            return_y_lim=return_y_lim,
            **kwargs
        )
        if return_y_lim:
            return y_lim
        else:
            return None

    def plot_counterfactual_rule(self, x, y_lim=None, **kwargs):
        # counterfactual rule plotted on x
        predicted_locations = self.shapelet_tree.locate(x)
        factual_id = self.shapelet_tree.find_leaf_id(x)
        counterfactual_rule = (
            self.shapelet_tree.decision_tree_queryable_.get_counterfactual_rule_by_idx(
                factual_id, as_contained=True, labels=self.shapelet_tree.labels
            )
        )
        shapelets_idxs = [premise.attribute for premise in counterfactual_rule.premises]
        starting_idxs = [predicted_locations[0, idx] for idx in shapelets_idxs]
        operators = [premise.operator for premise in counterfactual_rule.premises]
        shapelets = [
            self.shapelet_tree.get_shapelet_by_idx(idx) for idx in shapelets_idxs
        ]
        plot_shapelet_rule(
            x=kwargs.get("draw_on", x),
            shapelets_idxs=shapelets_idxs,
            shapelets=shapelets,
            starting_idxs=starting_idxs,
            condition_operators=operators,
            title="Counterfactual Rule\n" + str(counterfactual_rule),
            legend_label=r"$x$",
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
        counterfactuals_idxs = np.argwhere(leave_idxs == counterfactual_leaf)
        # choose one counterfactual among those in the leaf
        counterfactual_idx = counterfactuals_idxs[kwargs.get("z_tilde_idx", 0)]
        counterfactual_ts = self.shapelet_tree.X_[counterfactual_idx].reshape(1, -1)
        counterfactual_y = self.shapelet_tree.y_[counterfactual_idx][0]
        if counterfactual_y != counterfactual_rule.consequence:
            warnings.warn(
                "The real class is different from the one predicted by the tree.\ny_real=%s "
                "y_pred=%s" % (counterfactual_y, counterfactual_rule.consequence)
            )
        predicted_locations = self.shapelet_tree.locate(counterfactual_ts)
        starting_idxs = [predicted_locations[0, idx] for idx in shapelets_idxs]
        plot_shapelet_rule(
            x=counterfactual_ts,
            shapelets_idxs=shapelets_idxs,
            shapelets=shapelets,
            starting_idxs=starting_idxs,
            condition_operators=operators,
            title="Counterfactual Rule\n" + str(counterfactual_rule),
            legend_label=r"$\tilde{z}'$",
            forced_y_lim=y_lim,
            **kwargs
        )

    def plot(self, kind, **kwargs):
        if kind == "subsequences":
            self.plot_subsequences(**kwargs)
        elif kind == "subsequences_grid":
            self.plot_subsequences_grid(n=kwargs.pop("n"), m=kwargs.pop("m"), **kwargs)
        elif kind == "subsequences_binary_heatmap":
            self.plot_binary_heatmap(x_label=kwargs.pop("x_label"), **kwargs)
        elif kind == "subsequences_heatmap":
            self.plot_shapelet_heatmap(**kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        elif kind == "factual_rule":
            self.plot_factual_rule(x=kwargs.pop("x"), **kwargs)

        elif kind == "counterfactual_rule":
            self.plot_counterfactual_rule(x=kwargs.pop("x"), **kwargs)

        elif kind == "rules":
            x = kwargs.pop("x")
            y_lim = self.plot_factual_rule(
                x=x,
                return_y_lim=kwargs.pop(
                    "return_y_lim", True
                ),  # ensure that the plots have the same y scale
                **kwargs
            )
            self.plot_counterfactual_rule(x=x, y_lim=y_lim, **kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        else:
            raise Exception("Plot kind not valid.")
        return self


class ShapeletTreeEvaluator(Evaluator):
    METRICS = ["tree_coverage", "tree_precision"]

    def __init__(self, shapelet_tree: ShapeletTree):
        self.shapelet_tree = shapelet_tree

    def tree_coverage_score(self, x):
        leaf_id = self.shapelet_tree.find_leaf_id(x)  # FIXME: private method
        return coverage_score_scikit_tree(
            dt=self.shapelet_tree.decision_tree_, leaf_id=leaf_id
        )

    def tree_precision_score(self, X, x, y):
        leaf_id = self.shapelet_tree.find_leaf_id(x)  # FIXME: private method
        X = self.shapelet_tree.transform(X)
        return precision_score_scikit_tree(
            dt=self.shapelet_tree.decision_tree_, X=X, y=y, leaf_id=leaf_id
        )

    def accuracy_score(self, X, y):
        return self.shapelet_tree.score(X, y)

    def evaluate(self, metric, **kwargs):
        if metric == "tree_coverage":
            return self.tree_coverage_score(x=kwargs.get("x"))
        elif metric == "tree_precision":
            return self.tree_precision_score(
                X=kwargs.get("X"),
                x=kwargs.get("x"),
                y=kwargs.get("y"),
            )
        elif metric == "accuracy_score":
            return self.accuracy_score(X=kwargs.get("X"), y=kwargs.get("y"))
        else:
            raise Exception("Metric not valid.")


def save_shapelet_tree(shapelet_tree: ShapeletTree, folder="./"):
    path = make_path(folder)
    shapelet_tree.shapelet_model_.to_pickle(path=path / "shapelet_model.joblib")
    shapelet_tree.shapelet_model_ = None
    dump(shapelet_tree, path / "shapelet_tree.joblib")
    return


def load_shapelet_tree(folder):
    path = pathlib.Path(folder)
    shapelet_model = LearningShapelets().from_pickle(path / "shapelet_model.joblib")
    shapelet_tree = load(path / "shapelet_tree.joblib")
    shapelet_tree.shapelet_model_ = shapelet_model
    return shapelet_tree


if __name__ == "__main__":
    from lasts.datasets.datasets import build_cbf
    from lasts.surrogates.utils import generate_n_shapelets_per_size

    X_train, y_train, X_val, y_val, *_ = build_cbf(n_samples=600)

    n_shapelets_per_size = generate_n_shapelets_per_size(X_train.shape[1])
    shapelet_model_kwargs = {
        "l": 0.1,
        "r": 2,
        "optimizer": "sgd",
        "n_shapelets_per_size": n_shapelets_per_size,
        "weight_regularizer": 0.01,
        "max_iter": 100,
    }

    clf = ShapeletTree(
        random_state=0,
        shapelet_model_kwargs=shapelet_model_kwargs,
        prune_duplicate_tree_leaves=True,
        labels=["cyl", "bell", "fun"],
    )
    clf.fit(X_train, y_train)
    print(clf.score(X_val, y_val))
    clf.plot(kind="rules", x=X_train[1:2])
    # print(clf.evaluate("tree_coverage", x=X_train[1:2]))
    exp = clf.explain(x=X_train[1:2])
    clf.plot("subsequences_heatmap")
