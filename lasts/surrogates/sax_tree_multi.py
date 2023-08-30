import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings
from joblib import dump, load
import pathlib

from lasts.surrogates.sax_transformer import SaxTransformer
from lasts.surrogates.sax_tree import (
    SaxTree,
    SaxTreeEvaluator,
    SaxTreePlotter,
    get_subsequence_alignment,
)
from lasts.utils import (
    make_path,
    format_multivariate_input,
    compute_medoid,
    sliding_window_distance,
)
from lasts.plots import plot_multi_subsequence_rule
from lasts.explanations.tree import SklearnDecisionTreeConverter, prune


class SaxTreeMultivariate(SaxTree):
    def __init__(
        self,
        labels=None,
        random_state=None,
        custom_config=None,
        decision_tree_grid_search_kwargs={
            "min_samples_split": [0.002, 0.01, 0.05, 0.1, 0.2],
            "min_samples_leaf": [0.001, 0.01, 0.05, 0.1, 0.2],
            "max_depth": [None, 2, 4, 6, 8, 10, 12, 16],
        },
        prune_duplicate_tree_leaves=True,
        create_plotting_dictionaries=True,
        verbose=False,
    ):

        self.labels = labels
        self.random_state = random_state
        self.decision_tree_grid_search_kwargs = decision_tree_grid_search_kwargs
        self.custom_config = custom_config
        self.create_plotting_dictionaries = create_plotting_dictionaries
        self.plotter = SaxTreeMultivariatePlotter(self)
        self.evaluator = SaxTreeMultivariateEvaluator(self)
        self.verbose = verbose

        self.sax_transformers_ = None
        self.X_ = None
        self.X_transformed_ = None
        self.y_ = None
        self.subsequence_dict_ = None
        self.name_dict_ = None
        self.subsequences_norm_same_length_ = None
        self.decision_tree_ = None
        self.decision_tree_queryable_ = None
        self.seql_model_ = None
        self.subsequence_norm_dict_ = None
        self.explanation_ = None
        self.feature_mapping_dict_ = None

    def fit(self, X, y):
        X = format_multivariate_input(X)
        self.X_ = X
        self.y_ = y
        self.sax_transformers_ = list()
        prev_size = 0
        self.feature_mapping_dict_ = dict()
        for i, X_dim in enumerate(X):
            if self.verbose:
                print(i + 1, "/", len(X))
            sax_transformer = SaxTransformer(
                labels=self.labels,
                random_state=self.random_state,
                custom_config=self.custom_config[i],
            )
            sax_transformer.fit(X_dim, y)
            for feature_index in range(sax_transformer.X_transformed_.shape[1]):
                self.feature_mapping_dict_[feature_index + prev_size] = [
                    i,
                    feature_index,
                ]
            prev_size += sax_transformer.X_transformed_.shape[1]
            self.sax_transformers_.append(sax_transformer)
        self.X_transformed_ = list()
        for X_dim, transformer in zip(X, self.sax_transformers_):
            self.X_transformed_.append(transformer.transform(X_dim))
        self.X_transformed_ = np.concatenate(self.X_transformed_, axis=1)

        clf = DecisionTreeClassifier()
        param_grid = self.decision_tree_grid_search_kwargs
        grid = GridSearchCV(
            clf, param_grid=param_grid, scoring="accuracy", n_jobs=1, verbose=0
        )
        grid.fit(self.X_transformed_, y)

        clf = DecisionTreeClassifier(
            **grid.best_params_, random_state=self.random_state
        )
        clf.fit(self.X_transformed_, y)
        # prune_duplicate_leaves(clf)
        clf = prune(clf)

        self.decision_tree_ = clf
        self.decision_tree_queryable_ = SklearnDecisionTreeConverter(clf)
        return self

    def transform(self, X):
        X = format_multivariate_input(X)
        X_transformed = list()
        for X_dim, transformer in zip(X, self.sax_transformers_):
            X_transformed.append(transformer.transform(X_dim))
        X_transformed = np.concatenate(X_transformed, axis=1)
        return X_transformed

    def predict(self, X):
        X = format_multivariate_input(X)
        return self.decision_tree_.predict(self.transform(X))

    def find_counterfactual_tss(self, ts):
        if self.decision_tree_queryable_.n_nodes == 1:  # if the tree is a unique leaf
            return ts
        leave_idxs = self.decision_tree_.apply(self.X_transformed_)
        counterfactual_leaf = self.find_counterfactual_leaf_id(ts)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.nonzero(leave_idxs == counterfactual_leaf)
        # counterfactual_tss = [dim[counterfactuals_idxs] for dim in self.X_]
        # FIXME: this one would be correct but I would have to redo all the evaluate only for counterfactual
        #  precision and coverage. So for now this only works for multi ts
        counterfactual_tss = np.concatenate(
            [dim[counterfactuals_idxs] for dim in self.X_], axis=2
        )
        return counterfactual_tss

    def save(self, folder, name="", compress=None):
        path = make_path(folder)
        dump(self, path / (name + "_saxmultitree.joblib"), compress=compress)
        return self

    @classmethod
    def load(cls, folder, name=""):
        path = pathlib.Path(folder)
        sax_tree = load(path / (name + "_saxmultitree.joblib"))
        return sax_tree


class SaxTreeMultivariatePlotter(SaxTreePlotter):
    def __init__(self, sax_tree_multivariate: SaxTreeMultivariate):
        super().__init__(sax_tree=sax_tree_multivariate)

    def _contained_dictionaries(
        self, x, z_tilde_star, factual_rule, counterfactual_rule, **kwargs
    ):
        # subsequences that are contained in the 2 rules. If the factual and counterfactual have both the same
        # contained subsequences, only the one inside of x will be inserted
        factual_cont_subseq_dict = dict()

        # subsequences that are contained in the 2 rules. If the factual and counterfactual have both the same
        # contained subsequences, only the one inside of z_tilde_star will be inserted
        counterfactual_cont_subseq_dict = dict()

        # contained subseq idxs for the factual rule
        factual_contained_idxs = [
            premise.attribute
            for premise in factual_rule.premises
            if premise.operator == ">"
        ]
        for sub_idx in factual_contained_idxs:
            dim_idx, orig_idx = self.sax_tree.feature_mapping_dict_[sub_idx]
            start_idx, end_idx, subsequence, feature_name = get_subsequence_alignment(
                x[dim_idx],
                orig_idx,
                self.sax_tree.sax_transformers_[dim_idx].seql_model_,
            )
            factual_cont_subseq_dict[sub_idx] = subsequence
            counterfactual_cont_subseq_dict[sub_idx] = subsequence

        # contained subseq idxs for the counterfactual rule
        counterfactual_contained_idxs = [
            premise.attribute
            for premise in counterfactual_rule.premises
            if premise.operator == ">"
        ]
        for sub_idx in counterfactual_contained_idxs:
            dim_idx, orig_idx = self.sax_tree.feature_mapping_dict_[sub_idx]
            start_idx, end_idx, subsequence, feature_name = get_subsequence_alignment(
                z_tilde_star[dim_idx],
                orig_idx,
                self.sax_tree.sax_transformers_[dim_idx].seql_model_,
            )
            counterfactual_cont_subseq_dict[
                sub_idx
            ] = subsequence  # overwrite the x subsequence
            if sub_idx not in factual_cont_subseq_dict:
                factual_cont_subseq_dict[
                    sub_idx
                ] = subsequence  # don't overwrite the x subsequence
            else:
                # case in which a subsequence is contained both in x and in the counterexemplar (z_tilde_star).
                pass
        return factual_cont_subseq_dict, counterfactual_cont_subseq_dict

    def _map_to_dimension(self, x, rule, contained_subsequences_dict, plot_names=False):
        subsequences_idxs = [
            premise.attribute for premise in rule.premises
        ]  # tree idxs of shapelets
        dimensions_idxs = [
            self.sax_tree.feature_mapping_dict_[idx][0] for idx in subsequences_idxs
        ]
        orig_subsequences_idxs = [
            self.sax_tree.feature_mapping_dict_[idx][1] for idx in subsequences_idxs
        ]
        operators_list = [
            premise.operator for premise in rule.premises
        ]  # operators in the rule
        if plot_names:
            subsequences_names = [
                self.sax_tree.sax_transformers_[dim_idx]
                .name_dict_[sub_idx]
                .decode("utf-8")
                for dim_idx, sub_idx in zip(dimensions_idxs, orig_subsequences_idxs)
            ]
        else:
            subsequences_names = None

        subsequences = list()
        starting_idxs = list()

        for sub_idx, op, orig_idx, dim_idx in zip(
            subsequences_idxs, operators_list, orig_subsequences_idxs, dimensions_idxs
        ):
            if op == ">":  # if the subsequences is contained
                # if sub_idx in factual_cont_subseq_dict:
                subsequence = contained_subsequences_dict[sub_idx]
                start_idx = sliding_window_distance(x[dim_idx].ravel(), subsequence)
                end_idx = start_idx + len(subsequence)
                if end_idx == len(x[dim_idx].ravel()):
                    end_idx -= 1
                    subsequence = subsequence[:-1]
            else:  # if the subsequence is not-contained
                if (
                    sub_idx in contained_subsequences_dict
                ):  # if the subsequence is contained in the other rule
                    subsequence = contained_subsequences_dict[
                        sub_idx
                    ]  # use the saved subsequence
                    start_idx = sliding_window_distance(x[dim_idx].ravel(), subsequence)
                    end_idx = start_idx + len(subsequence)
                    if end_idx == len(x[dim_idx].ravel()):
                        end_idx -= 1
                        subsequence = subsequence[:-1]
                else:  # if the subsequence is always not-contained
                    #  find a representative of the not-contained subsequence
                    subs = self.sax_tree.sax_transformers_[dim_idx].subsequence_dict_[
                        orig_idx
                    ][:, :, 0]
                    subsequence = compute_medoid(subs)
                    #  find the alignment of the subsequence with x
                    start_idx = sliding_window_distance(x[dim_idx].ravel(), subsequence)
                    end_idx = start_idx + len(subsequence)
                    if end_idx == len(x[dim_idx].ravel()):
                        end_idx -= 1
                        subsequence = subsequence[:-1]
            subsequences.append(subsequence)
            starting_idxs.append(start_idx)

        return (
            subsequences_idxs,
            operators_list,
            subsequences,
            starting_idxs,
            dimensions_idxs,
            subsequences_names,
        )

    def plot_rules(self, x, plot_names=False, **kwargs):
        x = format_multivariate_input(x)
        draw_on = kwargs.get("draw_on", None)
        if draw_on is None:
            draw_on = x
        else:
            draw_on = format_multivariate_input(draw_on)
        factual_id = self.sax_tree.find_leaf_id(x)
        factual_rule = self.sax_tree.decision_tree_queryable_.get_factual_rule_by_idx(
            factual_id, as_contained=True, labels=self.sax_tree.labels
        )
        counterfactual_rule = (
            self.sax_tree.decision_tree_queryable_.get_counterfactual_rule_by_idx(
                factual_id, as_contained=True, labels=self.sax_tree.labels
            )
        )

        # counterfactual rule plotted on a counterexamplar z_tilde
        # get all the leave idxs
        (
            _,
            counterfactual_leaf,
        ) = self.sax_tree.decision_tree_queryable_._minimum_distance(
            self.sax_tree.decision_tree_queryable_._get_node_by_idx(factual_id)
        )  # FIXME: ugly
        leave_idxs = self.sax_tree.decision_tree_.apply(self.sax_tree.X_transformed_)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.argwhere(leave_idxs == counterfactual_leaf)
        # choose one counterfactual among those in the leaf
        counterfactual_idx = counterfactuals_idxs[kwargs.get("z_tilde_idx", 0)][0]
        counterfactual_ts = [
            self.sax_tree.X_[i][counterfactual_idx : counterfactual_idx + 1]
            for i in range(len(x))
        ]
        counterfactual_y = self.sax_tree.y_[counterfactual_idx]
        if counterfactual_y != counterfactual_rule.consequence:
            warnings.warn(
                "The real class is different from the one predicted by the tree.\ny_real=%s "
                "y_pred=%s" % (counterfactual_y, counterfactual_rule.consequence)
            )

        (
            factual_cont_subseq_dict,
            counterfactual_cont_subseq_dict,
        ) = self._contained_dictionaries(
            x, counterfactual_ts, factual_rule, counterfactual_rule, **kwargs
        )

        subs_idxs, ops, subs, start_idxs, dim_idxs, subs_names = self._map_to_dimension(
            x, factual_rule, factual_cont_subseq_dict, plot_names=plot_names
        )

        plot_multi_subsequence_rule(
            x=draw_on,
            shapelets_idxs=subs_idxs,
            shapelets=subs,
            starting_idxs=start_idxs,
            condition_operators=ops,
            dimension_idxs=dim_idxs,
            title="Factual Rule",
            legend_label=r"$X$",
            shapelets_names=subs_names,
            **kwargs
        )

        subs_idxs, ops, subs, start_idxs, dim_idxs, subs_names = self._map_to_dimension(
            x, counterfactual_rule, factual_cont_subseq_dict
        )

        plot_multi_subsequence_rule(
            x=draw_on,
            shapelets_idxs=subs_idxs,
            shapelets=subs,
            starting_idxs=start_idxs,
            condition_operators=ops,
            dimension_idxs=dim_idxs,
            title="Counterfactual Rule",
            legend_label=r"$X$",
            shapelets_names=subs_names,
            **kwargs
        )

        subs_idxs, ops, subs, start_idxs, dim_idxs, subs_names = self._map_to_dimension(
            counterfactual_ts, counterfactual_rule, counterfactual_cont_subseq_dict
        )
        plot_multi_subsequence_rule(
            x=counterfactual_ts,
            shapelets_idxs=subs_idxs,
            shapelets=subs,
            starting_idxs=start_idxs,
            condition_operators=ops,
            dimension_idxs=dim_idxs,
            title="Counterfactual Rule (on a counterexemplar)",
            legend_label=r"$\hat{X}'_\neq$",
            shapelets_names=subs_names,
            **kwargs
        )

    def plot(self, kind, **kwargs):
        if kind == "subsequences_heatmap":
            super().plot_shapelet_heatmap(**kwargs)

        elif kind == "subsequences_binary_heatmap":
            super().plot_binary_heatmap(x_label=kwargs.pop("x_label"), **kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        elif kind == "rules":
            self.plot_rules(x=kwargs.pop("x"), **kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        else:
            raise Exception("Plot kind not valid.")
        return self


class SaxTreeMultivariateEvaluator(SaxTreeEvaluator):
    def __init__(self, sax_tree_multi: SaxTreeMultivariate):
        super().__init__(sax_tree=sax_tree_multi)

    def tree_coverage_score(self, x):
        x = format_multivariate_input(x)
        return super().tree_coverage_score(x)

    def tree_precision_score(self, X, x, y):
        x = format_multivariate_input(x)
        X = format_multivariate_input(X)
        return super().tree_precision_score(X, x, y)


def main1():
    X_train = [np.random.random(size=(10, 235, 1)), np.random.random(size=(10, 20, 1))]
    y_train = np.random.randint(0, 2, size=10)

    clf = SaxTreeMultivariate(
        random_state=0,
        prune_duplicate_tree_leaves=True,
    )

    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    from lasts.datasets.datasets import build_multivariate_cbf

    (
        X_train,
        y_train,
        _,
        _,
        X_test,
        y_test,
        X_exp_train,
        y_exp_train,
        *_,
    ) = build_multivariate_cbf()

    clf = SaxTreeMultivariate(
        random_state=0,
        prune_duplicate_tree_leaves=True,
    )

    clf.fit(X_exp_train, y_exp_train)

    print(clf.score(X_test, y_test))
    i = 10
    x = X_test[i : i + 1]
    # print(clf.explain(X_test[i:i+1]))
    # clf.plotter.plot_factual_rule([X_train[0][0:1], X_train[1][0:1]], return_y_lim=False)
    print(clf.explain(x=x))
    clf.plot("rules", x=x, figsize=(10, 10))
    # clf.plot(kind="rules", x=X_train[1:2])
    # # print(clf.evaluate("tree_coverage", x=X_train[1:2]))
    # exp = clf.explain(x=X_train[1:2])

    # clf.plot("subsequences_heatmap", step=100)
