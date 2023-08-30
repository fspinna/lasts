from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.shapelet_based.mrseql.mrseql import (
    PySAX,
)  # custom fork of sktime
from sktime.utils.validation.panel import check_X_y, check_X
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import scipy
import numpy as np
import math
import warnings


from lasts.base import Plotter, Evaluator, Surrogate, RuleBasedExplanation
from lasts.utils import (
    coverage_score_scikit_tree,
    precision_score_scikit_tree,
    compute_medoid,
    convert_numpy_to_sktime,
    sliding_window_distance,
)
from lasts.explanations.tree import SklearnDecisionTreeConverter, prune
from lasts.plots import (
    plot_subsequences_grid,
    plot_subsequence_mapping,
    plot_binary_heatmap,
    plot_shapelet_rule,
    plot_sklearn_decision_tree,
    plot_shapelet_heatmap,
)


def find_feature(feature_idx, sequences):  # FIXME: check if the count is correct
    idxs = 0
    for i, config in enumerate(sequences):
        if idxs + len(config) - 1 < feature_idx:
            idxs += len(config)
            continue
        elif idxs + len(config) - 1 >= feature_idx:
            j = feature_idx - idxs
            feature = config[j]
            break
    return feature, i, j


def get_subsequence_alignment(x, word_idx, seql_model):
    start_idx, end_idx, word = map_word_idx_to_ts(x, word_idx, seql_model)
    if end_idx == len(x.ravel()):
        end_idx -= 1
    subsequence = x.ravel()[start_idx : end_idx + 1]
    return start_idx, end_idx, subsequence, word


def map_word_idx_to_ts(x, word_idx, seql_model):
    word, cfg_idx, _ = find_feature(word_idx, seql_model.sequences)
    start_idx, end_idx = map_word_to_ts(x, word, seql_model.config[cfg_idx])
    return start_idx, end_idx, word


def map_word_to_ts(x, word, cfg):
    word = [word]
    ps = PySAX(cfg["window"], cfg["word"], cfg["alphabet"])
    idx_set = ps.map_patterns(x.ravel(), word)[0]
    # print(idx_set)
    if len(idx_set) == 0:
        return None, None
    idx_set_sorted = sorted(list(idx_set))
    start_idx = idx_set_sorted[0]
    end_idx = math.floor(start_idx + (len(word[0]) * cfg["window"] / cfg["word"]) - 1)
    if end_idx < len(x) - 1:
        end_idx += 1
    return start_idx, end_idx


def create_subsequences_dictionary(X, X_transformed, n_features, seql_model):
    subsequence_dictionary = dict()
    subsequence_norm_dictionary = dict()
    name_dictionary = dict()
    for feature in range(n_features):
        subsequences = list()
        subsequences_norm = list()
        lengths = list()
        for i, x in enumerate(X):
            if X_transformed[i][feature] == 1:
                start_idx, end_idx, feature_string = map_word_idx_to_ts(
                    x, feature, seql_model
                )
                x_norm = scipy.stats.zscore(x)
                if start_idx is not None:
                    subsequence = x[start_idx:end_idx]
                    subsequences.append(subsequence)

                    subsequence_norm = x_norm[start_idx:end_idx]
                    subsequences_norm.append(subsequence_norm)

                    lengths.append(end_idx - start_idx)
        mode = scipy.stats.mode(np.array(lengths))[0][0]
        subsequences_same_length = list()
        subsequences_norm_same_length = list()
        for i, subsequence in enumerate(
            subsequences
        ):  # to avoid problems with sequences having slightly different lengths
            if len(subsequence) == mode:
                subsequences_same_length.append(subsequence)
                subsequences_norm_same_length.append(subsequences_norm[i])
        subsequence_dictionary[feature] = np.array(subsequences_same_length)
        subsequence_norm_dictionary[feature] = np.array(subsequences_norm_same_length)
        name_dictionary[feature] = feature_string
    return subsequence_dictionary, name_dictionary, subsequence_norm_dictionary


class SaxTree(Surrogate):
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
        create_plotting_dictionaries=True,
    ):
        self.labels = labels
        self.random_state = random_state
        self.decision_tree_grid_search_kwargs = decision_tree_grid_search_kwargs
        self.custom_config = custom_config
        self.create_plotting_dictionaries = create_plotting_dictionaries
        self.plotter = SaxTreePlotter(self)
        self.evaluator = SaxTreeEvaluator(self)

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

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        X = convert_numpy_to_sktime(X)
        seql_model = MrSEQLClassifier(
            seql_mode="fs", symrep="sax", custom_config=self.custom_config
        )
        seql_model.fit(X, y)
        X, _ = check_X_y(X, y, coerce_to_numpy=True)
        mr_seqs = seql_model._transform_time_series(X)
        X_transformed = seql_model._to_feature_space(mr_seqs)

        clf = DecisionTreeClassifier()
        param_grid = self.decision_tree_grid_search_kwargs
        grid = GridSearchCV(
            clf, param_grid=param_grid, scoring="accuracy", n_jobs=1, verbose=0
        )
        grid.fit(X_transformed, y)

        clf = DecisionTreeClassifier(
            **grid.best_params_, random_state=self.random_state
        )
        clf.fit(X_transformed, y)
        # prune_duplicate_leaves(clf)
        clf = prune(clf)

        self.X_transformed_ = X_transformed
        self.decision_tree_ = clf
        self.decision_tree_queryable_ = SklearnDecisionTreeConverter(clf)
        self.seql_model_ = seql_model
        if self.create_plotting_dictionaries:
            self._create_dictionaries()
        return self

    def transform(self, X):
        X = convert_numpy_to_sktime(X)
        X = check_X(X, coerce_to_numpy=True)
        mr_seqs = self.seql_model_._transform_time_series(X)
        X_transformed = self.seql_model_._to_feature_space(mr_seqs)
        return X_transformed

    def predict(self, X):
        X_transformed = self.transform(X)
        y = self.decision_tree_.predict(X_transformed)
        return y

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def find_leaf_id(self, ts):
        ts_transformed = self.transform(ts)
        leaf_id = self.decision_tree_.apply(ts_transformed)[0]
        return leaf_id

    def find_counterfactual_leaf_id(self, ts):
        leaf_id = self.find_leaf_id(ts)
        _, counterfactual_leaf = self.decision_tree_queryable_._minimum_distance(
            self.decision_tree_queryable_._get_node_by_idx(leaf_id)
        )  # FIXME: ugly
        return counterfactual_leaf

    def find_counterfactual_tss(self, ts):
        leave_idxs = self.decision_tree_.apply(self.X_transformed_)
        counterfactual_leaf = self.find_counterfactual_leaf_id(ts)
        # get all record in the counterfactual leaf
        counterfactuals_idxs = np.nonzero(leave_idxs == counterfactual_leaf)
        counterfactual_tss = self.X_[counterfactuals_idxs]
        return counterfactual_tss

    def _create_dictionaries(self):
        (
            self.subsequence_dict_,
            self.name_dict_,
            self.subsequence_norm_dict_,
        ) = create_subsequences_dictionary(
            self.X_, self.X_transformed_, self.X_transformed_.shape[1], self.seql_model_
        )

    def explain(self, x, **kwargs):
        factual_id = self.find_leaf_id(x)
        factual_rule = self.decision_tree_queryable_.get_factual_rule_by_idx(
            factual_id, as_contained=True, labels=self.labels
        )
        if len(factual_rule.premises) == 0:  # if the premises are empty
            counterfactual_rule = factual_rule
            warnings.warn("The decision tree is a unique leaf.")
        else:
            counterfactual_rule = (
                self.decision_tree_queryable_.get_counterfactual_rule_by_idx(
                    factual_id, as_contained=True, labels=self.labels
                )
            )
        explanation = RuleBasedExplanation(
            factual_rule=factual_rule, counterfactual_rule=counterfactual_rule
        )
        return explanation

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind, **kwargs)

    def evaluate(self, metric, **kwargs):
        return self.evaluator.evaluate(metric, **kwargs)


class SaxTreePlotter(Plotter):
    def __init__(self, sax_tree: SaxTree):
        self.sax_tree = sax_tree

    def plot_subsequences_grid(
        self, n, m, starting_idx=0, random=False, color="mediumblue", **kwargs
    ):
        subsequence_list = list()
        for key in self.sax_tree.subsequence_dict_:
            subsequence_list.append(
                self.sax_tree.subsequence_dict_[key].mean(axis=0).ravel()
            )
        plot_subsequences_grid(
            subsequence_list,
            n=n,
            m=m,
            starting_idx=starting_idx,
            random=random,
            color=color,
            **kwargs
        )

    def plot_rule_subsequences(self, rule, **kwargs):
        for attribute in rule.premises_attributes:
            plot_subsequence_mapping(
                self.sax_tree.subsequence_dict_,
                self.sax_tree.name_dict_,
                attribute,
                **kwargs
            )

    def plot_binary_heatmap(self, x_label, step, **kwargs):
        plot_binary_heatmap(
            x_label=x_label,
            y=self.sax_tree.y_,
            X_binary=self.sax_tree.X_transformed_,
            step=step,
            **kwargs
        )

    def plot_shapelet_heatmap(self, step, **kwargs):
        plot_shapelet_heatmap(
            X=self.sax_tree.X_transformed_, y=self.sax_tree.y_, step=step, **kwargs
        )

    def plot_rules(self, x, **kwargs):
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
        counterfactual_idx = counterfactuals_idxs[kwargs.get("z_tilde_idx", 0)]
        counterfactual_ts = self.sax_tree.X_[counterfactual_idx].reshape(1, -1)
        counterfactual_y = self.sax_tree.y_[counterfactual_idx][0]
        if counterfactual_y != counterfactual_rule.consequence:
            warnings.warn(
                "The real class is different from the one predicted by the tree.\ny_real=%s "
                "y_pred=%s" % (counterfactual_y, counterfactual_rule.consequence)
            )

        self._plot_rules(
            x=x,
            z_tilde_star=counterfactual_ts,
            factual_rule=factual_rule,
            counterfactual_rule=counterfactual_rule,
            **kwargs
        )

    def _plot_rules(self, x, z_tilde_star, factual_rule, counterfactual_rule, **kwargs):
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
            start_idx, end_idx, subsequence, feature_name = get_subsequence_alignment(
                x, sub_idx, self.sax_tree.seql_model_
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
            start_idx, end_idx, subsequence, feature_name = get_subsequence_alignment(
                z_tilde_star, sub_idx, self.sax_tree.seql_model_
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

        y_lim = self._plot_rule(
            rule=factual_rule,
            x=x,
            contained_subsequences_dict=factual_cont_subseq_dict,
            legend_label=r"$X$",
            title="Factual Rule",
            return_y_lim=True,
            **kwargs
        )

        self._plot_rule(
            rule=counterfactual_rule,
            x=x,
            contained_subsequences_dict=factual_cont_subseq_dict,  # factual it's right because we are plotting over x
            legend_label=r"$X$",
            title="Counterfactual Rule",
            return_y_lim=False,
            **kwargs
        )

        kwargs.pop("draw_on", None)  # don't want to plot over another ts
        self._plot_rule(
            rule=counterfactual_rule,
            x=z_tilde_star,
            contained_subsequences_dict=counterfactual_cont_subseq_dict,
            legend_label=r"$\hat{X}'_\neq$",
            title="Counterfactual Rule (on a counterexemplar)",
            return_y_lim=False,
            y_lim=y_lim,
            **kwargs
        )

    def _plot_rule(
        self,
        rule,
        x,
        contained_subsequences_dict,
        legend_label,
        title,
        return_y_lim,
        y_lim=None,
        plot_names=True,
        rules_in_title=True,
        **kwargs
    ):
        subsequences_idxs = [premise.attribute for premise in rule.premises]
        if plot_names:
            subsequences_names = [
                self.sax_tree.name_dict_[idx].decode("utf-8")
                for idx in subsequences_idxs
            ]
        else:
            subsequences_names = None
        operators = [premise.operator for premise in rule.premises]
        subsequences = list()
        starting_idxs = list()

        for sub_idx, op in zip(subsequences_idxs, operators):
            if op == ">":  # if the subsequences is contained
                # if sub_idx in factual_cont_subseq_dict:
                subsequence = contained_subsequences_dict[sub_idx]
                start_idx = sliding_window_distance(x.ravel(), subsequence)
                end_idx = start_idx + len(subsequence)
                if end_idx == len(x.ravel()):
                    end_idx -= 1
                    subsequence = subsequence[:-1]
            else:  # if the subsequence is not-contained
                if (
                    sub_idx in contained_subsequences_dict
                ):  # if the subsequence is contained in the other rule
                    subsequence = contained_subsequences_dict[
                        sub_idx
                    ]  # use the saved subsequence
                    start_idx = sliding_window_distance(x.ravel(), subsequence)
                    end_idx = start_idx + len(subsequence)
                    if end_idx == len(x.ravel()):
                        end_idx -= 1
                        subsequence = subsequence[:-1]
                else:  # if the subsequence is always not-contained
                    #  find a representative of the not-contained subsequence
                    subsequence = compute_medoid(
                        self.sax_tree.subsequence_dict_[sub_idx][:, :, 0]
                    )
                    #  find the alignment of the subsequence with x
                    start_idx = sliding_window_distance(x.ravel(), subsequence)
                    end_idx = start_idx + len(subsequence)
                    if end_idx == len(x.ravel()):
                        end_idx -= 1
                        subsequence = subsequence[:-1]
            subsequences.append(subsequence)
            starting_idxs.append(start_idx)
        y_lim = plot_shapelet_rule(
            x=kwargs.get("draw_on", x),
            shapelets_idxs=subsequences_idxs,
            shapelets=subsequences,
            starting_idxs=starting_idxs,
            condition_operators=operators,
            shapelets_names=subsequences_names,
            title=title + "\n" + str(rule) if rules_in_title else title,
            legend_label=legend_label,
            return_y_lim=return_y_lim,
            forced_y_lim=y_lim,
            **kwargs
        )
        if return_y_lim:
            return y_lim
        else:
            return None

    def plot_tree(self, **kwargs):
        plot_sklearn_decision_tree(dt=self.sax_tree.decision_tree_, **kwargs)

    def plot(self, kind, **kwargs):
        if kind == "subsequences_grid":
            self.plot_subsequences_grid(
                n=kwargs.pop("n"),
                m=kwargs.pop("m"),
                starting_idx=kwargs.pop("starting_idx", 0),
                random=kwargs.pop("random", False),
                **kwargs
            )

        elif kind == "rule_subsequences":
            self.plot_rule_subsequences(kwargs.pop("rule"), **kwargs)

        elif kind == "subsequences_binary_heatmap":
            self.plot_binary_heatmap(
                x_label=kwargs.pop("x_label"), step=kwargs.pop("step", 200), **kwargs
            )

        elif kind == "subsequences_heatmap":
            self.plot_shapelet_heatmap(step=kwargs.pop("step", 200), **kwargs)

        elif kind == "tree":
            self.plot_tree(**kwargs)

        elif kind == "factual_rule":
            pass

        elif kind == "counterfactual_rule":
            pass

        elif kind == "rules":
            self.plot_rules(x=kwargs.pop("x"), **kwargs)

        else:
            raise Exception("Plot kind not valid.")

        return self


class SaxTreeEvaluator(Evaluator):
    def __init__(self, sax_tree: SaxTree):
        self.sax_tree = sax_tree

    def tree_coverage_score(self, x):
        leaf_id = self.sax_tree.find_leaf_id(x)
        return coverage_score_scikit_tree(
            dt=self.sax_tree.decision_tree_, leaf_id=leaf_id
        )

    def tree_precision_score(self, X, x, y):
        leaf_id = self.sax_tree.find_leaf_id(x)
        X = self.sax_tree.transform(X)
        return precision_score_scikit_tree(
            dt=self.sax_tree.decision_tree_, X=X, y=y, leaf_id=leaf_id
        )

    def factual_rule_length_score(self, x, **kwargs):
        explanation = self.sax_tree.explain(x, **kwargs)
        return len(explanation.factual_rule)

    def counterfactual_rule_length_score(self, x, **kwargs):
        explanation = self.sax_tree.explain(x, **kwargs)
        return len(explanation.counterfactual_rule)

    def accuracy_score(self, X, y):
        return self.sax_tree.score(X, y)

    def evaluate(self, metric, **kwargs):
        if metric in ["tree_coverage", "tree_factual_coverage"]:
            return self.tree_coverage_score(x=kwargs.get("x"))
        elif metric in ["tree_precision"]:
            return self.tree_precision_score(
                X=kwargs.get("X"),
                x=kwargs.get("x"),
                y=kwargs.get("y"),
            )

        elif metric == "tree_factual_precision":
            return self.sax_tree.evaluate(
                metric="tree_precision",
                X=self.sax_tree.X_,
                x=kwargs.pop("x"),
                y=self.sax_tree.y_,
            )

        elif metric == "tree_counterfactual_precision":
            counterfactual_ts = self.sax_tree.find_counterfactual_tss(kwargs.pop("x"))[
                0:1
            ]
            return self.sax_tree.evaluate(
                metric="tree_precision",
                X=kwargs.get("X"),
                x=counterfactual_ts,
                y=kwargs.get("y"),
            )

        elif metric == "tree_counterfactual_coverage":
            counterfactual_ts = self.sax_tree.find_counterfactual_tss(kwargs.pop("x"))[
                0:1
            ]
            return self.sax_tree.evaluate(metric="tree_coverage", x=counterfactual_ts)

        elif metric == "accuracy_score":
            return self.accuracy_score(X=kwargs.get("X"), y=kwargs.get("y"))
        elif metric == "factual_rule_length":
            return self.factual_rule_length_score(kwargs.pop("x"), **kwargs)

        elif metric == "counterfactual_rule_length":
            return self.counterfactual_rule_length_score(kwargs.pop("x"), **kwargs)

        else:
            raise Exception("Metric not valid.")


if __name__ == "__main__":
    from lasts.datasets.datasets import build_cbf

    X_train, y_train, X_val, y_val, *_ = build_cbf(n_samples=600)

    clf = SaxTree(random_state=0, labels=["cyl", "bell", "fun"])
    clf.fit(X_train, y_train)
    print(clf.score(X_val, y_val))
    i = 0
    x = X_train[i : i + 1]
    exp = clf.explain(x)
    print(exp)
    clf.plot(kind="rules", x=x)
    # clf.plot(kind="factual_rule", x=x)
    # clf.plot(kind="counterfactual_rule", x=x)
    # clf.plot(kind="rule_subsequences", rule=exp.factual_rule)
    # clf.plot(kind="rule_subsequences", rule=exp.counterfactual_rule)
    print(clf.evaluate("tree_coverage", x=X_train[1:2]))
    print(clf.evaluate(metric="tree_precision", X=X_train, x=x, y=y_train))
    print(clf.evaluate("accuracy_score", X=X_train, y=y_train))
