import warnings

from anchor import anchor_tabular
import numpy as np
from lasts.base import BlackBox, ModularExplainer, RuleBasedExplanation, Evaluator
from lasts.explanations.rule import Inequality, Rule
from lasts.explainers.shapts_multi import from2dto3d, from3dto2d
from datetime import datetime


class AnchorBlackboxWrapper(BlackBox):
    def __init__(self, blackbox, shape_3d):
        self.blackbox = blackbox
        self.shape_3d = shape_3d

    def predict(self, X):
        return self.blackbox.predict(from2dto3d(X, shape_3d=self.shape_3d))


class AnchorTS(ModularExplainer):
    def __init__(self, blackbox, X, labels):
        self.blackbox = AnchorBlackboxWrapper(blackbox, shape_3d=X.shape)
        self.anchor = anchor_tabular.AnchorTabularExplainer(
            class_names=labels,
            feature_names=[str(i) for i in range(from3dto2d(X).shape[1])],
            train_data=from3dto2d(X),
        )
        self.labels = labels
        self.X = X
        self.plotter = None
        self.evaluator = AnchorTSEvaluator(self)

        self.explanation_ = None
        self.anchor_explanation_ = None
        self.runtime_ = None
        self.coverage_ = None
        self.precision_ = None

    def fit(self, x):
        start_time = datetime.now()
        x_label = self.blackbox.blackbox.predict(x)[0]
        self.anchor_explanation_ = self.anchor.explain_instance(
            from3dto2d(x), self.blackbox.predict, threshold=0.95
        )
        if x_label != self.anchor_explanation_.exp_map["prediction"]:
            warnings.warn("Incoherent predicted labels")
        self.explanation_ = RuleBasedExplanation(
            parse_anchor_rule(self.anchor_explanation_.names(), x_label, self.labels)
        )
        self.runtime_ = (datetime.now() - start_time).total_seconds()
        self.coverage_ = self.anchor_explanation_.coverage()
        self.precision_ = self.anchor_explanation_.precision()
        return self

    def explain(self):
        return self.explanation_

    def fit_explain(self, x):
        self.fit(x)
        return self.explain()

    def plot(self, kind, **kwargs):
        pass

    def evaluate(self, metric, **kwargs):
        return self.evaluator.evaluate(metric, **kwargs)


class AnchorTSEvaluator(Evaluator):
    METRICS = ["coverage", "precision", "rule_length", "runtime"]

    def __init__(self, anchor_explainer: AnchorTS):
        self.anchor_explainer = anchor_explainer

    def evaluate(self, metric, **kwargs):
        if metric == "coverage":
            return self.coverage_score()
        elif metric == "precision":
            return self.precision_score()
        elif metric == "rule_length":
            return self.rule_length_score()
        elif metric == "runtime":
            return self.anchor_explainer.runtime_
        else:
            raise Exception("Metric not valid.")

    def coverage_score(self):
        if self.evaluate("rule_length") > 0:
            return self.anchor_explainer.coverage_
        else:
            return np.nan

    def precision_score(self):
        if self.evaluate("rule_length") > 0:
            return self.anchor_explainer.precision_
        else:
            return np.nan

    def rule_length_score(self):
        return len(self.anchor_explainer.explanation_.factual_rule)


# def parse_anchor_inequality(inequality):
#     attribute, operator, threshold = inequality.split(" ")
#     attribute = int(attribute)
#     threshold = float(threshold)
#     return Inequality(attribute, operator, threshold)


def parse_anchor_inequality(inequality):
    splitted_inequality = inequality.split(" ")
    if len(splitted_inequality) == 3:  # e.g. feature <= 1.2
        attribute, operator, threshold = inequality.split(" ")
        attribute = int(attribute)
        threshold = float(threshold)
        inequalities = [Inequality(attribute, operator, threshold)]
    elif len(splitted_inequality) == 5:  # e.g. 0.5 < feature <= 1.2
        threshold1, operator1, attribute, operator2, threshold2 = inequality.split(" ")
        attribute = int(attribute)
        threshold1 = float(threshold1)
        threshold2 = float(threshold2)
        inequalities = [
            Inequality(attribute, operator1, threshold1),
            Inequality(attribute, operator2, threshold2),
        ]
    return inequalities


def parse_anchor_rule(inequality_list, consequence, labels=None):
    inequalities = list()
    for inequality in inequality_list:
        inequalities.extend(parse_anchor_inequality(inequality))
    rule = Rule(inequalities, consequence, labels=labels)
    return rule


def test_multivariate():
    from lasts.datasets.datasets import build_multivariate_cbf

    _, _, _, _, _, _, _, _, _, _, X_exp_test, y_exp_test = build_multivariate_cbf()

    blackbox = cached_blackbox_loader("cbfmulti_rocket.joblib")

    i = 0
    x = X_exp_test[i].ravel().reshape(1, -1, 1)

    anch = AnchorTS(blackbox, X_exp_test[1:], ["cyl", "bell", "fun"])

    exp = anch.fit_explain(x)

    print(exp)
    print(anch.evaluate("coverage"))
    print(anch.evaluate("precision"))
    print(anch.evaluate("rule_length"))
    return anch


if __name__ == "__main__":
    from lasts.blackboxes.loader import cached_blackbox_loader
    from lasts.datasets.loader import dataset_loader

    data = dataset_loader("Coffee")
    X_train, y_train, X_test, y_test = data()

    blackbox = cached_blackbox_loader("Coffee_rocket.joblib")

    i = 0
    x = X_test[i : i + 1]

    anch = AnchorTS(
        blackbox,
        np.median(X_train, axis=0)[np.newaxis, :, :],
        data.label_encoder.classes_,
    )

    exp = anch.fit_explain(x)

    print(exp)
    print(anch.evaluate("coverage"))
    print(anch.evaluate("precision"))
    print(anch.evaluate("rule_length"))
    print(anch.runtime_)

    # from lasts.datasets.datasets import build_multivariate_cbf
    # _, _, _, _, _, _, X_exp_train, y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test = build_multivariate_cbf()
    #
    # blackbox = cached_blackbox_loader("cbfmulti_rocket.joblib")
    #
    # i = 0
    # x = X_exp_test[i:i+1]
    #
    # anch = AnchorTS(blackbox, X_exp_test[1:11], ["cyl", "bell", "fun"])
    #
    # exp = anch.fit_explain(x)
    #
    # print(exp)
    # print(anch.evaluate("coverage"))
    # print(anch.evaluate("precision"))
    # print(anch.evaluate("rule_length"))
