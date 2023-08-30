from lasts.base import ModularExplainer, Plotter, Evaluator, Explanation
import numpy as np


class DummyExplainer(ModularExplainer):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        self.x_ = None
        self.explanation_ = None
        self.plotter = DummyExplainerPlotter(self)
        self.evaluator = DummyExplainerEvaluator(self)

    def fit(self, x):
        self.x_ = x
        return self

    def explain(self):
        return self.explanation_

    def fit_explain(self, x):
        self.fit(x)
        return self.explain()

    def plot(self, kind, **kwargs):
        self.plotter.plot(kind, **kwargs)

    def evaluate(self, metric, **kwargs):
        self.evaluator.evaluate(metric, **kwargs)


class DummyExplainerPlotter(Plotter):
    def __init__(self, dummy_explainer: DummyExplainer):
        self.dummy_explainer = dummy_explainer

    def plot(self, kind, **kwargs):
        pass


class DummyExplainerEvaluator(Evaluator):
    def __init__(self, dummy_explainer: DummyExplainer):
        self.dummy_explainer = dummy_explainer

    def evaluate(self, metric=None, **kwargs):
        return np.nan


class DummyExplanation(Explanation):
    def __init__(self):
        pass

    def explain(self, **kwargs):
        return np.nan
