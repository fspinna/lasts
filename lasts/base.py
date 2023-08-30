from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score
from joblib import dump, load
from lasts.utils import make_path
import pathlib
import importlib
import numpy as np


class Explanation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def explain(self, **kwargs):
        return


class Pickler(object):
    def __init__(self, object_to_pickle):
        self.object_to_pickle = object_to_pickle
        self.variables_names = None
        self.modules_paths = None
        self.classes_names = None

    def save(self, variables_names):
        self.variables_names = variables_names
        self.modules_paths = list()
        self.classes_names = list()
        for name in self.variables_names:
            self.modules_paths.append(
                getattr(self.object_to_pickle, name).__module__
            )  # absolute path to the module
            self.classes_names.append(
                type(getattr(self.object_to_pickle, name)).__name__
            )  # object class name
        return self

    def load(self):
        return self.variables_names, self.modules_paths, self.classes_names


class Module:
    __metaclass__ = ABCMeta
    pickler: Pickler

    @abstractmethod
    def predict(self, **kwargs):
        return

    def save(self, folder, name=""):
        path = make_path(folder)
        var_names, *_ = self.pickler.load()  # get the name of variables to pickle
        for var_name in var_names:
            var = getattr(self, var_name)  # get the variable
            var.save(folder, name)  # save the variable (and subvariables)
            setattr(self, var_name, None)  # set variable to None
        dump(self, path / (name + ".joblib"))  # dump self

    @classmethod
    def load(cls, folder, name="", custom_load=None):
        path = pathlib.Path(folder)
        module = load(path / (name + ".joblib"))  # load explainer
        var_names, mod_paths, class_names = (
            module.pickler.load() if custom_load is None else custom_load
        )
        for var_name, mod_path, class_name in zip(var_names, mod_paths, class_names):
            class_ = getattr(
                importlib.import_module(mod_path), class_name
            )  # retrieve the class to load
            var = class_.load(folder, name)  # load the instance of the class
            setattr(
                module, var_name, var
            )  # set the variable to the instance of the class
        return module


class ModularExplainer(Module):
    @abstractmethod
    def fit(self, x):
        return

    @abstractmethod
    def explain(self):
        return

    @abstractmethod
    def plot(self, kind, **kwargs):
        return

    @abstractmethod
    def evaluate(self, metric, **kwargs):
        return


class Surrogate(Module):
    labels: list or None

    @abstractmethod
    def fit(self, X, y):
        return

    @abstractmethod
    def explain(self, **kwargs):
        return

    @abstractmethod
    def plot(self, kind, **kwargs):
        return

    @abstractmethod
    def predict(self, X):
        return

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    @abstractmethod
    def evaluate(self, metric, **kwargs):
        return

    def set_labels(self, labels):
        self.labels = labels


class NeighborhoodGenerator(Module):
    labels: list or None

    @abstractmethod
    def fit(self, x):
        return

    @abstractmethod
    def predict(self):
        return

    @abstractmethod
    def plot(self, kind, **kwargs):
        return

    @abstractmethod
    def evaluate(self, metric, **kwargs):
        return

    @abstractmethod
    def explain(self):
        return

    def set_labels(self, labels):
        self.labels = labels


class BlackBox:
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, X):
        return

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    @abstractmethod
    def predict_proba(self, X):
        return


class Plotter:
    __metaclass__ = ABCMeta

    @abstractmethod
    def plot(self, kind, **kwargs):
        return


class Evaluator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, metric, **kwargs):
        return


class RuleBasedExplanation(Explanation):
    def __init__(self, factual_rule, counterfactual_rule=None):
        self.factual_rule = factual_rule
        self.counterfactual_rule = counterfactual_rule

    def __str__(self):
        return "factual_rule: %s\ncounterfactual_rule: %s" % (
            str(self.factual_rule),
            str(self.counterfactual_rule)
            if self.counterfactual_rule is not None
            else "None",
        )

    def explain(self, **kwargs):
        return self.factual_rule, self.counterfactual_rule


class ExampleBasedExplanation(Explanation):
    def __init__(self, exemplars, counterexemplars):
        self.exemplars = exemplars
        self.counterexemplars = counterexemplars

    def explain(self, **kwargs):
        return self.exemplars, self.counterexemplars


class SaliencyBasedExplanation(Explanation):
    def __init__(self, saliency):
        self.saliency = saliency

    def explain(self, **kwargs):
        return self.saliency


class Benchmark:
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, **kwargs):
        return
