from lasts.base import BlackBox, Pickler, Module
from lasts.utils import make_path
import numpy as np
from joblib import dump, load
import pathlib
import tensorflow as tf
import importlib


class SklearnClassifierWrapper(BlackBox):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        # change the input shape of the time series, from 3 dimensions to 2.
        return self.clf.predict(X[:, :, 0]).ravel()

    def predict_proba(self, X):
        # change the input shape of the time series, from 3 dimensions to 2.
        return self.clf.predict_proba(X[:, :, 0])


class KerasClassifierWrapper(BlackBox):
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        # here the input is 3-dimensional already
        y = self.clf.predict(X)
        # FIXME: not sure about this condition, check it
        if len(y.shape) > 1 and (
            y.shape[1] != 1
        ):  # not something like this: [[1],[0],[1],...]
            y = np.argmax(y, axis=1)
        return y.ravel()

    def predict_proba(self, X):
        # The probabilities are already in the predict
        return self.clf.predict(X)

    def save(self, folder, name=""):
        path = make_path(folder)
        self.clf.save(path / (name + "_keraswrapper.h5"))
        self.clf = None
        dump(self, path / (name + "_keraswrapper.joblib"))

    @classmethod
    def load(cls, folder, name=""):
        path = pathlib.Path(folder)
        wrapper = load(path / (name + "_keraswrapper.joblib"))
        wrapper.clf = tf.keras.models.load_model(path / (name + "_keraswrapper.h5"))
        return wrapper


class DecoderWrapper(Module):
    def __init__(self, decoder):
        self.decoder = decoder

    def predict(self, X):
        return self.decoder.predict(X)

    def save(self, folder, name=""):
        path = make_path(folder)
        self.decoder.save(path / (name + "_decoder.h5"))
        self.decoder = None
        dump(self, path / (name + "_decoder.joblib"))

    @classmethod
    def load(cls, folder, name=""):
        path = pathlib.Path(folder)
        wrapper = load(path / (name + "_decoder.joblib"))
        wrapper.decoder = tf.keras.models.load_model(path / (name + "_decoder.h5"))
        return wrapper


class DecoderBlackboxWrapper(Module):
    def __init__(self, decoder, blackbox, unpicklable_variables=None):
        self.decoder = decoder
        self.blackbox = blackbox
        self.pickler = Pickler(self)
        if unpicklable_variables is not None:
            self.pickler.save(unpicklable_variables)

    def predict(self, X):
        return self.blackbox.predict(self.decoder.predict(X))

    def save(self, folder, name=""):
        super().save(folder, name + "_decoderblackbox")

    @classmethod
    def load(cls, folder, name="", custom_load=None):
        return super().load(folder, name + "_decoderblackbox", custom_load)
