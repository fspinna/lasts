from joblib import load

import tensorflow as tf

from lasts.utils import get_project_root
from lasts.wrappers import SklearnClassifierWrapper, KerasClassifierWrapper
from lasts.blackboxes.rocket import RocketWrapper
from lasts.explainers.synth import SynthPatternClassifier


def cached_blackbox_list():
    root = get_project_root()
    path = (root / "blackboxes" / "cached").glob("**/*")
    return [x.name for x in path if x.is_file() and x.suffix in [".joblib", ".h5"]]


def cached_blackbox_loader(file_name, wrap=True):
    root = get_project_root()
    path = root / "blackboxes" / "cached" / file_name
    if path.suffix == ".joblib":
        if wrap:
            blackbox = load(path)
            if isinstance(blackbox, RocketWrapper):
                pass  # no need to wrap it
            elif isinstance(blackbox, SynthPatternClassifier):
                pass
            else:
                blackbox = SklearnClassifierWrapper(blackbox)
        else:
            blackbox = load(path)
    elif path.suffix == ".h5":
        if wrap:
            blackbox = KerasClassifierWrapper(tf.keras.models.load_model(path))
        else:
            blackbox = tf.keras.models.load_model(path)
    else:
        raise Exception("File extension not recognized.")
    return blackbox


if __name__ == "__main__":
    blackbox = cached_blackbox_loader("cbf_knn.joblib")
    print(blackbox)
