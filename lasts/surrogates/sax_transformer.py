from lasts.surrogates.sax_tree import SaxTree
from lasts.utils import convert_numpy_to_sktime
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.utils.validation.panel import check_X_y

# from sktime.datatypes._panel._convert import from_3d_numpy_to_nested


class SaxTransformer(SaxTree):
    def __init__(
        self,
        labels=None,
        random_state=None,
        create_plotting_dictionaries=True,
        custom_config=None,
    ):
        super().__init__(
            labels=labels,
            random_state=random_state,
            create_plotting_dictionaries=create_plotting_dictionaries,
            custom_config=custom_config,
        )

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        X = convert_numpy_to_sktime(X)
        # X = from_3d_numpy_to_nested(X.reshape(X.shape[0], X.shape[2], X.shape[1]))
        # seql_model = MrSEQLClassifier(seql_mode='fs', symrep='sax', custom_config=self.custom_config)
        seql_model = MrSEQLClassifier(
            seql_mode="clf", symrep="sax", custom_config=self.custom_config
        )
        seql_model.fit(X, y)
        X, _ = check_X_y(X, y, coerce_to_numpy=True)
        mr_seqs = seql_model._transform_time_series(X)
        X_transformed = seql_model._to_feature_space(mr_seqs)
        self.X_transformed_ = X_transformed
        self.seql_model_ = seql_model
        if self.create_plotting_dictionaries:
            self._create_dictionaries()
        return self

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def explain(self, x, **kwargs):
        pass
