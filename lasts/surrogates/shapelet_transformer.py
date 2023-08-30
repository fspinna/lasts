from tslearn.shapelets import grabocka_params_to_shapelet_size_dict, LearningShapelets
from lasts.surrogates.shapelet_tree import ShapeletTree


class ShapeletTransformer(ShapeletTree):
    def __init__(
        self,
        labels=None,
        random_state=None,
        shapelet_model_kwargs={
            "l": 0.1,
            "r": 2,
            "optimizer": "sgd",
            "n_shapelets_per_size": "heuristic",
            "weight_regularizer": 0.01,
            "max_iter": 100,
        },
    ):
        super().__init__(
            labels=labels,
            random_state=random_state,
            shapelet_model_kwargs=shapelet_model_kwargs,
        )

    def fit(self, X, y):
        self.X_ = X  # shape : list(np.array, np.array...) np.array being a different dimension of the mts
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
        self.shapelet_model_ = shp_clf

        return self

    def transform(self, X):
        X_transformed = self.shapelet_model_.transform(X)
        return X_transformed

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def find_leaf_id(self, ts):
        pass

    def explain(self, x):
        pass
