import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression  # noqa


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, records=False, pattern=None):
        self.columns = columns
        self.records = records
        self.pattern = pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")

        if self.pattern is not None:
            return X[[col for col in X.columns if self.pattern in col]]

        return X[self.columns]


class TotalBoughtUnitsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols, fillna=0):
        self.fillna = fillna
        self.cols = cols
        self.vocabulary_ = None

    def fit(self, X, y=None):
        X = X.loc[:, self.cols]
        X["_answer"] = y
        self.vocabulary_ = X.groupby(self.cols)["_answer"].apply(
            lambda x: x.sum()).to_dict()
        return self

    def transform(self, X):
        targets = list(zip(*X[self.cols].values.T))
        return np.array(list(map(self._translate, targets))).reshape(-1, 1)

    def _translate(self, target):
        return self.vocabulary_.get(target, self.fillna)


class DiscountExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, target, arg):
        self.target = target
        self.arg = arg
        self.vocabulary_ = None

    def fit(self, X, y=None):
        data = X.loc[:, [self.target, self.arg]]
        self.vocabulary_ = data.groupby(self.arg)[self.target].agg(
            lambda x: x.mode())
        return self

    def transform(self, X):
        target = X[self.arg]
        discount = target.map(self.vocabulary_) / X[self.target]
        return discount.values.reshape(-1, 1)


def categorical(colname):
    return make_pipeline(
        PandasSelector(colname, records=True),
        DictVectorizer(sparse=False),
        OneHotEncoder(categories='auto'),
    )


def build_model(classifier=LogisticRegression(solver="lbfgs")):
    model = make_pipeline(
        make_union(
            categorical([
                "i",
                "j"
            ]),
            TotalBoughtUnitsTransformer(["i", "j"]),
            DiscountExtractor(target="price", arg="j"),
            make_pipeline(
                PandasSelector(["t"]),
                FunctionTransformer(lambda x: x % 4, validate=True)
            ),
            make_pipeline(
                PandasSelector([
                    "t",
                    "advertised",
                    "price",
                    "weeks_passed_since_last_ad",
                ]),
                # StandardScaler()
            ),
            # PandasSelector(pattern="j_"),
        ),
        classifier
    )
    return model
